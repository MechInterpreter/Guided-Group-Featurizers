"""Choose a recipe using development data, freeze it, then repeat fresh starts."""
import copy
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training,run as native_run
from coco_adaptation import run,inputs


def config(device='cuda'):
    cfg=run.config('pilot')
    cfg.update(version=1,study='coco_stratified_confirmation',device=device,adaptation_variants=['unanchored'],
        images=dict(train=2000,select=400,confirm=400,mapping=400,test=800),
        minimum_images_per_category=dict(train=40,select=10,confirm=10,mapping=10,test=20),
        minimum_categories=60,candidate_images_per_category=600,scan_chunk=100,split_seed=62001,
        discovery_seed=1,confirmation_seeds=[2,3,4],fresh_test_images=800,fresh_test_seed=62002,
        gallery_seed=62003,gallery_per_category=1,bootstrap=200,proposal_images=2000,
        warm_updates=1500,refit_updates=500)
    return cfg


def native_start(master,packs,base,method,seed,lrs,cfg,source_sha):
    current=copy.deepcopy(base);current.update(methods=[method],seeds=[seed],lrs=lrs,device=cfg['device'],images=cfg['images'],dataset='COCO train2017 custom stratified splits')
    s=inputs.JournalStore(master.output/f'native_{method}_s{seed}',dict(config=current,source_sha=source_sha,
        data_plan=master.config['data_plan'],numerical_versions=master.config['numerical_versions']),
        master.rescue/f'native_{method}_s{seed}',master.require_drive)
    x=packs['train']['x'];selection=packs['select']['x']
    choices=native_run.train_baselines(s,x.reshape(-1,x.shape[-1]),selection.reshape(-1,x.shape[-1]),current)
    if not choices[0]['sparsity_matched']:
        s.write('sparsity_failure.json',choices[0]);raise RuntimeError(f'{method} seed {seed}: no matched-activity native checkpoint. Stop before adaptation; no fallback model is substituted.')
    return s,choices[0]


def arm_store(master,native,method,seed,cfg,source_sha):
    c=copy.deepcopy(cfg);c['methods']=[method]
    return run.build_store(master.output/f'adaptation_{method}_s{seed}',master.rescue/f'adaptation_{method}_s{seed}',c,native,source_sha,master.require_drive)


def fit_branch(master,packs,base,provenance,method,seed,lrs,cfg,source_sha):
    master.event('branch_start',method=method,seed=seed)
    native,choice=native_start(master,packs,base,method,seed,lrs,cfg,source_sha)
    store=arm_store(master,native,method,seed,cfg,source_sha);current=store.config['config']
    pp=dict(packs)
    run.prepare_protection(store,native,[choice],pp,provenance,current)
    arms=run.train_all(store,native,[choice],pp,provenance,current)
    master.event('branch_frozen',method=method,seed=seed,accepted_edits=sum(a.get('accepted_edits',0) for a in arms))
    return dict(store=store,native=native,choice=choice,arms=arms,cfg=current)


def select_recipe(master,branches,packs):
    rows=[]
    for name,b in branches.items():
        cfg=b['cfg'];store=b['store'];base=b['native'].config['config'];method=b['choice']['method']
        arms={a['arm']:a for a in b['arms']};targets=store.load(f"{method}_s{b['choice']['seed']}_targets")
        values={}
        for label in ('native','supervised_adaptive','supervised_fixed_equal_search_compute'):
            w=store.load(arms[label]['key'])['best'];values[label]=run.assess(w,method,base,packs['confirm'],targets,cfg)
        from coco_adaptation import core
        decision=core.compare(values['supervised_adaptive'],values['supervised_fixed_equal_search_compute'],values['native'],cfg,base,method)
        gain_capture=values['supervised_adaptive']['macro_capture']-values['supervised_fixed_equal_search_compute']['macro_capture']
        gain_ap=values['supervised_adaptive']['macro_ap']-values['supervised_fixed_equal_search_compute']['macro_ap']
        reasons=list(decision['reasons'])
        if arms['supervised_adaptive']['accepted_edits']==0:reasons.append('no_structural_edit_accepted')
        if gain_capture<=0:reasons.append('capture_not_improved_over_equal_budget_control')
        if gain_ap<0:reasons.append('detection_declined_over_equal_budget_control')
        rows.append(dict(method=method,lr=b['choice']['lr'],eligible=not reasons,reasons=reasons,capture_gain=gain_capture,ap_gain=gain_ap,
                         composite_gain=decision['composite_gain'],full_decision=decision))
    eligible=[r for r in rows if r['eligible']]
    chosen=max(eligible,key=lambda r:(r['composite_gain'],r['capture_gain'],r['method'])) if eligible else None
    record=dict(candidates=rows,selected=None if chosen is None else {k:chosen[k] for k in ('method','lr')},
        rule='Confirmation/development only: accepted structural edit, positive capture, nonnegative AP, existing reconstruction/activity/category gates, then largest composite gain. No test results used.',
        no_go=chosen is None,confirmation_reused=True)
    master.save('recipe_frozen',record);master.write('recipe_selection.json',record);return record


def train(master,packs,base,provenance,cfg,source_sha):
    branches={}
    for method in cfg['methods']:
        key=f"{method}_s{cfg['discovery_seed']}"
        branches[key]=fit_branch(master,packs,base,provenance,method,cfg['discovery_seed'],base['lrs'],cfg,source_sha)
    choice=master.load('recipe_frozen')
    if choice is None:choice=select_recipe(master,branches,packs)
    if choice['selected'] is not None:
        selected=choice['selected']
        for seed in cfg['confirmation_seeds']:
            key=f"{selected['method']}_s{seed}"
            branches[key]=fit_branch(master,packs,base,provenance,selected['method'],seed,[selected['lr']],cfg,source_sha)
    else:master.event('no_eligible_recipe',message='No recipe cleared the declared development checks. Skip repeated fits; audit every discovery branch and export the failures.')
    master.save('all_models_frozen',dict(models={k:b['arms'] for k,b in branches.items()},recipe=choice,test_used=False))
    return branches,choice


def evaluate(master,branches,packs,test,provenance,cfg):
    reports={};paired=[]
    for key,b in branches.items():
        rows=run.evaluate_all(b['store'],b['native'],b['arms'],packs,test,b['cfg'],provenance)
        run.plot(b['store'],rows);reports[key]=rows
        indexed={r['summary']['arm']:r for r in rows}
        a=indexed['supervised_adaptive'];fixed=indexed['supervised_fixed_equal_search_compute']
        aa={r['category_index']:r for r in a['categories']};ff={r['category_index']:r for r in fixed['categories']}
        deltas={};per_category=[]
        for kind,field in [('contrast','single_group_capture'),('detection','ap')]:
            missing=[c for c in cfg['category_indices'] if aa[c][kind] is None or ff[c][kind] is None]
            if missing:raise ValueError(f'Frozen category cohort missing {kind} metrics: {missing}; do not change the denominator after seeing results')
            values=[aa[c][kind][field]-ff[c][kind][field] for c in cfg['category_indices']]
            deltas[kind+'_gain']=float(np.mean(values))
            per_category.extend(dict(category=aa[c]['category'],metric=field,delta=float(v)) for c,v in zip(cfg['category_indices'],values))
        rec=a['summary']['reconstruction'];control=fixed['summary']['reconstruction']
        protected=json.loads((b['store'].results/'protected_group_fresh_audit.json').read_text())
        guard=next(r for r in protected['rows'] if r['arm']=='supervised_adaptive')
        paired.append(dict(branch=key,seed=b['choice']['seed'],method=b['choice']['method'],
            phase='discovery' if b['choice']['seed']==cfg['discovery_seed'] else 'fresh_start_confirmation',
            **deltas,category_deltas=per_category,accepted_edits=a['summary']['accepted_edits'],
            reconstruction_relative_mse_change=rec['mse']/control['mse']-1,active_group_difference=rec['active_groups']-control['active_groups'],
            protected_original_groups_preserved=guard['passed'],protected_pairs=len(guard['checks'])))
    repeated=[r for r in paired if r['phase']=='fresh_start_confirmation']
    summary=dict(comparisons=paired,confirmation_counts=dict(seeds=len(repeated),
        seeds_accepting_edits=sum(r['accepted_edits']>0 for r in repeated),
        seeds_improving_capture_and_ap=sum(r['contrast_gain']>0 and r['detection_gain']>0 for r in repeated),
        seeds_improving_and_preserving=sum(r['contrast_gain']>0 and r['detection_gain']>0 and r['protected_original_groups_preserved'] and r['reconstruction_relative_mse_change']<=cfg['accept_reconstruction_tolerance'] for r in repeated)),
        models=[r['summary'] for rows in reports.values() for r in rows],
        scope='Same coverage-enriched COCO splits and fixed category cohort. Discovery seed is excluded from repetition counts. All results, including failed seeds, retained. This does not measure whole concept-manifold recovery.')
    master.write('summary.json',summary)
    return reports,summary


def export(master,branches,source):
    path=master.output/f'agf_coco_stratified_{master.run_id[:12]}_results.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        roots=[('results',master.results),('source',Path(source))]
        for name,b in branches.items():roots.extend([(name+'/adaptation',b['store'].results),(name+'/native',b['native'].results)])
        for prefix,root in roots:
            for p in root.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,prefix+'/'+p.relative_to(root).as_posix())
    return path
