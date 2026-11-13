"""Frozen dependency exchanges from completed stronger mask-initialized GGF."""
import copy,csv,gc,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import inputs
from coco_baselines import training,metrics
from coco_baselines.storage import identity,sha_file
from coco_coordinated import run as memory
from . import mask_recovery,large_mask,mature_regroup as mature
from . import accumulating_regroup as accum,dependency_regroup as dep
from . import direction_regroup as prior,bank_experiment as bank
from . import fresh_preservation as fresh,independent_regroup as independent
from . import downstream_fidelity as fidelity,study
from . import strong_mask_planner as planner

def load_saved_confirmation(ctx):
    """Load the omitted historical split; never generate replacement data."""
    provenance = ctx['provenance']
    expected = list(provenance['splits']['confirm'])
    if not expected or len(expected) != len(set(expected)):
        raise ValueError('Historical confirmation IDs are empty or duplicated')
    for name, ids in provenance['splits'].items():
        if name != 'confirm' and set(expected).intersection(ids):
            raise ValueError('Historical confirmation overlaps ' + name)
    original = ctx['original']
    if original.run_id != ctx['source'].config['parent_run_id']:
        raise ValueError('Original activation lineage changed')
    norm = memory.required(original, 'normalization')
    if identity(norm) != identity(ctx['norm']):
        raise ValueError('Training normalization changed')
    ctx['store'].event('loading_saved_confirmation', images=len(expected))
    pack = memory.normalized_pack(
        original, 'confirm', provenance, norm, ctx['local'],
        original.config['config']['extract_chunk'])
    if list(pack['ids']) != expected:
        raise ValueError('Historical confirmation order changed')
    memory.assert_immutable(ctx['stamps'])
    ctx['packs']['confirm'] = pack
    for arm in ctx.get('arms', {}).values():
        arm['packs']['confirm'] = pack
    ctx['store'].write('saved_confirmation_input_fix.json', dict(
        fix='load_original_confirm_v1', activation_run=original.run_id,
        images=len(expected), ids_sha=identity(expected),
        cache_identity=pack['identity'], raw_float32_sha=pack['raw_float32_sha'],
        role='Historical development only; not a fresh test',
        new_extractions=0, new_training_updates=0))
    print('Loaded original confirmation:', len(expected), 'images. Rerun the failed planning cell and continue.')
    return pack


RECOVERY_ID='766a3dfee0e3'  # Full content-derived identity checked against its manifest below.
RECOVERY_SHA='a3f1e0df207a3637b866cda4b0faed0f9e8514dbffd5ed3e8d20402783d31e30'

def settings():
    return dict(version=3,study='strong_mask_ggf_dependency_exchange_confirmation',seeds=[21,22,23],
        coverage_protocol='historical_development_fresh_test_v3',strict_evidence_quotas=True,
        development_source='saved_original_confirm',
        start_arm='control',start_field='weights',start_updates=500,
        development_images=0,test_images=1500,development_min_images=0,test_min_images=20,
        plan_seed=106071,coverage_reserve=1.2,extension_limit_per_category=1500,
        solver_seconds=60,extract_chunk=20,extract_batch=8,bootstrap=200,
        recipe=mature.settings(),ridge_grid=[.001,.01,.1,1.],task_ap_margin=.001,
        source_choice='Retrospective exploratory choice of ordinary continuation; identical arm/endpoint for all seeds, no per-seed winners.',
        scope='One remeasured training graph/pass per seed; eight trials, at most five nonconflicting exchanges; no gradients or baseline retraining. Original saved development selects; all seeds freeze before untouched test. Legacy native +2% MSE eligibility remains a separate required flag for an official upgrade.')

def collect_ids(value):
    """Only explicitly named image-ID fields; never confuse category/atom IDs."""
    out=set()
    if isinstance(value,dict):
        for key,v in value.items():
            if key in ('ids','splits','image_ids'):
                def add(x):
                    if isinstance(x,dict):
                        for y in x.values():add(y)
                    elif isinstance(x,list):
                        for y in x:
                            if isinstance(y,(int,np.integer)):out.add(int(y))
                            elif isinstance(y,(list,dict)):add(y)
                add(v)
            else:out.update(collect_ids(v))
    elif isinstance(value,list):
        for v in value:out.update(collect_ids(v))
    return out

def prepare(source_output,parent_output,comparison_output,confirmation_output,recovery_output,
            output,local,sha,versions,device,history_root,require_drive=True):
    # Existing verified loader reads cached activations/native controls. It trains nothing.
    ctx=mask_recovery.prepare(source_output,parent_output,comparison_output,confirmation_output,
        Path(output)/'verified_inputs',local,sha,versions,mask_recovery.settings(),device,require_drive)
    matches=list((Path(recovery_output)/'results').glob(RECOVERY_ID+'*/manifest.json'))
    if len(matches)!=1:raise FileNotFoundError('Expected one completed recovery manifest, prefix '+RECOVERY_ID)
    manifest=json.loads(matches[0].read_text());run=manifest['run_id']
    source=memory.FrozenStore(Path(recovery_output)/'checkpoints'/run[:20],Path(local)/'read_recovery')
    if source.run_id!=run or source.config['source_sha']!=RECOVERY_SHA:
        raise ValueError('Wrong recovery source identity')
    source_frozen=memory.required(source,'all_models_frozen')
    if not source_frozen['complete'] or not memory.required(source,'recovery_summary')['complete']:
        raise ValueError('Recovery study must be complete; no retraining fallback')
    if source.config['parent_run']!=ctx['parent_mask'].run_id or source.config['original_control_run']!=ctx['source'].run_id:
        raise ValueError('Recovery parent/control lineage differs')
    if source.config['config']!=mask_recovery.settings():raise ValueError('Recovery protocol changed')
    if large_mask.training_versions(source.config['versions'])!=large_mask.training_versions(versions):
        raise ValueError('Training numerical versions differ')
    if source.config['device']!=device:raise ValueError('Use the same device category as the source')
    roots=[Path(x).resolve() for x in (source_output,parent_output,comparison_output,confirmation_output,recovery_output)]
    dest=Path(output).resolve()
    if any(dest==p or dest in p.parents or p in dest.parents for p in roots):raise ValueError('Output overlaps read-only inputs')
    cfg=settings();ctx.update(strong_cfg=cfg,fresh_cfg=cfg,cohort=ctx['cfg']['category_indices'],local=Path(local))
    ctx['norm']=memory.required(ctx['original'],'normalization')
    ctx['stamps']+=memory.source_snapshot([source]);ctx['strong_source']=source
    s=inputs.JournalStore(output,dict(config=cfg,source_sha=sha,versions=versions,device=device,
        recovery_run=run,activation_run=ctx['original'].run_id),Path(local)/'rescue',require_drive)
    ctx['store']=s;ctx['arms']={};catalog=[]
    for seed in cfg['seeds']:
        st=memory.required(source,f'recovery_s{seed}_control')
        if st['updates']!=500 or st['target']!=500:raise ValueError('Expected completed ordinary 500-update endpoint')
        row=next(r for r in source_frozen['models'] if r['seed']==seed and r['arm']=='control')
        if row['weights_sha']!=training.weight_sha(st['weights']):raise ValueError('Final endpoint differs from frozen source catalog')
        parent=memory.required(ctx['parent_mask'],f'large_mask_ggf_s{seed}')
        if st['initial_sha']!=training.weight_sha(parent['weights']) or st['reference_mse']!=parent['reference_mse']:
            raise ValueError('Endpoint start or native reconstruction reference changed')
        if tuple(st['head'].shape)!=(75,2) or not torch.isfinite(st['head']).all() or not all(torch.isfinite(w).all() for w in st['weights'].values()):
            raise ValueError('Nonfinite endpoint or wrong trained head')
        del parent
        targets=memory.required(ctx['parent_mask'],f'large_mask_targets_s{seed}')
        if identity(targets)!=st['target_sha']:raise ValueError('Source training readouts changed')
        snap=bank.native_snapshot(ctx,st['weights'],targets,st['head'],1024,3,16)
        rule=dict(cfg['recipe'],seed=seed,run_recovery=False)
        store=inputs.JournalStore(Path(output)/f'seeds/{seed}',dict(seed=seed,source_sha=sha,
            start_sha=accum.weight_sha(snap),spec_sha=identity(snap['spec']),recipe=rule),Path(local)/f'seed{seed}',require_drive)
        arm=dict(ctx,store=store,initial=snap,protocol=rule,prior=None,seed=seed,
            source_weights_sha=accum.weight_sha(snap),native_reference_mse=st['reference_mse'])
        arm['cfg']=dict(ctx['cfg'],**{k:rule[k] for k in ('lr','snr','lambda_capture','lambda_detection','grad_clip','images_per_update')})
        ctx['arms'][seed]=arm;store.save('verified_start',snap)
        v=st['history'][-1]['validation']
        row=dict(seed=seed,key=f'recovery_s{seed}_control',field='weights',weights_sha=accum.weight_sha(snap),
            head_sha=training.weight_sha({'head':st['head']}),targets_sha=identity(targets),guided_updates=3000,
            native_reference_mse=st['reference_mse'],native_MSE_limit=st['reference_mse']*1.02,
            source_development_mse=v['reconstruction']['mse'],legacy_eligible=bool(v['reconstruction']['mse']<=st['reference_mse']*1.02))
        catalog.append(row);del st;gc.collect()
    s.write('source_catalog.json',catalog);ctx['catalog']=catalog
    # Read all historical AGF/GGF split plans, plus original and scaled splits.
    used=set(collect_ids(ctx['provenance']));used.update(collect_ids(memory.required(ctx['source'],'fresh_plan')))
    files=[];historical_groups={}
    required=['AGF_COCO_independent_regroup_v1','AGF_COCO_repeated_loop_v1','AGF_COCO_fresh_preservation_v1']
    for name in required:
        if not (Path(history_root)/name/'results').is_dir():raise FileNotFoundError('Historical exclusion source missing: '+name)
    for folder in sorted(Path(history_root).iterdir()):
        if not folder.is_dir() or not folder.name.startswith(('AGF_COCO_','GGF_COCO_')) or folder.resolve()==dest:continue
        result_root=folder/'results'
        if not result_root.exists():continue
        for p in sorted(result_root.rglob('*.json')):
            if p.name not in ('split_roles.json','provenance.json','fresh_split_plan.json','fresh_plan.json','split_plan.json','fresh_activation_identity.json','confirm_activation_identity.json','test_activation_identity.json'):continue
            content=json.loads(p.read_text());ids=collect_ids(content)
            if ids:
                used.update(ids);files.append(dict(path=str(p),sha=sha_file(p),images=len(ids)))
                historical_groups[str(p.relative_to(history_root))]=sorted(ids)
    if not files:raise ValueError('No historical split records found')
    old=s.load('exclusions')
    record=dict(ids=sorted(used),files=files,ids_sha=identity(sorted(used)))
    if old is not None and old!=record:raise ValueError('Historical exclusions changed on resume; do not change inputs mid-run')
    s.save('exclusions',record);s.write('historical_exclusions.json',record)
    # Reuse the tested joint quota planner/extractor through explicit adapters.
    ctx['excluded_groups']=historical_groups;ctx['excluded_ids']=used
    s.write('protocol_settings.json',dict(settings=cfg,source_catalog=catalog,
        inference='Signed native Top-16, fixed width3; encoder column, bias and decoder atom move together. No scalar bank, coefficient refit, head training or dimension change.',
        gates='Actual TopK capture must increase on training and BOTH saved development sets; current/root AP and VE lose at most0.1pp, MSE grows at most2%, every category capture/AP loses at most1pp. Native reconstruction+2% eligibility is also logged unchanged; failed roots cannot become eligible by relabeling.',
        final='All three selections frozen before test extraction. Fixed training readouts primary; common historical calibration best-group metrics secondary. One shared fresh test, three trained starts. No test-driven fallback or retry.',
        comparison='Exact stronger GGF vs frozen AGF at identical capacity and gradient history; search costs are additional, logged separately. No wall-time-matched or capacity-matched SLiCS claim.'))
    load_saved_confirmation(ctx)
    memory.assert_immutable(ctx['stamps']);return ctx

def exclusions(ctx):return ctx['excluded_groups'],ctx['excluded_ids']

def plan(ctx,coco):
    return planner.plan(ctx,coco)

def verify(ctx,frozen):
    for r in frozen['models']:
        a=ctx['arms'][r['seed']]
        for key,expected in [('verified_start',r['root_sha']),('final_agf',r['final_sha'])]:
            if accum.weight_sha(memory.required(a['store'],key))!=expected:raise ValueError('Frozen model identity changed')

def extract(ctx,coco,token_callback,split):
    if split=='confirm':
        pack=ctx['packs']['confirm'];s=ctx['store'];plan=memory.required(s,'fresh_plan')
        planner.validate(plan,ctx['excluded_ids'],ctx['cohort'],ctx['fresh_cfg'])
        if pack['ids']!=plan['splits']['confirm']:raise ValueError('Historical development changed')
        observed=fresh.coverage.evidence(pack,ctx['cfg']).sum(0)
        if not np.array_equal(observed,np.asarray(plan['coverage']['split_evidence_counts']['confirm'])):
            raise ValueError('Historical development mask coverage changed')
        stamp=dict(ids=pack['ids'],normalized_sha=memory.streaming_array_sha(pack['x']),
                   normalization_sha=identity(ctx['norm']),role='Reused historical development; selection only')
        old=s.load('activation_identity_confirm')
        if old is not None and old!=stamp:raise ValueError('Historical development identity changed')
        s.save('activation_identity_confirm',stamp);s.write('confirm_activation_identity.json',stamp)
        s.event('historical_development_reused',images=len(pack['ids']),new_backbone_extractions=0)
        return pack
    old_ex=fresh.excluded_ids;old_verify=fresh.verify_frozen;old_validate=fresh.validate_plan
    fresh.excluded_ids=exclusions;fresh.verify_frozen=verify;fresh.validate_plan=planner.validate
    try:
        pack=fresh.extract(ctx,coco,token_callback,split)
        observed=fresh.coverage.evidence(pack,ctx['cfg']).sum(0)
        declared=memory.required(ctx['store'],'fresh_plan')['coverage']['split_evidence_counts'][split]
        if not np.array_equal(observed,np.asarray(declared)):
            raise ValueError('Extracted per-evidence coverage differs from frozen plan')
        return pack
    finally:
        fresh.excluded_ids=old_ex;fresh.verify_frozen=old_verify;fresh.validate_plan=old_validate

@torch.no_grad()
def select_arm(arm):
    s=arm['store'];saved=s.load('selection_frozen')
    if saved is not None:
        if saved['root_sha']!=accum.weight_sha(arm['initial']):raise ValueError('Resumed start changed')
        if accum.weight_sha(memory.required(s,'final_agf'))!=saved['final_sha']:raise ValueError('Resumed selection changed')
        return saved
    screen=dep.training_screen(arm);view=dict(arm,packs=dict(arm['packs'],train_screen=screen))
    splits=('select','confirm');root=arm['initial'];current=root;rule=arm['protocol']
    train_root=accum.measured(view,'train_root',root,'train_screen');train_current=train_root
    refs={k:accum.measured(arm,k+'_root',root,k) for k in splits};current_reports=dict(refs)
    steps=[];accepted=[];groups=set();targets=set()
    order=accum.proposal_order(arm['saved_screen'],rule);s.write('proposal_order.json',order)
    graph=memory.required(s,'dependency_graph') if order else None
    for i,p in enumerate(order):
        key=f'exchange{i:02d}';before=accum.weight_sha(current);row=s.load(key)
        if row is not None:
            if row['before_sha']!=before or row['proposal']!=p:raise ValueError('Resume decision chain changed')
        else:
            skip=accum.conflict(p,groups,targets)
            if len(accepted)>=rule['max_accepted']:skip='maximum_accepted_reached'
            row=dict(proposal=p,before_sha=before,accepted=False,skip=skip,checks={})
            if skip is None:
                candidate=prior.permute_snapshot(current,p['outgoing'],p['incoming'])
                tr=accum.measured(view,key+'_train',candidate,'train_screen')
                tg=accum.positive_gate(tr,train_current,train_root,rule,p['category_index'])
                row.update(train_report=tr,train_gate=tg,candidate_sha=accum.weight_sha(candidate))
                if tg['passed']:
                    for split in splits:
                        r=accum.measured(arm,key+'_'+split,candidate,split)
                        gate=accum.positive_gate(r,current_reports[split],refs[split],rule)
                        row['checks'][split]=dict(report=r,gate=gate,deltas_vs_root=accum.category_deltas(r,refs[split]))
                    row['accepted']=all(v['gate']['passed'] for v in row['checks'].values())
                    row['skip']=None if row['accepted'] else 'development_preservation_or_gain_failed'
                else:row['skip']='training_preservation_or_gain_failed'
                m=prior.model(arm,candidate)
                row['gate_effects']=dep.gate_effects(arm,m,screen,graph,p);del m;gc.collect()
                if row['accepted']:s.save(key+'_accepted',candidate)
            row['after_sha']=row.get('candidate_sha') if row['accepted'] else before
            s.save(key,row)
        if row['accepted']:
            current=memory.required(s,key+'_accepted')
            if accum.weight_sha(current)!=row['after_sha']:raise ValueError('Accepted snapshot changed')
            accepted.append(p);groups.update((p['receiver_group'],p['donor_group']));targets.add(p['category_index'])
            train_current=row['train_report'];current_reports={k:v['report'] for k,v in row['checks'].items()}
        elif accum.weight_sha(current)!=before:raise AssertionError('Rejected edit changed current model')
        steps.append(row);s.write('steps/'+key+'.json',row)
        s.event('exchange_decision',seed=arm['seed'],done=i+1,total=len(order),accepted=row['accepted'],skip=row['skip'],
            train_failures=row.get('train_gate',{}).get('failures',[]),development_failures={k:v['gate']['failures'] for k,v in row['checks'].items()})
    s.save('final_agf',current);s.save('accumulated_snapshot',current)
    legacy=current_reports['select']['reconstruction']['mse']<=arm['native_reference_mse']*1.02
    out=dict(seed=arm['seed'],complete=True,root_sha=accum.weight_sha(root),final_sha=accum.weight_sha(current),
        accepted_proposals=accepted,steps=steps,development_reports=current_reports,
        legacy_native_eligible=bool(legacy),native_limit=arm['native_reference_mse']*1.02,
        selected_architecture='AGF' if accepted else 'GGF unchanged',gradient_updates=0,test_used=False)
    s.save('selection_frozen',out);s.write('selection_frozen.json',out);return out

@torch.no_grad()
def freeze(ctx,confirm):
    s=ctx['store'];old=s.load('all_models_frozen')
    if old is not None:verify(ctx,old);return old
    if confirm['ids']!=memory.required(s,'fresh_plan')['splits']['confirm']:raise ValueError('Wrong development images')
    if memory.streaming_array_sha(confirm['x'])!=memory.required(s,'activation_identity_confirm')['normalized_sha']:raise ValueError('Development activations changed')
    rows=[];start=time.perf_counter()
    for seed,arm in ctx['arms'].items():
        arm['packs']=dict(arm['packs'],confirm=confirm)
        flight=prior.preflight(arm)
        if not flight['numerical_ok']:raise ValueError('Numerical preflight failed')
        mature.screen_and_graph(arm)
        rows.append(select_arm(arm))
        arm['store'].write('next_action.json',dict(
            stopping_reason='accepted cap reached' if len(rows[-1]['accepted_proposals'])>=arm['protocol']['max_accepted'] else 'bounded training-ranked shortlist exhausted',
            next='Remeasure training dependencies before any additional pass; no extra graph or retry in this run.',
            evidence='All direction profiles, initial removal dependencies, trial gate changes and per-category losses are retained.',
            no_global_convergence_claim=True))
    out=dict(complete=True,models=rows,test_used=False,seconds=time.perf_counter()-start,gradient_updates=0)
    s.save('all_models_frozen',out);s.write('all_models_frozen.json',out);verify(ctx,out)
    memory.assert_immutable(ctx['stamps']);return out

def fit_probe(ctx):
    s=ctx['store'];old=s.load('probe')
    if old is not None:return old
    packs={}
    for split in ('train','select'):
        p=ctx['packs'][split];x=np.array([np.asarray(xx,np.float64).mean(0) for xx in p['x']])
        packs[split]=dict(x=x,y=fidelity.presence(p['masks'],ctx['cohort']))
    model,history=fidelity.fit_probe(packs['train'],packs['select'],ctx['strong_cfg']['ridge_grid'])
    out=dict(model=model,history=history,training_ids_sha=identity(ctx['packs']['train']['ids']),
        select_ids_sha=identity(ctx['packs']['select']['ids']))
    s.save('probe',out);s.write('probe_training.json',dict(history=history,training_ids_sha=out['training_ids_sha'],select_ids_sha=out['select_ids_sha'],weights_sha=training.weight_sha({'weights':torch.from_numpy(model['weights'])})))
    return out

def task_scores(arm,snap,test,probe,key):
    s=arm['store'];old=s.load(key)
    sha=accum.weight_sha(snap);sig=identity(dict(weights=sha,ids=test['ids'],probe=probe))
    if old is not None:
        if old['signature']!=sig:raise ValueError('Task cache identity changed')
        return old['scores']
    m=prior.model(arm,snap);m.eval();values=[]
    with torch.inference_mode():
        for i,x in enumerate(test['x']):
            xx=torch.as_tensor(np.array(x,copy=True),device=arm['device'])
            values.append(m(xx)[0].double().mean(0).cpu().numpy())
            if (i+1)%100==0:s.event('task_inference',key=key,done=i+1,total=len(test['ids']))
    del m;gc.collect();scores=fidelity.predict(probe,np.array(values));s.save(key,dict(signature=sig,scores=scores));return scores

def evaluate(ctx,test):
    master=ctx['store'];frozen=memory.required(master,'all_models_frozen');verify(ctx,frozen)
    if test['ids']!=memory.required(master,'fresh_plan')['splits']['test']:raise ValueError('Wrong fresh test')
    if memory.streaming_array_sha(test['x'])!=memory.required(master,'activation_identity_test')['normalized_sha']:raise ValueError('Test activations changed')
    cached=master.load('experiment_complete')
    if cached is not None:return cached
    probe=fit_probe(ctx)['model'];y=fidelity.presence(test['masks'],ctx['cohort'])
    clean=fidelity.predict(probe,np.array([np.asarray(x,np.float64).mean(0) for x in test['x']]))
    clean_ap=fidelity.category_ap(y,clean);rows=[]
    for item in frozen['models']:
        arm=ctx['arms'][item['seed']];s=arm['store'];saved=s.load('fresh_result')
        if saved is not None:rows.append(saved);continue
        arm['packs']=dict(arm['packs'],fresh=test)
        root=memory.required(s,'verified_start');edit=memory.required(s,'final_agf')
        a=accum.measured(arm,'fresh_control',root,'fresh');b=accum.measured(arm,'fresh_agf',edit,'fresh')
        gate=accum.positive_gate(b,a,a,arm['protocol']);paired=independent.fresh_pair(arm,edit,root)
        control_scores=task_scores(arm,root,test,probe,'task_ggf');edited_scores=task_scores(arm,edit,test,probe,'task_agf')
        task=fidelity.paired_ap(y,edited_scores,control_scores,200,106072)
        task.update(clean_macro_ap=float(clean_ap.mean()),ggf_macro_ap=float(fidelity.category_ap(y,control_scores).mean()),agf_macro_ap=float(fidelity.category_ap(y,edited_scores).mean()))
        task['preserved']=bool(task['mean_gain']>=-ctx['strong_cfg']['task_ap_margin'])
        row=dict(seed=item['seed'],selection=item,ggf=a,agf=b,fresh_gate=gate,paired=paired,downstream=task,
            capture_gain=b['macro_capture']-a['macro_capture'],ap_gain=b['macro_ap']-a['macro_ap'],ve_gain=b['reconstruction']['ve']-a['reconstruction']['ve'],
            deltas=accum.category_deltas(b,a),all_failures=gate['failures'],test_used_for_selection=False)
        s.save('fresh_result',row);s.write('fresh_result.json',row);rows.append(row)
        master.event('fresh_seed_complete',seed=item['seed'],capture_gain_pp=100*row['capture_gain'],ap_gain_pp=100*row['ap_gain'],guard=gate['passed'],task_preserved=task['preserved'])
    advantage=all(r['selection']['accepted_proposals'] and r['fresh_gate']['passed'] and r['downstream']['preserved'] for r in rows)
    out=dict(complete=True,results=rows,all_three_structural_advantage=bool(advantage),
        official_upgrade_eligible=bool(advantage and all(r['selection']['legacy_native_eligible'] for r in rows)),
        native_reference_guard_unchanged=True,final_test_images=len(test['ids']),fresh_development_images=0,
        historical_development_images=len(ctx['packs']['confirm']['ids']),
        development_role='Reused original confirmation split; selection only, not an independent new confirmation',
        decision_frozen_before_test=True,new_gradient_updates=0,
        scope='Same capacity/gradient history; extra graph/scoring compute is NOT free or wall-time matched. Fixed training readouts. Single-group object contrast is a proxy, not full concept recovery. Source selection was retrospective. No capacity-matched SLiCS superiority claim.')
    master.save('experiment_complete',out);master.write('summary.json',out);charts(ctx,rows)
    memory.assert_immutable(ctx['stamps']);return out

def charts(ctx,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ctx['store'].results;tab=[]
    for row in rows:
        for arm in ('ggf','agf'):tab.append(dict(seed=row['seed'],arm=arm,**bank.scalar(row[arm])))
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(tab[0]));w.writeheader();w.writerows(tab)
    (folder/'figures').mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,key,title in zip(axes,('capture_gain','ap_gain','ve_gain'),('Capture gain','Detection AP gain','Reconstruction VE change')):
        ax.bar([str(r['seed']) for r in rows],[100*r[key] for r in rows]);ax.axhline(0,c='gray');ax.set_title(title);ax.set_ylabel('percentage points')
    fig.tight_layout();fig.savefig(folder/'figures/fresh_gains.png',dpi=160);plt.close(fig)

def export(ctx):
    s=ctx['store'];path=s.output/f'agf_strong_mask_{s.run_id[:12]}_results.zip';hashes={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for prefix,root in [('results',s.results)]+[(f'seeds/{k}',a['store'].results) for k,a in ctx['arms'].items()]:
            for p in sorted(root.rglob('*')):
                if p.is_file():
                    name=prefix+'/'+p.relative_to(root).as_posix();z.write(p,name);hashes[name]=sha_file(p)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise OSError('Export integrity failed')
    return path
