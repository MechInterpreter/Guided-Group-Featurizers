"""Frozen-recipe seed22/23 confirmation; seed21 is imported, never retrained."""
import copy
import csv
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from coco_baselines import training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from . import mask_initialization as pilot, study

PILOT_ID='324773f2e1952aa37a2c5573a5667dc3c7fd3c42aa8dbc1885b6e3ffc703e662'
PILOT_SOURCE='96fb13897c3a99ab9ecfdc49fb59037b3218cd089281858513e17973bdc2b259'
PILOT_WEIGHT='d07caf814cfe5bd2d76647032b6de176f245254f92e433345fdde67e4d347d18'
NEW_SEEDS=[22,23]
ALL_SEEDS=[21,22,23]
ARMS=['continued Vanilla','native-init GGF','SLiCS mask','SVD mask','mask-init GGF']

def settings():
    return dict(version=1,study='ggf_mask_initialization_confirmation',seeds=NEW_SEEDS.copy(),
                evaluate_every=500,bootstrap=200)

def prepare(source_output,parent_output,pilot_output,output,local,sha,versions,cfg,device,require_drive=True):
    if cfg!=settings():raise ValueError('Frozen seed22/23 confirmation recipe; do not tune after the pilot')
    p=memory.FrozenStore(Path(pilot_output)/'checkpoints'/PILOT_ID[:20],Path(local)/'read_pilot')
    if p.run_id!=PILOT_ID or p.config['source_sha']!=PILOT_SOURCE:
        raise ValueError('Wrong successful seed21 pilot')
    if p.config['versions']!=versions or p.config['device']!=device:
        raise ValueError('Match the pilot numerical versions/device type before confirmation')
    if p.config['config']!=dict(version=1,study='ggf_mask_initialization_pilot',seeds=[21],evaluate_every=500,bootstrap=200):
        raise ValueError('Pilot configuration changed')
    f=memory.required(p,'comparison_frozen')
    if not f['complete'] or {r['seed'] for r in f['models']}!={21}:
        raise ValueError('Pilot models incomplete')
    r=next(r for r in f['models'] if r['arm']=='mask-init GGF')
    state=memory.required(p,r['key'])
    if r['weights_sha']!=PILOT_WEIGHT or training.weight_sha(state[r['field']])!=PILOT_WEIGHT:
        raise ValueError('Successful pilot checkpoint changed')
    memory.required(p,'comparison_summary')
    for r in f['models']:
        for prefix in ('report_','compact_','mapping_'):
            memory.required(p,prefix+r['key']+'_top16')
    a,b=Path(output).resolve(),Path(pilot_output).resolve()
    if a==b or a in b.parents or b in a.parents:raise ValueError('Pilot input must remain read-only')
    ctx=pilot.prepare(source_output,parent_output,output,local,sha,versions,cfg,device,require_drive)
    ctx['completed_pilot']=p;ctx['stamps']+=memory.source_snapshot([p])
    # Preflight cached fitted comparators and init controls now; no solver or new fitting.
    old=ctx['parent_comparison'];frozen=memory.required(old,'comparison_frozen')
    for seed in ALL_SEEDS:
        for arm in ('SLiCS mask','SVD mask'):
            row=next(r for r in frozen['models'] if r['seed']==seed and r['arm']==arm)
            st=memory.required(old,row['key'])
            if pilot.close.model_hash(st[row['field']])!=row['weights_sha']:
                raise ValueError('Comparator model hash changed')
            for prefix in ('report_','compact_'):
                memory.required(old,prefix+row['key']+'_top16')
    d0=memory.required(p,'mask_initial_s21')['D']
    for seed in NEW_SEEDS:
        d=memory.required(old,f'svd_mask_s{seed}')['D']
        if pilot.close.model_hash(d)!=pilot.close.model_hash(d0):
            raise ValueError('Deterministic training-mask SVD changed across starts')
    ctx['store'].write('confirmation_protocol.json',dict(config=cfg,fit_recipe=ctx['comparison'],
        new_training_starts=NEW_SEEDS,imported_pilot_seed=21,pilot_checkpoint_sha=PILOT_WEIGHT,
        total_new_updates=10000,per_seed_warm=2500,per_seed_guided=2500,
        initializer='Same deterministic training-mask SVD for all starts; seeds vary shuffle and noise, not this initial dictionary.',
        masks='Training-only foreground SVD and training group assignment; annotations never enter encode at inference.',
        unchanged='75 x 3, Top16, data, normalization, optimizer, denoising, losses, training target policy and common native reconstruction guard.',
        historical_test='Same inspected1500-image holdout. New training trajectories on one dataset, not a new independent benchmark.',
        primary='Mask-init vs native-init GGF within each seed. Capture/AP/VE must be read jointly.',
        comparisons='Also show continued Vanilla, fitted SLiCS-mask and SVD-only. Preserve ineligible labels. This is our SLiCS-based adaptation, not published SLiCS.',
        statistical_scope='Image intervals are conditional on fitted models; initialization is identical across mask runs. No seed-population/SOTA claim.',
        interpretation='Replicate across new training orders before promoting initialization. This notebook does not test full1024-group capacity or structural AGF.',
        follow_up='If confirmed, freeze the initialization recipe and then test its larger-capacity extension separately.'))
    return ctx

def copy_audit(ctx,source,row):
    s=ctx['store'];tag=row['key']+'_top16'
    report=memory.required(source,'report_'+tag)
    if report['summary']['weights_sha']!=row['weights_sha']:
        raise ValueError('Cached report/model identity differs')
    if (report['summary']['images'],report['summary']['categories'])!=(1500,75):
        raise ValueError('Cached audit cohort differs')
    for prefix in ('report_','compact_'):
        key=prefix+tag
        if s.load(key) is None:s.save(key,memory.required(source,key))
    mapping=source.load('mapping_'+tag)
    if mapping is not None:
        if s.load('mapping_'+tag) is None:s.save('mapping_'+tag,mapping)
        s.write('mappings/'+tag+'.json',mapping['mappings'])
    s.write('reports/'+tag+'.json',report)
    return report

def train(ctx):
    s=ctx['store'];done=s.load('confirmation_models_frozen')
    if done is not None:return done
    new=pilot.train(ctx) # Only seeds22/23. Original fit functions and RNG schedule.
    rows={r['key']:copy.deepcopy(r) for r in new['models']}
    p=ctx['completed_pilot']
    for row in memory.required(p,'comparison_frozen')['models']:
        row=copy.deepcopy(row);row['reused_pilot']=True
        st=memory.required(p,row['key'])
        if training.weight_sha(st[row['field']])!=row['weights_sha']:
            raise ValueError('Pilot frozen weights changed')
        if s.load(row['key']) is None:s.save(row['key'],st)
        rows[row['key']]=row
        copy_audit(ctx,p,row)
    allrows=sorted(rows.values(),key=lambda r:(r['seed'],r['arm']))
    if len(allrows)!=9 or {r['seed'] for r in allrows}!=set(ALL_SEEDS):
        raise ValueError('Need exactly three native arms per start')
    out=dict(complete=True,models=allrows,test_used=False,groups=75,rank=3,active_groups=16,
        imported_pilot_used_test=True,source_sha=s.config['source_sha'])
    # Only native models enter the live scorer. NNLS comparators remain frozen cached audits.
    s.save('comparison_frozen',out);s.write('models_frozen.json',out)
    s.save('confirmation_models_frozen',out)
    memory.assert_immutable(ctx['stamps']);return out

def summarize(reports,comparisons):
    rows=[r['summary'] for r in reports]
    aggregates=[]
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        if len(rr)!=3 or {r['seed'] for r in rr}!=set(ALL_SEEDS):
            raise ValueError('Cannot hide a start or shrink the mean denominator')
        aggregates.append(dict(arm=arm,seeds=ALL_SEEDS,capture=float(np.mean([r['capture'] for r in rr])),
            ap=float(np.mean([r['ap'] for r in rr])),ve=float(np.mean([r['reconstruction']['ve'] for r in rr])),
            eligible_checkpoints=sum(r['eligible'] is True for r in rr),
            note='SVD is a zero-update reference with unmatched gradient budget.' if arm=='SVD mask' else None))
    primary=[c for c in comparisons if c['control']=='native-init GGF']
    if len(primary)!=3:raise ValueError('Missing primary paired seed comparison')
    heldout=[c for c in primary if c['seed'] in NEW_SEEDS]
    return dict(complete=True,models=rows,comparisons=comparisons,aggregate=aggregates,
        new_starts_positive=sum(c['positive_within_cost'] for c in heldout),new_starts_tested=2,
        all_starts_positive=sum(c['positive_within_cost'] for c in primary),
        new_start_image_intervals_positive=sum(c['paired_intervals']['capture_gain_ci95'][0]>0 and c['paired_intervals']['ap_gain_ci95'][0]>0 for c in heldout),
        decision='Read every start, eligibility, confidence interval and per-category loss. Confirmation on two new orders does not establish global optimality, full-capacity gains or published-SLiCS superiority.',
        pilot='Seed21 is historical and imported; only22/23 train here. All mask runs share one deterministic mask-derived dictionary and fixed dataset.',
        audit_scope='Reused previously inspected1500-image holdout; not a newly untouched confirmation.')

def evaluate(ctx):
    s=ctx['store'];cached=s.load('confirmation_summary')
    if cached is not None:
        reports=[memory.required(s,'report_'+r['key']+'_top16') for r in cached['models']]
        render(ctx,reports,cached);memory.assert_immutable(ctx['stamps']);return cached
    if s.load('confirmation_models_frozen') is None:raise ValueError('Freeze both new fits first')
    score_ctx=dict(ctx,pilot=dict(ctx['pilot'],seeds=ALL_SEEDS))
    native=pilot.evaluate(score_ctx)
    reports=[memory.required(s,'report_'+r['key']+'_top16') for r in native['models']]
    comparisons=copy.deepcopy(native['comparisons'])
    old=ctx['parent_comparison'];catalog=memory.required(old,'comparison_frozen')['models']
    for seed in ALL_SEEDS:
        a=next(r for r in reports if r['summary']['seed']==seed and r['summary']['arm']=='mask-init GGF')
        aa=memory.required(s,'compact_'+a['tag'])
        for arm in ('SLiCS mask','SVD mask'):
            row=next(r for r in catalog if r['seed']==seed and r['arm']==arm)
            b=copy_audit(ctx,old,row);reports.append(b)
            ci=study.paired(aa,memory.required(s,'compact_'+b['tag']),ctx['cfg'])
            av,bv=a['summary'],b['summary'];gain=av['capture']-bv['capture'];ap=av['ap']-bv['ap']
            cost=bv['reconstruction']['ve']-av['reconstruction']['ve']
            comparisons.append(dict(seed=seed,control=arm,capture_gain=gain,ap_gain=ap,ve_cost=cost,
                paired_intervals=ci,eligible=av['eligible'],control_eligible=bv['eligible'],
                positive_within_cost=bool(av['eligible'] and gain>0 and ap>=0 and cost<=.01),
                scope='Descriptive task-adapted comparator; ineligible control does not establish validated method superiority.' if arm=='SLiCS mask' else 'Untrained initialization control; gradient budget differs.'))
    out=summarize(reports,comparisons)
    s.save('confirmation_summary',out);s.write('summary.json',out)
    render(ctx,reports,out);memory.assert_immutable(ctx['stamps']);return out

def render(ctx,reports,summary):
    # Original CSV aggregation with csv explicitly imported in the pilot module.
    pilot.make_tables(ctx,reports)
    s=ctx['store']
    with (s.results/'aggregate.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['method','capture_pct','ap_pct','ve_pct','eligible_checkpoints']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in summary['aggregate']:
            w.writerow(dict(method=r['arm'],capture_pct=100*r['capture'],ap_pct=100*r['ap'],ve_pct=100*r['ve'],eligible_checkpoints=r['eligible_checkpoints']))
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    color={'mask-init GGF':'#008c82','native-init GGF':'#768bab','continued Vanilla':'#abb8c8','SLiCS mask':'#bc9bcb','SVD mask':'#d5b86d'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,field,title in zip(axes,['capture','ap','ve'],['Object contrast capture','Patch detection AP','Reconstruction VE']):
        for j,arm in enumerate(ARMS):
            vals=[100*r['summary']['reconstruction']['ve'] if field=='ve' else 100*r['summary'][field] for r in reports if r['summary']['arm']==arm]
            ax.bar(j,np.mean(vals),color=color[arm])
            ax.scatter(np.linspace(j-.12,j+.12,3),vals,c='black',s=13,zorder=3)
        ax.set_xticks(range(len(ARMS)),['Continued\nVanilla','Native-init\nGGF','SLiCS-based\nadaptation','SVD-only\ncontrol','Mask-init\nGGF'],fontsize=8)
        ax.set(title=title,ylabel='%');ax.grid(axis='y',alpha=.2)
    fig.suptitle('75 x 3, Top16 | Seed21 pilot reused; seeds22/23 new | Same inspected1500-image test',fontsize=10)
    fig.tight_layout();folder=s.results/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(folder/f'confirmation.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    s.write('summary.json',summary)

def export(ctx):
    s=ctx['store'];out=s.output/f'ggf_mask_confirmation_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in s.results.rglob('*'):
            if p.is_file():z.write(p,'results/'+p.relative_to(s.results).as_posix())
    return out

def preflight(ctx):
    v=pilot.preflight(ctx)
    return dict(v,only_new_seeds=NEW_SEEDS,imported_seed21=PILOT_WEIGHT,
        statistical_scope='Identical SVD initialization, varied order/noise. Saved comparators reused; no baseline/NNLS/DINO fitting.')
