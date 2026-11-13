"""Single-change, cached-control pilot: object-mask SVD initialization for GGF."""
import copy
import csv
import gc
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import slics_closeout as close, slics_comparison as old, study

PARENT_ID = 'ce65e7b00536a58e690f3fe5a807b1dfac2bc081667ebf1d644683be5601407a'
PARENT_SOURCE = '58a01a8852c6f15e4aaea2230449562450525611031182042399736961455a91'

def settings():
    return dict(version=1, study='ggf_mask_initialization_pilot', seeds=[21],
                evaluate_every=500, bootstrap=200)

def initialize(D):
    """Same tied encoder initialization as native Vanilla; signed projection."""
    if D.ndim != 2 or not torch.isfinite(D).all():
        raise ValueError('Finite matrix required')
    if not torch.allclose(D.norm(dim=1), torch.ones(len(D), device=D.device), atol=1e-4):
        raise ValueError('Decoder atoms must have unit norm')
    return dict(W_dec=D.detach().cpu().clone(),
                W_enc=D.detach().cpu().T.contiguous().clone(),
                b_enc=torch.zeros(len(D), dtype=D.dtype))

def geometry(weights, reference, rank=3):
    """Decoder geometry diagnostics; these are not semantic capture scores."""
    D=weights['W_dec'].double().reshape(-1,rank,reference.shape[1])
    R=reference.double().reshape_as(D)
    Q=torch.linalg.qr(D.transpose(1,2),mode='reduced').Q
    QR=torch.linalg.qr(R.transpose(1,2),mode='reduced').Q
    retention=((Q.transpose(1,2)@QR).square().sum((1,2))/rank).tolist()
    norm=weights['W_dec'].float()
    cross=(norm@norm.T).square()
    group=torch.arange(len(norm))//rank
    mask=group[:,None]!=group[None,:]
    return dict(mean_same_slot_subspace_overlap=float(np.mean(retention)),
                per_group_same_slot_subspace_overlap=retention,
                mean_cross_group_squared_cosine=float(cross[mask].mean()),
                note='Same slot retention only; no claim of category ownership after training.')

def prepare(source_output,parent_output,output,local,sha,versions,cfg,device,require_drive=True):
    if set(cfg)!=set(settings()) or cfg['seeds'] not in ([21],[22,23],[21,22,23]):
        raise ValueError('Choose the original pilot, the frozen22/23 confirmation, or all three starts')
    if cfg['evaluate_every']!=500 or cfg['bootstrap']!=200:
        raise ValueError('Keep frozen scoring cadence and bootstrap')
    parent=memory.FrozenStore(Path(parent_output)/'checkpoints'/PARENT_ID[:20],Path(local)/'read_comparison')
    if parent.run_id!=PARENT_ID or parent.config['source_sha']!=PARENT_SOURCE:
        raise ValueError('Wrong completed comparison source')
    if parent.config['versions']!=versions or parent.config['device']!=device:
        raise ValueError('Cached controls require the same numerical versions and device type. Read the environment cell; do not override this check.')
    if parent.config['config']!=close.settings():
        raise ValueError('Historical comparison recipe changed')
    # Validate availability before preparing any new fit. Never retrain missing controls.
    frozen=memory.required(parent,'comparison_frozen')
    if not frozen['complete']:raise ValueError('Parent comparison incomplete')
    memory.required(parent,'comparison_summary')
    for seed in cfg['seeds']:
        for key in (f'native_s{seed}',f'targets_s{seed}',f'ggf_s{seed}',f'continued_s{seed}',f'svd_mask_s{seed}',
                    f'report_ggf_s{seed}_top16',f'report_continued_s{seed}_top16'):
            memory.required(parent,key)
    for p in (Path(source_output),Path(parent_output)):
        a,b=Path(output).resolve(),p.resolve()
        if a==b or a in b.parents or b in a.parents:raise ValueError('Keep all input folders read-only')
    basecfg=close.settings()
    ctx=close.prepare(source_output,Path(output)/'input_verification',local,sha,versions,basecfg,device,require_drive)
    ctx['comparison']['seeds']=cfg['seeds']
    ctx['pilot']=copy.deepcopy(cfg);ctx['parent_comparison']=parent
    ctx['store']=inputs.JournalStore(output,dict(config=cfg,recipe=ctx['comparison'],source_sha=sha,
        versions=versions,device=device,parent_run=PARENT_ID,source_run=ctx['source'].run_id,
        activation_run=ctx['original'].run_id),Path(local)/'mask_initialization_rescue',require_drive)
    ctx['stamps']+=memory.source_snapshot([parent])
    ctx['store'].write('protocol_settings.json',dict(config=cfg,recipe=ctx['comparison'],
        train_images=2000,development_images=400,calibration_images=400,test_images=1500,
        groups=75,rank=3,active_groups=16,category_count=75,
        only_change='Training-mask foreground SVD dictionary replaces random dictionary at update zero. Encoder is its transpose, bias zero, exactly the native tied-init convention.',
        schedule='2500 reconstruction warm-up updates + 2500 guided updates; same ordered images and denoising RNG schedule as cached native-start GGF.',
        mapping='Both arms choose category training targets using the identical training-only mapping policy AFTER warm-up. Initial SVD category slots are not hard-wired into the loss.',
        checkpoint_rule='Warm-up: best development MSE. Guided: capture + 0.2 AP, guarded by common cached native warm-up MSE +2%, not by a weaker mask-start reconstruction floor.',
        initializer='All training-supported foreground patches, uncentered per-category second moment after frozen global normalization; no whitening, label-based test routing or new losses.',
        controls='Cached native-start GGF and continued reconstruction; identical numerical versions, device type and batch identities required. Reevaluated alongside new model.',
        scope='Small 75-group initialization diagnostic, not a replacement for the successful 1024-group result. Reused inspected holdout, not independent confirmation. SVD adds measured computation beyond matched gradient-update budget.'))
    return ctx

def initial(ctx,seed):
    s=ctx['store'];key=f'mask_initial_s{seed}';st=s.load(key)
    if st is not None:return st
    rec=memory.required(ctx['parent_comparison'],f'svd_mask_s{seed}')
    D=rec['D'].float();w=initialize(D)
    # Hash the exact training-only dictionary recorded by the parent frozen catalog.
    catalog=memory.required(ctx['parent_comparison'],'comparison_frozen')['models']
    row=next(r for r in catalog if r['key']==f'svd_mask_s{seed}')
    if close.model_hash(D)!=row['weights_sha']:raise ValueError('Training-only SVD dictionary hash changed')
    if tuple(D.shape)!=(225,768):raise ValueError('Expected 75 x 3 dictionary')
    m=core.new_model('vanilla',ctx['base'],w,ctx['device'])
    x,_,_=close.batch(ctx,seed,0);xx=x.flatten(0,1)
    with torch.no_grad():
        out,z=m(xx)
        diagnostics=dict(initial_batch_mse=float((out-xx).square().mean()),
            active_groups=float((z.norm(dim=-1)>0).float().sum(-1).mean()),
            finite=bool(torch.isfinite(out).all()),geometry=geometry(w,D),
            category_order=ctx['cfg']['category_indices'],
            no_masks_at_inference=True,encoder='D.T',bias='zero',coefficient_sign='signed')
    if not diagnostics['finite'] or diagnostics['active_groups']>16.0001:raise ValueError('Initialization numerical/activity check failed')
    st=dict(weights=w,weights_sha=training.weight_sha(w),D=D,diagnostics=diagnostics,
            source_dictionary_sha=row['weights_sha'],original_svd_seconds=rec['seconds'],report=rec['report'])
    s.save(key,st);s.write('initialization/'+key+'.json',{k:v for k,v in st.items() if k not in ('weights','D')})
    del m;return st

def preflight(ctx):
    s=ctx['store'];prev=s.load('preflight')
    if prev is not None:return prev
    seed=ctx['pilot']['seeds'][0];st=initial(ctx,seed)
    m=core.new_model('vanilla',ctx['base'],st['weights'],ctx['device'])
    # Diagnostic only: the initializer's category slots were fixed by TRAIN masks.
    # This never chooses loss coefficients, targets, warm-up duration or test groups.
    with close.progress(ctx,m,'initializer_development_diagnostic'):
        baseline=close.checked_dev(ctx,m,old.sl_targets(ctx,m))
    s.write('initialization/initializer_development.json',baseline)
    x,_,_=close.batch(ctx,seed,0);opt=torch.optim.Adam(m.parameters(),lr=ctx['comparison']['lr'])
    times=[]
    for _ in range(4):
        study.sync(ctx['device']);t=time.monotonic();loss,_=m.loss(x.flatten(0,1))
        opt.zero_grad();loss.backward();opt.step();m.normalize_decoder();study.sync(ctx['device'])
        times.append(time.monotonic()-t)
    out=dict(numerical_ok=bool(torch.isfinite(loss)),gradient_seconds_per_update=float(np.median(times[1:])),
             new_gradient_updates=5000*len(ctx['pilot']['seeds']),
             rough_training_minutes=float(np.median(times[1:])*5000*len(ctx['pilot']['seeds'])/60),
             excludes='Validation, training-group mapping, calibration, test scoring, bootstraps, cache loading and Drive. Timing extrapolation, not ETA.',
             initialization=st['diagnostics'],original_svd_seconds=st['original_svd_seconds'],
             initializer_development={k:v for k,v in baseline.items() if k!='categories'},
             cached_controls='No baseline training, NNLS or DINO extraction required.')
    del m,opt;s.save('preflight',out);s.write('preflight.json',out);return out

def train(ctx):
    s=ctx['store'];parent=ctx['parent_comparison'];done=s.load('comparison_frozen')
    if done is not None:return done
    rows=[]
    for seed in ctx['pilot']['seeds']:
        s.event('seed_start',seed=seed)
        init=initial(ctx,seed)
        reference=float(memory.required(parent,f'ggf_s{seed}')['initial_mse'])
        def observe(model,update,phase):
            g=geometry(training.weights(model),init['D'])
            s.write(f'geometry/{phase}_s{seed}_u{update:05d}.json',g)
        warm=close.fit_sgd(ctx,seed,f'mask_warm_s{seed}',warm_initial=init['weights'],observer=observe)
        oldwarm=memory.required(parent,f'native_s{seed}')
        if warm['batch_sha']!=oldwarm['batch_sha']:raise ValueError('Warm-up image stream differs from cached control')
        m=core.new_model('vanilla',ctx['base'],warm['best'],ctx['device'])
        key=f'mask_targets_s{seed}';targets=s.load(key)
        if targets is None:
            with close.progress(ctx,m,'training_group_assignment'):targets=old.fixed_category_targets(ctx,m)
            s.save(key,targets);s.write('targets/'+key+'.json',targets)
        s.write(f'initialization/warm_summary_s{seed}.json',dict(best_update=warm['best_update'],
            mask_warm_mse=-warm['best_score'],common_native_mse=reference,
            mask_warm_within_common_guard=(-warm['best_score']<=reference*1.02),
            same_initialization_slot_groups=sum(t['contrast_group']==ctx['cfg']['category_indices'].index(t['category_index']) for t in targets),
            distinct_contrast_groups=len(set(t['contrast_group'] for t in targets)),
            distinct_detector_groups=len(set(t['detector_group'] for t in targets))))
        del m
        guided=close.fit_sgd(ctx,seed,f'mask_ggf_s{seed}',warm['best'],targets,True,
                             reference_mse=reference,observer=observe)
        cached=memory.required(parent,f'ggf_s{seed}')
        if guided['batch_sha']!=cached['batch_sha']:raise ValueError('Guided image stream differs from cached control')
        eligible=any(h['eligible'] for h in guided['history'])
        field='best' if eligible else 'weights'
        rows.append(dict(key=f'mask_ggf_s{seed}',seed=seed,arm='mask-init GGF',kind='native',field=field,
            weights_sha=training.weight_sha(guided[field]),best_update=guided['best_update'] if eligible else None,
            eligible=eligible,training_seconds=init['original_svd_seconds']+warm['seconds']+guided['seconds'],
            updates=5000,batch_sha=guided['batch_sha'],parameters=345975,
            warm_start_sha=init['weights_sha'],common_reference_mse=reference))
        catalog=memory.required(parent,'comparison_frozen')['models']
        for arm in ('continued','ggf'):
            r=copy.deepcopy(next(r for r in catalog if r['key']==f'{arm}_s{seed}'))
            st=memory.required(parent,r['key'])
            if training.weight_sha(st[r['field']])!=r['weights_sha']:raise ValueError('Cached control hash differs')
            s.save(r['key'],st)
            r['arm']='native-init GGF' if arm=='ggf' else 'continued Vanilla'
            r['cached_control']=True;rows.append(r)
        s.event('seed_complete',seed=seed,new_model_eligible=eligible)
        del warm,guided,cached;gc.collect()
    out=dict(complete=True,models=rows,test_used=False,groups=75,rank=3,active_groups=16,
             source_sha=s.config['source_sha'])
    memory.assert_immutable(ctx['stamps']);s.save('comparison_frozen',out);s.write('models_frozen.json',out)
    return out

def evaluate(ctx):
    """Use original audited capture/AP/VE definitions and freeze mappings first."""
    s=ctx['store'];cached=s.load('comparison_summary')
    if cached is not None:
        reports=[memory.required(s,'report_'+r['key']+'_top16') for r in cached['models']]
        make_tables(ctx,reports);memory.assert_immutable(ctx['stamps']);return cached
    # close.evaluate's common scoring code is used verbatim by a small arm-neutral helper.
    reports=score_models(ctx)
    replay=[]
    for r in reports:
        v=r['summary']
        if v['arm']=='mask-init GGF':continue
        prior=memory.required(ctx['parent_comparison'],'report_'+r['tag'])['summary']
        diffs={k:float(v[k]-prior[k]) for k in ('capture','ap')}
        diffs['ve']=float(v['reconstruction']['ve']-prior['reconstruction']['ve'])
        replay.append(dict(key=r['tag'],differences=diffs,
                           within_tolerance=all(abs(x)<=1e-5 for x in diffs.values())))
    s.write('cached_control_metric_replay.json',replay)
    if not all(r['within_tolerance'] for r in replay):
        raise ValueError('Cached control scores failed numerical replay. Inspect cached_control_metric_replay.json before attributing differences to initialization.')
    comparisons=[]
    for seed in ctx['pilot']['seeds']:
        r={a['summary']['arm']:a for a in reports if a['summary']['seed']==seed}
        a=r['mask-init GGF'];aa=memory.required(s,'compact_'+a['tag'])
        for arm in ('native-init GGF','continued Vanilla'):
            b=r[arm];ci=study.paired(aa,memory.required(s,'compact_'+b['tag']),ctx['cfg'])
            av,bv=a['summary'],b['summary'];gain=av['capture']-bv['capture'];ap=av['ap']-bv['ap']
            cost=bv['reconstruction']['ve']-av['reconstruction']['ve']
            comparisons.append(dict(seed=seed,control=arm,capture_gain=gain,ap_gain=ap,ve_cost=cost,
                paired_intervals=ci,eligible=av['eligible'],
                positive_within_cost=bool(av['eligible'] and gain>0 and ap>=0 and cost<=.01)))
    out=dict(complete=True,models=[r['summary'] for r in reports],comparisons=comparisons,
        primary='Mask-init GGF vs native-init GGF, identical 75 x 3 Top-16 capacity and gradient schedule.',
        interpretation='Pilot on a reused inspected holdout; no full-capacity or independent replication claim.',
        decision='A gain in capture must be read alongside AP, VE, eligibility and paired intervals. One seed is a pilot; confirm all three before changing the recipe.')
    s.save('comparison_summary',out);s.write('summary.json',out)
    make_tables(ctx,reports);memory.assert_immutable(ctx['stamps']);return out

def export(ctx):
    s=ctx['store'];out=s.output/f'ggf_mask_initialization_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in s.results.rglob('*'):
            if p.is_file():z.write(p,'results/'+p.relative_to(s.results).as_posix())
    return out


from coco_baselines import metrics

def score_models(ctx):
    s=ctx['store']
    frozen=memory.required(s,'comparison_frozen');mapping,test=close.load_audit_packs(ctx);reports=[]
    for row in frozen['models']:
        modes=[False]
        if ctx['comparison']['include_dense_first_seed'] and row['seed']==ctx['comparison']['seeds'][0] and row['arm'].startswith('SLiCS'):modes+=[True]
        for dense in modes:
            tag=row['key']+('_dense' if dense else '_top16');prior=s.load('report_'+tag)
            if prior is not None:reports.append(prior);continue
            s.event('audit_start',key=tag,images=len(test['ids']))
            w=memory.required(s,row['key'])[row['field']]
            if close.model_hash(w)!=row['weights_sha']:raise ValueError('Frozen weights changed')
            m=close.SlicsModel(w.to(ctx['device']),ctx['comparison'],dense) if row['kind']=='slics' else core.new_model('vanilla',ctx['base'],w,ctx['device'])
            maps=s.load('mapping_'+tag)
            if maps is None:
                s.event('calibration_start',key=tag,images=len(mapping['ids']))
                with close.progress(ctx,m,tag+'_calibration'):a=metrics.audit_arrays(m,mapping['x'],mapping['masks'],mapping['ids'],ctx['cfg'],ctx['device'])
                maps=metrics.calibrate(a,ctx['cfg'],ctx['provenance']['categories']);del a;gc.collect();s.save('mapping_'+tag,maps)
                s.write('mappings/'+tag+'.json',maps['mappings'])
            started=time.monotonic()
            with close.progress(ctx,m,tag+'_test'):audit=metrics.audit_arrays(m,test['x'],test['masks'],test['ids'],ctx['cfg'],ctx['device'])
            per=metrics.evaluate(audit,maps,dict(ctx['cfg'],bootstrap=0),71002)
            indexed={r['category_index']:r for r in per};cohort=[indexed[c] for c in ctx['cfg']['category_indices']]
            if any(r['contrast'] is None or r['detection'] is None for r in cohort):raise ValueError('Missing test category; denominator cannot shrink')
            with close.progress(ctx,m,tag+'_reconstruction'):rec=training.measure(m,test['x'].reshape(-1,m.d),ctx['device'],512)
            if not dense and rec['active_groups']>16.00001:raise RuntimeError('Inference activity budget violated')
            compact=study.compact_audit(audit,maps,ctx['cfg']);s.save('compact_'+tag,compact)
            summary=dict(**row,readout='dense NNLS' if dense else 'Top-16',images=len(test['ids']),categories=len(cohort),
                capture=float(np.mean([r['contrast']['single_group_capture'] for r in cohort])),
                ap=float(np.mean([r['detection']['ap'] for r in cohort])),
                all_group_capture=float(np.mean([r['contrast']['all_group_capture'] for r in cohort])),reconstruction=rec,
                audit_seconds=time.monotonic()-started,solver=getattr(m,'solver_totals',None),
                inference_seconds=getattr(m,'inference_seconds',None),inference_patches=getattr(m,'inference_patches',None))
            result=dict(tag=tag,summary=summary,categories=per,skipped_objects=audit['skipped'])
            s.save('report_'+tag,result);s.write('reports/'+tag+'.json',result);reports.append(result);s.event('audit_complete',**summary)
            del m,audit,compact,w;gc.collect()
            if ctx['device']=='cuda':torch.cuda.empty_cache()
    return reports


def make_tables(ctx,reports):
    s=ctx['store'];flat=[]
    for r in reports:
        v=r['summary'];flat.append(dict(seed=v['seed'],method=v['arm'],readout=v['readout'],capture_pct=100*v['capture'],
            ap_pct=100*v['ap'],ve_pct=100*v['reconstruction']['ve'],active_groups=v['reconstruction']['active_groups'],
            checkpoint_eligible=v['eligible'],train_seconds=v['training_seconds'],audit_seconds=v['audit_seconds'],parameters=v['parameters']))
    with (s.results/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    with (s.results/'per_category.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['seed','method','readout','category','contrast_group','detector_group','capture','ap','objects','positive_images']);w.writeheader()
        for r in reports:
            v=r['summary']
            for c in r['categories']:
                if c['category_index'] not in ctx['cfg']['category_indices']:continue
                w.writerow(dict(seed=v['seed'],method=v['arm'],readout=v['readout'],category=c['category'],
                    contrast_group=c['contrast']['groups'][0],detector_group=c['detection']['group'],capture=c['contrast']['single_group_capture'],
                    ap=c['detection']['ap'],objects=c['contrast']['object_count'],positive_images=c['test_positive_images']))
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    arms=['continued Vanilla','native-init GGF','mask-init GGF'];fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,field,title in zip(axes,['capture','ap','ve'],['Object contrast capture','Patch detection AP','Reconstruction VE']):
        for j,arm in enumerate(arms):
            vals=[r[field+'_pct'] for r in flat if r['method']==arm and r['readout']=='Top-16']
            ax.bar(j,np.mean(vals),color='#008c82' if arm=='mask-init GGF' else '#9eaec8')
            ax.scatter(np.linspace(j-.1,j+.1,len(vals)),vals,c='black',s=16,zorder=3)
        ax.set_xticks(range(len(arms)),arms,rotation=35,ha='right',fontsize=8);ax.set(title=title,ylabel='%');ax.grid(axis='y',alpha=.2)
    fig.suptitle('75 groups × 3 directions; Top-16; initialization pilot; reused 1,500-image holdout');fig.tight_layout()
    folder=s.results/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(folder/f'comparison.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
