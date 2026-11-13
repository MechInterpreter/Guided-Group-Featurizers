"""Mask-informed initialization at the successful 1024 x 3 GGF capacity.

The only numerical treatment is replacing the first 75 initial groups with
training-mask SVD directions. Remaining random groups, fit engines, losses,
batch streams, optimizer schedules and common reconstruction floor are fixed.
"""
import copy
import csv
import gc
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training, metrics
from coco_baselines.storage import identity
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import loss_weight_sweep as reader, loss_ablation as lineage, study
from . import mask_initialization as pilot, slics_closeout as close
from .large_mask_fit import fit_warm, fit_guided

COMPARISON_ID=pilot.PARENT_ID
COMPARISON_SOURCE=pilot.PARENT_SOURCE
CONFIRM_ID='e3cf93dbe6e37a8232ddb258ba18245a3710bb45f4f662c83c4b5ead60ab6bb2'
CONFIRM_SOURCE='650499ce4739b03845d625bd0139ff0e9b4075140caace5401510d58f25c4da1'
LABELS=dict(continued_object='Continued Vanilla',ggf='Previous GGF',mask_ggf='Mask-initialized GGF')
NUMERICAL_PACKAGES=('numpy','torch','scipy','scikit-learn','einops')

def training_versions(record):
    missing=[k for k in NUMERICAL_PACKAGES if k not in record]
    if missing:raise ValueError('Missing training numerical versions: '+str(missing))
    return {k:record[k] for k in NUMERICAL_PACKAGES}

def settings():
    return dict(version=1,study='ggf_large_mask_initialization',seeds=[21,22,23],
                bootstrap=200,drive_save_every=50,initializer_groups=75,
                initializer_policy='75 mask-SVD groups + 949 original random groups')

def validate_settings(cfg):
    if cfg!=settings():
        raise ValueError('Keep the frozen three-start large-capacity initialization protocol')

def validate_inputs(ctx,comparison,confirmation,versions,device):
    if training_versions(ctx['source'].config['numerical_versions'])!=training_versions(versions):
        raise ValueError('Numerical versions differ from the large-model controls; do not mix cached scores')
    if ctx['source'].config['config']['device']!=device:
        raise ValueError('Use the original cuda device category for the matched controls')
    if comparison.run_id!=COMPARISON_ID or comparison.config['source_sha']!=COMPARISON_SOURCE:
        raise ValueError('Wrong completed mask-SVD source')
    if comparison.config['source_run']!=ctx['source'].run_id or comparison.config['activation_run']!=ctx['original'].run_id:
        raise ValueError('Small mask source and large model used different training activations')
    if confirmation.run_id!=CONFIRM_ID or confirmation.config['source_sha']!=CONFIRM_SOURCE:
        raise ValueError('Wrong completed mask confirmation source')
    if confirmation.config['parent_run']!=comparison.run_id:
        raise ValueError('Confirmation parent differs from mask-SVD source')
    q=memory.required(confirmation,'confirmation_summary')
    if not q['complete'] or q['new_starts_tested']!=2 or q['all_starts_positive']!=3:
        raise ValueError('Expected the completed successful three-start small-model confirmation')
    return q

def prepare(source_output,comparison_output,confirmation_output,output,local,sha,versions,cfg,device,require_drive=True):
    validate_settings(cfg)
    roots=[Path(source_output),Path(comparison_output),Path(confirmation_output),
           Path(source_output).parent/'AGF_COCO_stratified_confirmation_v1']
    dest=Path(output).resolve()
    for p in roots:
        p=p.resolve()
        if dest==p or p in dest.parents or dest in p.parents:
            raise ValueError('Separate output is required; all inputs stay read-only')
    # Existing verified lineage reader; no loss sweep or baseline training is run.
    rcfg=reader.settings();rcfg.update(seeds=cfg['seeds'],include_scratch=False)
    ctx=reader.prepare(source_output,Path(output)/'input_verification',local,sha,versions,rcfg,device,require_drive)
    comparison=memory.FrozenStore(Path(comparison_output)/'checkpoints'/COMPARISON_ID[:20],Path(local)/'read_svd')
    confirmation=memory.FrozenStore(Path(confirmation_output)/'checkpoints'/CONFIRM_ID[:20],Path(local)/'read_small_confirmation')
    validate_inputs(ctx,comparison,confirmation,versions,device)
    # Use original main-study numerical settings, not the smaller comparison's
    # constant-Adam/shuffled-image warm-up. That would change two things at once.
    inherited=copy.deepcopy(ctx['source'].config['config'])
    ctx['cfg']=dict(inherited,device=device,local_save_every=cfg['drive_save_every'],audit_image_batch=2,preview_images=0)
    ctx['norm']=memory.required(ctx['original'],'normalization')
    ctx['plan']=memory.required(ctx['source'],'fresh_plan')
    ctx['large']=copy.deepcopy(cfg);ctx['comparison_source']=comparison
    ctx['confirmation_source']=confirmation
    ctx['source_frozen']=memory.required(ctx['source'],'all_models_frozen')
    ctx['stamps']+=memory.source_snapshot([comparison,confirmation])
    ctx['store']=inputs.JournalStore(output,dict(config=cfg,fit_config=ctx['cfg'],native_config=ctx['base'],
        source_sha=sha,versions=versions,device=device,source_run=ctx['source'].run_id,
        activation_run=ctx['original'].run_id,mask_source_run=comparison.run_id,
        mask_source_versions=comparison.config['versions'],mask_source_device=comparison.config['device'],
        small_confirmation_run=confirmation.run_id),Path(local)/'large_mask_rescue',require_drive)
    s=ctx['store'];models=ctx['source_frozen']['models'];checks=[]
    for seed in cfg['seeds']:
        svd=memory.required(comparison,f'svd_mask_s{seed}')
        if tuple(svd['D'].shape)!=(225,768):raise ValueError('Mask-SVD capacity differs')
        if [r['category_index'] for r in svd['report']]!=ctx['cfg']['category_indices']:
            raise ValueError('Training-mask SVD category order differs')
        catalog=memory.required(comparison,'comparison_frozen')['models']
        svd_row=next(r for r in catalog if r['key']==f'svd_mask_s{seed}')
        if close.model_hash(svd['D'])!=svd_row['weights_sha']:raise ValueError('Saved training-mask dictionary changed')
        if seed==cfg['seeds'][0]:reference_D=svd['D'].clone()
        elif not torch.equal(reference_D,svd['D']):raise ValueError('Mask-SVD dictionaries must match across starts')
        for arm in ('native','ggf','continued_object'):
            row=next(r for r in models if r['seed']==seed and r['arm']==arm)
            st=memory.required(ctx['source'],row['key'])
            if training.weight_sha(st[row['field']])!=row['weights_sha']:raise ValueError('Cached large control weights changed')
            if arm=='native':
                expected=(len(ctx['packs']['train']['x'])*196//ctx['base']['batch'])*ctx['base']['epochs']
                if st['updates']!=expected or row['updates']!=expected:raise ValueError('Native pretraining budget differs')
            else:
                if st['updates']!=2500:raise ValueError('Cached branch training is incomplete')
                for prefix in ('report_','compact_','mapping_'):
                    memory.required(ctx['source'],prefix+f's{seed}_{arm}')
                report=memory.required(ctx['source'],'report_'+f's{seed}_{arm}')
                if report['summary']['weights_sha']!=row['weights_sha']:raise ValueError('Cached report points to different weights')
                if report['summary']['images']!=1500 or report['summary']['categories']!=75:
                    raise ValueError('Cached audit image/category denominator differs')
            del st
        checks.append(dict(seed=seed,native_updates=expected,guided_updates=2500,mask_dictionary_sha=svd_row['weights_sha']))
        del svd;gc.collect()
    if len(ctx['cfg']['category_indices'])!=75 or ctx['base']['epochs']!=40:
        raise ValueError('The frozen category cohort or 40-epoch large-model recipe changed')
    s.write('protocol_settings.json',dict(config=cfg,native_config=ctx['base'],guided_config=ctx['cfg'],checks=checks,
        treatment='Replace initial groups 0..74 with saved TRAIN-mask SVD; groups75..1023 retain same-seed original random initialization. Encoder is decoder transpose, bias zero.',
        budget='Same native40-epoch patch batches, OneCycleLR, denoising, native checkpoint cadence; then same2500 guided image batches and constant Adam as original GGF. SVD cost is extra and recorded.',
        groups='1024 groups x3 signed coordinates; at most16 groups per patch. No hard category ownership, scalar bank, routing change, anchoring or structural edit.',
        targets='After warm-up, the ORIGINAL main-study training-only mapping policy searches all1024 groups. SVD category slots are never hardwired into the loss.',
        guard='Common original native-start development MSE +2%, even if the new warm-up starts worse. Original score capture+.2AP; select before calibration/test.',
        controls='Read-only cached1024-group GGF and object-selected continued Vanilla for all3seeds; same numerical/device provenance. No controls or DINO are retrained.',
        audit='Previously inspected1500-image holdout. Calibration/test remain closed until every new model is frozen. Conditional paired whole-image intervals; not untouched independent confirmation.',
        scaling='This is a hybrid75-mask/949-random initialization extension. It is not identical to initializing every group in the75-group experiment.',
        initializer_provenance='Saved float32 mask-SVD tensor bytes are hash-checked and reused as input data, not recomputed. SVD source may have different library/CUDA versions; new training matches LARGE-control versions, recorded separately.',
        limits='Logs identify observed failures, not every causal explanation; no promise of optimality or a win.'))
    memory.assert_immutable(ctx['stamps'])
    return ctx

def hybrid_initial(base,D,seed):
    """Keep exactly the original same-seed random reserve, without saved learning."""
    torch.manual_seed(seed)
    model=training.make_model('vanilla',base,D.shape[1]);random=training.weights(model)
    n=len(D)
    if n%base['block_rank'] or n>=len(random['W_dec']):raise ValueError('Invalid mask-initialization capacity')
    if not torch.isfinite(D).all() or not torch.allclose(D.norm(dim=1),torch.ones(n),atol=1e-4):
        raise ValueError('Expected finite unit training-mask directions')
    dec=random['W_dec'].clone();dec[:n]=D.float()
    initial=dict(W_dec=dec,W_enc=dec.T.contiguous(),b_enc=torch.zeros_like(random['b_enc']))
    if not torch.equal(initial['W_dec'][n:],random['W_dec'][n:]):raise AssertionError('Random reserve changed')
    return initial,random

def initial(ctx,seed):
    s=ctx['store'];key=f'large_mask_initial_s{seed}';old=s.load(key)
    if old is not None:return old
    rec=memory.required(ctx['comparison_source'],f'svd_mask_s{seed}');D=rec['D'].float()
    w,random=hybrid_initial(ctx['base'],D,seed)
    q=dict(weights=w,D=D,weights_sha=training.weight_sha(w),random_initializer_sha=training.weight_sha(random),
           svd_sha=close.model_hash(D),svd_original_seconds=rec['seconds'],svd_report=rec['report'],
           mask_groups=len(D)//3,reserve_groups=ctx['base']['groups']-len(D)//3)
    s.save(key,q);s.write('initialization/'+key+'.json',{k:v for k,v in q.items() if k not in ('weights','D')})
    return q

def geometry(ctx,w,D,seed,phase,update):
    # Full3072x3072 similarity matrices add needless CPU work. The seeded75
    # subspaces are checked; the reserve's activity is logged by audits/targets.
    g=pilot.geometry({'W_dec':w['W_dec'][:len(D)]},D)
    g.update(seed=seed,phase=phase,update=update,mask_groups=75,
             decoder_atom_norm_min=float(w['W_dec'].norm(dim=1).min()),
             decoder_atom_norm_max=float(w['W_dec'].norm(dim=1).max()))
    ctx['store'].write(f'geometry/s{seed}_{phase}_u{update:05d}.json',g)

def choose_training_targets(ctx,seed,w):
    s=ctx['store'];key=f'large_mask_targets_s{seed}';old=s.load(key)
    if old is not None:return old
    m=core.new_model('vanilla',ctx['base'],w,ctx['device']);p=ctx['packs']['train']
    with close.progress(ctx,m,f's{seed}_training_group_assignment'):
        a=metrics.audit_arrays(m,p['x'],p['masks'],p['ids'],ctx['cfg'],ctx['device'])
    targets=core.choose_targets(a,ctx['cfg'],ctx['provenance']['categories'])
    if sorted(t['category_index'] for t in targets)!=ctx['cfg']['category_indices']:
        raise ValueError('Training assignment missed a declared category; denominator cannot shrink')
    heat=np.asarray(a['heat']);means=np.zeros(heat.shape[1],np.float64);on=means.copy()
    for lo in range(0,len(heat),2048):
        part=heat[lo:lo+2048];means+=part.sum(0,dtype=np.float64);on+=(part>1e-6).sum(0)
    stats=dict(seed=seed,mask_slot_capture_matches=sum(t['contrast_group']==j for j,t in enumerate(targets)),
        mask_slot_detector_matches=sum(t['detector_group']==j for j,t in enumerate(targets)),
        reserve_capture_assignments=sum(t['contrast_group']>=75 for t in targets),
        reserve_detector_assignments=sum(t['detector_group']>=75 for t in targets),
        distinct_capture_groups=len(set(t['contrast_group'] for t in targets)),
        distinct_detector_groups=len(set(t['detector_group'] for t in targets)),
        group_activity_mean=means/len(heat),positive_code_fraction=on/len(heat))
    stats['collisions']={}
    for field in ('contrast_group','detector_group'):
        groups={}
        for t in targets:groups.setdefault(t[field],[]).append(t['category'])
        stats['collisions'][field]=[dict(group=g,categories=c) for g,c in groups.items() if len(c)>1]
    s.save(key,targets);s.write('targets/'+key+'.json',targets);s.write(f'training_mapping_s{seed}.json',stats)
    del a,m;gc.collect()
    return targets

def preflight(ctx):
    s=ctx['store'];old=s.load('preflight')
    if old is not None:return old
    seed=ctx['large']['seeds'][0];init=initial(ctx,seed)
    m=core.new_model('vanilla',ctx['base'],init['weights'],ctx['device']);x=torch.as_tensor(np.array(ctx['packs']['train']['x'][:ctx['cfg']['images_per_update']],copy=True),device=ctx['device'])
    targets=ctx['targets'][seed] # Timing only; new targets are chosen AFTER warm-up.
    head=torch.nn.Parameter(torch.zeros(len(targets),2,device=ctx['device']))
    times={};opt=torch.optim.Adam(list(m.parameters())+[head],lr=ctx['cfg']['lr'])
    warm_rows=ctx['base']['batch'];warm_images=(warm_rows+195)//196
    warm_x=torch.as_tensor(np.array(ctx['packs']['train']['x'][:warm_images],copy=True),device=ctx['device']).flatten(0,1)[:warm_rows]
    for supervised in [False,True]:
        values=[]
        for i in range(4):
            study.sync(ctx['device']);start=time.monotonic();xx=x.flatten(0,1) if supervised else warm_x
            loss,_=m.loss(xx)
            if supervised:
                cap,det,_=core.supervised_loss(m,x,ctx['packs']['train']['masks'][:len(x)],targets,head,ctx['cfg'])
                loss=loss+.05*cap+.01*det
            opt.zero_grad();loss.backward();opt.step();m.normalize_decoder();study.sync(ctx['device'])
            if i:values.append(time.monotonic()-start)
        times['guided' if supervised else 'warm']=float(np.median(values))
    with torch.no_grad():pred,z=m(x.flatten(0,1))
    if not torch.isfinite(pred).all() or float((z.norm(dim=-1)>1e-6).sum(-1).float().mean())>16.00001:
        raise ValueError('Initialization numerical/activity check failed')
    tt=time.monotonic()
    with close.progress(ctx,m,'disposable_preflight_development'):
        v=core.validation(m,ctx['packs']['select'],targets,ctx['cfg'],ctx['device'])
    val_seconds=time.monotonic()-tt
    s.save('timing_checkpoint',dict(weights=training.weights(m)))
    n=len(ctx['large']['seeds']);warm=(len(ctx['packs']['train']['x'])*196//ctx['base']['batch'])*ctx['base']['epochs']
    out=dict(numerical_ok=True,new_models=n,new_updates_per_seed=warm+2500,
        new_warm_updates_per_seed=warm,new_guided_updates_per_seed=2500,
        total_new_updates=n*(warm+2500),seconds_per_update=times,development_seconds=val_seconds,
        rough_gradient_minutes=n*(warm*times['warm']+2500*times['guided'])/60,
        timing_scope='Actual native patch batch and guided image batch; disposable Adam timing. Mapping/calibration/audits/bootstraps/cache/Drive are excluded; not a guaranteed ETA.',
        inputs_reused=True,baseline_fits=0,DINO_extractions=0,
        parameters=sum(p.numel() for p in m.parameters()),auxiliary_head_parameters=head.numel())
    s.save('preflight',out);s.write('preflight.json',out)
    del m,head,opt,x,warm_x;gc.collect()
    if ctx['device'].startswith('cuda'):torch.cuda.empty_cache()
    return out

def train(ctx,max_warm_updates=None,max_guided_updates=None):
    s=ctx['store'];done=s.load('all_models_frozen')
    if done is not None:return done
    rows=[];source=ctx['source'];catalog=ctx['source_frozen']['models']
    for seed in ctx['large']['seeds']:
        s.event('seed_start',seed=seed,groups=1024)
        init=initial(ctx,seed);x=ctx['packs']['train']['x'].reshape(-1,768);val=ctx['packs']['select']['x'].reshape(-1,768)
        base=copy.deepcopy(ctx['base']);base['local_save_updates']=ctx['large']['drive_save_every']
        warm=fit_warm(s,f'large_mask_warm_s{seed}','vanilla',x,val,base,seed,base['lrs'][0],ctx['device'],
                      max_updates=max_warm_updates,initial=init['weights'],drive_every=ctx['large']['drive_save_every'])
        if warm['updates']!=warm['target']:return dict(complete=False,stage='warm',seed=seed)
        if warm['best'] is None:raise ValueError('No eligible warm-up checkpoint')
        geometry(ctx,warm['best'],init['D'],seed,'warm_best',warm['best_update'])
        targets=choose_training_targets(ctx,seed,warm['best'])
        reference=float(memory.required(source,f'ggf_s{seed}')['initial_mse'])
        s.write(f'warm_summary_s{seed}.json',dict(seed=seed,total_updates=warm['updates'],best_update=warm['best_update'],
            warm_mse=warm['best_score'],common_native_mse=reference,within_common_guard=warm['best_score']<=reference*1.02,
            warm_batch_sha=warm['sampled_batches_sha'],initial_weights_sha=init['weights_sha']))
        def observer(model,u):geometry(ctx,training.weights(model),init['D'],seed,'guided',u)
        guided=fit_guided(ctx,f'large_mask_ggf_s{seed}',warm['best'],targets,True,71000+seed,
            max_updates=max_guided_updates,reference_mse=reference,observer=observer)
        if guided['updates']!=2500:return dict(complete=False,stage='guided',seed=seed)
        expected=json.loads((source.results/f'paired_training_s{seed}.json').read_text())['minibatches_sha']
        if guided['sampled_batches_sha']!=expected:raise ValueError('Guided image stream differs from existing GGF controls')
        eligible=any(r.get('object_checkpoint_eligible',False) for r in guided['history'])
        field='best' if eligible else 'weights'
        rows.append(dict(key=f'large_mask_ggf_s{seed}',field=field,arm='mask_ggf',seed=seed,
            updates=warm['updates']+2500,warm_updates=warm['updates'],guided_updates=2500,
            best_update=guided['best_update'] if eligible else None,weights_sha=training.weight_sha(guided[field]),
            eligible=eligible,origin='new',parameters=sum(w.numel() for w in guided[field].values()),
            auxiliary_training_head_parameters=2*len(targets),common_reference_mse=reference,
            training_seconds=warm['seconds']+guided['seconds'],svd_original_seconds=init['svd_original_seconds'],
            batch_sha=expected,warm_batch_sha=warm['sampled_batches_sha']))
        for arm in ('continued_object','ggf'):
            item=copy.deepcopy(next(r for r in catalog if r['seed']==seed and r['arm']==arm))
            st=memory.required(source,item['key'])
            if training.weight_sha(st[item['field']])!=item['weights_sha']:raise ValueError('Reference checkpoint changed')
            item.update(origin='saved',source_tag=f's{seed}_{arm}',eligible=True,
                training_seconds=st['seconds'],native_pretraining_updates=warm['updates'],batch_sha=st['sampled_batches_sha'])
            if item['batch_sha']!=expected:raise ValueError('Reference image stream differs')
            rows.append(item);del st
        s.event('seed_complete',seed=seed,new_model_eligible=eligible,best_update=guided['best_update'])
        del warm,guided,init;gc.collect()
        if ctx['device'].startswith('cuda'):torch.cuda.empty_cache()
    out=dict(complete=True,models=rows,test_used=False,groups=1024,rank=3,active_groups=16,
             structural_edits=0,source_sha=s.config['source_sha'])
    memory.assert_immutable(ctx['stamps']);s.save('all_models_frozen',out);s.write('all_models_frozen.json',out)
    return out

def load_audit(ctx):
    if not memory.required(ctx['store'],'all_models_frozen')['complete']:
        raise RuntimeError('Freeze all new weights before calibration/test')
    mapping=memory.normalized_pack(ctx['original'],'mapping',ctx['provenance'],ctx['norm'],ctx['local'],ctx['cfg']['extract_chunk'])
    p=dict(splits=ctx['plan']['splits'])
    test=memory.normalized_pack(ctx['source'],'test',p,ctx['norm'],ctx['local'],ctx['cfg']['extract_chunk'])
    sets=[set(p['ids']) for p in [ctx['packs']['train'],ctx['packs']['select'],mapping,test]]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)):raise ValueError('Audit split overlap')
    if len(test['ids'])!=1500 or len(mapping['ids'])!=400:raise ValueError('Audit sizes changed')
    ctx['store'].write('audit_data_identity.json',dict(test_ids_sha=identity(test['ids']),calibration_ids_sha=identity(mapping['ids']),
        normalization_sha=identity(ctx['norm']),no_overlap=True,previously_inspected=True))
    return mapping,test

def score_models(ctx):
    s=ctx['store'];frozen=memory.required(s,'all_models_frozen');mapping,test=load_audit(ctx);reports=[]
    for row in frozen['models']:
        tag=f"s{row['seed']}_{row['arm']}";r=s.load('report_'+tag)
        if r is not None:reports.append(r);continue
        if row['origin']=='saved':
            r=copy.deepcopy(memory.required(ctx['source'],'report_'+row['source_tag']))
            ca=memory.required(ctx['source'],'compact_'+row['source_tag'])
            if r['summary']['weights_sha']!=row['weights_sha'] or ca['image_ids']!=test['ids'] or ca['category_indices']!=ctx['cfg']['category_indices']:
                raise ValueError('Reference audit identity differs')
            r['summary'].update(**row);s.save('compact_'+tag,ca)
            maps=memory.required(ctx['source'],'mapping_'+row['source_tag'])
            s.save('mapping_'+tag,maps);s.write('mappings/'+tag+'.json',maps)
            del ca
        else:
            st=memory.required(s,row['key']);w=st[row['field']];del st
            if training.weight_sha(w)!=row['weights_sha']:raise ValueError('Frozen new weights changed')
            m=core.new_model('vanilla',ctx['base'],w,ctx['device']);del w
            maps=s.load('mapping_'+tag)
            if maps is None:
                s.event('calibration_start',model=tag,images=400)
                with close.progress(ctx,m,tag+'_calibration'):
                    a=metrics.audit_arrays(m,mapping['x'],mapping['masks'],mapping['ids'],ctx['cfg'],ctx['device'])
                maps=metrics.calibrate(a,ctx['cfg'],ctx['provenance']['categories']);del a;gc.collect()
                s.save('mapping_'+tag,maps);s.write('mappings/'+tag+'.json',maps)
            s.event('test_scoring_start',model=tag,images=1500)
            with close.progress(ctx,m,tag+'_test'):
                a=metrics.audit_arrays(m,test['x'],test['masks'],test['ids'],ctx['cfg'],ctx['device'])
            per=metrics.evaluate(a,maps,dict(ctx['cfg'],bootstrap=0),ctx['cfg']['fresh_test_seed'])
            selected={v['category_index']:v for v in per};cohort=[selected[c] for c in ctx['cfg']['category_indices']]
            if any(v['contrast'] is None or v['detection'] is None for v in cohort):raise ValueError('Missing test category; denominator cannot shrink')
            rec=training.measure(m,test['x'].reshape(-1,768),ctx['device'],ctx['cfg']['batch'])
            if rec['active_groups']>16.00001:raise ValueError('Top16 budget exceeded')
            ca=study.compact_audit(a,maps,ctx['cfg']);s.save('compact_'+tag,ca)
            summary=dict(**row,groups=1024,block_rank=3,images=1500,categories=75,reconstruction=rec,
                mean_single_group_contrast_capture=float(np.mean([v['contrast']['single_group_capture'] for v in cohort])),
                mean_single_group_ap=float(np.mean([v['detection']['ap'] for v in cohort])),
                mean_single_group_iou=float(np.mean([v['detection']['iou'] for v in cohort])),
                mean_multi_group_contrast_capture=float(np.mean([v['contrast']['multi_group_capture'] for v in cohort])),
                mean_all_group_contrast_capture=float(np.mean([v['contrast']['all_group_capture'] for v in cohort])))
            r=dict(summary=summary,categories=per,skipped_objects=a['skipped'])
            del m,a,ca;gc.collect()
            if ctx['device'].startswith('cuda'):torch.cuda.empty_cache()
        s.save('report_'+tag,r);s.write('reports/'+tag+'.json',r);reports.append(r)
        s.event('model_audit_complete',seed=row['seed'],arm=row['arm'],eligible=row['eligible'])
    return reports

def evaluate(ctx):
    s=ctx['store'];old=s.load('large_summary')
    if old is not None:
        reports=[memory.required(s,f"report_s{r['seed']}_{r['arm']}") for r in old['models']]
        tables_and_plots(ctx,reports,old);memory.assert_immutable(ctx['stamps']);return old
    reports=score_models(ctx);comparisons=[]
    for seed in ctx['large']['seeds']:
        lookup={r['summary']['arm']:r['summary'] for r in reports if r['summary']['seed']==seed}
        a=lookup['mask_ggf'];ca=memory.required(s,f'compact_s{seed}_mask_ggf')
        for arm in ('ggf','continued_object'):
            b=lookup[arm];key=f'paired_s{seed}_mask_vs_{arm}';ci=s.load(key)
            if ci is None:
                s.event('paired_bootstrap_start',seed=seed,control=arm)
                cb=memory.required(s,f'compact_s{seed}_{arm}')
                ci=study.paired(ca,cb,dict(ctx['cfg'],bootstrap=ctx['large']['bootstrap']))
                s.save(key,ci);del cb
            cg=a['mean_single_group_contrast_capture']-b['mean_single_group_contrast_capture']
            ag=a['mean_single_group_ap']-b['mean_single_group_ap'];vc=b['reconstruction']['ve']-a['reconstruction']['ve']
            comparisons.append(dict(seed=seed,control=arm,capture_gain=cg,ap_gain=ag,ve_cost=vc,
                eligible=a['eligible'],positive_within_predeclared_cost=bool(a['eligible'] and cg>0 and ag>=0 and vc<=ctx['cfg']['max_reconstruction_ve_drop']),
                paired_intervals=ci))
        del ca;gc.collect()
    primary=[c for c in comparisons if c['control']=='ggf']
    out=dict(complete=True,models=[r['summary'] for r in reports],comparisons=comparisons,
        positives_vs_previous_GGF=sum(c['positive_within_predeclared_cost'] for c in primary),starts=3,
        scope='Hybrid mask initialization at1024x3, Top16; same update schedule and original losses. Three new trajectories, same inspected1500-image holdout.',
        interpretation='Promote only after reading capture, AP, VE and each start/category. A gain here does not demonstrate structural adaptation, complete concept recovery, or downstream superiority.')
    tables_and_plots(ctx,reports,out);diagnostics(ctx,reports)
    memory.assert_immutable(ctx['stamps']);s.save('large_summary',out);s.write('summary.json',out)
    return out

def tables_and_plots(ctx,reports,summary):
    s=ctx['store'];rows=[];per=[]
    for r in reports:
        v=r['summary'];rows.append(dict(seed=v['seed'],method=LABELS[v['arm']],eligible=v['eligible'],
            capture_pct=100*v['mean_single_group_contrast_capture'],ap_pct=100*v['mean_single_group_ap'],
            ve_pct=100*v['reconstruction']['ve'],best_update=v.get('best_update'),
            active_groups=v['reconstruction']['active_groups'],dead_groups=v['reconstruction']['dead_groups']))
        for c in r['categories']:
            if c['category_index'] not in ctx['cfg']['category_indices']:continue
            per.append(dict(seed=v['seed'],method=LABELS[v['arm']],category=c['category'],
                contrast_group=c['contrast']['groups'][0],detector_group=c['detection']['group'],
                capture_pct=100*c['contrast']['single_group_capture'],ap_pct=100*c['detection']['ap'],
                objects=c['contrast']['object_count'],positive_images=c['test_positive_images']))
    aggregate=[]
    for arm,label in LABELS.items():
        q=[r for r in rows if r['method']==label]
        if len(q)!=3:raise ValueError('All three starts are required, including failures')
        aggregate.append(dict(method=label,**{k:float(np.mean([v[k] for v in q])) for k in ['capture_pct','ap_pct','ve_pct']},eligible=sum(v['eligible'] for v in q)))
    for name,values in [('comparison',rows),('per_category',per),('aggregate',aggregate)]:
        with (s.results/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    folder=s.results/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,k,title in zip(axes,['capture_pct','ap_pct','ve_pct'],['Object contrast capture','Detection AP','Reconstruction VE']):
        for j,r in enumerate(aggregate):
            ax.bar(j,r[k],color='#008c82' if j==2 else '#9eafc4')
            vals=[v[k] for v in rows if v['method']==r['method']];ax.scatter(np.linspace(j-.1,j+.1,3),vals,c='black',s=18,zorder=3)
        ax.set_xticks(range(3),[r['method'] for r in aggregate],rotation=25,ha='right',fontsize=8);ax.set(title=title,ylabel='%');ax.grid(axis='y',alpha=.15)
    fig.suptitle('1024 groups x 3 directions; Top16; mask initialization; reused1500-image test');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/('comparison.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,9))
    for i,seed in enumerate(ctx['large']['seeds']):
        for arm in ('mask_ggf','ggf','continued_object'):
            st=memory.required(s,f'large_mask_ggf_s{seed}') if arm=='mask_ggf' else memory.required(ctx['source'],f"{'ggf' if arm=='ggf' else 'continued'}_s{seed}")
            for ax,k in zip(axes[i],['macro_capture','macro_ap','ve']):
                vals=[100*(h['validation']['reconstruction']['ve'] if k=='ve' else h['validation'][k]) for h in st['history']]
                ax.plot([h['update'] for h in st['history']],vals,label=LABELS[arm]);ax.set(title=f'Seed{seed}: {k}',xlabel='Guided/continued updates',ylabel='%');ax.grid(alpha=.15)
            del st
        axes[i,0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/('learning_curves.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig)

def diagnostics(ctx,reports):
    s=ctx['store'];out=[]
    for seed in ctx['large']['seeds']:
        st=memory.required(s,f'large_mask_ggf_s{seed}');h=st['history']
        eligible=any(v['object_checkpoint_eligible'] for v in h)
        chosen=next(v for v in h if v['update']==st['best_update']) if eligible else h[-1]
        out.append(dict(seed=seed,eligible=eligible,selected_update=st['best_update'] if eligible else None,updates=st['updates'],
            guard_rejections=sum(not v['object_checkpoint_eligible'] for v in h),
            selected_validation=chosen['validation'],final_validation=h[-1]['validation'],
            weighted_gradient_conflicts=[v['component_gradients'] for v in h if v.get('component_gradients')]))
        new=next(r for r in reports if r['summary']['seed']==seed and r['summary']['arm']=='mask_ggf')
        old=next(r for r in reports if r['summary']['seed']==seed and r['summary']['arm']=='ggf')
        olds={v['category_index']:v for v in old['categories']};deltas=[]
        for v in new['categories']:
            if v['category_index'] not in ctx['cfg']['category_indices']:continue
            b=olds[v['category_index']]
            deltas.append(dict(category=v['category'],capture_gain=v['contrast']['single_group_capture']-b['contrast']['single_group_capture'],
                               ap_gain=v['detection']['ap']-b['detection']['ap']))
        s.write(f'category_changes_s{seed}.json',dict(capture_wins=sum(v['capture_gain']>0 for v in deltas),
            AP_wins=sum(v['ap_gain']>0 for v in deltas),categories=75,all_changes=deltas,
            capture_losses=sorted([v for v in deltas if v['capture_gain']<0],key=lambda v:v['capture_gain'])))
        del st
    s.write('component_diagnostics.json',dict(starts=out,
        note='Inspect warm reconstruction, target collisions/slot retention, gradient conflicts, selection guard and category deltas. These are measured clues, not exhaustive causal explanations.'))

def export(ctx):
    s=ctx['store'];p=s.output/f'ggf_large_mask_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for f in s.results.rglob('*'):
            if f.is_file():z.write(f,'results/'+f.relative_to(s.results).as_posix())
    return p
