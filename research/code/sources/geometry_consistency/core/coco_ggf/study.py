"""Bounded scaled GGF confirmation with unchanged Goodfire Vanilla classes.

No structural edits, Gaussian fitting, EM, or preservation penalties are run.
The historical activation source is read-only. Fresh-test choices use masks,
never model scores, and final test inference waits for all weights to freeze.
"""
import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from coco_baselines import data, metrics, training
from coco_baselines.storage import identity, sha_file
from coco_adaptation import core, coverage, inputs, run as adaptation
from coco_coordinated import run as memory
from coco_stratified import dataset, heatmaps
from coco_ggf import holdout

EXPECTED_PARENT = memory.EXPECTED_PARENT
LABELS = dict(native='Native BSF start', continued_mse='Continued BSF: reconstruction selection',
              continued_object='Continued BSF: object selection', ggf='GGF: object-guided training')


def config(device='cuda'):
    c = adaptation.config('pilot')
    c.update(version=1, study='ggf_scaled_fixed_group_confirmation', device=device,
        methods=['vanilla'], groups=1024, block_rank=3, active_groups=16,
        seeds=[21,22,23], updates=2500, native_epochs=40, native_save_updates=500,
        native_local_save_updates=100, evaluate_every=250, local_save_every=50,
        diagnostic_grad_every=250, fresh_test_images=1500, fresh_test_seed=71002,
        gallery_seed=71003, gallery_categories=12, bootstrap=100,
        max_estimated_hours=4., max_reconstruction_ve_drop=.01,
        lambda_contribution_anchor=0., lambda_gram_anchor=0.,
        anchor_ablation=False, structural_edits=False, dimension_adaptation=False,
        target_audit_images=2000, min_category_images=5, fresh_coverage_extension_limit=500,
        native_lr_source='Historical Vanilla winner selected by reconstruction only')
    return c


def prepare(parent_output, output, local, cfg, source_sha, versions, require_drive=True):
    parent = memory.FrozenStore(Path(parent_output)/'checkpoints'/EXPECTED_PARENT[:20], Path(local)/'read_parent')
    if parent.run_id != EXPECTED_PARENT: raise ValueError('Wrong stratified activation source')
    if Path(output).resolve() == Path(parent_output).resolve() or Path(parent_output).resolve() in Path(output).resolve().parents:
        raise ValueError('GGF output must be separate from the historical source')
    plan = json.loads((parent.results/'coverage_plan.json').read_text())
    provenance = json.loads((parent.results/'provenance.json').read_text())
    if identity({k:v for k,v in plan.items() if k!='fingerprint'}) != plan['fingerprint']:
        raise ValueError('Coverage plan fingerprint mismatch')
    if plan['splits'] != provenance['splits'] or parent.config['data_plan'] != plan['fingerprint']:
        raise ValueError('Historical data identity mismatch')
    cfg = copy.deepcopy(cfg); cfg['category_indices'] = plan['coverage']['category_indices']
    native = memory._discover(Path(parent_output)/'native_vanilla_s1',
        lambda c:c['config']['methods']==['vanilla'] and c['config']['seeds']==[1]
          and c['data_plan']==parent.config['data_plan'] and c['source_sha']==parent.config['source_sha'], Path(local)/'read_native')
    choices = json.loads((native.results/'models_frozen.json').read_text())
    if not choices['complete'] or len(choices['choices']) != 1: raise ValueError('Historical native recipe not frozen')
    choice = choices['choices'][0]
    if not choice['sparsity_matched']: raise ValueError('Historical native recipe has unmatched sparsity')
    base = copy.deepcopy(native.config['config'])
    base.update(groups=cfg['groups'], block_rank=cfg['block_rank'], active_groups=cfg['active_groups'],
        epochs=cfg['native_epochs'], seeds=cfg['seeds'], methods=['vanilla'], lrs=[choice['lr']],
        save_updates=cfg['native_save_updates'], local_save_updates=cfg['native_local_save_updates'], device=cfg['device'])
    norm = memory.required(parent,'normalization')
    if not np.isfinite(norm['scale']) or norm['scale']<=0: raise ValueError('Invalid saved normalization')
    store = inputs.JournalStore(output, dict(config=cfg, native_config=base, source_sha=source_sha,
        parent_run_id=parent.run_id, numerical_versions=versions, data_plan=plan['fingerprint'],
        recipe_lr=choice['lr'], upstream=training.make_model('vanilla',base,6).__class__.__module__),
        Path(local)/'rescue', require_drive)
    memory.memory_safe_metrics()
    packs={}
    for split in ('train','select','mapping'):
        packs[split]=memory.normalized_pack(parent,split,provenance,norm,local,parent.config['config']['extract_chunk'])
        store.event('reused_verified_activations',split=split,images=len(packs[split]['ids']))
    if len(packs['train']['ids']) != cfg['target_audit_images']: raise ValueError('Declared training audit size differs from saved training set')
    sets=[set(v) for v in provenance['splits'].values()]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)): raise ValueError('Historical split overlap')
    stamps=memory.source_snapshot([parent,native])
    ctx=dict(store=store,parent=parent,cfg=cfg,base=base,packs=packs,provenance=provenance,
        norm=norm,plan=plan,local=Path(local),stamps=stamps,parent_output=Path(parent_output))
    store.write('protocol_settings.json',dict(config=cfg,native_config=base,native_lr=choice['lr'],
        target_mapping='All training images, labels only. Group assignments freeze before branch training.',
        checkpoint_selection='GGF and object-selected native control use identical training-fixed category mappings and object score, under the same reconstruction guard.',
        supervision='COCO masks train GGF. Native BSF losses contain no object supervision; object-selected native is additionally label-selected.',
        nomenclature='GGF has fixed boundaries and dimensions. AGF is reserved for demonstrated structural and block-dimension adaptation.',
        historical_results='Historical 256-group scores are contextual, not the matched 1024-group comparator.',
        publication='A positive bounded confirmation is evidence, not a novelty assessment or publication guarantee.'))
    return ctx


def fresh_plan(ctx,coco=None):
    """Reuse mask-only scans. Exclude EVERY image used by the historical study."""
    s,cfg=ctx['store'],ctx['cfg'];old=s.load('fresh_plan')
    if old is not None: return old
    source=memory._discover(ctx['parent_output']/'coverage',
        lambda c:c['source_sha']==ctx['parent'].config['source_sha']
          and c['annotation_sha']==ctx['provenance']['annotation_sha'],ctx['local']/'read_coverage')
    used=set(i for ids in ctx['provenance']['splits'].values() for i in ids)
    ids=[];rows=[]
    folders=sorted(source.root.glob('coverage_scan_*'))
    s.event('historical_mask_coverage_start',chunks=len(folders))
    for chunk,folder in enumerate(folders):
        p=memory.required(source,folder.name)
        for i,e in zip(p['ids'],p['evidence']):
            if i not in used:ids.append(i);rows.append(e)
        if chunk%20==0 or chunk+1==len(folders):
            s.event('historical_mask_coverage_progress',chunks_done=chunk+1,total_chunks=len(folders),unused_candidate_images=len(ids))
    if len(ids)!=len(set(ids)) or not ids: raise ValueError('Coverage scan missing or duplicates')
    evidence=np.stack(rows);cohort=cfg['category_indices'];extensions=[]
    # The old reserve is smaller than a whole new holdout for a rare category.
    # Extend ONLY deficient categories using fresh annotation-only mask scans.
    supply=evidence.all(2).sum(0);seen=set(ids)|used
    for c in cohort:
        if supply[c]>=24:continue
        if coco is None:raise ValueError('Fresh holdout needs an annotation-only coverage extension for category '+str(c)+'. Load the official annotations first.')
        cat_id=ctx['provenance']['categories'][c]['id']
        candidates=sorted(set(coco.getImgIds(catIds=[cat_id]))-seen)
        candidates=np.random.default_rng(cfg['fresh_test_seed']+100+c).permutation(candidates)[:cfg['fresh_coverage_extension_limit']].tolist()
        added=0
        for lo in range(0,len(candidates),25):
            chosen=candidates[lo:lo+25];key=f'holdout_extra_c{c}_chunk{lo}'
            part=s.load(key)
            if part is None:
                ev=[]
                for i in chosen:
                    im=coco.imgs[i];mask=data.masks_for_image(coco,i,data.geometry(im['width'],im['height']),sorted(coco.cats))
                    ev.append(coverage.evidence(dict(ids=[i],masks=[mask]),cfg)[0])
                part=dict(ids=chosen,evidence=np.stack(ev));s.save(key,part)
            if part['ids']!=chosen:raise ValueError('Coverage extension order changed')
            ids.extend(chosen);rows.extend(part['evidence']);seen.update(chosen)
            supply+=part['evidence'].all(2).sum(0);added+=len(chosen)
            s.event('fresh_mask_coverage_extension',category_index=c,additional_images=added,usable_candidate_images=int(supply[c]))
            if supply[c]>=36:break
        extensions.append(dict(category_index=c,additional_mask_scans=added,usable_candidate_images=int(supply[c])))
    evidence=np.stack(rows)
    # Freeze the historical cohort; do not silently replace difficult categories.
    limited=evidence.copy();limited[:,[c for c in range(evidence.shape[1]) if c not in cohort]]=False
    splits,report=holdout.allocate(ids,limited,dict(test=cfg['fresh_test_images']),dict(test=20),
        cfg['fresh_test_seed'],len(cohort),dataset.allocate,s.event)
    if report['category_indices']!=cohort: raise ValueError('Fresh holdout cannot cover the declared cohort')
    index={i:j for j,i in enumerate(ids)};ev=evidence[[index[i] for i in splits['test']]]
    examples=heatmaps.select_examples(splits['test'],ev,cohort,cfg['gallery_seed'])
    # An annotation-only random category subset bounds gallery cost; every class is still scored.
    order=np.random.default_rng(cfg['gallery_seed']+1).permutation(len(examples))[:cfg['gallery_categories']]
    examples=[examples[int(i)] for i in order]
    report['split_counts']['test']=ev.all(2).sum(0).tolist()
    out=dict(splits=splits,coverage=report,categories=ctx['provenance']['categories'],gallery=examples,annotation_only_extensions=extensions,
        excluded_historical_ids=len(used),rule='Mask coverage only; 20 usable images per frozen category; no model scores; all old train/development/mapping/test IDs excluded.')
    if set(splits['test'])&used: raise AssertionError('Fresh-test leakage')
    s.save('fresh_plan',out);s.write('fresh_test_plan.json',out);return out


def sync(device):
    if device.startswith('cuda'): torch.cuda.synchronize()


def component_gradients(model,terms):
    """Measure the actual three-term objective, including reconstruction conflicts."""
    params=[p for p in model.parameters() if p.requires_grad];vectors={}
    for name,term in terms.items():
        grads=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
        vectors[name]=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])
    norms={k:float(v.norm().detach()) for k,v in vectors.items()};cosines={}
    for a,b in [('native','capture'),('native','detection'),('capture','detection')]:
        den=norms[a]*norms[b]
        cosines[a+'_vs_'+b]=None if den<=1e-20 else float((vectors[a]@vectors[b]).detach())/den
    return dict(weighted_gradient_norms=norms,gradient_cosines=cosines,
        interpretation='A negative cosine indicates locally conflicting update directions; it does not prove a unique cause of test deterioration.')


def preflight(ctx):
    """Measure real supervised updates, full validation, mapping and one Drive save."""
    s,cfg,base=ctx['store'],ctx['cfg'],ctx['base'];cached=s.load('preflight')
    if cached is not None:return cached
    device=cfg['device'];torch.manual_seed(71010)
    m=training.make_model('vanilla',base,ctx['packs']['train']['x'].shape[-1]).to(device)
    sample=inputs.split_pack(ctx['packs']['train'],np.arange(min(128,len(ctx['packs']['train']['ids']))))
    begun=time.monotonic();a=metrics.audit_arrays(m,sample['x'],sample['masks'],sample['ids'],cfg,device)
    targets=core.choose_targets(a,dict(cfg,target_min_images=1),ctx['provenance']['categories'])
    mapping_seconds=time.monotonic()-begun;del a
    head=torch.nn.Parameter(torch.zeros(len(targets),2,device=device));times={}
    for supervised in (False,True):
        opt=torch.optim.Adam(list(m.parameters())+([head] if supervised else []),lr=cfg['lr']);tt=[]
        for i in range(8):
            idx=np.arange(cfg['images_per_update']);x=torch.as_tensor(np.array(ctx['packs']['train']['x'][idx],copy=True),device=device)
            sync(device);t=time.monotonic();m.train();xx=x.reshape(-1,m.d)
            loss,_=m.loss(xx,target=xx)
            if supervised:
                cap,det,_=core.supervised_loss(m,x,[ctx['packs']['train']['masks'][j] for j in idx],targets,head,cfg)
                loss=loss+cfg['lambda_capture']*cap+cfg['lambda_detection']*det
            opt.zero_grad();loss.backward();opt.step();m.normalize_decoder();sync(device)
            if i>=3:tt.append(time.monotonic()-t)
        times['ggf' if supervised else 'continued']=float(np.median(tt))
    t=time.monotonic();validation=core.validation(m,ctx['packs']['select'],targets,cfg,device)
    val_seconds=time.monotonic()-t
    t=time.monotonic();s.save('timing_checkpoint',dict(weights=training.weights(m),optimizer=opt.state_dict()))
    save_seconds=time.monotonic()-t
    native_timing=training.preflight(ctx['packs']['train']['x'].reshape(-1,m.d),base,device)
    native_updates=len(ctx['packs']['train']['x'].reshape(-1,m.d))//base['batch']*base['epochs']
    native_sec=native_timing['measurements'][0]['seconds_per_update'];n=len(cfg['seeds'])
    validation_calls=n*(math.ceil(native_updates/base['save_updates'])+2*(1+math.ceil(cfg['updates']/cfg['evaluate_every'])))
    target_scale=max(1.,len(cfg['category_indices'])/len(targets))
    gradient=n*(native_updates*native_sec+cfg['updates']*(times['continued']+times['ggf']*target_scale))
    # Scale sample mapping to full training; real category density/Drive can change the estimate.
    mappings=n*mapping_seconds*len(ctx['packs']['train']['ids'])/len(sample['ids'])
    final_audit=n*4*val_seconds*3
    saves=n*(math.ceil(native_updates/base['save_updates'])+2*math.ceil(cfg['updates']/cfg['evaluate_every']))*save_seconds
    measured_allowance=gradient+validation_calls*val_seconds+mappings+final_audit+saves
    result=dict(native_updates_per_seed=native_updates,branch_updates_per_seed=2*cfg['updates'],
        fits=3*n,final_models=4*n,groups=base['groups'],directions=base['groups']*base['block_rank'],
        seconds_per_update=times,supervision_category_timing_scale=target_scale,native_seconds_per_update=native_sec,full_selection_seconds=val_seconds,
        target_mapping_128_images_seconds=mapping_seconds,checkpoint_seconds=save_seconds,
        rough_remaining_hours=[measured_allowance/3600,measured_allowance*2/3600],
        cap_hours=cfg['max_estimated_hours'],time_pass=measured_allowance*2/3600<=cfg['max_estimated_hours'],
        numerical_ok=all(np.isfinite(validation['reconstruction'][f]) for f in ('mse','ve')),
        excludes='Fresh-image downloads/DINO extraction, annotation planning, heatmap rendering, device throttling and stalls. Measured extrapolation with allowances, NOT a guaranteed runtime bound.')
    s.save('preflight',result);s.write('preflight.json',result);del m,head,opt;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result


def fit(ctx,key,initial,targets,supervised,seed,max_updates=None):
    """Identical batches/noise/horizon for paired arms; uninterrupted optimizer."""
    s,cfg,base=ctx['store'],ctx['cfg'],ctx['base'];device=cfg['device'];target=cfg['updates']
    torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    m=core.new_model('vanilla',base,initial,device);head=torch.nn.Parameter(torch.zeros(len(targets),2,device=device))
    opt=torch.optim.Adam(list(m.parameters())+([head] if supervised else []),lr=cfg['lr'])
    rng=np.random.default_rng(seed);st=s.load(key)
    if st is None:
        v=core.validation(m,ctx['packs']['select'],targets,cfg,device)
        st=dict(updates=0,target=target,initial_sha=training.weight_sha(initial),supervised=supervised,
            target_sha=identity(targets),weights=training.weights(m),head=head.detach().cpu(),
            best=training.weights(m),best_score=v['score'],best_update=0,
            best_mse=training.weights(m),best_mse_score=v['reconstruction']['mse'],best_mse_update=0,
            initial_mse=v['reconstruction']['mse'],history=[dict(update=0,validation=v)],seconds=0.,
            sampled_batches_sha=hashlib.sha256().hexdigest())
    else:
        if (st['target'],st['initial_sha'],st['supervised'],st['target_sha'])!=(target,training.weight_sha(initial),supervised,identity(targets)):
            raise ValueError('Branch identity changed')
        m.load_state_dict(st['weights']);head.data.copy_(st['head'].to(device));opt.load_state_dict(st['optimizer'])
        rng.bit_generator.state=st['sample_rng'];torch.set_rng_state(st['torch_rng'])
        if device.startswith('cuda'):torch.cuda.set_rng_state_all(st['cuda_rng'])
    limit=target if max_updates is None else min(target,max_updates)
    print(key,'resume',st['updates'],'/',target,flush=True)
    def save(drive=True):
        st.update(weights=training.weights(m),head=head.detach().cpu().clone(),optimizer=copy.deepcopy(opt.state_dict()),
            sample_rng=copy.deepcopy(rng.bit_generator.state),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
        s.save(key,st,drive)
    try:
        while st['updates']<limit:
            idx=rng.integers(len(ctx['packs']['train']['ids']),size=cfg['images_per_update'])
            st['sampled_batches_sha']=hashlib.sha256(st['sampled_batches_sha'].encode()+idx.tobytes()).hexdigest()
            x=torch.as_tensor(np.array(ctx['packs']['train']['x'][idx],copy=True),device=device)
            m.train();began=time.monotonic();xx=x.reshape(-1,m.d)
            noisy=xx+cfg['snr']*xx.std()*torch.randn_like(xx) if cfg['snr'] else xx
            native,info=m.loss(noisy,target=xx);cap=det=native*0;objects=0
            if supervised:cap,det,objects=core.supervised_loss(m,x,[ctx['packs']['train']['masks'][i] for i in idx],targets,head,cfg)
            terms=dict(native=native,capture=cfg['lambda_capture']*cap,detection=cfg['lambda_detection']*det)
            u=st['updates']+1;diagnostics=None
            if supervised and u%cfg['diagnostic_grad_every']==0:
                diagnostics=component_gradients(m,terms)
            loss=sum(terms.values())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite GGF/native loss')
            opt.zero_grad();loss.backward()
            norms={name:float(p.grad.norm()) for name,p in m.named_parameters() if p.grad is not None}
            grad=float(torch.nn.utils.clip_grad_norm_(list(m.parameters())+([head] if supervised else []),cfg['grad_clip']))
            if not np.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
            opt.step();m.normalize_decoder();sync(device);st['seconds']+=time.monotonic()-began;st['updates']=u
            if u%cfg['evaluate_every']==0 or u==target:
                val=core.validation(m,ctx['packs']['select'],targets,cfg,device)
                eligible=val['reconstruction']['mse']<=st['initial_mse']*(1+cfg['fit_reconstruction_tolerance'])
                if eligible and val['score'] is not None and val['score']>st['best_score']:
                    st.update(best=training.weights(m),best_score=val['score'],best_update=u)
                if val['reconstruction']['mse']<st['best_mse_score']:
                    st.update(best_mse=training.weights(m),best_mse_score=val['reconstruction']['mse'],best_mse_update=u)
                st['history'].append(dict(update=u,validation=val,object_checkpoint_eligible=bool(eligible),
                    rejection_reasons=[] if eligible else ['reconstruction_exceeded_predeclared_guard'],
                    native_loss=float(native.detach()),capture_loss=float(cap.detach()),detection_loss=float(det.detach()),
                    total_loss=float(loss.detach()),native_loss_info=info,gradient_norm=grad,parameter_gradient_norms=norms,
                    component_gradients=diagnostics,objects_in_batch=objects,seconds_per_update=st['seconds']/u))
                save();s.write('histories/'+key+'.json',st['history'])
                s.event('branch_checkpoint',key=key,update=u,total=target,best_object_update=st['best_update'],
                    best_mse_update=st['best_mse_update'],capture=val['macro_capture'],AP=val['macro_ap'],VE=val['reconstruction']['ve'])
            elif u%cfg['local_save_every']==0:
                save(False);print(key,u,'/',target,flush=True)
        save()
    except BaseException as exc:
        save(False)
        try:save();s.event('fit_error',key=key,update=st['updates'],error_type=type(exc).__name__,error=str(exc))
        except BaseException:pass
        raise
    del m,head,opt;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return st


def train(ctx):
    s,cfg,base=ctx['store'],ctx['cfg'],ctx['base'];old=s.load('all_models_frozen')
    if old is not None:return old
    models=[]
    for seed in cfg['seeds']:
        key=f'native_s{seed}';x=ctx['packs']['train']['x'].reshape(-1,ctx['packs']['train']['x'].shape[-1])
        selection=ctx['packs']['select']['x'].reshape(-1,x.shape[-1])
        native=training.fit(s,key,'vanilla',x,selection,base,seed,base['lrs'][0],cfg['device'])
        start=native['best'];native_update=native['best_update'];native_total=native['updates']
        if start is None:raise RuntimeError('No eligible native model')
        history=native['history'];latest=history[-1]['selection']['mse']
        previous=history[-2]['selection']['mse'] if len(history)>1 else None
        s.write(f'native_maturity_s{seed}.json',dict(total_updates=native_total,best_update=native_update,
            epochs=base['epochs'],best_selection_mse=native['best_score'],last_selection_mse=latest,
            final_checkpoint_relative_improvement=None if previous is None else (previous-latest)/max(previous,1e-12),
            note='A learning-curve diagnostic, not proof of global convergence or optimal learning rate at the larger capacity.'))
        target_key=f'targets_s{seed}';targets=s.load(target_key)
        if targets is None:
            s.event('training_group_mapping_start',seed=seed,groups=base['groups'],images=len(ctx['packs']['train']['ids']))
            m=core.new_model('vanilla',base,start,cfg['device'])
            audit=metrics.audit_arrays(m,ctx['packs']['train']['x'],ctx['packs']['train']['masks'],ctx['packs']['train']['ids'],cfg,cfg['device'])
            targets=core.choose_targets(audit,cfg,ctx['provenance']['categories']);del m,audit;gc.collect()
            if sorted(t['category_index'] for t in targets)!=cfg['category_indices']:raise ValueError('Training mapping missed frozen categories')
            s.save(target_key,targets);s.write('targets/'+target_key+'.json',targets)
        collisions={}
        for field in ('contrast_group','detector_group'):
            grouped={}
            for t in targets:grouped.setdefault(t[field],[]).append(t['category'])
            collisions[field]=[dict(group=g,categories=c) for g,c in grouped.items() if len(c)>1]
        s.write(f'mapping_collisions_s{seed}.json',dict(collisions=collisions,
            note='Multiple categories can map to the same group. This does not prove independent concept recovery; all such assignments remain visible.'))
        models.append(dict(key=key,field='best',arm='native',seed=seed,updates=native_total,best_update=native_update,
            weights_sha=training.weight_sha(start),parameters=native['parameters'],auxiliary_training_head_parameters=0))
        del native
        continuation=fit(ctx,f'continued_s{seed}',start,targets,False,71000+seed)
        for field,arm,upd in [('best_mse','continued_mse','best_mse_update'),('best','continued_object','best_update')]:
            models.append(dict(key=f'continued_s{seed}',field=field,arm=arm,seed=seed,updates=cfg['updates'],best_update=continuation[upd],
                weights_sha=training.weight_sha(continuation[field]),parameters=models[-1]['parameters'],auxiliary_training_head_parameters=0))
        batch_sha=continuation['sampled_batches_sha'];del continuation
        guided=fit(ctx,f'ggf_s{seed}',start,targets,True,71000+seed)
        if guided['sampled_batches_sha']!=batch_sha:raise ValueError('Paired branches did not see identical minibatches')
        models.append(dict(key=f'ggf_s{seed}',field='best',arm='ggf',seed=seed,updates=cfg['updates'],best_update=guided['best_update'],
            weights_sha=training.weight_sha(guided['best']),parameters=models[-1]['parameters'],auxiliary_training_head_parameters=2*len(targets)))
        s.write(f'paired_training_s{seed}.json',dict(common_start_sha=training.weight_sha(start),
            minibatches_sha=batch_sha,matched_gradient_updates=cfg['updates'],native_pretrain_updates=native_total,
            checkpoint_rules=dict(continued_mse='Minimum selection reconstruction MSE',
                continued_object='Maximum capture + 0.2 AP under reconstruction guard',ggf='Same object rule and guard'),
            object_selected_control_uses_validation_labels=True,ggf_auxiliary_head_parameters=2*len(targets),
            shared_dictionary_capacity=True,structural_edits=0))
        del start,guided;gc.collect()
    out=dict(complete=True,models=models,test_used=False,structural_edits=0,dimensions_changed=False)
    s.save('all_models_frozen',out);s.write('all_models_frozen.json',out)
    memory.assert_immutable(ctx['stamps']);return out


def compact_audit(audit, frozen, cfg):
    """Retain exact one-group scores and per-image contrast errors for paired CIs.

    Full all-group test codes are transient, keeping RAM/Drive costs bounded.
    Native codes, decoder contributions and calibration-chosen IDs are unchanged.
    """
    cohort=cfg['category_indices'];C=len(cohort);n=len(audit['image_ids'])
    y,valid=metrics.binary_labels(audit,cfg)
    scores=np.zeros((len(y),C),np.float32);den=np.zeros((n,C),np.float64);err=den.copy()
    cats=np.array([o['category'] for o in audit['objects']]);index={r['category_index']:r for r in frozen['mappings']}
    for j,c in enumerate(cohort):
        item=index[c]
        if item['detector'] is None or item['contrast'] is None:raise ValueError('Missing calibration mapping for frozen category '+str(c))
        scores[:,j]=audit['heat'][:,item['detector']['group']]
        take=np.flatnonzero(cats==c);dx=audit['dx'][take].astype(np.float64)
        im=np.array([audit['objects'][i]['image_index'] for i in take],int)
        g=item['contrast']['groups'][0];q=audit['dz'][take,g]@audit['atoms'][g]
        den[:,j]=np.bincount(im,weights=np.square(dx).sum(1),minlength=n)
        err[:,j]=np.bincount(im,weights=np.square(dx-q).sum(1),minlength=n)
    return dict(image_ids=audit['image_ids'],category_indices=cohort,detection_scores=scores,
        positive=y[:,cohort],valid=valid[:,cohort],patch_image_index=audit['image_index'],
        denominator=den,error=err,objects=[o['annotation_id'] for o in audit['objects']])


def prepared_ap(scores, y, valid, images):
    order=np.argsort(-scores,kind='stable');sorted_scores=scores[order]
    ends=np.r_[np.flatnonzero(sorted_scores[:-1]!=sorted_scores[1:]),len(order)-1]
    return dict(ends=ends,y=(y&valid)[order],valid=valid[order],images=images[order])


def weighted_prepared_ap(p, image_counts):
    weights=image_counts[p['images']]
    tp=np.cumsum(p['y']*weights,dtype=np.float64);known=np.cumsum(p['valid']*weights,dtype=np.float64)
    if tp[-1]==0 or tp[-1]==known[-1]:return None
    ends=p['ends'];precision=np.divide(tp[ends],known[ends],out=np.zeros(len(ends)),where=known[ends]>0)
    return float(np.sum(np.diff(np.r_[0.,tp[ends]])*precision)/tp[-1])


def paired(a,b,cfg):
    if (a['image_ids'],a['category_indices'],a['objects'])!=(b['image_ids'],b['category_indices'],b['objects']):raise ValueError('Paired audit identities differ')
    for k in ('positive','valid','patch_image_index','denominator'):
        if not np.array_equal(a[k],b[k]):raise ValueError('Paired audit data differ: '+k)
    n=len(a['image_ids']);C=len(a['category_indices']);rng=np.random.default_rng(cfg['fresh_test_seed']+91)
    pa=[prepared_ap(a['detection_scores'][:,j],a['positive'][:,j],a['valid'][:,j],a['patch_image_index']) for j in range(C)]
    pb=[prepared_ap(b['detection_scores'][:,j],b['positive'][:,j],b['valid'][:,j],b['patch_image_index']) for j in range(C)]
    caps=[];aps=[];invalid=0
    for _ in range(cfg['bootstrap']):
        counts=np.bincount(rng.integers(n,size=n),minlength=n);den=counts@a['denominator']
        cap=np.divide(counts@(b['error']-a['error']),den,out=np.full(C,np.nan),where=den>1e-12)
        ap=[None if (x:=weighted_prepared_ap(pa[j],counts)) is None or (y:=weighted_prepared_ap(pb[j],counts)) is None else x-y for j in range(C)]
        if np.isfinite(cap).all() and all(v is not None for v in ap):caps.append(cap);aps.append(ap)
        else:invalid+=1
    if not caps:raise ValueError('No full-cohort paired bootstrap replicate; inspect category coverage')
    ca,da=np.asarray(caps),np.asarray(aps)
    return dict(capture_gain_ci95=metrics.interval(ca.mean(1)),ap_gain_ci95=metrics.interval(da.mean(1)),
        per_category=[dict(category_index=c,capture_gain_ci95=metrics.interval(ca[:,j]),ap_gain_ci95=metrics.interval(da[:,j])) for j,c in enumerate(a['category_indices'])],
        valid_bootstrap_replicates=len(caps),invalid_full_cohort_replicates=invalid,
        note='Paired whole-image bootstrap with the same fixed category denominator. Conditional on trained weights and this coverage-enriched sample; not a training-seed or population guarantee.')


def load_test(ctx,coco,token_callback):
    if ctx['store'].load('all_models_frozen') is None:raise RuntimeError('Freeze all models before test inference')
    plan=fresh_plan(ctx)
    raw=dataset.extract(ctx['store'],coco,plan,'test',ctx['cfg'],ctx['provenance'],ctx['local'],token_callback)
    raw['x']=data.normalize(raw['x'],ctx['norm'])
    return raw


def evaluate(ctx,test):
    s,cfg,base=ctx['store'],ctx['cfg'],ctx['base'];frozen=memory.required(s,'all_models_frozen');reports=[]
    for arm in frozen['models']:
        tag=f"s{arm['seed']}_{arm['arm']}";cached=s.load('report_'+tag)
        if cached is not None:reports.append(cached);print('Reused audit',tag,flush=True);continue
        w=memory.required(s,arm['key'])[arm['field']]
        if training.weight_sha(w)!=arm['weights_sha']:raise ValueError('Frozen model changed')
        m=core.new_model('vanilla',base,w,cfg['device']);del w
        mapping=s.load('mapping_'+tag)
        if mapping is None:
            s.event('calibration_start',model=tag)
            p=ctx['packs']['mapping'];a=metrics.audit_arrays(m,p['x'],p['masks'],p['ids'],cfg,cfg['device'])
            mapping=metrics.calibrate(a,cfg,ctx['provenance']['categories']);del a;gc.collect()
            s.save('mapping_'+tag,mapping)
            s.write('mappings/'+tag+'.json',dict(mappings=mapping['mappings'],weights_sha=arm['weights_sha'],
                label_use='Disjoint mapping images select evaluation group IDs/thresholds only; neither model weights nor losses. GGF training separately uses training masks.'))
        s.event('test_scoring_start',model=tag)
        audit=metrics.audit_arrays(m,test['x'],test['masks'],test['ids'],cfg,cfg['device'])
        # Pairwise intervals below are primary; avoid repeatedly sorting scores for per-model bootstraps.
        rows=metrics.evaluate(audit,mapping,dict(cfg,bootstrap=0),cfg['fresh_test_seed'])
        indexed={r['category_index']:r for r in rows};selected=[indexed[c] for c in cfg['category_indices']]
        if any(r['contrast'] is None or r['detection'] is None for r in selected):raise ValueError('Frozen category denominator missing test evidence')
        rec=training.measure(m,test['x'].reshape(-1,m.d),cfg['device'],cfg['batch'])
        compact=compact_audit(audit,mapping,cfg);s.save('compact_'+tag,compact)
        summary=dict(**arm,method='vanilla',label=LABELS[arm['arm']],groups=base['groups'],block_rank=base['block_rank'],
            reconstruction=rec,mean_single_group_contrast_capture=float(np.mean([r['contrast']['single_group_capture'] for r in selected])),
            mean_single_group_ap=float(np.mean([r['detection']['ap'] for r in selected])),
            mean_single_group_iou=float(np.mean([r['detection']['iou'] for r in selected])),
            mean_multi_group_contrast_capture=float(np.mean([r['contrast']['multi_group_capture'] for r in selected])),
            mean_all_group_contrast_capture=float(np.mean([r['contrast']['all_group_capture'] for r in selected])),
            categories=len(selected),images=len(test['ids']),structural_edits=0,dimensions_changed=False)
        report=dict(summary=summary,categories=rows,skipped_objects=audit['skipped'])
        s.save('report_'+tag,report);s.write('reports/'+tag+'.json',report);reports.append(report)
        s.event('model_audit_complete',**summary);del m,audit,compact;gc.collect()
        if cfg['device'].startswith('cuda'):torch.cuda.empty_cache()
    comparisons=[]
    for seed in cfg['seeds']:
        r={v['summary']['arm']:v for v in reports if v['summary']['seed']==seed};a=r['ggf']['summary']
        ca=memory.required(s,f'compact_s{seed}_ggf')
        for other in ('native','continued_mse','continued_object'):
            b=r[other]['summary'];tag=f's{seed}_vs_{other}';ci=s.load('paired_'+tag)
            if ci is None:
                s.event('paired_bootstrap_start',seed=seed,control=other)
                ci=paired(ca,memory.required(s,f'compact_s{seed}_{other}'),cfg);s.save('paired_'+tag,ci)
            gain=a['mean_single_group_contrast_capture']-b['mean_single_group_contrast_capture']
            apgain=a['mean_single_group_ap']-b['mean_single_group_ap'];vedrop=b['reconstruction']['ve']-a['reconstruction']['ve']
            comparisons.append(dict(seed=seed,control=other,capture_gain=gain,ap_gain=apgain,
                reconstruction_ve_drop=vedrop,active_group_difference=a['reconstruction']['active_groups']-b['reconstruction']['active_groups'],
                positive_within_predeclared_reconstruction_cost=gain>0 and apgain>=0 and vedrop<=cfg['max_reconstruction_ve_drop'],
                paired_intervals=ci))
        del ca
    primary=[r for r in comparisons if r['control']=='continued_object']
    summary=dict(models=[r['summary'] for r in reports],comparisons=comparisons,
        primary_control='continued_object: same object-based checkpoint rule as GGF, native loss only',
        fresh_start_count=len(primary),seeds_improving_capture=sum(r['capture_gain']>0 for r in primary),
        seeds_improving_capture_and_ap_within_ve_cost=sum(r['positive_within_predeclared_reconstruction_cost'] for r in primary),
        mean_primary_capture_gain=float(np.mean([r['capture_gain'] for r in primary])),
        primary_capture_gain_range=[min(r['capture_gain'] for r in primary),max(r['capture_gain'] for r in primary)],
        mean_primary_ap_gain=float(np.mean([r['ap_gain'] for r in primary])),
        scope='Fixed groups, object-supervised GGF on final DINOv3 ViT-B/16 patches. Direct unchanged Goodfire Vanilla class controls on identical COCO data/capacity/sparsity. Not published ImageNet replication, full manifold recovery, or an exact feature-splitting rate.',
        independence='Three new dictionary starts, one reused training/development dataset and one new coverage-enriched test set. No seed filtering. Test is evaluated only after all model weights freeze.',
        positive_evidence_rule='Exploratory: positive capture, nonnegative AP, at most 1pp reconstruction VE cost, assessed seed by seed against the object-selected native continuation. Report continuous effects/CIs even if this descriptive rule fails.',
        limitations='No guarantee of publication/novelty, optimality, adaptation, downstream task preservation or generalization beyond the measured benchmark. Three starts are not three independent datasets.')
    s.write('summary.json',summary)
    adaptation.export_categories(s,reports)
    plots(ctx,reports)
    memory.assert_immutable(ctx['stamps'])
    return reports,summary


def plots(ctx,reports):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    folder=ctx['store'].results/'figures';folder.mkdir(parents=True,exist_ok=True)
    arms=list(LABELS);fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,field,title in zip(axes,['mean_single_group_contrast_capture','mean_single_group_ap','VE'],
        ['Single-group object contrast capture','Object detection AP','Reconstruction VE']):
        for j,arm in enumerate(arms):
            values=[100*(r['summary']['reconstruction']['ve'] if field=='VE' else r['summary'][field]) for r in reports if r['summary']['arm']==arm]
            ax.bar(j,np.mean(values),color=['#bac7d1','#9ab1c1','#b59ccb','#008d83'][j],alpha=.8)
            ax.scatter(np.linspace(j-.10,j+.10,len(values)),values,c='black',s=25,zorder=3)
        ax.set_xticks(range(len(arms)),[LABELS[k] for k in arms],rotation=30,ha='right',fontsize=8)
        ax.set(title=title,ylabel='%');ax.grid(axis='y',alpha=.15);ax.axhline(0,color='#444',lw=.5)
    fig.suptitle('GGF versus matched native Goodfire Vanilla: all fresh starts shown');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'ggf_vs_bsf.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(len(ctx['cfg']['seeds']),3,figsize=(15,3.5*len(ctx['cfg']['seeds'])),squeeze=False)
    for i,seed in enumerate(ctx['cfg']['seeds']):
        for arm,label in [('continued','Continued native BSF'),('ggf','GGF')]:
            st=ctx['store'].load(f'{arm}_s{seed}');h=st['history']
            for ax,f,title in zip(axes[i],['macro_capture','macro_ap','VE'],['Selection capture','Selection AP','Selection reconstruction VE']):
                values=[100*(r['validation']['reconstruction']['ve'] if f=='VE' else r['validation'][f]) for r in h]
                ax.plot([r['update'] for r in h],values,label=label);ax.set(title=f'Seed {seed}: {title}',xlabel='Additional updates',ylabel='%');ax.grid(alpha=.15)
        axes[i,0].legend()
    fig.tight_layout();fig.savefig(folder/'learning_curves.png',dpi=160);plt.close(fig)


def gallery(ctx,test,coco):
    """Small annotation-fixed gallery, identical calibration color scale across arms."""
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    import csv
    import html
    s,cfg=ctx['store'],ctx['cfg'];examples=fresh_plan(ctx)['gallery'];ix={i:j for j,i in enumerate(test['ids'])}
    models=memory.required(s,'all_models_frozen')['models'];folder=s.results/'heatmaps';folder.mkdir(exist_ok=True)
    records=[];figures=[]
    for seed in cfg['seeds']:
        done=s.load(f'gallery_s{seed}')
        if done is not None and all((folder/n).exists() and sha_file(folder/n)==h for n,h in done['hashes'].items()):
            records+=done['rows'];figures+=done['figures'];continue
        maps={};heats={};scales={}
        for arm in ('continued_object','ggf'):
            item=next(r for r in models if r['seed']==seed and r['arm']==arm)
            state=memory.required(s,item['key']);m=core.new_model('vanilla',ctx['base'],state[item['field']],cfg['device']);del state
            mapping=memory.required(s,f'mapping_s{seed}_{arm}');maps[arm]={r['category_index']:r for r in mapping['mappings']}
            heats[arm]=[heatmaps.group_heat(m,np.array(test['x'][ix[e['image_id']]],copy=True),cfg['device']) for e in examples]
            gs=sorted({maps[arm][e['category_index']]['detector']['group'] for e in examples});hh=[]
            for image in ctx['packs']['mapping']['x']:hh.append(heatmaps.group_heat(m,np.array(image,copy=True),cfg['device'])[:,gs])
            h=np.concatenate(hh);scales[arm]={g:max(float(np.quantile(h[:,j],.99)),1e-8) for j,g in enumerate(gs)};del m,h,hh
        start=len(records);fst=len(figures)
        np.savez_compressed(folder/f's{seed}_all_group_heat.npz',image_ids=[e['image_id'] for e in examples],**{k:np.stack(v) for k,v in heats.items()})
        for j,e in enumerate(examples):
            i=ix[e['image_id']];c=e['category_index'];name=ctx['provenance']['categories'][c]['name'];im=coco.imgs[e['image_id']]
            path=data.download(data.COCO_BUCKET+'train2017/'+im['file_name'],ctx['local']/'images'/im['file_name'])
            with Image.open(path) as image:crop=np.asarray(data.transform_image(image.convert('RGB'),data.geometry(*image.size)))
            mm=test['masks'][i]['fraction'][:,c].astype(float)/255;scale=max(scales[arm][maps[arm][c]['detector']['group']] for arm in heats)
            fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].imshow(crop);axes[0].imshow(mm.reshape(14,14),extent=(0,224,224,0),cmap='Greens',alpha=.45,vmin=0,vmax=1);axes[0].set_title(name+' mask')
            for k,arm in enumerate(heats,1):
                heat=heats[arm][j];g=maps[arm][c]['detector']['group'];cg=maps[arm][c]['contrast']['groups'][0]
                axes[k].imshow(crop);axes[k].imshow(heat[:,g].reshape(14,14),extent=(0,224,224,0),cmap='magma',alpha=.65,vmin=0,vmax=scale)
                axes[k].set_title(f'{LABELS[arm]}\nDetector {g}; capture {cg}',fontsize=9)
                known=test['masks'][i]['crowd']<=cfg['negative_fraction'];pos=(mm>=cfg['positive_fraction'])&known;neg=(mm<=cfg['negative_fraction'])&known
                energy=heat[pos].mean(0)
                for rank,h in enumerate(np.argsort(-energy)[:5],1):records.append(dict(seed=seed,arm=arm,category=name,image_id=e['image_id'],detector_group=g,capture_group=cg,
                    region_rank=rank,group_index=int(h),mean_object_activation=float(energy[h]),mean_other_activation=float(heat[neg,h].mean())))
            for ax in axes:ax.axis('off')
            fig.suptitle(f'Seed {seed}: annotation-fixed example {e["image_id"]}; shared calibration scale');fig.tight_layout()
            namefile=f's{seed}_c{c}_im{e["image_id"]}.png';fig.savefig(folder/namefile,dpi=140);plt.close(fig)
            figures.append(dict(seed=seed,category=name,path=namefile,image_id=e['image_id']))
        names=[r['path'] for r in figures[fst:]]+[f's{seed}_all_group_heat.npz']
        s.save(f'gallery_s{seed}',dict(rows=records[start:],figures=figures[fst:],hashes={n:sha_file(folder/n) for n in names}))
        s.event('gallery_seed_complete',seed=seed,examples=len(examples))
    with (folder/'region_group_indices.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    cards=''.join(f'<h2>Seed {r["seed"]}: {html.escape(r["category"])}</h2><img loading="lazy" src="{r["path"]}">' for r in figures)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GGF heatmaps</title><style>body{max-width:1200px;margin:32px auto;font:16px system-ui}img{width:100%}</style><h1>GGF versus continued native BSF</h1><p>Examples selected from masks before training. Detector/capture IDs selected on separate calibration images. Shared calibration color scales. Group IDs are local to each model. Brightness is activation strength, not proof of concept purity or causal importance.</p>'+cards,encoding='utf-8')
    s.write('heatmap_manifest.json',dict(figures=figures,indices='heatmaps/region_group_indices.csv',raw_maps='heatmaps/*_all_group_heat.npz'))
    return folder/'index.html'


def export(ctx,source):
    import zipfile
    import shutil
    s=ctx['store'];local=s.rescue/'ggf_results.zip'
    with zipfile.ZipFile(local,'w',zipfile.ZIP_DEFLATED) as z:
        for prefix,root in [('results',s.results),('source',Path(source))]:
            for p in root.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,prefix+'/'+p.relative_to(root).as_posix())
    dest=s.output/f'ggf_coco_scaled_{s.run_id[:12]}_results.zip';s.check_mount();shutil.copyfile(local,dest)
    s.write('export.json',dict(zip=str(dest),sha256=sha_file(dest),checkpoints=str(s.root),
        note='Full model/optimizer/RNG checkpoints, activation shards, compact paired-bootstrap arrays and calibration mappings remain in Drive. ZIP includes every seed/category, plots, heatmaps, histories, diagnostics, source and protocol.'))
    return dest
