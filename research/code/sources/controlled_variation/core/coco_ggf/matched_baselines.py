"""Common-cohort, budget-matched native baselines for the saved final AGF.

Only missing reconstruction fits are run. No backbone, dataset download,
token callback, initialization SVD, guided fit or exchange search is invoked.
"""
import copy, csv, gc, importlib.metadata, json, time, traceback, zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from coco_baselines import data, metrics, training
from coco_baselines.storage import identity, serial, sha_file
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import study, direction_regroup as prior
from .matched_baselines_fit import fit
from .matched_baselines_warm import fit_warm

SEEDS = [21,22,23]
NUMERICAL = ('numpy','torch','scipy','scikit-learn','einops')
LABELS = dict(vanilla='Vanilla BSF', grassmannian='Grassmannian BSF',
    group_lasso='Released Group Lasso', mask_reconstruction='Mask + reconstruction',
    mask_ggf='Mask GGF', agf='Final AGF')
ARMS = list(LABELS)


def versions():
    return {k:importlib.metadata.version(k) for k in NUMERICAL}


def discover(output, local, run=None, prefix=None):
    matches = []
    for p in (Path(output)/'results').glob('*/manifest.json'):
        m = json.loads(p.read_text())
        if (run is None or m['run_id']==run) and (prefix is None or m['run_id'].startswith(prefix)):
            matches.append(m)
    if len(matches)!=1:
        raise FileNotFoundError(f'Expected one source manifest at {output}, run={run}, prefix={prefix}; found {len(matches)}. Restore the saved run; no retraining fallback.')
    m=matches[0]
    return memory.FrozenStore(Path(output)/'checkpoints'/m['run_id'][:20], local)


def discover_seed(output, local, seed, row, master_config):
    """Choose the seed run by its master protocol and both frozen identities.

    Input-fix retries can leave several legitimate manifests in one seed
    folder. Directory count and modification time do not identify the model.
    """
    recipe=dict(master_config['config']['recipe'],seed=seed,run_recovery=False)
    matches=[];diagnostics=[]
    for path in sorted((Path(output)/'results').glob('*/manifest.json')):
        record=json.loads(path.read_text());cfg=record['config']
        mismatch=[key for key,value in dict(seed=seed,start_sha=row['root_sha'],
            source_sha=master_config['source_sha'],recipe=recipe).items() if cfg.get(key)!=value]
        if mismatch:
            diagnostics.append(dict(run=record['run_id'],reason='manifest mismatch',fields=mismatch));continue
        try:
            candidate=memory.FrozenStore(Path(output)/'checkpoints'/record['run_id'][:20],local)
            root=memory.required(candidate,'verified_start');final=memory.required(candidate,'final_agf')
            valid=(training.weight_sha(root['weights'])==row['root_sha'] and
                   training.weight_sha(final['weights'])==row['final_sha'] and
                   identity(root['spec'])==cfg['spec_sha'])
            del root,final
            if valid:matches.append(candidate)
            else:diagnostics.append(dict(run=record['run_id'],reason='frozen snapshot identity mismatch'))
        except (OSError,ValueError,KeyError,RuntimeError) as exc:
            diagnostics.append(dict(run=record['run_id'],reason=str(exc)))
    if len(matches)!=1:
        raise ValueError(f'Expected one seed{seed} run matching the master protocol and root/final hashes; found {len(matches)} in {output}. Candidates: {diagnostics}')
    print('Verified saved seed run:',seed,matches[0].run_id,flush=True)
    return matches[0]


def complete(st, updates):
    if st['updates']!=updates or st['target']!=updates:
        raise ValueError(f'Incomplete source: {st["updates"]}/{st["target"]}; expected {updates}')


def valid_weights(w, method, base):
    if not w or not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in w.values()):
        raise ValueError('Empty or nonfinite native weights')
    G,r=base['groups'],base['block_rank']
    if method=='grassmannian':
        if tuple(w['B_raw'].shape)!=(G,768,r): raise ValueError('Wrong Grassmannian capacity')
    elif tuple(w['W_dec'].shape)!=(G*r,768) or tuple(w['W_enc'].shape)!=(768,G*r):
        raise ValueError('Wrong independent-encoder capacity')


def check_targets(t, cfg):
    if sorted(v['category_index'] for v in t)!=cfg['category_indices'] or len(t)!=75:
        raise ValueError('A fixed training readout is missing or repeats categories')
    for v in t:
        for k in ('contrast_group','detector_group'):
            if not 0<=v[k]<1024: raise ValueError('Invalid readout group')
    # Strict JSON serialization also catches nonfinite scale/AP/capture values.
    identity(t)


def snapshot_weights(snap):
    return {k:snap['weights'][k] for k in ('W_dec','W_enc','b_enc')}


def source_key_stamp(s, key):
    rows=s._records(s.root,key)
    if not rows: raise FileNotFoundError('Missing/corrupt saved checkpoint '+str(s.root/key))
    _,meta,payload=rows[0]
    return dict(store=s.run_id,key=key,meta_path=str(meta),payload_path=str(payload),
                meta_sha=sha_file(meta),payload_sha=sha_file(payload))


def assert_sources(ctx):
    memory.assert_immutable(ctx['stamps'])
    for v in ctx['checkpoint_stamps']:
        if sha_file(v['meta_path'])!=v['meta_sha'] or sha_file(v['payload_path'])!=v['payload_sha']:
            raise RuntimeError('Read-only source changed: '+v['key'])


@contextmanager
def progress(ctx, model, stage):
    """Heartbeat during long bounded-batch inference, including group mapping."""
    original=model.encode; started=time.monotonic(); last=started; patches=0
    def encode(x):
        nonlocal last,patches
        z=original(x);patches+=len(x)
        if time.monotonic()-last>=20:
            ctx['store'].event('inference_progress',operation=stage,patches=patches,seconds=time.monotonic()-started)
            last=time.monotonic()
        return z
    model.encode=encode
    try: yield
    finally: model.encode=original


def prepare(paths, output, local, source_sha, require_drive=True):
    local=Path(local);local.mkdir(parents=True,exist_ok=True)
    roots={k:Path(v).resolve() for k,v in paths.items()}
    dest=Path(output).resolve()
    if any(dest==r or dest in r.parents or r in dest.parents for r in roots.values()):
        raise ValueError('New output overlaps a read-only source')
    src={}
    src['strong']=discover(roots['strong'],local/'read_strong',prefix='0995cf6f6f1a')
    q=src['strong'].config
    src['recovery']=discover(roots['recovery'],local/'read_recovery',run=q['recovery_run'])
    src['large']=discover(roots['large'],local/'read_large',run=src['recovery'].config['parent_run'])
    src['scaled']=discover(roots['scaled'],local/'read_scaled',run=src['large'].config['source_run'])
    src['original']=discover(roots['original'],local/'read_original',run=src['large'].config['activation_run'])
    if src['original'].run_id!=memory.EXPECTED_PARENT or src['scaled'].config['parent_run_id']!=src['original'].run_id:
        raise ValueError('Original activation lineage differs')
    if src['recovery'].config['original_control_run']!=src['scaled'].run_id or q['activation_run']!=src['original'].run_id:
        raise ValueError('Recovery/AGF activation lineage differs')
    actual=versions()
    for s,field in ((src['strong'],'versions'),(src['recovery'],'versions'),(src['large'],'versions'),(src['scaled'],'numerical_versions')):
        if {k:s.config[field][k] for k in NUMERICAL}!=actual:
            raise RuntimeError('Numerical versions differ from '+s.run_id+'. Run setup and restart first.')
    device=q['device']
    if device!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Use a CUDA GPU runtime, matching the saved study. CPU is only supported by local validation tests.')
    if any(s.config['device']!=device for s in (src['large'],src['recovery'])) or src['scaled'].config['config']['device']!=device:
        raise ValueError('Source device categories differ')
    for name,key in (('large','large_summary'),('recovery','recovery_summary'),('strong','experiment_complete')):
        if not memory.required(src[name],key)['complete']: raise ValueError('Source study is not complete: '+name)
    frozen={k:memory.required(src[k],'all_models_frozen') for k in ('scaled','large','recovery','strong')}
    if not all(v['complete'] for v in frozen.values()): raise ValueError('Missing frozen source models')
    base=copy.deepcopy(src['large'].config['native_config'])
    cfg=copy.deepcopy(src['large'].config['fit_config'])
    cfg.update(device=device,evaluate_every=50,local_save_every=50,audit_image_batch=2,bootstrap=200,fresh_test_seed=106051,preview_images=0)
    if (base['groups'],base['block_rank'],base['active_groups'],base['epochs'],base['batch'])!=(1024,3,16,40,2048):
        raise ValueError('Saved native budget/capacity differs')
    if (cfg['lr'],cfg['snr'],cfg['images_per_update'],cfg['grad_clip'])!=(3e-4,.1,8,5.):
        raise ValueError('Saved continuation recipe differs')
    norm=memory.required(src['original'],'normalization')
    provenance=json.loads((src['original'].results/'provenance.json').read_text())
    plan=json.loads((src['original'].results/'coverage_plan.json').read_text())
    if identity({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint'] or provenance['splits']!=plan['splits']:
        raise ValueError('Saved stratification plan disagrees')
    if src['original'].config['data_plan']!=plan['fingerprint'] or cfg['category_indices']!=plan['coverage']['category_indices'] or len(cfg['category_indices'])!=75:
        raise ValueError('Training categories/coverage changed')
    if np.asarray(norm['mean']).shape!=(196,768) or not np.isfinite(norm['mean']).all() or not np.isfinite(norm['scale']) or norm['scale']<=0:
        raise ValueError('Invalid normalization')
    test_plan=memory.required(src['strong'],'fresh_plan');test_stamp=memory.required(src['strong'],'activation_identity_test')
    test_ids=test_plan['splits']['test']
    if test_stamp['ids']!=test_ids or len(test_ids)!=1500 or test_stamp['normalization_sha']!=identity(norm):
        raise ValueError('Wrong saved final test cohort')
    train_ids=provenance['splits']['train'];select_ids=provenance['splits']['select']
    if (len(train_ids),len(select_ids))!=(2000,400) or any(len(v)!=len(set(v)) for v in (train_ids,select_ids,test_ids)):
        raise ValueError('Wrong split counts or duplicate IDs')
    if set(train_ids)&set(select_ids) or set(test_ids)&(set(train_ids)|set(select_ids)):
        raise ValueError('Train/development/test overlap')
    stamps=memory.source_snapshot(list(src.values()));key_stamps=[]
    stamps += [dict(path=str(p),sha256=sha_file(p)) for p in
               (src['original'].results/'provenance.json',src['original'].results/'coverage_plan.json')]
    for source,keys in ((src['original'],('normalization',)),
            (src['strong'],('fresh_plan','activation_identity_test','all_models_frozen','experiment_complete')),
            (src['scaled'],('all_models_frozen',)),(src['large'],('all_models_frozen','large_summary')),
            (src['recovery'],('all_models_frozen','recovery_summary'))):
        key_stamps += [source_key_stamp(source,k) for k in keys]
    # Verify all required saved payloads before expensive fitting. Test payload
    # checksums are read here; features and labels are only opened after freeze.
    for split in ('train','select'):
        for lo in range(0,len(provenance['splits'][split]),src['original'].config['config']['extract_chunk']):
            key_stamps.append(source_key_stamp(src['original'],f'features_{split}_{lo:06d}'))
    chunk=q['config']['extract_chunk']
    for lo in range(0,len(test_ids),chunk):key_stamps.append(source_key_stamp(src['strong'],f'features_test_{lo:06d}'))
    seed_stores={};reused=[];native_updates=(2000*196//2048)*40
    for seed in SEEDS:
        row=next(v for v in frozen['strong']['models'] if v['seed']==seed)
        sd=discover_seed(roots['strong']/f'seeds/{seed}',local/f'read_seed{seed}',seed,row,q)
        seed_stores[seed]=sd;stamps+=memory.source_snapshot([sd])
        if sd.config['start_sha']!=row['root_sha'] or sd.config['seed']!=seed or row['gradient_updates']!=0:
            raise ValueError('Wrong frozen AGF seed identity')
        a=memory.required(sd,'verified_start');b=memory.required(sd,'final_agf')
        for snap,sha in ((a,row['root_sha']),(b,row['final_sha'])):
            if training.weight_sha(snap['weights'])!=sha or (snap['spec']['groups'],snap['spec']['rank'],snap['spec']['active'])!=(1024,3,16):
                raise ValueError('Frozen AGF snapshot differs')
            valid_weights(snapshot_weights(snap),'vanilla',base);check_targets(snap['spec']['targets'],cfg)
        if a['spec']['targets']!=b['spec']['targets'] or not torch.equal(a['weights']['head'],b['weights']['head']):
            raise ValueError('AGF changed readouts or trained heads')
        rec=memory.required(src['recovery'],f'recovery_s{seed}_control');complete(rec,500)
        lg=memory.required(src['large'],f'large_mask_ggf_s{seed}');complete(lg,2500)
        warm=memory.required(src['large'],f'large_mask_warm_s{seed}');complete(warm,native_updates)
        native=memory.required(src['scaled'],f'native_s{seed}');complete(native,native_updates)
        cont=memory.required(src['scaled'],f'continued_s{seed}');complete(cont,2500)
        ts=memory.required(src['scaled'],f'targets_s{seed}');tm=memory.required(src['large'],f'large_mask_targets_s{seed}')
        check_targets(ts,cfg);check_targets(tm,cfg)
        if warm['best'] is None or native['best'] is None:raise ValueError('Saved reconstruction-selected warm-up missing')
        if cont['initial_sha']!=training.weight_sha(native['best']) or lg['initial_sha']!=training.weight_sha(warm['best']):
            raise ValueError('Warm-up to continuation lineage differs')
        if rec['initial_sha']!=training.weight_sha(lg['weights']) or training.weight_sha(rec['weights'])!=training.weight_sha(snapshot_weights(a)):
            raise ValueError('Final GGF root differs from ordinary recovery endpoint')
        if not torch.equal(rec['head'],a['weights']['head']) or rec['target_sha']!=identity(tm) or a['spec']['targets']!=tm:
            raise ValueError('Final GGF head/readouts differ')
        if rec['reference_mse']!=lg['reference_mse'] or rec['reference_mse']!=memory.required(src['scaled'],f'ggf_s{seed}')['initial_mse']:
            raise ValueError('Original native reconstruction reference changed')
        for w in (warm['best'],native['best'],cont['weights'],lg['weights'],rec['weights']):valid_weights(w,'vanilla',base)
        reused.append(dict(seed=seed,native_updates=native_updates,native_best_update=native['best_update'],
            mask_warm_best_update=warm['best_update'],native_reference_mse=rec['reference_mse'],
            vanilla_2500_sha=training.weight_sha(cont['weights']),mask_warm_sha=training.weight_sha(warm['best']),
            mask_ggf_sha=training.weight_sha(rec['weights']),agf_sha=training.weight_sha(snapshot_weights(b)),
            source_root_sha=row['root_sha'],source_final_sha=row['final_sha'],accepted_exchanges=len(row['accepted_proposals'])))
        for source,keys in ((sd,('verified_start','final_agf')),(src['scaled'],(f'native_s{seed}',f'continued_s{seed}',f'targets_s{seed}')),
                (src['large'],(f'large_mask_warm_s{seed}',f'large_mask_ggf_s{seed}',f'large_mask_targets_s{seed}')),
                (src['recovery'],(f'recovery_s{seed}_control',))):
            key_stamps += [source_key_stamp(source,k) for k in keys]
        del a,b,rec,lg,warm,native,cont;gc.collect()
    payload=dict(protocol='agf_matched_native_baselines_v1',seeds=SEEDS,arms=ARMS,source_sha=source_sha,
        sources={k:s.run_id for k,s in src.items()},native_config=base,fit_config=cfg,versions=actual,device=device,
        reused_catalog=reused,test_identity=test_stamp,endpoint='final 3000 continuation updates for every arm',
        readouts='Training-selected after warm-up, fixed through continuation and exchanges; no test calibration',
        test_status='Previously inspected final AGF cohort; retrospective common-cohort comparison, not a new untouched holdout')
    s=inputs.JournalStore(output,payload,local/'rescue',require_drive)
    ctx=dict(store=s,sources=src,seed_stores=seed_stores,base=base,cfg=cfg,device=device,local=local,
        norm=norm,provenance=provenance,test_ids=test_ids,test_stamp=test_stamp,test_chunk=chunk,
        stamps=stamps,checkpoint_stamps=key_stamps,reused=reused,source_sha=source_sha,packs={})
    memory.memory_safe_metrics()
    for split in ('train','select'):
        s.event('loading_saved_activations',split=split,images=len(provenance['splits'][split]))
        ctx['packs'][split]=memory.normalized_pack(src['original'],split,provenance,norm,local,src['original'].config['config']['extract_chunk'])
    s.write('protocol.json',payload);s.write('source_checkpoint_hashes.json',key_stamps)
    s.write('split_roles.json',dict(train_ids=train_ids,development_ids=select_ids,test_ids=test_ids,
        test_used_for_new_fit=False,test_previously_inspected=True))
    assert_sources(ctx);return ctx


def warm_choice(st, method):
    if st['best'] is not None:return st['best'],st['best_update'],True
    if method!='group_lasso' or st['fallback'] is None:raise ValueError('No finite warm-up checkpoint')
    row=min(st['history'],key=lambda r:(r['selection']['mse'],r['update']))
    return st['fallback'],row['update'],False


def training_targets(ctx, method, w, key):
    s=ctx['store'];sha=training.weight_sha(w);old=s.load(key)
    if old is not None:
        if old['warm_sha']!=sha:raise ValueError('Training mapping warm-up changed')
        check_targets(old['targets'],ctx['cfg']);return old['targets']
    s.event('training_readouts_start',method=method,key=key,images=len(ctx['packs']['train']['ids']))
    m=core.new_model(method,ctx['base'],w,ctx['device']);p=ctx['packs']['train']
    with progress(ctx,m,key):
        a=metrics.audit_arrays(m,p['x'],p['masks'],p['ids'],ctx['cfg'],ctx['device'])
        t=core.choose_targets(a,ctx['cfg'],ctx['provenance']['categories'])
    check_targets(t,ctx['cfg']);s.save(key,dict(warm_sha=sha,targets=t));s.write('targets/'+key+'.json',t)
    del m,a;gc.collect();torch.cuda.empty_cache();return t


def catalog_row(ctx,seed,arm,w,targets,**extra):
    method=arm if arm in ('grassmannian','group_lasso') else 'vanilla'
    m=core.new_model(method,ctx['base'],w,ctx['device'])
    params=sum(v.numel() for v in m.parameters());del m
    return dict(seed=seed,arm=arm,label=LABELS[arm],method=method,weights_sha=training.weight_sha(w),
        targets=targets,targets_sha=identity(targets),groups=1024,rank=3,directions=3072,
        core_parameters=params,auxiliary_trained_head_parameters=150 if arm in ('mask_ggf','agf') else 0,
        post_warm_gradient_updates=3000,native_updates=7640,gate='learned threshold, mean activity target16' if method=='group_lasso' else 'Top16',
        **extra)


def freeze(ctx):
    s=ctx['store'];old=s.load('all_models_frozen')
    if old is not None:
        verify_catalog(ctx,old);return old
    rows=[];base=ctx['base'];x=ctx['packs']['train']['x'].reshape(-1,768);v=ctx['packs']['select']['x'].reshape(-1,768)
    for seed in SEEDS:
        ref=next(r for r in ctx['reused'] if r['seed']==seed)['native_reference_mse']
        for arm in ('vanilla','grassmannian','group_lasso','mask_reconstruction'):
            st=None;warm=None;warm_sparse=True
            if arm=='vanilla':
                warm=memory.required(ctx['sources']['scaled'],f'native_s{seed}')
                st=memory.required(ctx['sources']['scaled'],f'continued_s{seed}')
                start=st['weights'];targets=memory.required(ctx['sources']['scaled'],f'targets_s{seed}')
                warm_update=next(r for r in ctx['reused'] if r['seed']==seed)['native_best_update']
                compute_before=st['seconds'];phase1_sha=st['sampled_batches_sha']
                s.write(f'histories/reused_vanilla_warm_s{seed}.json',warm['history'])
                s.write(f'histories/reused_vanilla_phase1_s{seed}.json',st['history'])
            else:
                if arm=='mask_reconstruction':
                    warm=memory.required(ctx['sources']['large'],f'large_mask_warm_s{seed}')
                    start,warm_update,warm_sparse=warm_choice(warm,'vanilla')
                    targets=memory.required(ctx['sources']['large'],f'large_mask_targets_s{seed}')
                    s.write(f'histories/reused_mask_warm_s{seed}.json',warm['history'])
                else:
                    warm_cfg=dict(base,local_save_updates=50)
                    warm=fit_warm(s,f'warm_{arm}_s{seed}',arm,x,v,warm_cfg,seed,base['lrs'][0],ctx['device'],drive_every=50)
                    complete(warm,7640);start,warm_update,warm_sparse=warm_choice(warm,arm)
                    targets=training_targets(ctx,arm,start,f'targets_{arm}_s{seed}')
                method=arm if arm in ('grassmannian','group_lasso') else 'vanilla'
                st=fit(ctx,f'phase1_{arm}_s{seed}',method,start,2500,71000+seed)
                complete(st,2500);start=st['weights'];compute_before=st['seconds'];phase1_sha=st['sampled_batches_sha']
            method=arm if arm in ('grassmannian','group_lasso') else 'vanilla'
            key=f'phase2_{arm}_s{seed}'
            tail=fit(ctx,key,method,start,500,82000+seed);complete(tail,500)
            rows.append(catalog_row(ctx,seed,arm,tail['weights'],targets,origin='new_tail',store='current',key=key,field='weights',
                warm_selected_update=warm_update,warm_sparsity_eligible=warm_sparse,compute_seconds=compute_before+tail['seconds'],
                warm_compute_seconds=None if warm is None else warm['seconds'],phase1_batch_sha=phase1_sha,phase2_batch_sha=tail['sampled_batches_sha'],
                best_mse_phase2_update=tail['best_mse_update'],native_reference_mse=ref,structural_exchanges=0))
            del start,tail,st,warm;gc.collect();torch.cuda.empty_cache()
        sd=ctx['seed_stores'][seed]
        for arm,key in (('mask_ggf','verified_start'),('agf','final_agf')):
            snap=memory.required(sd,key);w=snapshot_weights(snap)
            src=memory.required(ctx['sources']['recovery'],f'recovery_s{seed}_control')
            first=memory.required(ctx['sources']['large'],f'large_mask_ggf_s{seed}')
            warm=memory.required(ctx['sources']['large'],f'large_mask_warm_s{seed}')
            reuse=next(r for r in ctx['reused'] if r['seed']==seed)
            rows.append(catalog_row(ctx,seed,arm,w,snap['spec']['targets'],origin='saved_final',store='seed',key=key,field='snapshot',
                snapshot_sha=training.weight_sha(snap['weights']),warm_selected_update=reuse['mask_warm_best_update'],warm_sparsity_eligible=True,
                compute_seconds=first['seconds']+src['seconds'],warm_compute_seconds=warm['seconds'],phase1_batch_sha=first['sampled_batches_sha'],phase2_batch_sha=src['sampled_batches_sha'],
                native_reference_mse=ref,structural_exchanges=reuse['accepted_exchanges'] if arm=='agf' else 0,
                adaptation_cost='Additional saved graph and acceptance scoring, not included in gradient seconds' if arm=='agf' else None))
            s.write(f'histories/reused_mask_ggf_phase1_s{seed}.json',first['history'])
            s.write(f'histories/reused_mask_ggf_phase2_s{seed}.json',src['history'])
            del snap,w,src,first,warm;gc.collect()
        seed_rows=[r for r in rows if r['seed']==seed]
        if len({r['phase1_batch_sha'] for r in seed_rows})!=1 or len({r['phase2_batch_sha'] for r in seed_rows})!=1:
            raise ValueError('Matched continuation image schedules diverged')
        s.write(f'paired_schedule_s{seed}.json',dict(phase1_batch_sha=seed_rows[0]['phase1_batch_sha'],
            phase2_batch_sha=seed_rows[0]['phase2_batch_sha'],note='Identical sampled image schedules and noise rule; random initialization consumption differs across native classes, so exact cross-class noise equality is not claimed.'))
    out=dict(complete=True,models=rows,test_used_for_new_training=False,test_previously_inspected=True,
        endpoint_rule='Final 3000-update endpoint, no best-on-test selection',new_gradient_updates=6*7640+3*(3*2500+4*500),
        new_guided_updates=0,new_exchanges=0)
    verify_catalog(ctx,out);assert_sources(ctx);s.save('all_models_frozen',out);s.write('all_models_frozen.json',out);return out


def row_weights(ctx,row):
    if row['store']=='current':w=memory.required(ctx['store'],row['key'])[row['field']]
    else:
        snap=memory.required(ctx['seed_stores'][row['seed']],row['key'])
        if training.weight_sha(snap['weights'])!=row['snapshot_sha']:raise ValueError('Saved final snapshot changed')
        w=snapshot_weights(snap)
    if training.weight_sha(w)!=row['weights_sha']:raise ValueError('Frozen native weights changed')
    return w


def verify_catalog(ctx,frozen):
    if not frozen['complete'] or {(r['seed'],r['arm']) for r in frozen['models']}!={(s,a) for s in SEEDS for a in ARMS} or len(frozen['models'])!=18:
        raise ValueError('All eighteen fixed models are required before test scoring')
    for r in frozen['models']:
        check_targets(r['targets'],ctx['cfg']);w=row_weights(ctx,r);valid_weights(w,r['method'],ctx['base']);del w


def load_test(ctx):
    frozen=memory.required(ctx['store'],'all_models_frozen');verify_catalog(ctx,frozen)
    source=ctx['sources']['strong'];ids=ctx['test_ids'];stamp=ctx['test_stamp'];norm=ctx['norm']
    folder=ctx['local']/'normalized';folder.mkdir(exist_ok=True)
    path=folder/('final_test_'+stamp['normalized_sha']+'.npy');meta=path.with_suffix('.json')
    valid=path.exists() and meta.exists() and json.loads(meta.read_text()).get('file_sha')==sha_file(path)
    target=None
    if not valid: target=np.lib.format.open_memmap(path.with_suffix('.pending.npy'),mode='w+',dtype=np.float32,shape=(1500,196,768))
    masks=[]
    for lo in range(0,len(ids),ctx['test_chunk']):
        shard=memory.required(source,f'features_test_{lo:06d}');want=ids[lo:lo+ctx['test_chunk']]
        if shard['ids']!=want or shard['normalization_sha']!=identity(norm) or len(shard['masks'])!=len(want):
            raise ValueError('Final cached shard identity changed')
        raw=shard['x'].float().numpy() if torch.is_tensor(shard['x']) else np.asarray(shard['x'],np.float32)
        if raw.shape!=(len(want),196,768) or not np.isfinite(raw).all():raise ValueError('Invalid final cached activations')
        if target is not None:target[lo:lo+len(want)]=data.normalize(raw,norm)
        masks.extend(shard['masks']);del shard,raw
        if lo%200==0:ctx['store'].event('loading_cached_test',completed=lo+len(want),total=1500)
    if target is not None:
        target.flush();del target
        import os
        os.replace(path.with_suffix('.pending.npy'),path)
    x=np.load(path,mmap_mode='r')
    if x.shape!=(1500,196,768) or x.dtype!=np.float32 or memory.streaming_array_sha(x)!=stamp['normalized_sha']:
        raise ValueError('Final test does not match the saved AGF activation identity')
    meta.write_text(json.dumps(dict(file_sha=sha_file(path),normalized_sha=stamp['normalized_sha'])))
    ctx['packs']['test']=dict(x=x,ids=ids,masks=masks)
    assert_sources(ctx);return ctx['packs']['test']


def mapping(targets):
    return dict(mappings=[dict(category_index=t['category_index'],detector=dict(group=t['detector_group']),
        contrast=dict(groups=[t['contrast_group']])) for t in targets])


@torch.no_grad()
def activity(model,pack,device,batch):
    model.eval();group_hist=np.zeros(model.n_groups+1,np.int64);coordinate_hist=np.zeros(model.n_groups*model.group_size+1,np.int64)
    group_total=coordinate_total=n=0;x=pack['x'].reshape(-1,model.d)
    for lo in range(0,len(x),batch):
        z=model.encode(torch.as_tensor(np.array(x[lo:lo+batch],copy=True),device=device))
        g=(z.norm(dim=-1)>1e-6).sum(1).cpu().numpy();c=(z.abs()>1e-6).sum((1,2)).cpu().numpy()
        group_hist+=np.bincount(g,minlength=len(group_hist));coordinate_hist+=np.bincount(c,minlength=len(coordinate_hist))
        group_total+=int(g.sum());coordinate_total+=int(c.sum());n+=len(z)
    return dict(patches=n,mean_active_groups=group_total/n,mean_active_coordinates=coordinate_total/n,
        group_histogram=group_hist.tolist(),coordinate_histogram=coordinate_hist.tolist(),threshold=1e-6)


def audit_one(ctx,row):
    s=ctx['store'];tag=f"s{row['seed']}_{row['arm']}";p=ctx['packs']['test']
    sig=identity(dict(weights=row['weights_sha'],targets=row['targets_sha'],test=ctx['test_stamp'],cfg=ctx['cfg']))
    old=s.load('audit_'+tag)
    if old is not None:
        if old['identity']!=sig:raise ValueError('Cached final audit identity changed')
        return old['report']
    # Every model is scored by the same native inference path. Existing GGF/AGF
    # reports are checked for agreement; re-scoring is inference only.
    m=core.new_model(row['method'],ctx['base'],row_weights(ctx,row),ctx['device']);start=time.monotonic()
    s.event('common_test_scoring_start',model=tag,images=1500)
    with progress(ctx,m,tag):
        report=core.validation(m,p,row['targets'],ctx['cfg'],ctx['device'],with_arrays=True)
        compact=study.compact_audit(report.pop('_audit'),mapping(row['targets']),ctx['cfg'])
        act=activity(m,p,ctx['device'],ctx['cfg']['batch'])
    if len(report['categories'])!=75 or any(r['capture'] is None or r['ap'] is None for r in report['categories']):
        raise ValueError('Missing test evidence for a frozen category')
    dev=training.measure(m,ctx['packs']['select']['x'].reshape(-1,768),ctx['device'],ctx['cfg']['batch'])
    report.update(seed=row['seed'],arm=row['arm'],label=row['label'],weights_sha=row['weights_sha'],
        targets_sha=row['targets_sha'],test_identity=ctx['test_stamp']['normalized_sha'],activity=act,
        development_reconstruction=dev,native_reference_mse=row['native_reference_mse'],
        native_MSE_limit=row['native_reference_mse']*1.02,
        legacy_native_eligible=bool(dev['mse']<=row['native_reference_mse']*1.02),
        activity_target_eligible=bool(abs(act['mean_active_groups']-16)<=ctx['base']['sparsity_tolerance']*16),
        core_parameters=row['core_parameters'],auxiliary_trained_head_parameters=row['auxiliary_trained_head_parameters'],
        scoring_seconds=time.monotonic()-start,structural_exchanges=row['structural_exchanges'])
    if row['arm'] in ('mask_ggf','agf'):
        source=memory.required(ctx['seed_stores'][row['seed']],'fresh_result')[('ggf' if row['arm']=='mask_ggf' else 'agf')]
        diffs={k:report[k]-source[k] for k in ('macro_capture','macro_ap')}
        diffs['ve']=report['reconstruction']['ve']-source['reconstruction']['ve']
        report['saved_score_deltas']=diffs
        if max(abs(v) for v in diffs.values())>1e-5:raise ValueError('Common scorer does not reproduce saved GGF/AGF scores: '+str(diffs))
    s.save('compact_'+tag,compact);s.save('audit_'+tag,dict(identity=sig,report=report));s.write('reports/'+tag+'.json',report)
    folder=s.results/'image_contrasts';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/(tag+'.npz'),image_ids=np.asarray(compact['image_ids']),category_indices=np.asarray(compact['category_indices']),
        denominator=compact['denominator'],error=compact['error'])
    s.event('common_test_scoring_complete',model=tag,capture=report['macro_capture'],AP=report['macro_ap'],VE=report['reconstruction']['ve'])
    del m,compact;gc.collect();torch.cuda.empty_cache();return report


def evaluate(ctx):
    frozen=memory.required(ctx['store'],'all_models_frozen');verify_catalog(ctx,frozen)
    if 'test' not in ctx['packs']:raise RuntimeError('Load the verified cached test after freezing every model')
    s=ctx['store'];reports=[audit_one(ctx,r) for r in frozen['models']]
    pairs=[('mask_reconstruction','vanilla'),('mask_ggf','mask_reconstruction'),('agf','mask_ggf')]
    pairs += [(a,b) for a in ('mask_ggf','agf') for b in ('vanilla','grassmannian','group_lasso')]
    comparisons=[]
    for seed in SEEDS:
        lookup={r['arm']:r for r in reports if r['seed']==seed}
        for a,b in pairs:
            tag=f's{seed}_{a}_vs_{b}';cached=s.load('paired_'+tag)
            if cached is None:
                s.event('paired_intervals_start',seed=seed,treatment=a,control=b,replicates=200)
                ca=memory.required(s,f'compact_s{seed}_{a}');cb=memory.required(s,f'compact_s{seed}_{b}')
                cached=study.paired(ca,cb,ctx['cfg']);s.save('paired_'+tag,cached);del ca,cb;gc.collect()
            comparisons.append(dict(seed=seed,treatment=a,control=b,
                capture_gain=lookup[a]['macro_capture']-lookup[b]['macro_capture'],
                ap_gain=lookup[a]['macro_ap']-lookup[b]['macro_ap'],
                ve_gain=lookup[a]['reconstruction']['ve']-lookup[b]['reconstruction']['ve'],
                paired_intervals=cached))
            s.write('paired/'+tag+'.json',comparisons[-1])
    out=dict(complete=True,models=reports,comparisons=comparisons,
        test_status='Previously inspected final AGF cohort, identical scoring data; not a new untouched test',
        match='Decoder capacity, training data, continuation image schedule and gradient budgets. Native gate/parameter/compute differences remain explicit.',
        endpoint='Final endpoint after 3000 continuation updates; native warm-ups selected on development MSE',
        uncertainty='Whole-image paired intervals conditional on each fixed seed; three seeds share one dataset; no multiple-comparison correction',
        no_claim_of_universal_dominance=True,no_new_guided_training=True,no_new_exchanges=True)
    tables_and_plots(ctx,out);s.save('experiment_complete',out);s.write('summary.json',out);assert_sources(ctx);return out


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def tables_and_plots(ctx,out):
    s=ctx['store'];rows=[];categories=[]
    for r in out['models']:
        rows.append(dict(seed=r['seed'],model=r['label'],arm=r['arm'],capture_pct=100*r['macro_capture'],
            patch_AP_pct=100*r['macro_ap'],VE_pct=100*r['reconstruction']['ve'],MSE=r['reconstruction']['mse'],
            mean_active_groups=r['activity']['mean_active_groups'],mean_active_coordinates=r['activity']['mean_active_coordinates'],
            core_parameters=r['core_parameters'],training_head_parameters=r['auxiliary_trained_head_parameters'],
            legacy_native_eligible=r['legacy_native_eligible'],activity_target_eligible=r['activity_target_eligible'],
            structural_exchanges=r['structural_exchanges']))
        for c in r['categories']:categories.append(dict(seed=r['seed'],model=r['label'],**c))
    csv_write(s.results/'comparison.csv',rows);csv_write(s.results/'all_categories.csv',categories)
    paired=[]
    for p in out['comparisons']:
        ci=p['paired_intervals'];paired.append(dict(seed=p['seed'],treatment=p['treatment'],control=p['control'],
            capture_gain_pp=100*p['capture_gain'],AP_gain_pp=100*p['ap_gain'],VE_gain_pp=100*p['ve_gain'],
            capture_ci_lo_pp=100*ci['capture_gain_ci95'][0],capture_ci_hi_pp=100*ci['capture_gain_ci95'][1],
            AP_ci_lo_pp=100*ci['ap_gain_ci95'][0],AP_ci_hi_pp=100*ci['ap_gain_ci95'][1],
            valid_resamples=ci['valid_bootstrap_replicates'],invalid_resamples=ci['invalid_full_cohort_replicates']))
    csv_write(s.results/'paired_comparison.csv',paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=s.results/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    colors=['#879fb0','#d4a86a','#779992','#bdacd4','#5b77ae','#008779']
    for ax,field,title in zip(axes,('capture_pct','patch_AP_pct','VE_pct'),('Object contrast capture','Patch detection AP','Reconstruction VE')):
        for j,arm in enumerate(ARMS):
            vals=[r[field] for r in rows if r['arm']==arm];mean=float(np.mean(vals))
            ax.bar(j,mean,color=colors[j]);ax.scatter([j-.09,j,j+.09],vals,s=15,color='#222',zorder=3)
            ax.text(j,mean+1,f'{mean:.2f}%',ha='center',fontsize=8)
        ax.set_xticks(range(6),[LABELS[a] for a in ARMS],rotation=35,ha='right',fontsize=8)
        ax.set_ylabel('%');ax.set_title(title,fontsize=10);ax.set_ylim(min(0,min(r[field] for r in rows)),max(100,max(r[field] for r in rows)*1.12));ax.grid(axis='y',alpha=.15)
    fig.suptitle('Same 1,500 images and 1,024 x 3 dictionary; all three seeds shown',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(folder/('matched_baselines.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    means=[]
    for arm in ARMS:
        vals=[r for r in rows if r['arm']==arm]
        means.append(dict(model=LABELS[arm],**{k:float(np.mean([r[k] for r in vals])) for k in ('capture_pct','patch_AP_pct','VE_pct','mean_active_groups','mean_active_coordinates')}))
    csv_write(s.results/'three_seed_means.csv',means)


def preflight(ctx):
    """Compare saved wrapper/native outputs; time disposable native updates."""
    s=ctx['store'];old=s.load('preflight')
    if old is not None:return old
    timings={};torch_state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state_all()
    try:
        x=torch.as_tensor(np.array(ctx['packs']['train']['x'][:8],copy=True),device=ctx['device']).reshape(-1,768)
        for seed in SEEDS:
            for key in ('verified_start','final_agf'):
                snap=memory.required(ctx['seed_stores'][seed],key)
                native=core.new_model('vanilla',ctx['base'],snapshot_weights(snap),ctx['device']);wrapper=prior.model(ctx,snap)
                native.eval();wrapper.eval()
                with torch.no_grad():
                    pn,zn=native(x);pw,zw=wrapper(x)
                if not torch.allclose(pn,pw,rtol=1e-5,atol=1e-6) or not torch.allclose(zn,zw,rtol=1e-5,atol=1e-6):
                    raise ValueError('Saved wrapper and native inference differ')
                del snap,native,wrapper,pn,pw,zn,zw
        for method in ('vanilla','grassmannian','group_lasso'):
            torch.manual_seed(21);m=training.make_model(method,ctx['base'],768).to(ctx['device'])
            opt=torch.optim.Adam(m.parameters(),lr=ctx['cfg']['lr']);tt=[]
            for i in range(6):
                torch.cuda.synchronize();start=time.monotonic();m.train()
                noisy=x+ctx['cfg']['snr']*x.std()*torch.randn_like(x);loss,_=m.loss(noisy,target=x)
                if not torch.isfinite(loss):raise FloatingPointError('Native preflight failed')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),ctx['cfg']['grad_clip']);opt.step();m.normalize_decoder()
                torch.cuda.synchronize()
                if i>=2:tt.append(time.monotonic()-start)
            t=time.monotonic();rec=training.measure(m,ctx['packs']['select']['x'].reshape(-1,768),ctx['device'],ctx['cfg']['batch']);torch.cuda.synchronize()
            timings[method]=dict(seconds_per_update=float(np.median(tt)),selection_seconds=time.monotonic()-t,measurement=rec)
            del m,opt;gc.collect();torch.cuda.empty_cache()
        pending={};hours=0.
        for method in ('grassmannian','group_lasso'):
            count=0
            for seed in SEEDS:
                st=s.load(f'warm_{method}_s{seed}');count+=max(0,7640-(0 if st is None else st['updates']));del st
            pending[method+'_warm']=count
            # Warm batches are2048 rather than1568; proportional allowance only.
            hours+=count*timings[method]['seconds_per_update']*(2048/1568)/3600
        for arm in ('vanilla','grassmannian','group_lasso','mask_reconstruction'):
            method=arm if arm in ('grassmannian','group_lasso') else 'vanilla';remaining=0
            for seed in SEEDS:
                phases=[(2,500)] if arm=='vanilla' else [(1,2500),(2,500)]
                for phase,n in phases:
                    st=s.load(f'phase{phase}_{arm}_s{seed}');remaining+=max(0,n-(0 if st is None else st['updates']));del st
            pending[arm+'_continuation']=remaining;hours+=remaining*timings[method]['seconds_per_update']/3600
        out=dict(wrapper_conformance=True,numerical_finite=True,measurements=timings,pending_updates=pending,
            estimated_gradient_hours=hours,
            note='Gradient-only extrapolation. Development scoring, mapping, Drive writes, final inference and27 paired200-resample intervals add time; not a guaranteed finish time.')
        s.save('preflight',out);s.write('preflight.json',out);assert_sources(ctx);return out
    finally:
        torch.set_rng_state(torch_state);torch.cuda.set_rng_state_all(cuda_state)


def export(ctx,source_path):
    s=ctx['store'];path=s.output/f'agf_matched_baselines_{s.run_id[:12]}_results.zip'
    files=[p for p in sorted(s.results.rglob('*')) if p.is_file()]
    locations={k:dict(run=v.run_id,checkpoints=str(v.root)) for k,v in ctx['sources'].items()}
    locations['current']=dict(run=s.run_id,checkpoints=str(s.root))
    s.write('checkpoint_locations.json',locations)
    files=[p for p in sorted(s.results.rglob('*')) if p.is_file()]
    hashes={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name='results/'+p.relative_to(s.results).as_posix();z.write(p,name);hashes[name]=sha_file(p)
        z.write(source_path,'source.zip');hashes['source.zip']=sha_file(source_path)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise OSError('Results archive CRC failed')
        for name,digest in hashes.items():
            import hashlib
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise OSError('Archive hash failed: '+name)
    return path


def failure(ctx,stage,exc,source_path):
    try:
        ctx['store'].write('failure.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            resume='Rerun unchanged after correcting the environment/input issue. Mid-update failures replay the last complete verified checkpoint.'))
        print('Partial results:',export(ctx,source_path),flush=True)
    except BaseException as export_error:print('Partial export failed:',type(export_error).__name__,str(export_error),flush=True)
