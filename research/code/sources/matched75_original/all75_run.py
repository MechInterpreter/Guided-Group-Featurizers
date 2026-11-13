"""Train missing genuine75-group endpoints; never launch audit scoring here."""
import argparse, copy, gc, hashlib, json, os, signal, sys, threading, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'core')]
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched
import cd_common, cd_data, cd_train

read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def configure(regime):
    if regime not in ['full_category','category_split']:raise ValueError('Unknown regime')
    full=regime=='full_category'
    cd_common.SPEC.update(revision='all75-'+regime+'-v1',seeds=[21,22,23],groups=75,width=12,top_k=32,
        warm_updates=7640,guided_updates=3000,lr=.0003,continuation_lr=.001 if full else .0003,
        cap=.6 if full else .4,snr=0. if full else .1,methods=['vanilla','supervised_vanilla','ggf'],
        inference='Exactly75 native BSF groups, width12, Top32. Native SLiCS reference remains separate.',
        scope='Full-category semantic study' if full else 'Original reversed category-supervision split at75 BSF groups')
    if full:cd_common.SPEC['sampling']='epochs'
    else:cd_common.SPEC.pop('sampling',None)

def check_weights(weights,groups=75,width=12,d=768):
    expected=dict(W_dec=(groups*width,d),W_enc=(d,groups*width),b_enc=(groups*width,))
    if set(weights)!=set(expected) or any(tuple(weights[k].shape)!=v for k,v in expected.items()):
        raise ValueError('Not a genuine declared-capacity checkpoint')
    if any(not torch.isfinite(w).all() for w in weights.values()):raise ValueError('Nonfinite checkpoint')

def mounted_path(drive,path):
    value=str(path).replace('\\','/')
    if '/MyDrive/' in value:value=value.split('/MyDrive/',1)[1]
    elif value.startswith('MyDrive/'):value=value[len('MyDrive/'):]
    else:raise ValueError('Expected recorded MyDrive source path')
    return Path(drive)/value

def reuse_full_catalog(options):
    """Exact serialized and native tensor checks; no permissive architecture reuse."""
    result=[]
    for r in read(HERE/'REUSE_CATALOG.json')['models']:
        pinned=mounted_path(options['drive_root'],r['drive_path'])
        candidates=[pinned,*sorted(pinned.parent.glob('slot*.pt'))]
        candidates=list(dict.fromkeys(candidates))
        exact=[p for p in candidates if p.is_file() and sha(p)==r['file_sha256']]
        if not exact:raise FileNotFoundError('Exact existing payload unavailable; restore original, do not silently retrain: '+str(pinned))
        p=exact[0];s=torch.load(p,map_location='cpu',weights_only=False)
        if r['collector_method']=='slics':
            if tuple(s['D'].shape)!=(900,768) or s['updates']!=10640:raise ValueError('SLiCS capacity/endpoint mismatch')
            w={'D':s['D']}
        else:
            if (s['groups'],s['active'],s['updates'])!=(75,32,3000):raise ValueError('GGF capacity/endpoint mismatch')
            w=s['weights'];check_weights(w)
        if training.weight_sha(w)!=r['weights_sha']:raise ValueError('Reused tensor identity differs')
        result.append(dict(r,path=str(p),status='mounted_payload_verified',native_groups=75))
        del s,w;gc.collect()
    if len(result)!=6:raise ValueError('Missing reusable full-category endpoint')
    return result

def reuse_native_category(options,folds):
    result=[]
    for r in read(HERE/'REUSE_CATALOG.json')['category_references']:
        row=copy.deepcopy(r['source_row']);row['output']=str(mounted_path(options['drive_root'],row['output']))
        store=memory.FrozenStore(Path(row['output'])/'checkpoints'/row['run_id'][:20],Path(options['local'])/'read_native_reference')
        state=memory.required(store,'frozen')
        if state['guide']!=folds[r['fold']]['guide'] or state['excluded']!=folds[r['fold']]['excluded']:
            raise ValueError('SLiCS reference category partition differs')
        weights=state['weights'];g=len(state['guide'])
        if tuple(weights.shape)!=(g*12,768) or training.weight_sha({'D':weights})!=row['weights_sha']:
            raise ValueError('Native reference payload mismatch')
        result.append(dict(row,native_groups=g,width=12,capacity_matched=False,status='mounted_payload_verified',role='native_reference_only'))
        del state,weights;gc.collect()
    return result

ORIGINAL_CONTEXT=cd_train.training_context
def training_context(ctx,fold,seed,kind,store):
    if kind=='vanilla':
        # Truly label-free: even preparing a reconstruction warm-up must not
        # rebuild annotation-derived guidance/crowd masks.
        from coco_ggf.width_balance import settings
        packs={name:dict(x=ctx['packs'][name]['x'],ids=ctx['packs'][name]['ids'],
            identity=identity(dict(ids=ctx['packs'][name]['ids'],normalization=identity(ctx['norm'])))) for name in ['train','select']}
        return dict(store=store,packs=packs,cfg=dict(ctx['cfg'],batch=512),
            rule=dict(cd_common.SPEC,final_guided_updates=cd_common.SPEC['guided_updates'],local_save_every=50),
            balance_recipe=copy.deepcopy(settings()['balance']),device=ctx['device'],norm=ctx['norm'],
            provenance=dict(categories=ctx['provenance']['categories']),local=Path(ctx['options']['local'])/f'label_free_s{seed}',
            options=ctx['options'],active_key=None)
    sub=ORIGINAL_CONTEXT(ctx,fold,seed,kind,store)
    if ctx['options']['regime']=='full_category':sub['rule']['known_only']=True
    return sub

def checked_endpoint(ctx,row):
    s=cd_train.get_store(ctx,row['fold'],row['seed'],row['kind']);state=memory.required(s,'frozen')
    check_weights(state['weights'],d=ctx['packs']['train']['x'].shape[-1])
    if training.weight_sha(state['weights'])!=row['weights_sha']:raise ValueError('Frozen tensor mismatch')
    final=memory.required(s,'continuation');warm=memory.required(s,'warm') if row['kind']!='supervised_vanilla' else memory.required(cd_train.get_store(ctx,0,row['seed'],'vanilla'),'warm')
    if (final['updates'],final['groups'],final['active'],warm['updates'])!=(3000,75,32,7640):raise ValueError('Incomplete or wrong native endpoint')
    return dict(row,groups=75,width=12,top_k=32,normalization_identity=identity(ctx['norm']),
        total_updates=10640,core_parameters=sum(v.numel() for v in state['weights'].values()),
        training_seconds=warm['seconds']+final['seconds'],
        continuation_batches_sha=final['sampled_batches_sha'],continuation_noise_sha=final['sampled_noise_sha'],
        status='new_frozen_verified',evaluation_status='pending_response_collection_and_annotation')

def run(options,require_drive=True):
    configure(options['regime']);cd_train.training_context=training_context
    if not torch.cuda.is_available():raise RuntimeError('Production training requires a GPU; no CPU fallback')
    torch.set_num_threads(1);torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    lease=cd_common.Lease(options['output']);lease.acquire();stop=threading.Event();errors=[];ctx=None
    def beat():
        while not stop.wait(30):
            try:lease.beat()
            except Exception as e:errors.append(e);os.kill(os.getpid(),signal.SIGTERM);return
    worker=threading.Thread(target=beat,daemon=True);worker.start()
    def interrupted(*_):raise KeyboardInterrupt('Retaining last complete training generation')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        reused=reuse_full_catalog(options) if options['regime']=='full_category' else []
        ctx=cd_data.prepare(options,require_drive)
        if reused and any(r['normalization_identity']!=identity(ctx['norm']) for r in reused):
            raise ValueError('Reusable endpoints and prepared feature normalization differ')
        if options['regime']=='full_category':ctx['folds']=[dict(fold=0,guide=list(ctx['eligible']),excluded=[])]
        references=reuse_native_category(options,ctx['folds']) if options['regime']=='category_split' else []
        # Freeze the ACTUAL regime and category folds in the writable master;
        # the prior input-verification journal never masquerades as this study.
        payload=dict(protocol=read(HERE/'PROTOCOL.json'),runtime_spec=copy.deepcopy(cd_common.SPEC),
            regime=options['regime'],folds=ctx['folds'],normalization=identity(ctx['norm']),versions=ctx['versions'],
            train=ctx['packs']['train']['identity'],development=ctx['packs']['select']['identity'],
            calibration=ctx['packs']['confirm']['identity'],audit=ctx['test_stamp'],source_sha=options['experiment_sha'])
        ctx['store']=cd_common.make_store(Path(options['output'])/'training',Path(options['local'])/'training_journal',payload,require_drive)
        ctx['store'].write('PROTOCOL.json',payload)
        ctx['store'].write('REUSED_ENDPOINTS.json',dict(models=reused,references=references))
        ctx['store'].write('ENVIRONMENT.json',dict(torch=torch.__version__,GPU=torch.cuda.get_device_name(),TF32=False,deterministic=True))
        rows=[]
        for seed in [21,22,23]:
            tasks=[(0,'vanilla')]+[(f,k) for f in range(len(ctx['folds'])) for k in (['supervised_vanilla'] if options['regime']=='full_category' else ['supervised_vanilla','ggf'])]
            for fold,kind in tasks:
                if errors:raise errors[0]
                torch.cuda.reset_peak_memory_stats()
                print('ALL75 FIT',dict(regime=options['regime'],seed=seed,fold=fold,kind=kind),flush=True)
                row=cd_train.fit_bsf(ctx,fold,seed,kind);row=checked_endpoint(ctx,row)
                row['observed_peak_GPU_bytes']=torch.cuda.max_memory_allocated()
                rows.append(row)
                same=[r for r in rows if r['seed']==seed]
                if len({r['continuation_batches_sha'] for r in same})!=1 or len({r['continuation_noise_sha'] for r in same})!=1:
                    raise ValueError('Paired BSF image/noise schedules differ')
                ctx['store'].write('ENDPOINT_PROGRESS.json',dict(models=rows,complete=False))
                gc.collect();torch.cuda.empty_cache()
        expected=6 if options['regime']=='full_category' else 15
        if len(rows)!=expected:raise ValueError('Incomplete endpoint matrix')
        receipt=dict(regime=options['regime'],complete=True,new_endpoints=rows,reused_endpoints=reused,
            native_references=references,source_sha=options['experiment_sha'],protocol_sha256=sha(HERE/'PROTOCOL.json'),
            dashboard_status='Pending response collection, label locks and blinded semantic/target review',
            no_test_inference_or_semantic_scores=True)
        ctx['store'].write('TRAINING_COMPLETE.json',receipt)
        cd_common.write_json(Path(options['output'])/'TRAINING_COMPLETE.json',receipt)
        matched.assert_sources(ctx)
        print('TRAINING COMPLETE',options['regime'],expected,'new endpoints; evaluation pending',flush=True)
    finally:
        try:
            if ctx is not None:
                stores={str(s.root):s for s in [ctx['store']]+ctx.get('writable_stores',[])}
                for store in stores.values():cd_common.flush(store)
        finally:
            stop.set();worker.join(timeout=2);lease.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(read(a.config))
