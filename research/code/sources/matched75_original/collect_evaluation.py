"""Consume immutable training receipts; collect bounded native maps with resume."""
import argparse,copy,gc,json,os,signal,threading,time
from pathlib import Path
import numpy as np
import torch
from eval_common import *
import cd_common,cd_data,all75_run
from coco_baselines import data
from coco_ggf import matched_baselines as matched

def load_audit(ctx):
    ids=ctx['test_ids'];source=ctx['sources']['strong'];stamp=ctx['test_stamp']
    folder=Path(ctx['options']['local'])/'evaluation';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'audit_features.npy';target=None;masks=[];records={}
    if not path.exists():target=np.lib.format.open_memmap(path.with_suffix('.pending.npy'),mode='w+',dtype=np.float32,shape=(len(ids),196,768))
    for lo in range(0,len(ids),ctx['test_chunk']):
        key=f'features_test_{lo:06d}';v=memory.required(source,key);want=ids[lo:lo+ctx['test_chunk']]
        if v['ids']!=want or v['normalization_sha']!=identity(ctx['norm']):raise ValueError('Audit feature lineage differs')
        x=v['x'].cpu().float().numpy() if torch.is_tensor(v['x']) else np.asarray(v['x'],np.float32)
        if target is not None:target[lo:lo+len(want)]=data.normalize(x,ctx['norm'])
        masks.extend(v['masks'])
        for j,i in enumerate(want):records[str(i)]=dict(image_id=i,role='audit',geometry=v['geometry'][j],original_sha256=v['image_hashes'][j],source_url=f'https://s3.amazonaws.com/images.cocodataset.org/train2017/{i:012d}.jpg')
        print('EVALUATION INPUT',dict(role='audit',done=lo+len(want),total=len(ids)),flush=True)
    if target is not None:target.flush();del target;os.replace(path.with_suffix('.pending.npy'),path)
    x=np.load(path,mmap_mode='r')
    if x.shape!=(len(ids),196,768) or memory.streaming_array_sha(x)!=stamp['normalized_sha']:raise ValueError('Audit normalization/cache checksum differs')
    return dict(x=x,ids=ids,masks=masks,identity=identity(stamp)),records

def calibration_records(ctx):
    source=ctx['sources']['original'];chunk=source.config['config']['extract_chunk'];ids=ctx['packs']['confirm']['ids'];records={}
    for lo in range(0,len(ids),chunk):
        v=memory.required(source,f'features_confirm_{lo:06d}')
        if v['ids']!=ids[lo:lo+chunk]:raise ValueError('Calibration metadata order differs')
        for j,i in enumerate(v['ids']):records[str(i)]=dict(image_id=i,role='calibration',geometry=v['geometry'][j],original_sha256=v['image_hashes'][j],source_url=f'https://s3.amazonaws.com/images.cocodataset.org/train2017/{i:012d}.jpg')
    return records

def save_labels(root,packs,categories):
    # COCO mask labels support annotated AP only; these are never promoted to
    # independent human semantic-presence judgments.
    for role,p in packs.items():
        path=root/'labels'/f'{role}_masks.npz'
        fraction=np.stack([m['fraction'][:,categories] for m in p['masks']]);crowd=np.stack([m['crowd'] for m in p['masks']])
        if not path.exists():npz(path,ids=np.array(p['ids']),fraction=fraction,crowd=crowd)
        with np.load(path) as z:
            if z['ids'].tolist()!=p['ids'] or not np.array_equal(z['fraction'],fraction) or not np.array_equal(z['crowd'],crowd):raise ValueError('Mask export differs')
        save(path.with_suffix('.json'),dict(path=str(path.relative_to(root)),sha256=sha(path),role='annotation_proxy_AP_only',categories=categories))

def collect_bank(root,row,packs,categories,local,device='cuda',chunk=16):
    row=copy.deepcopy(row);signature=digest(dict(row=row,pack_identities={k:p['identity'] for k,p in packs.items()},collector_sha256=sha(__file__),native_sha256=sha(HERE/'native_units.py'),chunk=chunk))
    folder=root/'responses'/row['bank_id'];folder.mkdir(parents=True,exist_ok=True);complete=folder/'BANK_COMPLETE.json'
    if complete.exists():
        old=read(complete)
        if old['signature']!=signature:raise ValueError('Completed bank identity changed')
        for rr in old['splits'].values():
            for r in rr:
                if sha(resolve(root,r['path']))!=r['sha256']:raise ValueError('Completed response changed')
        return old
    native=Native(row,local,device);splits={};start=time.monotonic()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    try:
        for role,pack in packs.items():
            records=[]
            for lo in range(0,len(pack['ids']),chunk):
                ids=pack['ids'][lo:lo+chunk];path=folder/f'{role}_{lo:06d}.npz';meta=path.with_suffix('.json')
                if meta.exists():
                    rec=read(meta)
                    if rec['signature']!=signature or rec['image_ids']!=ids or sha(path)!=rec['sha256']:raise ValueError('Resume chunk identity differs')
                else:
                    chunk_started=time.monotonic()
                    maps=[];pooled=[];err=np.zeros((len(categories),row['groups']));den=np.zeros(len(categories));counts=np.zeros(len(categories),np.int64);control=[]
                    for j,i in enumerate(ids):
                        x=np.asarray(pack['x'][lo+j],np.float32);s,z,D,pred=native.factor(x)
                        maps.append(s);pooled.append(np.partition(s,-4,axis=0)[-4:].mean(0))
                        e,d,n=capture_errors(x,z,D,pack['masks'][lo+j],categories);err+=e;den+=d;counts+=n
                        mse=float(np.square(np.asarray(pred,np.float64)-x).sum()) if pred is not None else np.nan
                        control.append([mse,*[float(v) for v in [np.square(x.astype(np.float64)).sum(),x.size,(s>1e-6).sum()]]])
                    npz(path,ids=np.array(ids),scores=np.stack(maps),pooled=np.stack(pooled),capture_errors=err,capture_den=den,capture_count=counts,controls=np.array(control))
                    rec=dict(signature=signature,path=str(path.relative_to(root)),sha256=sha(path),image_ids=ids,role=role,
                        completed_chunk_wall_seconds=time.monotonic()-chunk_started,
                        observed_peak_GPU_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
                    save(meta,rec)
                records.append(rec);print('NATIVE RESPONSES',dict(bank=row['bank_id'],role=role,done=lo+len(ids),total=len(pack['ids']),elapsed_minutes=round((time.monotonic()-start)/60,2)),flush=True)
            splits[role]=records
            if role=='calibration':
                from score_evaluation import lock_numeric_readouts
                readout_path=lock_numeric_readouts(root,dict(row,splits=splits),categories)
        bank=dict(row,signature=signature,complete=True,splits=splits,core_parameters=native.parameters,solver=native.solver,numeric_readout_sha256=sha(readout_path),
            collection_wall_seconds=sum(r['completed_chunk_wall_seconds'] for rr in splits.values() for r in rr),session_wall_seconds=time.monotonic()-start,
            timing_scope='Sum of committed chunk wall times across resume sessions; includes inference, control accumulation and chunk write, excludes calibration selector and source-loading overhead',
            observed_peak_GPU_bytes=max([r['observed_peak_GPU_bytes'] for rr in splits.values() for r in rr if r['observed_peak_GPU_bytes'] is not None],default=None),
            reconstruction_interface='native_additive_BSF' if row['kind']!='slics' else 'independent_concept_projections_no_joint_additive_reconstruction')
        save(complete,bank);return bank
    finally:native.close()

def collect(options):
    cd_common.safe_output(options['output'],[options['training_root'],*options['paths'].values()])
    pins=read(HERE/'EVALUATION_PINS.json');protocol_sha=sha(HERE/'PROTOCOL.json')
    if protocol_sha!=pins['training_protocol_sha256']:raise ValueError('Training metric protocol changed')
    receipts={}
    # Both regimes must freeze before the shared audit pool is opened.
    pilot=options.get('seed21_pilot',False);receipt_name='PILOT_SEED21_COMPLETE.json' if pilot else 'TRAINING_COMPLETE.json';seeds=(21,) if pilot else (21,22,23)
    for regime in ['full_category','category_split']:
        p=Path(options['training_root'])/regime/receipt_name
        if not p.exists():raise FileNotFoundError('Wait for training completion: '+str(p))
        receipts[regime]=verify_training(read(p),regime,pins['training_source_sha256'],protocol_sha,seeds=seeds)
    torch.set_num_threads(1);torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime for collection')
    root=Path(options['output']);root.mkdir(parents=True,exist_ok=True)
    lease=cd_common.Lease(root);lease.acquire();stop=threading.Event();errors=[]
    def beat():
        while not stop.wait(30):
            try:lease.beat()
            except Exception as e:errors.append(e);os.kill(os.getpid(),signal.SIGTERM);return
    worker=threading.Thread(target=beat,daemon=True);worker.start()
    def interrupted(*_):raise KeyboardInterrupt('Collection stopped; committed chunks are reusable')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        all75_run.configure('category_split')
        ctx=cd_data.prepare(dict(options,regime='category_split',output=str(root/'input_verification')))
        audit,records=load_audit(ctx);records.update(calibration_records(ctx));packs={'calibration':ctx['packs']['confirm'],'audit':audit}
        ids={k:p['ids'] for k,p in packs.items()}
        from dashboard import validate_splits
        validate_splits(ids['calibration'],ids['audit'])
        if any(set(p['ids'])&set(ids[k]) for p in [ctx['packs']['train'],ctx['packs']['select']] for k in ids):raise ValueError('Evaluation/training image overlap')
        cats=ctx['eligible'];definitions=read(HERE/'TARGET_DEFINITIONS.json')
        if [r['category_index'] for r in definitions['categories']]!=cats:raise ValueError('Target definition cohort differs')
        norm=identity(ctx['norm']);rows=[]
        for regime,receipt in receipts.items():rows.extend(registry(receipt,regime))
        expected_banks=11 if pilot else 29
        if len(rows)!=expected_banks:raise ValueError('Native endpoint matrix differs from the selected main/pilot plan')
        # Reverify EVERY native endpoint before the first calibration response.
        for row in rows:
            verify_fit_lineage(row,ctx,pins)
            _,w=endpoint_state(row,options['local']);del w
            if row.get('normalization_identity',norm)!=norm:raise ValueError('Model normalization differs')
        save(root/'ENDPOINT_LOCK.json',dict(protocol_sha256=protocol_sha,scope='seed21_pilot' if pilot else 'three_seed_comparison',training_receipt_sha256={k:sha(Path(options['training_root'])/k/receipt_name) for k in receipts},banks=rows,normalization_identity=norm))
        manifest=dict(schema='all75-native-response-input-v1',ids=ids,categories=cats,definitions_sha256=sha(HERE/'TARGET_DEFINITIONS.json'),normalization_identity=norm,
            feature_identities={k:p['identity'] for k,p in packs.items()},folds=ctx['folds'],records=records,
            allocation_status='Previously inspected source cohorts; not a fresh confirmatory sample',source_stamp=ctx['test_stamp'])
        save(root/'INPUTS.json',manifest);save(root/'TARGET_DEFINITIONS.json',definitions);save(root/'PROTOCOL.json',read(HERE/'PROTOCOL.json'))
        save_labels(root,packs,cats)
        completed=[]
        for row in rows:
            if errors:raise errors[0]
            bank=collect_bank(root,row,packs,cats,options['local'],chunk=options.get('response_chunk',16));completed.append(bank)
            save(root/'COLLECTION_PROGRESS.json',dict(completed_banks=len(completed),planned=expected_banks,complete=False),immutable=False)
        # Shared audit mean is the same for every native additive BSF.
        total=np.zeros(768);square=0.;n=0
        for x in audit['x']:
            a=np.asarray(x,np.float64);total+=a.sum(0);square+=np.square(a).sum();n+=len(a)
        manifest=dict(schema='all75-native-response-collection-v1',complete=True,scope='seed21_pilot' if pilot else 'three_seed_comparison',training_seeds=list(seeds),full_three_seed_study_complete=not pilot,banks=completed,inputs_sha256=sha(root/'INPUTS.json'),protocol_sha256=protocol_sha,
            audit_total_variation=float(square-np.square(total).sum()/n),audit_values=n*768,
            semantic_scores_status='pending_blinded_reviews_and_presence_locks',collector_sha256=sha(__file__))
        save(root/'COLLECTION_COMPLETE.json',manifest);matched.assert_sources(ctx)
        print('RESPONSE COLLECTION COMPLETE',len(completed),'banks; original sources unchanged',flush=True)
    finally:stop.set();worker.join(timeout=2);lease.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();collect(read(a.config))
