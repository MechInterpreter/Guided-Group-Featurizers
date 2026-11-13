"""Read-only whole-image object-presence fidelity audit of frozen GGF/AGF.

Fits one ridge probe to clean training activations; selects its regularization
on clean development only. Never updates a featurizer or selects on test.
"""
import csv
import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from coco_adaptation import inputs
from coco_baselines.storage import identity, serial, sha_file
from coco_coordinated import run as memory
from . import accumulating_regroup as accum, bank_experiment as bank

PARENT_RUN='5271fd4aee0806adb19d86ba2529e354210739647eff863921e2d6bf6e9030f0'
PARENT_SOURCE='b4fd1dd0f53bd92e5bb41ccb0b095b4e2dbbd0c1053ea18b50285cfcaf301a7b'

def settings():
    return dict(version=1,study='frozen_agf_whole_image_object_presence_fidelity',
        seeds=[21,22,23],primary_seeds=[22,23],reference_seed=21,
        ridge_grid=[.001,.01,.1,1.],chunk=20,inference_batch=512,
        bootstrap=200,bootstrap_seed=106061,task_ap_margin=.001,
        label_rule='At least one retained instance mask has mass >=2 patch equivalents after excluding crowd patches at 0.1.',
        scope='Additional task readout on the already inspected independent-regroup holdout; not another fresh test or a SLiCS comparison.')

def validate_ids(ids):
    sets=[]
    for split,values in ids.items():
        if len(values)!=len(set(values)):raise ValueError('Duplicate images in '+split)
        sets.append(set(values))
    if any(a&b for i,a in enumerate(sets) for b in sets[:i]):
        raise ValueError('Training/development/test images overlap')

def presence(masks,cohort):
    """Observable cropped-image object labels, not latent concept ground truth."""
    out=np.zeros((len(masks),len(cohort)),dtype=np.uint8)
    lookup={c:j for j,c in enumerate(cohort)}
    for i,mm in enumerate(masks):
        known=np.asarray(mm['crowd'])<=.1
        for obj in mm['instances']:
            c=obj['category']
            if c not in lookup:continue
            weights=np.asarray(obj['fraction'],np.float64).copy()
            weights[~known]=0
            if weights.sum()>=2:
                out[i,lookup[c]]=1
    return out

def category_ap(y,scores):
    y=np.asarray(y);scores=np.asarray(scores)
    if y.shape!=scores.shape or not np.isfinite(scores).all():raise ValueError('Invalid task scores')
    supported=(y.sum(0)>0)&(y.sum(0)<len(y))
    if not supported.all():raise ValueError('Every fixed category needs positive and negative images; never drop failed categories')
    return np.array([average_precision_score(y[:,c],scores[:,c]) for c in range(y.shape[1])])

def ridge(x,y,alpha):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha<=0:raise ValueError('Invalid probe inputs')
    mean=x.mean(0);offset=y.mean(0);xx=x-mean
    w=np.linalg.solve(xx.T@xx/len(x)+alpha*np.eye(x.shape[1]),xx.T@(y-offset)/len(x))
    return dict(mean=mean,offset=offset,weights=w,alpha=alpha)

def predict(probe,x):return (np.asarray(x,np.float64)-probe['mean'])@probe['weights']+probe['offset']

def fit_probe(train,select,grid):
    """No holdout argument exists: regularization selection cannot use test."""
    rows=[];best=None
    for alpha in grid:
        p=ridge(train['x'],train['y'],alpha)
        ap=category_ap(select['y'],predict(p,select['x']))
        rows.append(dict(alpha=alpha,development_macro_ap=float(ap.mean()),per_category_ap=ap.tolist()))
        if best is None or ap.mean()>best[0]:best=(ap.mean(),p)
    return best[1],rows

def paired_ap(y,a,b,count,seed):
    point=category_ap(y,a)-category_ap(y,b)
    rng=np.random.default_rng(seed);values=[];invalid=0
    for _ in range(count):
        ix=rng.integers(len(y),size=len(y));yy=y[ix]
        if np.any(yy.sum(0)==0) or np.any(yy.sum(0)==len(yy)):
            invalid+=1;continue
        values.append(category_ap(yy,a[ix])-category_ap(yy,b[ix]))
    if len(values)<.9*count:raise ValueError('Too few supported whole-image bootstrap draws')
    values=np.array(values)
    return dict(mean_gain=float(point.mean()),gain_ci95=np.quantile(values.mean(1),[.025,.975]).tolist(),
        per_category_gain=point.tolist(),per_category_gain_ci95=np.quantile(values,[.025,.975],axis=0).T.tolist(),
        valid=len(values),invalid=invalid,
        scope='Conditional paired image interval; fixed probe/weights and one inspected coverage-enriched holdout. Not seed uncertainty.')

def open_source(output,run,local):
    src=memory.FrozenStore(Path(output)/'checkpoints'/run[:20],local)
    if src.run_id!=run:raise ValueError('Wrong completed parent')
    return src

def prepare(parent_output,output,local,sha,versions,device='cuda',require_drive=True):
    parent_output=Path(parent_output);output=Path(output);local=Path(local);cfg=settings()
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):raise ValueError('Choose an available CPU or CUDA device')
    parent=open_source(parent_output,PARENT_RUN,local/'read_parent')
    if parent.config['source_sha']!=PARENT_SOURCE:raise ValueError('Independent-regroup source differs')
    summary=json.loads((parent.results/'summary.json').read_text())
    frozen=memory.required(parent,'all_models_frozen')
    if not summary['complete'] or not frozen['complete'] or summary['new_test_images']!=1500:
        raise ValueError('Independent-regroup parent is incomplete')
    original_output=parent_output.parent/'AGF_COCO_stratified_confirmation_v1'
    original=open_source(original_output,parent.config['activation_run'],local/'read_original')
    for root in (parent_output.resolve(),original_output.resolve()):
        dest=output.resolve()
        if dest==root or dest in root.parents or root in dest.parents:raise ValueError('Output overlaps a read-only source')
    provenance=json.loads((original.results/'provenance.json').read_text())
    plan=memory.required(parent,'fresh_plan')
    cohort=plan['coverage']['category_indices']
    if len(cohort)!=75:raise ValueError('Expected the fixed 75-category cohort')
    ids=dict(train=provenance['splits']['train'],select=provenance['splits']['select'],test=plan['splits']['test'])
    validate_ids(ids)
    if tuple(map(len,ids.values()))!=(2000,400,1500):raise ValueError('Unexpected image split sizes')
    children={};models=[]
    for item in frozen['models']:
        seed=item['seed'];folder=parent_output/f'seed_{seed}'
        # Read source-linked manifests, never choose a run using its score.
        matches=[]
        for mpath in (folder/'results').glob('*/manifest.json'):
            rec=json.loads(mpath.read_text())
            if rec['config'].get('source_sha')==PARENT_SOURCE and rec['config']['config'].get('seed')==seed:
                matches.append(rec['run_id'])
        if len(matches)!=1:raise ValueError('Ambiguous or missing seed parent '+str(seed))
        child=open_source(folder,matches[0],local/f'read_seed_{seed}');children[seed]=child
        for arm,key,h in [('GGF',item['root'],item['root_sha']),('AGF',item['candidate'],item['candidate_sha'])]:
            snap=memory.required(child,key)
            if accum.weight_sha(snap)!=h:raise ValueError('Frozen source weights changed')
            if (snap['spec']['groups'],snap['spec']['rank'],snap['spec']['active'])!=(1024,3,16):raise ValueError('Source capacity changed')
            prior_report=next(r for r in summary['results'] if r['seed']==seed)['control' if arm=='GGF' else 'edited']
            models.append(dict(seed=seed,arm=arm,key=key,sha=h,store=child,expected_patch_mse=float(prior_report['reconstruction']['mse'])))
    if sorted(set(v['seed'] for v in models))!=cfg['seeds']:raise ValueError('Cannot omit a failed seed')
    store=inputs.JournalStore(output,dict(config=cfg,source_sha=sha,versions=versions,device=device,
        parent_run=PARENT_RUN,activation_run=original.run_id,
        source_models=[{k:v for k,v in r.items() if k!='store'} for r in models]),local/'rescue',require_drive)
    store.write('protocol_settings.json',dict(settings=cfg,ids=ids,categories=[dict(index=c,name=provenance['categories'][c]['name']) for c in cohort],
        probe='One 768-to-75 multi-output ridge readout trained only on clean mean-pooled patch vectors. Same frozen readout for original, GGF and AGF.',
        evaluation='Whole-image category AP; not the earlier patch/group detection AP. Mask-derived cropped-image presence is supervised.',
        selection='Choose ridge alpha on clean development AP only; do not fit separate probes to each reconstruction or tune on test.',
        scope=cfg['scope'],featurizer_updates=0,backbone_extraction=False,baseline_retraining=False))
    norm=memory.required(original,'normalization')
    if np.asarray(norm['mean']).shape!=(196,768) or not np.isfinite(norm['mean']).all() or not np.isfinite(norm['scale']) or norm['scale']<=0:raise ValueError('Invalid saved normalization')
    return dict(store=store,parent=parent,original=original,children=children,models=models,ids=ids,cohort=cohort,
        provenance=provenance,norm=norm,cfg=cfg,device=device,local=local,
        stamps=memory.source_snapshot([parent,original,*children.values()]))

def normalized_shard(ctx,split,lo):
    cache=ctx['local']/'normalized_test_shards'/f'{lo:06d}.pt'
    meta=cache.with_suffix('.json')
    signature=identity(dict(parent=PARENT_RUN,split=split,lo=lo,ids=ctx['ids'][split][lo:lo+ctx['cfg']['chunk']],
        mean=memory.streaming_array_sha(ctx['norm']['mean']),scale=float(ctx['norm']['scale'])))
    if split=='test' and cache.is_file() and meta.is_file():
        record=json.loads(meta.read_text())
        if record['identity']==signature and record['sha256']==sha_file(cache):
            pack=torch.load(cache,map_location='cpu',weights_only=False)
            return pack['x'],pack['masks'],pack['ids'],None
    source=ctx['parent'] if split=='test' else ctx['original']
    shard=memory.required(source,f'features_{split}_{lo:06d}')
    expected=ctx['ids'][split][lo:lo+ctx['cfg']['chunk']]
    if list(shard['ids'])!=expected or len(shard['masks'])!=len(expected):raise ValueError('Shard IDs/masks changed')
    raw=shard['x'].detach().cpu().float().numpy() if torch.is_tensor(shard['x']) else np.asarray(shard['x'],np.float32)
    if raw.shape!=(len(expected),196,768) or not np.isfinite(raw).all():raise ValueError('Invalid frozen activation shard')
    x=np.asarray((raw-ctx['norm']['mean'])*ctx['norm']['scale'],np.float32)
    if split=='test':
        cache.parent.mkdir(parents=True,exist_ok=True)
        pending=cache.with_suffix('.pending.pt')
        torch.save(dict(x=x,masks=shard['masks'],ids=expected),pending);pending.replace(cache)
        meta.write_text(json.dumps(dict(identity=signature,sha256=sha_file(cache))),encoding='utf-8')
    return x,shard['masks'],expected,raw

def clean_pool(ctx,split):
    s=ctx['store'];key='clean_'+split;old=s.load(key)
    if old is not None:
        if old['ids']!=ctx['ids'][split]:raise ValueError('Cached probe split changed')
        s.event('reused_clean_pool',split=split,images=len(old['ids']));return old
    xs=[];ys=[];ids=[];n=len(ctx['ids'][split])
    raw_hash=hashlib.sha256(str((n,196,768)).encode()+str(np.dtype('float32')).encode())
    norm_hash=hashlib.sha256(str((n,196,768)).encode()+str(np.dtype('float32')).encode())
    for lo in range(0,n,ctx['cfg']['chunk']):
        x,masks,want,raw=normalized_shard(ctx,split,lo)
        if raw is not None:raw_hash.update(np.ascontiguousarray(raw).tobytes())
        norm_hash.update(np.ascontiguousarray(x).tobytes())
        xs.append(x.mean(1,dtype=np.float64));ys.append(presence(masks,ctx['cohort']));ids+=want
        s.event('clean_pool_progress',split=split,done=len(ids),total=n)
    if split=='train' and raw_hash.hexdigest()!=ctx['norm']['train_sha']:raise ValueError('Training activation/normalization hash mismatch')
    if split=='test':
        expected=json.loads((ctx['parent'].results/'fresh_activation_identity.json').read_text())['normalized_sha']
        if norm_hash.hexdigest()!=expected:raise ValueError('Fresh normalized activation bytes changed')
    result=dict(x=np.concatenate(xs),y=np.concatenate(ys),ids=ids,normalized_sha=norm_hash.hexdigest())
    category_ap(result['y'],result['y'])
    s.save(key,result);s.write('coverage_'+split+'.json',dict(images=n,positive_images=result['y'].sum(0).tolist(),category_indices=ctx['cohort']))
    return result

def probe(ctx):
    s=ctx['store'];old=s.load('frozen_probe')
    if old is not None:s.event('reused_frozen_probe',alpha=old['probe']['alpha']);return old
    train=clean_pool(ctx,'train');select=clean_pool(ctx,'select')
    p,rows=fit_probe(train,select,ctx['cfg']['ridge_grid'])
    result=dict(probe=p,selection=rows,train_ids_sha=identity(train['ids']),select_ids_sha=identity(select['ids']),
        weights_sha=identity({k:memory.streaming_array_sha(v) if isinstance(v,np.ndarray) else v for k,v in p.items()}),test_used=False)
    s.save('frozen_probe',result);s.write('probe_selection.json',result);s.event('probe_frozen',alpha=p['alpha'],weights_sha=result['weights_sha'])
    return result

@torch.inference_mode()
def reconstruct_pool(ctx,row,lo):
    s=ctx['store'];key=f"pool_s{row['seed']}_{row['arm']}_{lo:06d}";old=s.load(key)
    if old is not None:
        if old['source_sha']!=row['sha']:raise ValueError('Reconstruction cache source changed')
        return old
    live=ctx.get('_live_model')
    if live is None or live['sha']!=row['sha']:
        release_model(ctx)
        snap=memory.required(row['store'],row['key'])
        if accum.weight_sha(snap)!=row['sha']:raise ValueError('Frozen model changed')
        model=bank.make(ctx,snap);model.eval()
        ctx['_live_model']=dict(model=model,spec=snap['spec'],sha=row['sha'])
        del snap
    model=ctx['_live_model']['model']
    x,masks,ids,_=normalized_shard(ctx,'test',lo)
    pooled=np.zeros((len(x),768),np.float64);sse=0.;active=0.;started=time.monotonic()
    # Do not construct the complete 1500-image reconstruction or code tensor.
    for i in range(len(x)):
        image_sum=np.zeros(768,np.float64)
        for p in range(0,196,ctx['cfg']['inference_batch']):
            xx=torch.as_tensor(np.array(x[i,p:p+ctx['cfg']['inference_batch']],copy=True),device=ctx['device'])
            rec,z=model(xx)
            if not torch.isfinite(rec).all():raise FloatingPointError('Nonfinite reconstruction')
            image_sum+=rec.double().sum(0).cpu().numpy()
            sse+=float((rec-xx).double().square().sum());active+=float((z.norm(dim=-1)>1e-6).sum())
        pooled[i]=image_sum/196
    if ctx['device']=='cuda':torch.cuda.synchronize()
    result=dict(x=pooled,ids=ids,source_sha=row['sha'],sse=sse,active_groups=active/(len(x)*196),seconds=time.monotonic()-started)
    s.save(key,result);s.event('reconstruction_pool_progress',seed=row['seed'],arm=row['arm'],done=lo+len(ids),total=len(ctx['ids']['test']),seconds=result['seconds'])
    return result

def release_model(ctx):
    live=ctx.pop('_live_model',None)
    if live is not None:
        if accum.weight_sha(bank.snapshot(live['model'],live['spec']))!=live['sha']:
            raise ValueError('Inference modified the frozen weights')
        del live;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def timing(ctx):
    """First real reusable shard, not a disposable training job."""
    old=ctx['store'].load('timing')
    if old is not None:return old
    row=next(r for r in ctx['models'] if r['seed']==22 and r['arm']=='GGF')
    started=time.monotonic();part=reconstruct_pool(ctx,row,0);wall=time.monotonic()-started
    result=dict(compute_seconds_for_20_images=part['seconds'],wall_seconds_for_20_images=wall,
        rough_remaining_reconstruction_minutes=wall*(len(ctx['ids']['test'])/20*len(ctx['models'])-1)/60,
        excludes='Clean cache pooling, probe fitting, paired intervals and exports. Drive latency varies; not a guaranteed ETA.',
        reused_for_final_scoring=True)
    ctx['store'].save('timing',result);ctx['store'].write('timing.json',result);return result

def evaluate(ctx):
    s=ctx['store'];old=s.load('complete')
    if old is not None:s.event('reused_complete');return old
    frozen=probe(ctx);p=frozen['probe'];clean=clean_pool(ctx,'test');clean_scores=predict(p,clean['x'])
    baseline=category_ap(clean['y'],clean_scores);rows=[];scores={};pooled={}
    for row in ctx['models']:
        key=f"task_s{row['seed']}_{row['arm']}";saved=s.load(key)
        if saved is None:
            parts=[reconstruct_pool(ctx,row,lo) for lo in range(0,len(clean['ids']),ctx['cfg']['chunk'])]
            ids=[v for part in parts for v in part['ids']]
            if ids!=clean['ids']:raise ValueError('Pooled task image order differs')
            xx=np.concatenate([part['x'] for part in parts]);pred=predict(p,xx);ap=category_ap(clean['y'],pred)
            saved=dict(seed=row['seed'],arm=row['arm'],role='primary' if row['seed'] in (22,23) else 'previously inspected reference',
                source_sha=row['sha'],macro_image_presence_ap=float(ap.mean()),per_category_ap=ap.tolist(),
                clean_macro_ap=float(baseline.mean()),ap_change_from_clean=float(ap.mean()-baseline.mean()),
                task_score_mse_to_clean=float(np.square(pred-clean_scores).mean()),
                pooled_activation_mse_to_clean=float(np.square(xx-clean['x']).mean()),
                full_patch_mse=sum(part['sse'] for part in parts)/(len(ids)*196*768),
                inference_seconds=sum(part['seconds'] for part in parts),scores=pred,pool=xx,ids=ids)
            if 'expected_patch_mse' in row:
                saved.update(prior_report_patch_mse=row['expected_patch_mse'],reconstruction_replay_difference=saved['full_patch_mse']-row['expected_patch_mse'])
                if not np.isclose(saved['full_patch_mse'],row['expected_patch_mse'],rtol=1e-5,atol=1e-7):
                    raise ValueError('Frozen reconstruction does not replay the completed parent MSE; inspect source, normalization and numerical versions')
            s.save(key,saved)
            s.write('tasks/'+key+'.json',{k:v for k,v in saved.items() if k not in ('scores','pool','ids')})
        scores[row['seed'],row['arm']]=saved['scores'];pooled[row['seed'],row['arm']]=saved['pool']
        rows.append({k:v for k,v in saved.items() if k not in ('scores','pool','ids')})
        release_model(ctx)
        s.event('task_model_complete',seed=row['seed'],arm=row['arm'],macro_image_presence_ap=saved['macro_image_presence_ap'])
    pairs=[]
    for seed in ctx['cfg']['seeds']:
        key=f'paired_task_s{seed}';pair=s.load(key)
        if pair is None:
            pair=paired_ap(clean['y'],scores[seed,'AGF'],scores[seed,'GGF'],ctx['cfg']['bootstrap'],ctx['cfg']['bootstrap_seed'])
            pair.update(seed=seed,reference_seed=seed==21,
                point_within_declared_task_margin=pair['mean_gain']>=-ctx['cfg']['task_ap_margin'],
                lower_interval_within_declared_task_margin=pair['gain_ci95'][0]>=-ctx['cfg']['task_ap_margin'])
            s.save(key,pair);s.write('paired/task_s'+str(seed)+'.json',pair)
        pairs.append(pair)
    np.savez_compressed(s.results/'image_task_audits.npz',image_ids=np.array(clean['ids']),labels=clean['y'],
        clean_scores=clean_scores,category_indices=np.array(ctx['cohort']),**{f"s{seed}_{arm}":v for (seed,arm),v in scores.items()})
    result=dict(complete=True,clean_macro_image_presence_ap=float(baseline.mean()),models=rows,paired=pairs,
        primary_seeds=[22,23],test_images=len(clean['ids']),probe_sha=frozen['weights_sha'],
        scope=ctx['cfg']['scope'],featurizer_gradient_updates=0,
        comparison='Exact frozen GGF vs AGF starts; no architecture selection or candidate retries.',
        limits='One fixed linear object-presence probe, mean-pooled patches, previously inspected 1500-image holdout. Not ImageNet/segmentation/depth, causal fidelity, or supervised competitor superiority.')
    memory.assert_immutable(ctx['stamps'])
    with (s.results/'comparison.csv').open('w',newline='') as f:
        compact=[{k:v for k,v in r.items() if k!='per_category_ap'} for r in rows]
        writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    categories=[]
    for pair in pairs:
        seed=pair['seed'];a=next(r for r in rows if r['seed']==seed and r['arm']=='AGF');b=next(r for r in rows if r['seed']==seed and r['arm']=='GGF')
        for j,c in enumerate(ctx['cohort']):
            categories.append(dict(seed=seed,category_index=c,category=ctx['provenance']['categories'][c]['name'],
                positive_images=int(clean['y'][:,j].sum()),clean_image_ap=float(baseline[j]),
                GGF_image_ap=b['per_category_ap'][j],AGF_image_ap=a['per_category_ap'][j],
                AGF_minus_GGF_image_ap=pair['per_category_gain'][j],
                conditional_ci_low=pair['per_category_gain_ci95'][j][0],conditional_ci_high=pair['per_category_gain_ci95'][j][1]))
    with (s.results/'category_task_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(categories[0]));writer.writeheader();writer.writerows(categories)
    s.save('complete',result);s.write('summary.json',result);s.event('complete',primary_task_gains=[r['mean_gain'] for r in pairs if r['seed'] in (22,23)])
    return result

def export(ctx):
    s=ctx['store'];out=s.output/f'agf_downstream_fidelity_{s.run_id[:12]}_results.zip';hashes={}
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(s.results.rglob('*')):
            if path.is_file():
                name='results/'+path.relative_to(s.results).as_posix();z.write(path,name);hashes[name]=sha_file(path)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    return out
