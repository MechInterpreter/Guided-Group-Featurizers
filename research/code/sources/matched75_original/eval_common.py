"""Immutable artifacts and native inference for the separately packaged evaluation."""
from pathlib import Path
import copy,gc,hashlib,json,os,sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'core')]
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from scale_models import model
from native_units import partial_nnls
import all75_run

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x,immutable=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(clean(x),indent=2,allow_nan=False).encode()
    if p.exists() and immutable:
        if p.read_bytes()!=raw:raise ValueError('Frozen artifact differs: '+str(p))
        return sha(p)
    tmp=p.with_suffix(p.suffix+'.pending');tmp.write_bytes(raw);os.replace(tmp,p);return sha(p)
def npz(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending.npz')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,p);return sha(p)
def resolve(root,p):
    p=Path(p);return p if p.is_absolute() else Path(root)/p

def verify_training(receipt,regime,training_source_sha,protocol_sha,seeds=(21,22,23)):
    if receipt.get('complete') is not True or receipt.get('regime')!=regime or receipt.get('source_sha')!=training_source_sha or receipt.get('protocol_sha256')!=protocol_sha:
        raise ValueError('Training receipt does not identify the frozen all75 release')
    seeds=tuple(seeds)
    if seeds not in [(21,),(21,22,23)]:raise ValueError('Unknown frozen seed plan')
    if seeds==(21,):
        if receipt.get('scope')!='seed21_pilot' or receipt.get('required_seeds')!=[21] or receipt.get('full_three_seed_study_complete') is not False:
            raise ValueError('Explicit single-seed pilot receipt required')
        if receipt.get('execution_revision_sha256')!=sha(HERE/'pilot_seed21_run.py') or receipt.get('pilot_amendment_sha256')!=sha(HERE/'PILOT_AMENDMENT.json'):
            raise ValueError('Pilot execution/amendment identity changed')
    elif receipt.get('scope')=='seed21_pilot':raise ValueError('Pilot is not a completed three-seed study')
    expected={(s,0,k) for s in seeds for k in ['vanilla','supervised_vanilla']} if regime=='full_category' else (
        {(s,0,'vanilla') for s in seeds}|{(s,f,k) for s in seeds for f in [0,1] for k in ['ggf','supervised_vanilla']})
    rows=receipt['new_endpoints'];actual={(r['seed'],r['fold'],r['kind']) for r in rows}
    if actual!=expected or len(rows)!=len(expected):raise ValueError('Missing/duplicate training endpoint')
    if any((r['groups'],r['width'],r['top_k'],r['total_updates'])!=(75,12,32,10640) for r in rows):raise ValueError('Wrong architecture/budget')
    reused=receipt.get('reused_endpoints',[])
    if regime=='full_category':
        keys={(r['seed'],r['collector_method']) for r in reused}
        if len(reused)!=2*len(seeds) or keys!={(s,k) for s in seeds for k in ['ggf75','slics']}:raise ValueError('Incomplete reused checkpoint matrix')
        pins={(r['seed'],r['collector_method']):r for r in read(HERE/'REUSE_CATALOG.json')['models']}
        for r in reused:
            old=pins[r['seed'],r['collector_method']]
            if any(r.get(k)!=old[k] for k in ['weights_sha','file_sha256','normalization_identity']):raise ValueError('Reused checkpoint differs from the original inventory')
    elif reused:raise ValueError('Full-category supervised checkpoints cannot enter category split')
    return receipt

def endpoint_state(row,local):
    if row.get('origin')=='reused':
        p=Path(row['path'])
        if sha(p)!=row['file_sha256']:raise ValueError('Reused payload changed')
        state=torch.load(p,map_location='cpu',weights_only=False)
        w=state['D'] if row['kind']=='slics' else state['weights']
    else:
        store=memory.FrozenStore(Path(row['output'])/'checkpoints'/row['run_id'][:20],Path(local)/'read_endpoints')
        state=memory.required(store,'frozen');w=state['weights']
        if not state.get('training_complete'):raise ValueError('Endpoint not frozen')
        if row['kind']!='slics':
            continuation=memory.required(store,'continuation')
            if (continuation['updates'],continuation['groups'],continuation['active'])!=(3000,75,32) or training.weight_sha(continuation['weights'])!=row['weights_sha']:
                raise ValueError('Frozen endpoint differs from its completed continuation')
    expected=training.weight_sha({'D':w} if row['kind']=='slics' else w)
    if expected!=row['weights_sha']:raise ValueError('Endpoint tensor hash changed')
    if row['kind']=='slics':
        if tuple(w.shape)!=(row['groups']*12,768):raise ValueError('Native dictionary geometry differs')
    else:all75_run.check_weights(w)
    return state,w

def verify_fit_lineage(row,ctx,pins):
    if row['origin']=='reused':return
    store=memory.FrozenStore(Path(row['output'])/'checkpoints'/row['run_id'][:20],Path(ctx['options']['local'])/'read_lineage')
    cfg=store.config
    if cfg['kind']!=row['kind'] or cfg['seed']!=row['seed']:raise ValueError('Checkpoint method/seed mapping differs')
    if row['kind']!='vanilla' and cfg['fold']!=row['fold']:raise ValueError('Checkpoint fold mapping differs')
    if cfg['normalization']!=identity(ctx['norm']) or cfg['train']!=ctx['packs']['train']['identity'] or cfg['development']!=ctx['packs']['select']['identity']:
        raise ValueError('Training data/normalization lineage differs')
    if row['capacity_matched']:
        if cfg['code_sha256']!=pins['training_source_sha256'] or any(cfg['protocol'][k]!=v for k,v in dict(groups=75,width=12,top_k=32,warm_updates=7640,guided_updates=3000).items()):
            raise ValueError('Training source/architecture recipe differs')
    if row['kind']=='vanilla':
        if cfg['guide'] or cfg['fold']!='label_free_shared':raise ValueError('Vanilla is not the label-free shared fit')
    else:
        guide=ctx['eligible'] if row['regime']=='full_category' else ctx['folds'][row['fold']]['guide']
        excluded=[] if row['regime']=='full_category' else ctx['folds'][row['fold']]['excluded']
        if cfg['guide']!=guide or cfg['excluded']!=excluded:raise ValueError('Guidance/held-out category lineage differs')

def registry(receipt,regime):
    rows=[]
    for r in receipt['new_endpoints']:
        rows.append(dict(r,origin='new',regime=regime,evaluation_folds=[0,1] if regime=='category_split' and r['kind']=='vanilla' else [r['fold']],capacity_matched=True))
    for r in receipt.get('reused_endpoints',[]):
        kind='slics' if r['collector_method']=='slics' else 'ggf'
        rows.append(dict(r,origin='reused',regime=regime,kind=kind,fold=0,groups=75,width=12,top_k=None if kind=='slics' else 32,evaluation_folds=[0],capacity_matched=True))
    for r in receipt.get('native_references',[]):
        rows.append(dict(r,origin='new',regime=regime,groups=r['native_groups'],top_k=None,evaluation_folds=[r['fold']],capacity_matched=False))
    for r in rows:
        r['bank_id']=regime+'_'+r['kind']+'_s'+str(r['seed'])+'_f'+str(r['fold'])
    if len({r['bank_id'] for r in rows})!=len(rows):raise ValueError('Duplicate native banks')
    return rows

def object_weights(mask,category):
    known=np.asarray(mask['crowd'])<=.1;bg=(np.asarray(mask['fraction'])[:,category]/255<=.1)&known
    out=[]
    for obj in mask['instances']:
        if obj['category']!=category:continue
        fg=np.asarray(obj['fraction'],float)*known
        if fg.sum()>=2 and bg.sum()>=4:out.append(fg/fg.sum()-bg/bg.sum())
    return out

class Native:
    def __init__(self,row,local,device='cuda'):
        _,w=endpoint_state(row,local);self.row=row;self.device=device;self.model=None;self.D=None
        if row['kind']=='slics':self.D=w.reshape(row['groups'],12,768).to(device)
        else:
            self.model=model('released_vanilla',75,12,32,768).to(device);self.model.load_state_dict(w);self.model.eval().requires_grad_(False)
            self.D=self.model.atoms().detach()
        self.parameters=sum(v.numel() for v in w.values()) if isinstance(w,dict) else w.numel()
        self.solver=dict(calls=0,max_relative_kkt=0.,cpu_fallback_rows=0)
    @torch.inference_mode()
    def factor(self,x):
        a=torch.as_tensor(np.array(x,copy=True),dtype=torch.float32,device=self.device)
        if self.model is None:
            z,info=partial_nnls(a,self.D);D=self.D.to(z.dtype);pred=None
            self.solver['calls']+=1;self.solver['max_relative_kkt']=max(self.solver['max_relative_kkt'],info['max_relative_kkt']);self.solver['cpu_fallback_rows']+=info['cpu_fallback_rows']
        else:z=self.model.encode(a);D=self.D;pred=self.model.decode(z)
        scores=z.norm(dim=-1).cpu().numpy()
        if not np.isfinite(scores).all() or (scores<0).any():raise ValueError('Invalid native norm')
        return scores,z.cpu().numpy(),D.cpu().numpy(),None if pred is None else pred.cpu().numpy()
    def close(self):
        self.model=None;self.D=None;gc.collect()
        if self.device.startswith('cuda'):torch.cuda.empty_cache()

def capture_errors(x,z,D,mask,categories):
    """All-unit regional Capture errors, separately from response/category AP."""
    errors=np.zeros((len(categories),len(D)),np.float64);den=np.zeros(len(categories));count=np.zeros(len(categories),np.int64)
    for j,c in enumerate(categories):
        ww=object_weights(mask,c)
        if not ww:continue
        W=np.asarray(ww,np.float64);target=W@np.asarray(x,np.float64)
        code=np.einsum('op,pgr->ogr',W,z,optimize=True)
        part=np.einsum('ogr,grd->ogd',code,D,optimize=True)
        errors[j]=np.square(part-target[:,None,:]).sum((0,2));den[j]=np.square(target).sum();count[j]=len(W)
    return errors,den,count

def read_maps(root,bank,role):
    records=bank['splits'][role];ids=[];scores=[];pooled=[]
    errors=[];den=[];count=[];control=[]
    for r in records:
        p=resolve(root,r['path'])
        if sha(p)!=r['sha256']:raise ValueError('Response chunk changed')
        with np.load(p,allow_pickle=False) as z:
            if z['ids'].tolist()!=r['image_ids']:raise ValueError('Response chunk IDs changed')
            ids.extend(r['image_ids']);scores.append(z['scores']);pooled.append(z['pooled'])
            errors.append(z['capture_errors']);den.append(z['capture_den']);count.append(z['capture_count']);control.append(z['controls'])
    if len(set(ids))!=len(ids):raise ValueError('Repeated response IDs')
    return dict(ids=ids,scores=np.concatenate(scores),pooled=np.concatenate(pooled),
        capture_errors=np.sum(errors,axis=0),capture_den=np.sum(den,axis=0),capture_count=np.sum(count,axis=0),controls=np.concatenate(control))

def read_pooled(root,bank,role):
    p=Path(root)/'portable'/bank['bank_id']/'POOLED.npz';meta=p.with_suffix('.json')
    if p.exists() and meta.exists():
        r=read(meta)
        if sha(p)!=r['sha256'] or r['signature']!=bank['signature']:raise ValueError('Portable pooled bank changed')
        master=Path(root)/'PORTABLE_LOCK.json'
        if master.exists():
            expected=next(x for x in read(master)['banks'] if x['bank_id']==bank['bank_id'])
            if r['sha256']!=expected['pooled_sha256']:raise ValueError('Portable master hash differs')
        with np.load(p) as z:return dict(ids=z[role+'_ids'].tolist(),pooled=z[role+'_pooled'])
    return read_maps(root,bank,role)

def cache_views(root,bank):
    folder=Path(root)/'portable'/bank['bank_id'];folder.mkdir(parents=True,exist_ok=True)
    views={};arrays={}
    for role in ['calibration','audit']:
        a=read_maps(root,bank,role);arrays[role+'_ids']=np.array(a['ids']);arrays[role+'_pooled']=a['pooled'];roleviews={}
        for u in range(bank['groups']):
            indices=sorted(range(len(a['ids'])),key=lambda j:(-float(a['pooled'][j,u]),int(a['ids'][j])))[:6];selected=[]
            for j in indices:
                s=a['scores'][j,:,u];top=sorted(range(196),key=lambda p:(-float(s[p]),p))[:4]
                selected.append(dict(image_id=int(a['ids'][j]),pooled=float(a['pooled'][j,u]),patches=[p for p in top if s[p]>0]))
            roleviews[str(u)]=selected
        views[role]=roleviews
    p=folder/'POOLED.npz';npz(p,**arrays);save(p.with_suffix('.json'),dict(sha256=sha(p),signature=bank['signature']))
    save(folder/'VIEWS.json',dict(signature=bank['signature'],bank_id=bank['bank_id'],roles=views,
        selection='Six largest signed mean-top4 values per native unit; full196 positions; positive top4 patches only; no object masks'))
    return dict(bank_id=bank['bank_id'],pooled_sha256=sha(p),views_sha256=sha(folder/'VIEWS.json'))
