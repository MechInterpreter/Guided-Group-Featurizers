"""Synthetic factor-supervised GGF; archived BSFs are read-only evidence.

This is a declared synthetic analogue of object guidance, not a claim that
synthetic component targets are available for COCO. No COCO data are opened.
"""
import argparse,contextlib,copy,gc,hashlib,json,os,signal,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bsf import VanillaBSF
from coco_baselines.storage import identity,sha_file
from coco_baselines.training import weight_sha
from coco_adaptation.inputs import JournalStore
from coco_ggf.width_balance_fit import Balancer,encoder_diagnostics
from zoo_benchmark import data,metrics


def recipe(cfg):
    return dict(version=1,config=cfg,warm_epochs=40,cap=.60,seeds=[0,1,2],
        balance=dict(alpha=1.5,weight_lr=.01,every=10,reference_batches=20,
            bounds=dict(capture=[.0125,.60],detection=[.0025,.60])),
        initialization='Training mixture second moments conditional on factor presence; no true component vectors used for initialization.',
        guidance='Known component reconstruction, equally averaged over active instances within factor and then factors; balanced factor-presence BCE.',
        assignment='First128 groups fixed to128 known factors; reserve groups remain unlabelled.',
        selection='Minimum selection-set reconstruction MSE over logged checkpoints and two learning rates, before opening synthetic test.',
        final_endpoint='Final200-epoch endpoint of the same selected learning-rate trial is a separate diagnostic.',
        scope='Factor-supervised synthetic analogue; stronger labels than COCO object masks. No guarantee of perfect learned recovery.')


def model(cfg,device):
    return VanillaBSF(cfg['dimension'],cfg['groups'],cfg['block_rank'],cfg['active_factors']).to(device)


def cpu_weights(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}


def initialize(m,pack,cfg):
    """Uses only observed TRAIN mixtures and their presence labels."""
    width=cfg['block_rank'];moments=[]
    for f in range(cfg['factors']):
        rows=np.flatnonzero((pack['ids']==f).any(1));x=pack['x'][rows].astype(np.float64)
        if len(x)<width:raise ValueError('Insufficient training presence samples')
        _,v=np.linalg.eigh(x.T@x/len(x));axes=v[:,-width:][:,::-1].T.copy()
        for j in range(width):
            if axes[j,np.abs(axes[j]).argmax()]<0:axes[j]*=-1
        moments.append(axes)
    with torch.no_grad():
        atoms=m.W_dec.view(cfg['groups'],width,cfg['dimension'])
        atoms[:cfg['factors']].copy_(torch.as_tensor(np.array(moments),dtype=atoms.dtype,device=atoms.device))
        m.W_enc.copy_(m.W_dec.T);m.b_enc.zero_()
    m.normalize_decoder()


@torch.no_grad()
def head_stats(m,pack,cfg,device):
    total=np.zeros(cfg['factors']);square=total.copy();count=0
    for lo in range(0,len(pack['x']),cfg['batch']):
        x=torch.as_tensor(pack['x'][lo:lo+cfg['batch']],device=device)
        a=m.encode(x)[:,:cfg['factors']].norm(dim=-1).double().cpu().numpy()
        total+=a.sum(0);square+=(a*a).sum(0);count+=len(a)
    mean=total/count;scale=np.sqrt(np.maximum(square/count-mean*mean,0)).clip(1e-5)
    return np.stack([mean,scale]).astype(np.float32)


def losses(m,x,ids,coords,frames,head,stats):
    """Sparse targets: never materialize N x factors x input dimension."""
    z=m.encode(x);rec=(m.decode(z)-x).square().mean()
    n,k=ids.shape;rows=torch.arange(n,device=x.device)[:,None]
    predicted=torch.einsum('nkb,nkbd->nkd',z[rows,ids],m.W_dec.view(m.n_groups,m.group_size,m.d)[ids])
    target=torch.einsum('nkr,nkrd->nkd',coords,frames[ids])
    residual=(predicted-target).square().sum(-1).flatten()
    energy=target.square().sum(-1).flatten();index=ids.flatten();nf=frames.shape[0]
    sums=x.new_zeros(nf).scatter_add(0,index,residual)
    denom=x.new_zeros(nf).scatter_add(0,index,energy)
    count=x.new_zeros(nf).scatter_add(0,index,torch.ones_like(energy))
    active=count>0
    # Ratio of mean squared vector error to mean target energy per factor.
    contrast=(sums[active]/denom[active].clamp_min(1e-12)).mean()
    norm=z[:,:nf].norm(dim=-1);normalized=(norm-stats[0])/stats[1]
    logits=normalized*F.softplus(head[:,0])+head[:,1]
    y=torch.zeros_like(logits).scatter_(1,ids,1.)
    raw=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    pos=y.sum(0);neg=(1-y).sum(0);valid=(pos>0)&(neg>0)
    if not valid.any():raise ValueError('No balanced factor-presence pools in batch')
    detection=(.5*(raw*y).sum(0)/pos.clamp_min(1)+.5*(raw*(1-y)).sum(0)/neg.clamp_min(1))[valid].mean()
    return [rec,contrast,detection]


@torch.no_grad()
def measure(m,x,batch,device):
    error=0.
    for lo in range(0,len(x),batch):
        a=torch.as_tensor(x[lo:lo+batch],device=device);error+=float((m(a)[0]-a).double().square().sum())
    return error/x.size


@contextlib.contextmanager
def committed_step():
    """A notebook Stop waits until optimizer, scheduler and counters agree."""
    pending=[];old=signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT,lambda *args:pending.append(True))
    try:yield
    finally:signal.signal(signal.SIGINT,old)
    if pending:raise KeyboardInterrupt


def fit(store,key,cfg,ds,seed,lr,device='cuda',stop=None):
    torch.manual_seed(seed)
    m=model(cfg,device);initialize(m,ds['packs']['train'],cfg)
    head=torch.nn.Parameter(torch.zeros(cfg['factors'],2,device=device))
    control=Balancer(recipe(cfg)['balance'],device)
    optimizer=torch.optim.Adam([*m.parameters(),head],lr=lr)
    nb=cfg['train_rows']//cfg['batch'];target=nb*cfg['epochs'];warm=nb*min(40,cfg.get('warm_epochs',40))
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,total_steps=target,
        pct_start=cfg['warmup'],anneal_strategy='cos',div_factor=25,final_div_factor=1e4)
    state=store.load(key);fp=identity(dict(recipe=recipe(cfg),seed=seed,lr=lr,data=ds['fingerprints']))
    if state is None:
        state=dict(fingerprint=fp,updates=0,target_updates=target,best=None,best_mse=None,best_update=None,
            permutation=None,stats=None,history=[],seconds=0.,seed=seed,lr=lr)
    else:
        if state['fingerprint']!=fp or state['target_updates']!=target:raise ValueError('Synthetic fit identity changed')
        m.load_state_dict(state['weights']);head.data.copy_(state['head'].to(device));control.load_state_dict(state['controller'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'])
        if device.startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
    pack=ds['packs']['train'];frames=torch.as_tensor(ds['truth']['frames'],dtype=torch.float32,device=device)
    tx=torch.from_numpy(pack['x']);ti=torch.from_numpy(pack['ids'].astype('int64'));tc=torch.from_numpy(pack['coords'])
    limit=target if stop is None else min(target,stop)
    def checkpoint(drive=True):
        state.update(weights=cpu_weights(m),head=head.detach().cpu().clone(),controller=control.state_dict(),
            optimizer=copy.deepcopy(optimizer.state_dict()),scheduler=copy.deepcopy(scheduler.state_dict()),
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
        store.save(key,state,drive=drive)
    print('ZOO',key,'resume',state['updates'],'/',target,flush=True)
    in_step=False
    try:
        while state['updates']<limit:
            with committed_step():
                in_step=True;u=state['updates'];bi=u%nb;started=time.monotonic()
                if bi==0:state['permutation']=torch.randperm(len(tx))
                batch=state['permutation'][bi*cfg['batch']:(bi+1)*cfg['batch']]
                x=tx[batch].to(device)
                if u>=warm:
                    if state['stats'] is None:state['stats']=head_stats(m,pack,cfg,device)
                    stats=torch.as_tensor(state['stats'],device=device)
                    ls=losses(m,x,ti[batch].to(device),tc[batch].to(device),frames,head,stats)
                    ready=control.observe(ls)
                    if ready and (u-warm)%10==0:
                        norms,_=encoder_diagnostics(m,ls);control.adapt(ls,norms)
                    w=control.weights().detach();loss=sum(a*b for a,b in zip(w,ls))
                else:loss=m.loss(x)[0]
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite synthetic loss')
                optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step();m.normalize_decoder()
                if device.startswith('cuda'):torch.cuda.synchronize()
                state['updates']=u+1;state['seconds']+=time.monotonic()-started
                ep=(u+1)//nb
                if (u+1)%nb==0 and (ep==1 or ep%cfg['log_epochs']==0 or ep==cfg['epochs']):
                    mse=measure(m,ds['packs']['select']['x'],cfg['batch'],device)
                    row=dict(update=u+1,epoch=ep,selection_mse=mse,weights=control.weights().detach().cpu().tolist(),
                        phase='Guided Learning' if u>=warm else 'Initialization warm-up')
                    state['history'].append(row)
                    if state['best_mse'] is None or mse<state['best_mse']:
                        state.update(best=cpu_weights(m),best_mse=mse,best_update=u+1)
                    store.event('zoo_training',key=key,target=target,**row,seconds_per_update=state['seconds']/(u+1))
                in_step=False
            if state['updates']%250==0 or state['updates']==limit:checkpoint()
            elif state['updates']%50==0:checkpoint(False)
    except KeyboardInterrupt:
        if not in_step:
            checkpoint();print('ZOO verified Drive checkpoint:',key,state['updates'],flush=True)
        raise
    # Other failures retain the last valid journal generation; no mixed-step save.
    return state


@torch.no_grad()
def encode(m,x,cfg,device):
    return np.concatenate([m.encode(torch.as_tensor(x[lo:lo+cfg['batch']],device=device)).cpu().numpy()
        for lo in range(0,len(x),cfg['batch'])])


def evaluate_arrays(ds,zc,zt,atoms):
    from sklearn.metrics import average_precision_score
    cal=ds['packs']['calibration'];test=ds['packs']['test'];truth=ds['truth'];nf=len(truth['kinds'])
    cc=metrics.correlation_match(cal,zc,nf);ct=metrics.correlation_match(test,zt,nf);rows=[]
    for f,kind in enumerate(truth['kinds']):
        ci,cy=data.factor(cal,truth,f);ix,y=data.factor(test,truth,f);cg=int(cc[f].argmax());tg=int(ct[f].argmax())
        assigned=metrics.r2(y,zt[ix,f]@atoms[f]);locked=metrics.r2(y,zt[ix,cg]@atoms[cg])
        calr2=metrics.r2(cy,zc[ci,cg]@atoms[cg]);presence=(test['ids']==f).any(1)
        rows.append(dict(factor=f,kind=kind,assigned_r2=assigned,heldout_matched_r2=locked,
            paper_test_matched_r2=metrics.r2(y,zt[ix,tg]@atoms[tg]),calibration_group=cg,test_matched_group=tg,
            compact_locked=calr2>.8 and locked>.8,factor_presence_ap=float(average_precision_score(presence,np.linalg.norm(zt[:,f],axis=1))),
            active_test_samples=len(ix),assigned_above_80=assigned>.8,assigned_above_90=assigned>.9))
    reconstruction=np.einsum('ngr,grd->nd',zt,atoms,optimize=True)
    ve=1-float(np.sum((test['x']-reconstruction)**2)/np.sum((test['x']-ds['packs']['train']['x'].mean(0))**2))
    return dict(factors=rows,assigned_mean_r2=float(np.mean([r['assigned_r2'] for r in rows])),
        heldout_mean_r2=float(np.mean([r['heldout_matched_r2'] for r in rows])),paper_mean_r2=float(np.mean([r['paper_test_matched_r2'] for r in rows])),
        ve=ve,compact_count=sum(r['compact_locked'] for r in rows),assigned_above_80_count=sum(r['assigned_above_80'] for r in rows),
        assigned_above_90_count=sum(r['assigned_above_90'] for r in rows),factor_presence_ap=float(np.mean([r['factor_presence_ap'] for r in rows])),
        mean_active_groups=float((np.linalg.norm(zt,axis=2)>1e-6).sum(1).mean()))


def audit(store,key,cfg,ds,weights,device):
    old=store.load(key)
    if old is not None:
        if old['weights_sha']!=weight_sha(weights):raise ValueError('Synthetic audited weights changed')
        return old
    m=model(cfg,device);m.load_state_dict(weights)
    zc=encode(m,ds['packs']['calibration']['x'],cfg,device);zt=encode(m,ds['packs']['test']['x'],cfg,device)
    out=evaluate_arrays(ds,zc,zt,m.atoms().cpu().numpy());out['weights_sha']=weight_sha(weights)
    store.save(key,out);store.write(key+'.json',out);del m,zc,zt;gc.collect()
    return out


def csv_write(path,rows):
    import csv
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def fingerprints(ds):
    return {name:hashlib.sha256(p['x'].tobytes()+p['ids'].tobytes()+p['coords'].tobytes()).hexdigest()
        for name,p in ds['packs'].items()}


def original_dataset(drive,ref):
    """Bounded, read-only lookup of the original trusted run's data artifact."""
    direct=ref.get('original_dataset_relative_path')
    if direct is not None:
        root=Path(drive).resolve();payload=(root/direct).resolve()
        if root not in payload.parents:raise ValueError('Original dataset path escaped Drive')
        if payload.is_file():
            if sha_file(payload)!=ref['original_dataset_sha256']:raise ValueError('Pinned original synthetic dataset checksum differs')
            ds=torch.load(payload,map_location='cpu',weights_only=False)
            if fingerprints(ds)!=ref['original_data_fingerprints']:raise ValueError('Pinned original synthetic arrays differ')
            ds['fingerprints']=fingerprints(ds)
            print('Reusing pinned original synthetic dataset:',payload,flush=True)
            return ds
    run='20559c96e5df8f92f9f6539f4fefb76235ee1ab68c4306c6e0b5b6f0d9c570a3'
    # Only two known archive layouts below each top-level Drive folder.
    for parent in sorted(Path(drive).iterdir()):
        if not parent.is_dir():continue
        for root in [parent/'checkpoints'/run[:20],parent/run[:20]]:
            manifest=root/'manifest.json'
            if not manifest.is_file():continue
            config=json.loads(manifest.read_text())
            digest=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
            if digest!=run:continue
            candidates=[]
            for meta in (root/'manifold_zoo').glob('*.json'):
                info=json.loads(meta.read_text());payload=meta.with_suffix('.pt')
                if info.get('run')==run and payload.is_file():candidates.append((info.get('generation',0),payload,info))
            for _,payload,info in sorted(candidates,reverse=True):
                if sha_file(payload)!=info['sha256']:continue
                ds=torch.load(payload,map_location='cpu',weights_only=False)
                if fingerprints(ds)!=ref['original_data_fingerprints']:raise ValueError('Original synthetic data payload has unexpected arrays')
                ds['fingerprints']=fingerprints(ds)
                print('Reusing verified original synthetic arrays:',payload,flush=True)
                return ds
    return None


def dataset_for_reference(drive,ref):
    ds=original_dataset(drive,ref)
    if ds is None:
        ds=data.generate(ref['config'],'goodfire','independent',ref['config']['data_seed'])
        if fingerprints(ds)!=ref['original_data_fingerprints']:
            raise ValueError('Original Zoo dataset was not found and regeneration differs byte-for-byte (BLAS/platform rounding can cause this). Restore the original20559c96e5df8f92f9f6/manifold_zoo artifact on Drive before resuming; do not compare changed arrays to archived BSFs.')
    return ds


def archive(store,options):
    dest=store.output/f'ggf_zoo_{store.run_id[:12]}_results.zip';hashes={}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(store.results.rglob('*')):
            if p.is_file():
                name='results/'+p.relative_to(store.results).as_posix();z.write(p,name);hashes[name]=sha_file(p)
        p=Path(options['wrapper_zip']);z.write(p,'source.zip');hashes['source.zip']=sha_file(p)
        p=Path(options['source_zip']);z.write(p,'core_source.zip');hashes['core_source.zip']=sha_file(p)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(dest) as z:
        if z.testzip() or any(hashlib.sha256(z.read(n)).hexdigest()!=v for n,v in hashes.items()):raise OSError('Synthetic ZIP failed verification')
    return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);o=json.loads(Path(p.parse_args().config).read_text())
    here=Path(__file__).parent
    subprocess.check_call([sys.executable,'-m','pytest','-q',str(here/'test_zoo.py'),'--disable-warnings',
        '--basetemp',str(Path(o['local'])/('zoo_tests_'+os.urandom(4).hex()))])
    from threadpoolctl import threadpool_limits
    from coco_ggf.matched_baselines import versions
    if versions()!=o['expected_versions']:raise ValueError('Pinned numerical versions differ')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Select a T4 GPU')
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    ref=json.loads((here/'zoo_reference.json').read_text());cfg=ref['config'];protocol=recipe(cfg)
    s=JournalStore(o['zoo_output'],dict(wrapper_sha=o['wrapper_sha'],reference_sha=sha_file(here/'zoo_reference.json'),
        recipe=protocol,versions=versions()),Path(o['local'])/'zoo_rescue',True)
    try:
        done=s.load('zoo_complete')
        if done is None:
            s.write('PROTOCOL.json',protocol);s.write('archived_baselines.json',ref)
            ds=s.load('dataset')
            if ds is None:
                s.event('zoo_generating',rows={k:cfg[k+'_rows'] for k in ['train','select','calibration','test']})
                ds=dataset_for_reference(Path(o['zoo_output']).parent,ref)
                s.save('dataset',ds)
            if fingerprints(ds)!=ref['original_data_fingerprints']:raise ValueError('Saved Zoo fingerprints differ')
            s.write('data_audit.json',data.audit(ds))
            # Freeze every selected seed before the synthetic test is opened.
            catalog=s.load('zoo_selected')
            if catalog is None:
                catalog=[]
                for seed in protocol['seeds']:
                    trials=[]
                    for lr in cfg['lrs']:
                        key=f'zoo_s{seed}_lr{int(lr*1e7)}';state=fit(s,key,cfg,ds,seed,lr)
                        trials.append(dict(key=key,lr=lr,selection_mse=state['best_mse'],best_update=state['best_update'],
                            updates=state['updates'],weights_sha=weight_sha(state['best']),final_sha=weight_sha(state['weights'])))
                        del state;gc.collect()
                    chosen=min(trials,key=lambda t:(t['selection_mse'],t['lr']))
                    catalog.append(dict(seed=seed,**chosen,trials=trials))
                s.save('zoo_selected',catalog);s.write('zoo_selected.json',catalog)
            rows=[];factor_rows=[]
            for chosen in catalog:
                state=s.load(chosen['key'])
                if weight_sha(state['best'])!=chosen['weights_sha'] or weight_sha(state['weights'])!=chosen['final_sha']:raise ValueError('Frozen Zoo weights changed')
                for field,label in [('best','ggf_selected'),('weights','ggf_final_diagnostic')]:
                    a=audit(s,f"audit_s{chosen['seed']}_{label}",cfg,ds,state[field],'cuda')
                    row={k:v for k,v in a.items() if k!='factors'};row.update(seed=chosen['seed'],method=label,lr=chosen['lr'],
                        selected_update=chosen['best_update'],trained_updates=state['updates'],parameters=sum(x.numel() for x in state[field].values()),
                        additional_head_parameters=2*cfg['factors'],factor_labels_used=True)
                    rows.append(row);factor_rows.extend([dict(seed=chosen['seed'],method=label,**r) for r in a['factors']])
                del state;gc.collect()
            rows.extend([dict(r,factor_labels_used=False,archived=True) for r in ref['baseline_rows']])
            means=[]
            for method in dict.fromkeys(r['method'] for r in rows):
                selected=[r for r in rows if r['method']==method];row=dict(method=method,seeds=len(selected))
                for k in ['assigned_mean_r2','heldout_mean_r2','paper_mean_r2','ve','factor_presence_ap']:
                    vals=[100*r[k] for r in selected if k in r]
                    if len(vals)==len(selected):row[k+'_pct']=float(np.mean(vals));row[k+'_seed_sd_pp']=float(np.std(vals,ddof=1))
                row['compact_locked_factors_pct']=100*float(np.mean([r.get('compact_count',0)/cfg['factors'] for r in selected]))
                row['mean_active_groups']=float(np.mean([r['mean_active_groups'] for r in selected]));means.append(row)
            geometry=[]
            for method in ['ggf_selected','ggf_final_diagnostic']:
                for kind in sorted(set(r['kind'] for r in factor_rows)):
                    a=[r for r in factor_rows if r['method']==method and r['kind']==kind]
                    geometry.append(dict(method=method,kind=kind,factor_seed_pairs=len(a),
                        assigned_r2_pct=100*float(np.mean([r['assigned_r2'] for r in a])),
                        calibration_locked_r2_pct=100*float(np.mean([r['heldout_matched_r2'] for r in a]))))
            csv_write(s.results/'zoo_models.csv',rows);csv_write(s.results/'zoo_means.csv',means)
            csv_write(s.results/'zoo_factors.csv',factor_rows);csv_write(s.results/'zoo_geometry.csv',geometry)
            done=dict(complete=True,models=rows,means=means,selection_catalog=catalog,baseline_fits_repeated=0,
                new_ggf_fits=6,total_updates_each=29200,COCO_holdout_opened=False,
                interpretation='Ground-truth sum audit is an arithmetic ceiling, not a trained model. Recovery thresholds are distinct from mean R2. Factor supervision is stronger than object masks.',
                numerical_environment_note=ref['scope'])
            s.save('zoo_complete',done);s.write('zoo_complete.json',done)
        out=archive(s,o)
        receipt=dict(complete=True,result_zip=str(out),results=str(s.results),wrapper_sha=o['wrapper_sha'],baseline_fits_repeated=0,new_ggf_fits=6)
        Path(o['zoo_receipt']).write_text(json.dumps(receipt,indent=2))
        print('SYNTHETIC ZOO COMPLETE. Results:',out,flush=True)
        print(json.dumps(done['means'],indent=2),flush=True)
    except BaseException:
        try:print('Partial synthetic evidence ZIP:',archive(s,o),flush=True)
        except BaseException as extra:print('Synthetic archive issue:',repr(extra),flush=True)
        raise


if __name__=='__main__':main()
