"""Read-only historical inputs; converged NNLS repair at the original 75x3 capacity.

No fresh-holdout extraction or GGF retraining. Native SLiCS and a separately
labelled Top-16 sensitivity use identical frozen dictionaries.
"""
import argparse,copy,csv,gc,hashlib,json,math,signal,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import nnls as scipy_nnls
from coco_baselines import metrics,training
from coco_baselines.storage import identity,sha_file,serial
from coco_adaptation import core,inputs
from coco_coordinated import run as memory
from . import slics_closeout as old,study

SEEDS=[21,22,23]
PARENT='ce65e7b00536a58e690f3fe5a807b1dfac2bc081667ebf1d644683be5601407a'
MASK='e3cf93dbe6e37a8232ddb258ba18245a3710bb45f4f662c83c4b5ead60ab6bb2'
SCALE='95e7186b193ed3744ced01585a0ff0f4a198c7318a53ab23cde20b4adb5082fd'

def settings():
    return dict(version=1,seeds=SEEDS,groups=75,rank=3,active_groups=16,
        updates=5000,images_per_update=4,evaluate_every=500,drive_every=25,
        nnls_iterations=2000,nnls_tolerance=1e-5,nnls_fallback_iterations=20000,
        nnls_batch=512,probe_images=16,bootstrap=1000,
        selection='minimum full development dense-NNLS MSE including update zero; earliest tie wins',
        historical_audit=True,extract_features=False,train_ggf=False,
        background_extension=False,fresh_test=False)

class Solver:
    """FISTA with per-row KKT checks and verified float64 active-set fallback.

    No ridge penalty, shrinkage, label-dependent inference, or accepted
    unconverged rows. Empty permitted supports have the exact zero solution.
    """
    def __init__(self,D,cfg):
        if D.ndim!=2 or not torch.isfinite(D).all():raise ValueError('Finite dictionary required')
        self.D=D.detach().clone();self.cfg=cfg
        self.gram=self.D@self.D.T
        self.L=float(torch.linalg.eigvalsh(self.gram).max().clamp_min(1e-8))
    def _errors(self,z,x,allowed):
        # Verify the original objective in float64, not merely a surrogate gap.
        D=self.D.double();zz=z.double();cross=x.double()@D.T
        grad=(zz@D-x.double())@D.T
        pg=(zz-(zz-grad).clamp_min(0))*allowed
        denom=(cross*allowed).abs().amax(1).clamp_min(1e-12)
        return pg.abs().amax(1)/denom
    @torch.no_grad()
    def solve(self,x,allowed=None,force_fallback=False):
        x=x.to(device=self.D.device,dtype=self.D.dtype)
        if x.ndim!=2 or x.shape[1]!=self.D.shape[1] or len(x)==0:raise ValueError('Nonempty matching samples required')
        if allowed is None:allowed=torch.ones((len(x),len(self.D)),dtype=torch.bool,device=x.device)
        if allowed.dtype!=torch.bool or allowed.shape!=(len(x),len(self.D)):raise ValueError('Invalid support')
        allowed=allowed.to(x.device)
        cross=x@self.D.T;z=torch.zeros_like(cross);y=z.clone();t=1.;used=0
        tolerance=self.cfg['nnls_tolerance']
        # Fast iterations use the pinned source precision; terminal validation is double.
        if not force_fallback:
            for it in range(self.cfg['nnls_iterations']):
                new=(y-(y@self.gram-cross)/self.L).clamp_min(0)*allowed
                nt=(1+math.sqrt(1+4*t*t))/2
                y=new+(t-1)/nt*(new-z);z=new;t=nt;used=it+1
                if used%50==0 or used==self.cfg['nnls_iterations']:
                    grad=z@self.gram-cross
                    residual=(z-(z-grad).clamp_min(0))*allowed
                    denom=(cross*allowed).abs().amax(1).clamp_min(1e-12)
                    if float((residual.abs().amax(1)/denom).max())<=tolerance*.5:break
        error=self._errors(z,x,allowed);bad=torch.nonzero(error>tolerance).flatten()
        if force_fallback:bad=torch.nonzero(allowed.any(1)).flatten()
        for i in bad.tolist():
            take=torch.nonzero(allowed[i]).flatten()
            if len(take)==0:continue
            A=self.D[take].double().cpu().numpy().T
            b=x[i].double().cpu().numpy()
            try:answer,_=scipy_nnls(A,b,maxiter=self.cfg['nnls_fallback_iterations'])
            except RuntimeError as exc:raise RuntimeError('NNLS active-set fallback failed; no unconverged result accepted') from exc
            z[i]=0;z[i,take]=torch.as_tensor(answer,device=z.device,dtype=z.dtype)
        error=self._errors(z,x,allowed)
        if not torch.isfinite(z).all() or bool((z[~allowed]!=0).any()) or float(error.max())>tolerance:
            raise RuntimeError('NNLS failed the per-row float64 KKT check; source model unchanged')
        return z,dict(converged=True,rows=len(x),iterations=used,cpu_fallback_rows=len(bad),
            max_relative_kkt=float(error.max()),unsupported_rows=int((~allowed.any(1)).sum()),
            minimum_coefficient=float(z.min()),forbidden_max=float(z[~allowed].abs().max()) if bool((~allowed).any()) else 0.)

class Model(torch.nn.Module):
    def __init__(self,D,cfg,mode='dense'):
        super().__init__();self.register_buffer('D',D.clone());self.cfg=cfg;self.mode=mode
        self.group_size=cfg['rank'];self.n_groups=len(D)//self.group_size;self.d=D.shape[1]
        if mode not in ('dense','top16'):raise ValueError('Inference mode')
        self.solver=Solver(self.D,cfg);self.stats=dict(calls=0,rows=0,cpu_fallback_rows=0,max_relative_kkt=0.,seconds=0.)
    def atoms(self):return self.D.reshape(self.n_groups,self.group_size,self.d)
    def decoder_atoms(self):return self.D
    @torch.no_grad()
    def encode(self,x):
        start=time.monotonic();z,a=self.solver.solve(x);infos=[a]
        if self.mode=='top16':
            h=z.reshape(-1,self.n_groups,self.group_size).norm(dim=-1)
            ix=h.topk(min(self.cfg['active_groups'],self.n_groups),dim=1).indices
            allow=torch.zeros_like(h,dtype=torch.bool).scatter_(1,ix,True).repeat_interleave(self.group_size,1)
            z,b=self.solver.solve(x,allow);infos.append(b)
        for info in infos:
            self.stats['calls']+=1;self.stats['rows']+=info['rows'];self.stats['cpu_fallback_rows']+=info['cpu_fallback_rows']
            self.stats['max_relative_kkt']=max(self.stats['max_relative_kkt'],info['max_relative_kkt'])
        if x.device.type=='cuda':torch.cuda.synchronize()
        self.stats['seconds']+=time.monotonic()-start
        return z.reshape(-1,self.n_groups,self.group_size)
    def forward(self,x):
        z=self.encode(x);return z.flatten(1)@self.D,z

def frozen(output,run,local):
    s=memory.FrozenStore(Path(output)/'checkpoints'/run[:20],local)
    if s.run_id!=run:raise ValueError('Wrong source run')
    return s

def prepare(paths,local,sha,versions,cfg=None,device='cuda',require_drive=True):
    cfg=settings() if cfg is None else cfg
    if cfg!=settings():raise ValueError('Frozen small-study recipe changed; use a new protocol')
    local=Path(local);local.mkdir(parents=True,exist_ok=True)
    outputs=Path(paths['output']).resolve();sources=[Path(paths[k]).resolve() for k in ('comparison','mask','features')]
    if any(outputs==p or p in outputs.parents or outputs in p.parents for p in sources):raise ValueError('Output overlaps an immutable input')
    pins=json.loads((Path(__file__).resolve().parents[1]/'slics_repair_pins.json').read_text())
    stores=[frozen(paths[k],r,local/('read_'+k)) for k,r in [('comparison',PARENT),('mask',MASK),('features',SCALE)]]
    parent,mask,source=stores
    for i,s in enumerate(stores):
        if s.config!=pins[f'{i}_manifest.json']['config']:raise ValueError('Pinned source manifest changed')
    if versions!=parent.config['versions']:raise ValueError('Use the pinned numerical versions')
    if parent.config['device']!=device:raise ValueError('Reuse requires the same CPU/CUDA category')
    # No feature extraction. Only the historical train/development split is opened now.
    original=frozen(Path(paths['features']).parent/'AGF_COCO_stratified_confirmation_v1',source.config['parent_run_id'],local/'read_activations')
    stores.append(original);memory.memory_safe_metrics()
    provenance=json.loads((original.results/'provenance.json').read_text());norm=memory.required(original,'normalization')
    fit=copy.deepcopy(source.config['config']);fit.update(groups=75,block_rank=3,active_groups=16,
        device=device,audit_image_batch=2,preview_images=0,bootstrap=cfg['bootstrap'],batch=cfg['nnls_batch'],max_audit_groups=4,min_category_images=5)
    packs={split:memory.normalized_pack(original,split,provenance,norm,local,fit['extract_chunk']) for split in ('train','select')}
    if (len(packs['train']['ids']),len(packs['select']['ids']))!=(2000,400):raise ValueError('Unexpected historical cohort')
    if set(packs['train']['ids'])&set(packs['select']['ids']):raise ValueError('Train/development overlap')
    s=inputs.JournalStore(paths['output'],dict(recipe=cfg,source_sha=sha,versions=versions,device=device,
        input_runs=[v.run_id for v in stores]),local/'repair_rescue',require_drive)
    ctx=dict(store=s,parent=parent,mask=mask,source=source,original=original,local=local,
        cfg=fit,comparison=dict(old.settings(),seeds=SEEDS),recipe=cfg,device=device,provenance=provenance,
        packs=packs,base=dict(groups=75,block_rank=3,active_groups=16),stamps=memory.source_snapshot(stores),references=[])
    # Verify all source model identities before doing any new fit.
    for j,src in enumerate([parent,mask]):
        catalog=memory.required(src,'comparison_frozen')
        if catalog!=pins[f'{j}_models_frozen.json']:raise ValueError('Frozen model catalog changed')
        for seed in SEEDS:
            arms=['SVD mask','SLiCS mask'] if j==0 else ['mask-init GGF','continued Vanilla']
            for arm in arms:
                row=next(r for r in catalog['models'] if r['seed']==seed and r['arm']==arm)
                st=memory.required(src,row['key']);w=st[row['field']]
                if old.model_hash(w)!=row['weights_sha']:raise ValueError('Source checkpoint changed: '+row['key'])
                if torch.is_tensor(w) and tuple(w.shape)!=(225,768):raise ValueError('Unexpected decoder capacity')
                if isinstance(w,dict) and (tuple(w['W_dec'].shape),tuple(w['W_enc'].shape))!=((225,768),(768,225)):raise ValueError('Unexpected saved encoder capacity')
                ctx['references'].append(dict(**row,source='parent' if j==0 else 'mask',saved_batch_sha=st.get('batch_sha')))
                s.save('ref_'+row['key'],w) # new small copies only; source remains read-only
    s.write('PROTOCOL.json',dict(recipe=cfg,source_models=ctx['references'],
        scope='Historical75x3 supervised task-adaptation repair. Not the current4096x12D adaptive model or published SLiCS retrieval replication.',
        primary='Native dense NNLS. Same frozen dictionary with Top16 prune/refit is a separately labelled activity-budget sensitivity.',
        training='Three new fits from saved TRAIN-mask SVD. Paper Algorithm1 nonzero-support SVD/sign/clamp; converged NNLS replaces finite unchecked solves. No new background group or extra labels.',
        selection='Every saved checkpoint including0 is eligible. Min development native-NNLS MSE is primary. Object-score and three-metric Pareto candidates are retained as sensitivities, without a GGF-derived error guard.',
        controls='GGF/Vanilla are fixed historical endpoints, not retuned under the new checkpoint rule. Fixed weights .05/.01; these are not the later adaptive-weight controls.',
        budgets='SLiCS5000 four-image alternating batches and old GGF2500warm+2500guided gradient updates. Same epoch order; neither compute nor total parameter counts match.',
        evaluation='All configurations freeze before calibration/test. Existing inspected1500-image test is a descriptive audit only; the ongoing confirmation fresh holdout is never opened. Independent final confirmation remains necessary.',
        resume='New training dictionaries commit only after a full sweep; Drive checkpoints every25updates and on handled interruption. Abrupt loss may replay up to24updates; a partially completed model audit restarts.'))
    return ctx

def get_reference(ctx,seed,arm):
    row=next(r for r in ctx['references'] if r['seed']==seed and r['arm']==arm)
    return row,memory.required(ctx['store'],'ref_'+row['key'])

def variance(pack):
    x=pack['x'].reshape(-1,pack['x'].shape[-1]);total=np.zeros(x.shape[1]);energy=0.
    for lo in range(0,len(x),512):
        a=np.asarray(x[lo:lo+512],np.float64);total+=a.sum(0);energy+=float(np.square(a).sum())
    return max(energy-float(total@total)/len(x),1e-12)

class Meter(torch.nn.Module):
    def __init__(self,m):
        super().__init__();self.m=m;self.n_groups=m.n_groups;self.group_size=m.group_size;self.d=m.d
        self.sse=0.;self.n=0;self.active=0.;self.alive=np.zeros(m.n_groups,bool)
    def atoms(self):return self.m.atoms()
    @torch.no_grad()
    def encode(self,x):
        z=self.m.encode(x);D=self.m.atoms().to(x.device).reshape(-1,self.d)
        pred=z.flatten(1)@D;self.sse+=float((pred-x).double().square().sum());self.n+=len(x)
        active=z.norm(dim=-1)>1e-6;self.active+=float(active.sum());self.alive|=active.any(0).cpu().numpy()
        return z
    def reconstruction(self,total):
        return dict(mse=self.sse/(self.n*self.d),ve=1-self.sse/total,active_groups=self.active/self.n,
            dead_groups=int((~self.alive).sum()),patches=self.n)

def arrays(ctx,m,pack,label):
    wrapped=Meter(m);start=time.monotonic()
    with old.progress(ctx,wrapped,label):
        audit=metrics.audit_arrays(wrapped,pack['x'],pack['masks'],pack['ids'],ctx['cfg'],ctx['device'])
    if wrapped.n!=len(pack['ids'])*196:raise ValueError('Not every patch was evaluated')
    rec=wrapped.reconstruction(variance(pack))
    return audit,rec,dict(seconds=time.monotonic()-start,solver=copy.deepcopy(getattr(m,'stats',None)))

def validation(ctx,D,key):
    cached=ctx['store'].load(key)
    if cached is not None:
        if cached['dictionary_sha']!=old.model_hash(D.cpu()):raise ValueError('Cached validation identity changed')
        return cached
    m=Model(D,ctx['recipe']);a,rec,timing=arrays(ctx,m,ctx['packs']['select'],key)
    targets=old.old.sl_targets(ctx,m);scores=core.category_scores(a,targets,ctx['cfg'])
    if len(scores['categories'])!=75 or any(r['capture'] is None or r['ap'] is None for r in scores['categories']):
        raise ValueError('Development coverage insufficient; category denominator cannot shrink')
    result=dict(dictionary_sha=old.model_hash(D.cpu()),reconstruction=rec,**scores,**timing)
    ctx['store'].save(key,result);ctx['store'].write('development/'+key+'.json',result)
    return result

def probe(ctx):
    s=ctx['store'];prior=s.load('solver_probe')
    if prior is not None:return prior
    p=ctx['packs']['select'];indices=np.sort(np.random.default_rng(161021).choice(len(p['ids']),ctx['recipe']['probe_images'],replace=False))
    x=torch.as_tensor(np.array(p['x'][indices]).reshape(-1,p['x'].shape[-1]),device=ctx['device'])
    rows=[];svd_seen=set()
    for seed in SEEDS:
        for arm in ['SVD mask','SLiCS mask']:
            ref,D=get_reference(ctx,seed,arm)
            # SVD is shared; record aliasing instead of presenting deterministic copies as independent seeds.
            if arm=='SVD mask' and old.model_hash(D) in svd_seen:continue
            svd_seen.add(old.model_hash(D));D=D.to(ctx['device'])
            for mode in ['legacy_top16','top16','dense']:
                cfg=dict(ctx['comparison'],nnls_iterations=500,nnls_tolerance=1e-4)
                m=old.SlicsModel(D,cfg) if mode=='legacy_top16' else Model(D,ctx['recipe'],mode)
                start=time.monotonic();sse=0.;active=0.;count=0
                for lo in range(0,len(x),ctx['recipe']['nnls_batch']):
                    xx=x[lo:lo+ctx['recipe']['nnls_batch']]
                    with torch.no_grad():z=m.encode(xx);pred=z.flatten(1)@D
                    sse+=float((pred-xx).double().square().sum());active+=float((z.norm(dim=-1)>1e-6).sum());count+=len(xx)
                row=dict(seed=seed,arm=arm,inference=mode,mse=sse/(count*x.shape[1]),active_groups=active/count,
                    seconds=time.monotonic()-start,solver=copy.deepcopy(getattr(m,'stats',getattr(m,'solver_totals',None))),
                    weights_sha=ref['weights_sha'])
                rows.append(row);s.event('solver_probe_complete',**row)
    # Explain empty training supports, without interpreting them as solver failures.
    unsupported=patches=0;energy_empty=energy_total=0.
    for lo in range(0,len(ctx['packs']['train']['ids']),4):
        p=ctx['packs']['train'];mm=p['masks'][lo:lo+4]
        allowed,known=old.support(mm,ctx['cfg']['category_indices'],ctx['comparison'],'mask','cpu')
        v=np.asarray(p['x'][lo:lo+4]).reshape(-1,p['x'].shape[-1]);empty=(~allowed.any(1)).numpy() & known
        unsupported+=int(empty.sum());patches+=int(known.sum())
        energy_empty+=float(np.square(v[empty].astype(float)).sum());energy_total+=float(np.square(v[known].astype(float)).sum())
    result=dict(rows=rows,probe_ids=[p for j,p in enumerate(ctx['packs']['select']['ids']) if j in indices],
        unsupported_known_patch_fraction=unsupported/patches,empty_support_energy_fraction=energy_empty/max(energy_total,1e-12),
        interpretation='Unsupported training patches force zero contributions; this is a support/representation constraint, not NNLS nonconvergence. Probe is development only and reports reconstruction, not all-category Capture/AP.')
    s.save('solver_probe',result);s.write('solver_probe.json',result);return result

def preflight(ctx):
    s=ctx['store'];done=s.load('preflight')
    if done is not None:return done
    timings=[]
    for seed in SEEDS:
        _,D=get_reference(ctx,seed,'SVD mask');D=D.to(ctx['device']);start=time.monotonic()
        x,mm,_=old.batch(ctx,seed,0);allow,known=old.support(mm,ctx['cfg']['category_indices'],ctx['comparison'],'mask',ctx['device'])
        take=torch.as_tensor(known,device=ctx['device']);xx=x.flatten(0,1)[take];allow=allow[take]
        z,info=Solver(D,ctx['recipe']).solve(xx,allow)
        candidate,update=old.atom_sweep(xx,D.clone(),z,allow,False)
        if not torch.isfinite(candidate).all():raise FloatingPointError('Preflight atoms')
        timings.append(dict(seed=seed,full_batch_seconds=time.monotonic()-start,solver=info,atom_update=update))
    result=dict(timings=timings,training_only_lower_bound_hours=sum(t['full_batch_seconds']*ctx['recipe']['updates'] for t in timings)/3600,
        excluded='Development inference, calibration/test audits, SciPy fallbacks, Drive and runtime variation. Disposable first-batch measurement, not a completion ETA.',
        source_unchanged=True,new_training_models=3,new_gradient_models=0)
    s.save('preflight',result);s.write('preflight.json',result);return result

def pareto(history):
    keep=[]
    for a in history:
        va=a['validation'];v=np.array([va['macro_capture'],va['macro_ap'],va['reconstruction']['ve']])
        dominated=False
        for b in history:
            vb=b['validation'];w=np.array([vb['macro_capture'],vb['macro_ap'],vb['reconstruction']['ve']])
            if (w>=v).all() and (w>v).any():dominated=True;break
        if not dominated:keep.append(a['update'])
    return keep

def safe_save(s,key,st):
    try:s.save(key,st)
    except BaseException:
        s.save(key,st,False);raise

def fit_seed(ctx,seed,max_updates=None):
    s=ctx['store'];cfg=ctx['recipe'];key=f'repaired_s{seed}'
    _,initial=get_reference(ctx,seed,'SVD mask');initial_sha=old.model_hash(initial)
    st=s.load(key)
    if st is None:
        st=dict(D=initial.clone(),updates=0,initial_sha=initial_sha,total=cfg['updates'],history=[],
            best=None,best_mse=None,best_update=None,best_object=None,best_object_score=None,
            seconds=0.,batch_sha=identity([]),cpu_fallback_rows=0,max_kkt=0.,unsupported_rows=0,
            atom_increase_batches=0,last_evaluated=-1,optimization_history=[])
    if (st['initial_sha'],st['total'])!=(initial_sha,cfg['updates']):raise ValueError('Resume recipe/initialization changed')
    limit=cfg['updates'] if max_updates is None else min(max_updates,cfg['updates'])
    def evaluate():
        u=st['updates'];v=validation(ctx,st['D'].to(ctx['device']),f'validation_s{seed}_u{u}')
        if st['best_mse'] is None or v['reconstruction']['mse']<st['best_mse']:
            st.update(best=st['D'].clone(),best_mse=v['reconstruction']['mse'],best_update=u)
        if st['best_object_score'] is None or v['score']>st['best_object_score']:
            st.update(best_object=st['D'].clone(),best_object_score=v['score'],best_object_update=u)
        st['history'].append(dict(update=u,validation=v));st['last_evaluated']=u
        s.save(f'candidate_s{seed}_u{u}',dict(D=st['D'].clone(),update=u,validation=v))
        s.write('histories/'+key+'.json',dict(history=st['history'],pareto_updates=pareto(st['history'])))
        safe_save(s,key,st);s.event('repair_checkpoint',seed=seed,update=u,total=cfg['updates'],
            capture=v['macro_capture'],AP=v['macro_ap'],VE=v['reconstruction']['ve'],best_update=st['best_update'],cpu_fallback_rows=st['cpu_fallback_rows'])
    try:
        if st['last_evaluated']!=st['updates'] and st['updates']%cfg['evaluate_every']==0:evaluate()
        while st['updates']<limit:
            u=st['updates'];x,mm,bh=old.batch(ctx,seed,u)
            allow,known=old.support(mm,ctx['cfg']['category_indices'],ctx['comparison'],'mask',ctx['device'])
            take=torch.as_tensor(known,device=ctx['device']);xx=x.flatten(0,1)[take];allow=allow[take]
            if len(xx)==0:raise ValueError('All-crowd batch')
            start=time.monotonic();D=st['D'].to(ctx['device']).clone()
            z,solve=Solver(D,cfg).solve(xx,allow)
            revised,atom=old.atom_sweep(xx,D,z,allow,False)
            if not torch.isfinite(revised).all():raise FloatingPointError('Atom sweep')
            if revised.device.type=='cuda':torch.cuda.synchronize()
            # Commit only after a complete sweep. An interrupted sweep cannot overwrite st.D.
            committed=dict(D=revised.cpu().clone(),updates=u+1,seconds=st['seconds']+time.monotonic()-start,
                batch_sha=identity([st['batch_sha'],bh]),cpu_fallback_rows=st['cpu_fallback_rows']+solve['cpu_fallback_rows'],
                max_kkt=max(st['max_kkt'],solve['max_relative_kkt']),unsupported_rows=st['unsupported_rows']+solve['unsupported_rows'],
                atom_increase_batches=st['atom_increase_batches']+int(atom['residual_sse_after']>atom['residual_sse_before']+1e-7*max(1.,atom['residual_sse_before'])))
            st.update(committed)
            st['optimization_history'].append(dict(update=u+1,solver=solve,atom=atom,
                empty_support_sse=float(xx[~allow.any(1)].double().square().sum())))
            if st['updates']%cfg['drive_every']==0 or st['updates']==limit:
                s.write('optimization/'+key+'.json',st['optimization_history'])
                safe_save(s,key,st);s.event('repair_training',seed=seed,update=st['updates'],total=cfg['updates'],
                    solver=solve,atom=atom,training_seconds=st['seconds'])
            if st['updates']%cfg['evaluate_every']==0:evaluate()
        safe_save(s,key,st)
    except BaseException:
        safe_save(s,key,st);raise
    return st

def train(ctx):
    s=ctx['store'];done=s.load('repair_frozen')
    if done is not None:return done
    models=[]
    for seed in SEEDS:
        st=fit_seed(ctx,seed)
        if st['updates']!=ctx['recipe']['updates'] or st['last_evaluated']!=st['updates']:raise ValueError('Incomplete training/development')
        old_row=next(r for r in ctx['references'] if r['seed']==seed and r['arm']=='SLiCS mask')
        if old_row['saved_batch_sha'] and st['batch_sha']!=old_row['saved_batch_sha']:raise ValueError('Historical and new SLiCS image order differs')
        models.append(dict(seed=seed,key=f'repaired_s{seed}',field='best',weights_sha=old.model_hash(st['best']),
            selected_update=st['best_update'],training_updates=st['updates'],training_seconds=st['seconds'],
            object_weights_sha=old.model_hash(st['best_object']),object_selected_update=st['best_object_update'],
            cpu_fallback_rows=st['cpu_fallback_rows'],atom_increase_batches=st['atom_increase_batches'],
            initialization_selected=st['best_update']==0,pareto_updates=pareto(st['history'])))
    done=dict(complete=True,models=models,test_opened=False,fresh_holdout_opened=False,
        selection=ctx['recipe']['selection'],source_controls=ctx['references'])
    s.save('repair_frozen',done);s.write('models_frozen.json',done);memory.assert_immutable(ctx['stamps']);return done

def load_audit_packs(ctx):
    if ctx['store'].load('repair_frozen') is None:raise RuntimeError('Freeze all three seeds before calibration or test')
    norm=memory.required(ctx['original'],'normalization')
    mapping=memory.normalized_pack(ctx['original'],'mapping',ctx['provenance'],norm,ctx['local'],ctx['cfg']['extract_chunk'])
    # Historical only. The current baseline-confirmation holdout is never searched or accessed.
    plan=memory.required(ctx['source'],'fresh_plan')
    test=memory.normalized_pack(ctx['source'],'test',dict(splits=plan['splits']),norm,ctx['local'],ctx['source'].config['config']['extract_chunk'])
    packs=[ctx['packs']['train'],ctx['packs']['select'],mapping,test];sets=[set(p['ids']) for p in packs]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)):raise ValueError('Audit split overlap')
    if len(test['ids'])!=1500 or len(mapping['ids'])!=400:raise ValueError('Unexpected historical audit cohort')
    return mapping,test

def audit_model(ctx,m,row,mapping,test):
    s=ctx['store'];key=row['tag'];prior=s.load('report_'+key)
    if prior is not None:
        if prior['summary']['weights_sha']!=row['weights_sha']:raise ValueError('Frozen audit identity changed')
        return prior
    signature=identity(dict(weights=row['weights_sha'],inference=row['inference'],
        calibration_ids=mapping['ids'],audit_ids=test['ids'],configuration=ctx['cfg']))
    index=s.load('audit_cache_index') or {}
    if signature in index:
        origin=index[signature];base=memory.required(s,'report_'+origin)
        summary=dict(base['summary']);summary.update(row,reused_inference=True,inference_origin_tag=origin)
        result=dict(base,summary=summary)
        for prefix in ['mapping_','compact_']:s.save(prefix+key,memory.required(s,prefix+origin))
        s.save('report_'+key,result);s.write('reports/'+key+'.json',result);return result
    maps=s.load('mapping_'+key)
    if maps is None:
        a,_,_=arrays(ctx,m,mapping,key+'_calibration')
        maps=metrics.calibrate(a,ctx['cfg'],ctx['provenance']['categories']);s.save('mapping_'+key,maps)
        del a;gc.collect()
    a,rec,timing=arrays(ctx,m,test,key+'_historical_test')
    per=metrics.evaluate(a,maps,dict(ctx['cfg'],bootstrap=0),71002)
    indexed={r['category_index']:r for r in per};cohort=[indexed[c] for c in ctx['cfg']['category_indices']]
    if any(r['contrast'] is None or r['detection'] is None for r in cohort):raise ValueError('All75 categories must be scored')
    summary=dict(**row,capture=float(np.mean([r['contrast']['single_group_capture'] for r in cohort])),
        ap=float(np.mean([r['detection']['ap'] for r in cohort])),reconstruction=rec,**timing)
    result=dict(summary=summary,categories=per)
    compact=study.compact_audit(a,maps,ctx['cfg']);s.save('compact_'+key,compact)
    s.save('report_'+key,result);s.write('reports/'+key+'.json',result);s.event('historical_audit_complete',**summary)
    index[signature]=key;s.save('audit_cache_index',index)
    return result

def evaluate(ctx):
    s=ctx['store'];prior=s.load('repair_summary')
    if prior is not None:render(ctx,prior);return prior
    frozen=memory.required(s,'repair_frozen');mapping,test=load_audit_packs(ctx);reports=[]
    for seed in SEEDS:
        candidates=[]
        for arm in ['SVD mask','SLiCS mask']:
            ref,D=get_reference(ctx,seed,arm)
            for mode in ['dense','top16']:
                candidates.append((dict(seed=seed,method='SVD only' if arm=='SVD mask' else 'Historical SLiCS',
                    inference=mode,tag=f'{ref["key"]}_{mode}',weights_sha=ref['weights_sha'],selected_update=ref['best_update'],
                    training_updates=ref['updates'],training_seconds=ref['training_seconds'],parameters=225*768),D))
        new=next(r for r in frozen['models'] if r['seed']==seed);st=memory.required(s,new['key'])
        for field,name,sha_key,update_key in [('best','Repaired SLiCS','weights_sha','selected_update'),
                ('best_object','Repaired SLiCS (object-selected)','object_weights_sha','object_selected_update')]:
            for mode in ['dense','top16']:
                candidates.append((dict(seed=seed,method=name,inference=mode,tag=f'repaired_s{seed}_{field}_{mode}',
                    weights_sha=new[sha_key],selected_update=new[update_key],training_updates=new['training_updates'],
                    training_seconds=new['training_seconds'],parameters=225*768),st[field]))
        for row,D in candidates:
            if old.model_hash(D)!=row['weights_sha']:raise ValueError('Frozen dictionary changed')
            m=Model(D.to(ctx['device']),ctx['recipe'],row['inference'])
            reports.append(audit_model(ctx,m,row,mapping,test));del m;gc.collect()
        for arm in ['mask-init GGF','continued Vanilla']:
            ref,w=get_reference(ctx,seed,arm);m=core.new_model('vanilla',ctx['base'],w,ctx['device'])
            row=dict(seed=seed,method='Saved GGF' if arm=='mask-init GGF' else 'Saved Vanilla',inference='top16',
                tag=f'{ref["key"]}_repair_top16',weights_sha=ref['weights_sha'],selected_update=2500+ref['best_update'],
                training_updates=ref['updates'],training_seconds=ref['training_seconds'],parameters=ref['parameters'])
            reports.append(audit_model(ctx,m,row,mapping,test));del m;gc.collect()
    comparisons=[]
    for seed in SEEDS:
        a=next(r['summary'] for r in reports if r['summary']['seed']==seed and r['summary']['method']=='Saved GGF')
        for b in [r['summary'] for r in reports if r['summary']['seed']==seed and r['summary']['method'] not in ('Saved GGF','Saved Vanilla')]:
            key='ci_'+a['tag']+'_'+b['tag'];ci=s.load(key)
            if ci is None:
                ci=study.paired(memory.required(s,'compact_'+a['tag']),memory.required(s,'compact_'+b['tag']),ctx['cfg']);s.save(key,ci)
            comparisons.append(dict(seed=seed,control=b['method'],inference=b['inference'],
                capture_gain=a['capture']-b['capture'],ap_gain=a['ap']-b['ap'],ve_gain=a['reconstruction']['ve']-b['reconstruction']['ve'],paired_intervals=ci))
    result=dict(complete=True,models=[r['summary'] for r in reports],comparisons=comparisons,
        fresh_test=False,historical_test=True,source_models_unchanged=True,
        inference='Native dense SLiCS is primary. Top16 reuses the same selected dictionary; no claim dense activity is matched to GGF.',
        limitations='Small75x3 task adaptation; fixed historical GGF controls and different checkpoint rules; unequal parameters/compute; same deterministic SVD geometry, varied data orders; inspected historical test. Independent final confirmation remains necessary.')
    s.save('repair_summary',result);s.write('summary.json',result);render(ctx,result);memory.assert_immutable(ctx['stamps']);return result

def render(ctx,summary):
    s=ctx['store'];rows=[]
    for r in summary['models']:
        rows.append(dict(seed=r['seed'],method=r['method'],inference=r['inference'],capture_pct=100*r['capture'],
            ap_pct=100*r['ap'],ve_pct=100*r['reconstruction']['ve'],active_groups=r['reconstruction']['active_groups'],
            selected_update=r['selected_update'],training_updates=r['training_updates'],train_seconds=r['training_seconds'],
            audit_seconds=r['seconds'],parameters=r['parameters'],cpu_fallback_rows=(r['solver'] or {}).get('cpu_fallback_rows',0)))
    def csvwrite(path,rr):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    csvwrite(s.results/'comparison.csv',rows)
    aggregate=[]
    for method,mode in sorted({(r['method'],r['inference']) for r in rows}):
        rr=[r for r in rows if (r['method'],r['inference'])==(method,mode)]
        if sorted(r['seed'] for r in rr)!=SEEDS:raise ValueError('Cannot shrink the seed denominator')
        a=dict(method=method,inference=mode,seeds=3)
        for metric in ['capture_pct','ap_pct','ve_pct','active_groups']:
            vals=np.array([r[metric] for r in rr]);a[metric]=float(vals.mean());a[metric+'_seed_sd']=float(vals.std(ddof=1))
        aggregate.append(a)
    csvwrite(s.results/'aggregate.csv',aggregate)
    category_rows=[]
    for r in summary['models']:
        report=memory.required(s,'report_'+r['tag'])
        for c in report['categories']:
            if c['category_index'] in ctx['cfg']['category_indices']:
                category_rows.append(dict(seed=r['seed'],method=r['method'],inference=r['inference'],
                    category=c['category'],capture=c['contrast']['single_group_capture'],ap=c['detection']['ap'],
                    contrast_group=c['contrast']['groups'][0],detector_group=c['detection']['group']))
    csvwrite(s.results/'per_category.csv',category_rows)
    from scipy.stats import t as student_t
    seed_rows=[]
    for method,mode in sorted({(r['control'],r['inference']) for r in summary['comparisons']}):
        rr=[r for r in summary['comparisons'] if (r['control'],r['inference'])==(method,mode)]
        if sorted(r['seed'] for r in rr)!=SEEDS:raise ValueError('Missing comparison seed')
        for k in ['capture_gain','ap_gain','ve_gain']:
            values=np.array([r[k] for r in rr])*100;mean=float(values.mean());margin=float(student_t.ppf(.975,2)*values.std(ddof=1)/math.sqrt(3))
            seed_rows.append(dict(control=method,inference=mode,metric=k,mean_pp=mean,ci95_lower_pp=mean-margin,ci95_upper_pp=mean+margin,
                scope='Three training orders on one fixed dataset/initializer; not geometry or dataset uncertainty'))
    csvwrite(s.results/'seed_intervals.csv',seed_rows)
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,5));labels=[r['method']+'\n'+r['inference'] for r in aggregate]
    for ax,k,title in zip(axes,['capture_pct','ap_pct','ve_pct'],['Capture','Patch AP','Reconstruction VE']):
        vals=[r[k] for r in aggregate];ax.barh(range(len(vals)),vals,color=['#00877c' if r['method']=='Saved GGF' else '#8a76ab' for r in aggregate])
        ax.set_yticks(range(len(vals)),labels,fontsize=7);ax.set_title(title);ax.set_xlabel('%');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    fig.suptitle('Historical75x3 repair | 3 training orders | native dense and Top16 shown separately');fig.tight_layout()
    (s.results/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(s.results/'figures'/('comparison.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

def export(ctx,source_zip):
    s=ctx['store'];dest=s.output/f'ggf_slics_repair_{s.run_id[:12]}_results.zip';hashes={}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(s.results.rglob('*')):
            if p.is_file():name='results/'+p.relative_to(s.results).as_posix();z.write(p,name);hashes[name]=sha_file(p)
        z.write(source_zip,'source.zip');hashes['source.zip']=sha_file(source_zip)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:raise OSError('Export CRC failed')
        for name,digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise OSError('Export hash failed')
    return dest
