"""Prospective supervised-only SLiCS/GGF pair and read-only current GGF anchors."""
import copy,gc,hashlib,json,math,re,time,zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
from coco_baselines import training,metrics
from coco_baselines.storage import identity,sha_file
from coco_adaptation import inputs,core
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched,study
from coco_ggf.drive_recovery import install_drive_recovery
from coco_ggf.slics_repair import Solver
from coco_ggf.slics_closeout import atom_sweep,support
from scale_data import prepare as base_prepare,mask_dictionary,holdout_plan,load_holdout
from scale_models import model
from scale_run import development,flush
from supervised_fit import fit
from object_schedule import PrefixReuse, check_legacy, object_report, final_reconstruction, reserve_predecessor_holdout
from focused_protocol import CORE_SHA,STUDY_RUN
from focused_inputs import backbone_preflight,annotations

SEEDS=list(range(21,28));SCREEN=SEEDS[:3];WARM=7640;GUIDED=3000;TOTAL=WARM+GUIDED
LRS=[.0001,.0003,.001]

def native_solver_config():return dict(nnls_iterations=2000,nnls_tolerance=1e-5,nnls_fallback_iterations=20000)

def protocol():
    return dict(version=3,evaluation_revision='object-only-development-reservation-fix-v1',
        holdout_policy='Reuse the pinned, verified unopened predecessor reservation; never reuse an opened test cohort',
        holdout_reservation_sha='af34b44ced90edaae136b8108e287c8a4aabec3cadf45cb1f147f122212c07aa',seeds=SEEDS,screen_seeds=SCREEN,groups=75,width=12,k=32,cap=.6,current_anchor_groups=4096,
        warm=WARM,guided=GUIDED,total=TOTAL,new_bsf_controls=False,new_detection_only_bsf=False,
        small_groups=75,small_width=12,small_k=32,small_lrs=LRS,
        selection='capture + 0.2 AP on the fixed development set; exact ties retain earliest checkpoint, then declared LR grid order; no MSE selector',
        routine_validation='object metrics only for SLiCS; full joint reconstruction only on the frozen final holdout',
        checkpoint_schedule='SLiCS: 0, every 500, 7640 and 10640; GGF guided: 0, every 500 and 3000; GGF candidates include a shared 7640-update warm-up',
        training_arithmetic_changed=False,secondary_joint_object_readout=False,
        predecessor_run='d8a0a7c7bd2e6ffb880cee729c8fdfba2adea8ca3709f887e5c58e5ae77fb0fd',
        primary_contrasts=[['matched_ggf','slics_object_partial']],
        confirmatory_capture_AP_tests=2,holm_family_alpha=.05,
        current_capacity_anchor_seeds=SCREEN,current_anchor_retraining=False,
        fresh_seed=126109,fresh_images=1500,previous_test_for_selection=False,
        source_control_training=False,primary_small_readouts='fixed category slots for BOTH methods',
        seed_scope='Independent training orders conditional on a common deterministic SVD initializer and fixed data',
        scope='Tests specified supervised controls; no claim to settle all possible supervised methods or guarantee acceptance.')

def finite_report(v):
    if not np.isfinite([v['macro_capture'],v['macro_ap']]).all():raise ValueError('Nonfinite development report')
    rec=v['reconstruction']
    if rec.get('status')=='deferred':
        if rec['mse'] is not None or rec['ve'] is not None:raise ValueError('Deferred reconstruction must have null metrics')
    elif not np.isfinite([rec['mse'],rec['ve']]).all():raise ValueError('Nonfinite development report')
    return v

def select_trials(trials,policy='object'):
    if policy!='object':raise ValueError('The reconstruction-selected policy has been removed')
    if not trials:raise ValueError('No development candidates')
    for trial in trials:
        if sorted(r['seed'] for r in trial['models'])!=SCREEN:raise ValueError('Every candidate needs all three screening seeds')
        for row in trial['models']:finite_report(row['development'])
        trial['mean_capture']=float(np.mean([r['development']['macro_capture'] for r in trial['models']]))
        trial['mean_ap']=float(np.mean([r['development']['macro_ap'] for r in trial['models']]))
        trial['score']=trial['mean_capture']+.2*trial['mean_ap']
    return min(enumerate(trials),key=lambda item:(-item[1]['score'],item[0]))[1]

class Reuse:
    """Source stores are FrozenStores; no write is forwarded to them."""
    def __init__(self,job,sources,pins):self.job=job;self.sources=sources;self.pins=pins
    def __getattr__(self,n):return getattr(self.job,n)
    def load(self,key):
        v=self.job.load(key)
        if v is not None:return v
        allowed=bool(re.fullmatch(r'(?:dev_)?g4096_k32_s(?:21|22|23|24|25|26|27)_[A-Za-z0-9_]+',key)
                     or key=='mask_dictionary12' or re.fullmatch(r'initializer_cat\d+',key))
        if not allowed:return None
        for src in self.sources:
            v=src.load(key)
            if v is not None:
                if isinstance(v,dict) and 'updates' in v:
                    limit=WARM if key.endswith('_warm') else GUIDED
                    if v['updates']>limit:raise ValueError('Source is beyond requested endpoint; restore exact checkpoint: '+key)
                if key in self.pins and training.weight_sha(v['weights'])!=self.pins[key]['weights_sha']:
                    raise ValueError('Pinned pilot changed')
                # Copy metadata pin, not huge tensors, into the provenance record.
                stamp=matched.source_key_stamp(src,key)
                self.job.write('reused_keys/'+key+'.json',dict(source_run=src.run_id,key=key,source_stamp=stamp))
                return v
        return None
    def save(self,key,v,*a,**kw):
        if key in self.pins:raise PermissionError('Cannot rewrite a frozen pilot endpoint')
        return self.job.save(key,v,*a,**kw)
    @property
    def _width_drive_recovery_original_save(self):return self.force_save
    def force_save(self,key,v,*a,**kw):
        if key in self.pins:raise PermissionError('Cannot rewrite a frozen pilot endpoint')
        return self.job._width_drive_recovery_original_save(key,v,*a,**kw)

def focused_sources(o,local):
    found=[]
    for p in sorted((Path(o['baseline_output'])/'results').glob('*/manifest.json')):
        m=json.loads(p.read_text());c=m['config']
        if c.get('core_sha')==CORE_SHA and c.get('source_run')==STUDY_RUN and c.get('recipe',{}).get('groups')==4096:
            if c.get('wrapper_sha')!=o['focused_sha']:raise ValueError('Baseline source changed; inspect source identity before reuse')
            found.append(memory.FrozenStore(Path(o['baseline_output'])/'checkpoints'/m['run_id'][:20],local/('read_'+m['run_id'][:12])))
    if len(found)>1:raise ValueError('Multiple compatible baseline journals; choose the exact source rather than guessing')
    return found

def current_catalog(path,pins):
    from pilot_inputs import catalog
    rows=[r for r in catalog(json.loads(Path(path).read_text())) if r['kind']=='agf_g']
    for r in rows:
        if r['key'] not in pins or any(r[k]!=v for k,v in pins[r['key']].items()):raise ValueError('Current GGF catalog differs from exact pins')
    if {(r['seed'],r['kind']) for r in rows}!={(s,'agf_g') for s in SCREEN}:raise ValueError('Need three exact current GGF pilot endpoints')
    return rows

def prepare(o,require_drive=True):
    from coco_ggf.cap_followup import safe_destination
    legacy=check_legacy(o)
    safe_destination(o['output'],[o['source_output'],o['baseline_output'],*o['paths'].values()])
    base=base_prepare(dict(o,output=str(Path(o['output'])/'input_verification')),require_drive)
    local=Path(o['local']);source=matched.discover(o['source_output'],local/'read_scale',run=STUDY_RUN)
    if source.config['source_sha']!=CORE_SHA or source.config['versions']!=matched.versions():raise ValueError('Saved control environment mismatch')
    pins={k:r for k,r in json.loads((Path(__file__).parent/'pilot_pins.json').read_text()).items() if r['kind']=='agf_g'}
    from pilot_inputs import validate_targets,validate_state
    catalog_path=source.results/'priority_pilot_summary.json'
    rows=current_catalog(catalog_path,pins)
    parents=focused_sources(o,local)+[source]
    base['stamps']+=memory.source_snapshot(parents)
    base['stamps'].append(dict(path=str(catalog_path),sha256=sha_file(catalog_path)))
    payload=dict(study='ggf_prospective_supervised_controls',source_sha=o['wrapper_sha'],core_sha=CORE_SHA,
        numerical_versions=matched.versions(),protocol=protocol(),source_runs=[p.run_id for p in parents],
        normalization_sha=identity(base['norm']),train_identity=base['packs']['train']['identity'],
        development_identity=base['packs']['select']['identity'],pilot_pins=pins)
    job=inputs.JournalStore(o['output'],payload,local/'rescue',require_drive);install_drive_recovery(job)
    base.update(store=Reuse(job,parents,pins),source=source,source_stores=parents,pilot_rows=rows,local=local,active_key=None)
    base['rule']=copy.deepcopy(base['rule']);base['rule'].update(fresh_seed=126109,final_holdout_images=1500)
    base['cfg']=dict(base['cfg'],bootstrap=1000,audit_image_batch=1,max_audit_groups=4,preview_images=0)
    # Source model fingerprints use the ORIGINAL core rule; holdout rules are kept separately.
    base['core_rule']=copy.deepcopy(source.config['recipe'])
    if (base['core_rule']['width'],base['core_rule']['images_per_update'],base['core_rule']['warm_updates'])!=(12,8,WARM):raise ValueError('Unexpected current GGF recipe')
    for r in rows:
        validate_targets(r,base['cfg']['category_indices'])
        st=memory.required(source,r['key'])
        validate_state(r,st)
        if st['updates']!=GUIDED or training.weight_sha(st['weights'])!=r['weights_sha'] or identity(r['targets'])!=r['targets_sha']:
            raise ValueError('Restore the exact frozen pilot weights and readouts')
        base['checkpoint_stamps'].append(matched.source_key_stamp(source,r['key']));del st;gc.collect()
    # Import only numerically compatible training states into this NEW run.
    base['store']=PrefixReuse(base['store'],legacy,base)
    job.write('protocol.json',payload)
    job.write('evaluation_revision.json',dict(previous_run=legacy.run_id,new_run=job.run_id,scientific_protocol_changed=True,training_arithmetic_changed=False,minimum_MSE_selector_removed=True,old_settings_and_test_results_reused=False,prior_object_checkpoint_evidence_reused=True))
    print('VERIFIED NEW EVALUATION PROTOCOL:',job.run_id,'; compatible saved training states will be imported from',legacy.run_id,flush=True)
    matched.assert_sources(base);return base

def current(ctx,seed,kind):
    if kind!='agf_g':raise ValueError('This notebook runs only supervised methods, never BSF controls')
    prior=next((r for r in ctx['pilot_rows'] if r['seed']==seed and r['kind']==kind),None)
    if prior is not None:r=copy.deepcopy(prior)
    else:raise ValueError('Only exact saved current GGF anchors are reused; no new large-scale fits')
    r['arm']='current_ggf';r['selected_total_updates']=TOTAL
    if 'development' not in r:r['development']=development(ctx,r)['report']
    return r


def row_from_state(ctx,key,seed,arm,groups,targets,st,**extra):
    r=dict(key=key,seed=seed,arm=arm,method='released_vanilla',groups=groups,rank=12,active_groups=32,
        weights_sha=training.weight_sha(st['weights']),targets=targets,targets_sha=identity(targets),
        total_training_updates=TOTAL,selected_total_updates=extra.pop('selected_total_updates',TOTAL),**extra)
    r['development']=evaluate_state(ctx,r,ctx['packs']['select'],False)
    return r


def fixed_targets(ctx,groups=75):
    if groups!=75 or len(ctx['cfg']['category_indices'])!=75:raise ValueError('Declared 75-label task required')
    return [dict(category_index=c,category=ctx['provenance']['categories'][c]['name'],contrast_group=j,detector_group=j,
                 heat_mean=0.,heat_scale=1.) for j,c in enumerate(ctx['cfg']['category_indices'])]

def native_initializer(ctx):
    key='native_svd12';old=ctx['store'].load(key)
    if old is not None:return old['D']
    D=mask_dictionary(ctx).clone().cpu();b=12
    if D.shape!=(900,ctx['packs']['train']['x'].shape[-1]):raise ValueError('Expected 75x12 native initializer')
    # Same uncentered SVD vectors; choose native SLiCS polarity by positive coefficient energy.
    for j,c in enumerate(ctx['cfg']['category_indices']):
        pos=np.zeros(b);neg=np.zeros(b);block=D[j*b:(j+1)*b].numpy()
        for x,mm in zip(ctx['packs']['train']['x'],ctx['packs']['train']['masks']):
            good=(mm['crowd']<=.1)&(mm['fraction'][:,c].astype(float)/255>=.5)
            values=np.asarray(x[good],np.float64)@block.astype(np.float64).T
            pos+=np.square(np.maximum(values,0)).sum(0);neg+=np.square(np.minimum(values,0)).sum(0)
        D[j*b:(j+1)*b]*=torch.as_tensor(np.where(pos>=neg,1.,-1.),dtype=D.dtype)[:,None]
    ctx['store'].save(key,dict(D=D,sha=training.weight_sha({'D':D})));return D

def small_context(ctx):
    rule=dict(ctx['core_rule'],snr=0.,warm_updates=WARM,final_guided_updates=GUIDED,sampling='epochs',known_only=True)
    return dict(ctx,rule=rule)

def small_warm(ctx,seed):
    sub=small_context(ctx);key=f'small75_s{seed}_warm';torch.manual_seed(seed)
    m=model('released_vanilla',75,12,32,ctx['packs']['train']['x'].shape[-1]);D=native_initializer(ctx)
    with torch.no_grad():m.W_dec.copy_(D);m.W_enc.copy_(D.T);m.b_enc.zero_()
    start=training.weights(m);del m
    st=fit(sub,key,'released_vanilla',75,32,seed,start,WARM,WARM,stream=0)
    targetkey=f'small75_s{seed}_targets';targets=ctx['store'].load(targetkey)
    if targets is None:
        targets=fixed_targets(ctx);m=model('released_vanilla',75,12,32,D.shape[1]).to(ctx['device']);m.load_state_dict(st['weights'])
        count=0;total=np.zeros(75);square=np.zeros(75)
        with torch.no_grad():
            for x in ctx['packs']['train']['x']:
                h=m.encode(torch.as_tensor(np.array(x,copy=True),device=ctx['device'])).norm(dim=-1).cpu().numpy().astype(float)
                count+=len(h);total+=h.sum(0);square+=np.square(h).sum(0)
        mean=total/count;sd=np.sqrt(np.maximum(square/count-mean**2,0))
        for j,t in enumerate(targets):t.update(heat_mean=float(mean[j]),heat_scale=max(float(sd[j]),1e-3))
        ctx['store'].save(targetkey,targets);del m;gc.collect()
    return st,targets

def small_fit(ctx,seed,lr,policy='object'):
    key=f'small75_s{seed}_lr{int(lr*1e7)}_ggf'
    cache=ctx['store'].load('row_'+key+'_'+policy)
    if cache is not None:return cache
    if policy!='object':raise ValueError('Only the object selector is supported')
    sub=small_context(ctx);warm,targets=small_warm(ctx,seed)
    st=fit(sub,key,'released_vanilla',75,32,seed,warm['weights'],GUIDED,GUIDED,
        targets=targets,cap=.6,learning_rate=lr,stream=1,keep_candidates=True,readout_targets=targets)
    chosen=memory.required(ctx['store'],key+'_'+policy)
    r=row_from_state(ctx,key+'_'+policy,seed,'matched_ggf',75,targets,chosen,
        selected_total_updates=WARM+chosen['updates'],lr=lr,head_parameters=150,
        fitted_key=key,selection_policy=policy)
    ctx['store'].save('row_'+key+'_'+policy,r);del st,warm;gc.collect();return r

@torch.no_grad()
def partial_nnls(x,D,cfg,width):
    """Vectorized Eq.3: independent small NNLS per concept, never sum as a reconstruction."""
    groups=len(D)//width;B=D.reshape(groups,width,-1)
    gram=B@B.transpose(1,2);cross=(x@D.T).reshape(len(x),groups,width)
    L=torch.linalg.eigvalsh(gram).amax(-1).clamp_min(1e-8)[None,:,None]
    z=torch.zeros_like(cross);y=z.clone();t=1.
    for it in range(cfg['nnls_iterations']):
        grad=torch.einsum('ngb,gbc->ngc',y,gram)-cross
        new=(y-grad/L).clamp_min(0);nt=(1+math.sqrt(1+4*t*t))/2
        y=new+(t-1)/nt*(new-z);z=new;t=nt
        if (it+1)%50==0:
            grad=torch.einsum('ngb,gbc->ngc',z,gram)-cross
            error=(z-(z-grad).clamp_min(0)).abs().amax(-1)/cross.abs().amax(-1).clamp_min(1e-12)
            if float(error.max())<=cfg['nnls_tolerance']*.5:break
    BB=B.double();gram64=BB@BB.transpose(1,2);cross64=(x.double()@D.double().T).reshape_as(cross)
    def errors():
        zz=z.double();grad=torch.einsum('ngb,gbc->ngc',zz,gram64)-cross64
        return (zz-(zz-grad).clamp_min(0)).abs().amax(-1)/cross64.abs().amax(-1).clamp_min(1e-12)
    bad=torch.nonzero(errors()>cfg['nnls_tolerance'])
    from scipy.optimize import nnls
    for i,g in bad.tolist():
        result,_=nnls(B[g].double().cpu().numpy().T,x[i].double().cpu().numpy(),maxiter=cfg['nnls_fallback_iterations'])
        z[i,g]=torch.as_tensor(result,device=z.device,dtype=z.dtype)
    error=errors()
    if not torch.isfinite(z).all() or float(error.max())>cfg['nnls_tolerance']:raise RuntimeError('Partial NNLS failed strict KKT verification')
    return z,dict(cpu_fallback_rows=len(bad),max_relative_kkt=float(error.max()))

class NativeSLiCS(torch.nn.Module):
    def __init__(self,D,cfg,readout='joint'):
        super().__init__();self.register_buffer('D',D);self.d=D.shape[1];self.group_size=12;self.n_groups=len(D)//12
        self.cfg=cfg;self.readout=readout;self.solver=Solver(D,cfg);self.stats=dict(cpu_fallback_rows=0,calls=0,max_relative_kkt=0.,readout_rows=0,readout_active_group_sum=0.)
    def atoms(self):return self.D.reshape(self.n_groups,12,self.d)
    def solve(self,x,D=None):
        solver=self.solver if D is None else Solver(D,self.cfg)
        z,info=solver.solve(x);self.stats['calls']+=1;self.stats['cpu_fallback_rows']+=info['cpu_fallback_rows']
        self.stats['max_relative_kkt']=max(self.stats['max_relative_kkt'],info['max_relative_kkt']);return z
    @torch.no_grad()
    def encode(self,x):
        if self.readout=='joint':z=self.solve(x).reshape(-1,self.n_groups,12)
        elif self.readout=='partial':
            z,info=partial_nnls(x,self.D,self.cfg,12)
            self.stats['calls']+=self.n_groups;self.stats['cpu_fallback_rows']+=info['cpu_fallback_rows']
            self.stats['max_relative_kkt']=max(self.stats['max_relative_kkt'],info['max_relative_kkt'])
        else:raise ValueError('Unknown SLiCS readout')
        self.stats['readout_rows']+=len(x);self.stats['readout_active_group_sum']+=float((z.norm(dim=-1)>1e-6).sum())
        return z
    @torch.no_grad()
    def forward(self,x):
        # Eq.3 partial concept projections are NOT additive reconstructions.
        z=self.solve(x).reshape(-1,self.n_groups,12);return z.flatten(1)@self.D,z

def batch_indices(n,seed,u):
    stream=0 if u<WARM else 1;offset=u if stream==0 else u-WARM
    from supervised_fit import epoch_indices
    return epoch_indices(n,seed,offset,stream)

def native_fit(ctx,seed):
    s=ctx['store'];key=f'native_slics12_s{seed}';prior_rows=s.load('native_rows_s'+str(seed))
    if prior_rows is not None:return prior_rows
    cfg=native_solver_config()
    D0=native_initializer(ctx);ident=identity(dict(source=training.weight_sha({'D':D0}),train=ctx['packs']['train']['identity'],
        select=ctx['packs']['select']['identity'],seed=seed,solver=cfg,total=TOTAL,protocol=protocol()))
    st=s.load(key)
    if st is not None and st['fingerprint']!=ident:raise ValueError('SLiCS resume identity changed')
    if st is None:st=dict(fingerprint=ident,updates=0,target=TOTAL,D=D0,history=[],seconds=0.,selected={},sampled_batches_sha=hashlib.sha256().hexdigest(),phase_batch_shas={})
    def save():s._width_drive_recovery_original_save(key,st,drive=True)
    def score():
        u=st['updates'];D=st['D'].to(ctx['device']);m=NativeSLiCS(D,cfg,'partial')
        val=checked_native_evaluate(ctx,m,ctx['packs']['select'],False)
        for policy,value in [('object',val['macro_capture']+.2*val['macro_ap'])]:
            if policy not in st['selected'] or value>st['selected'][policy]['score']:
                st['selected'][policy]=dict(score=value,updates=u)
                s.save(key+'_'+policy,dict(D=st['D'].clone(),updates=u,validation=val,score=value))
        st['history'].append(dict(update=u,validation=val,solver=copy.deepcopy(m.stats)));st['last_evaluated']=u
        s.write('histories/'+key+'.json',st['history']);save();del m,D;gc.collect()
    # O(offset) schedule replay is avoided in the training loop by restoring NumPy state.
    rng=np.random.default_rng(126000+seed+100000*(st['updates']>=WARM))
    if 'sample_rng' in st:rng.bit_generator.state=st['sample_rng']
    elif st['updates']:raise ValueError('Missing exact sampling RNG')
    try:
        if st.get('last_evaluated')!=st['updates'] and (st['updates']%500==0 or st['updates'] in [WARM,TOTAL]):score()
        while st['updates']<TOTAL:
            u=st['updates']
            if u==WARM and not st.get('phase_reset'):
                rng=np.random.default_rng(126000+seed+100000);st['phase_reset']=True
                st['phase_batch_shas']['warm']=st['sampled_batches_sha'];st['sampled_batches_sha']=hashlib.sha256().hexdigest()
                st['sample_rng']=copy.deepcopy(rng.bit_generator.state)
            rng_before=copy.deepcopy(rng.bit_generator.state);ids=batch_indices(len(ctx['packs']['train']['ids']),seed,u)
            pack=ctx['packs']['train'];xx=torch.as_tensor(np.array(pack['x'][ids],copy=True),device=ctx['device']).reshape(-1,D0.shape[1])
            allow,known=support([pack['masks'][i] for i in ids],ctx['cfg']['category_indices'],dict(rank=12),'mask',ctx['device'])
            take=torch.as_tensor(known,device=ctx['device']);xx=xx[take];allow=allow[take];D=st['D'].to(ctx['device']).clone();start=time.monotonic()
            try:
                # Native training support: remove globally forbidden columns before any Gram allocation.
                cols=allow.any(0);z=torch.zeros((len(xx),900),device=ctx['device'])
                info=dict(cpu_fallback_rows=0,max_relative_kkt=0.,unsupported_rows=len(xx))
                if bool(cols.any()):z[:,cols],info=Solver(D[cols],cfg).solve(xx,allow[:,cols])
                changed,atom=atom_sweep(xx,D,z,allow,False)
                if not torch.isfinite(changed).all():raise FloatingPointError('SLiCS atoms')
                torch.cuda.synchronize() if str(ctx['device']).startswith('cuda') else None
            except BaseException:
                rng.bit_generator.state=rng_before;raise
            st.update(D=changed.cpu(),updates=u+1,seconds=st['seconds']+time.monotonic()-start,sample_rng=copy.deepcopy(rng.bit_generator.state),
                sampled_batches_sha=hashlib.sha256(st['sampled_batches_sha'].encode()+ids.tobytes()).hexdigest())
            st.setdefault('optimization_history',[]).append(dict(update=u+1,solver=info,atom=atom))
            if st['updates']%500==0 or st['updates'] in [WARM,TOTAL]:score()
            elif st['updates']%50==0:
                s.save(key,st);s.event('slics_progress',seed=seed,update=st['updates'],total=TOTAL,step_seconds=st['seconds']/st['updates'],solver=info)
            del D,z,xx,allow;gc.collect() if st['updates']%250==0 else None
        save()
    except BaseException:save();raise
    result={}
    for policy in ['object']:
        chosen=memory.required(s,key+'_'+policy)
        for readout in ['partial']:
            name=f'slics_{policy}_{readout}';r=dict(key=key+'_'+policy,seed=seed,arm=name,method='native_slics',groups=75,rank=12,active_groups=None,
                weights_sha=training.weight_sha({'D':chosen['D']}),targets=fixed_targets(ctx),targets_sha=identity(fixed_targets(ctx)),
                total_training_updates=TOTAL,selected_total_updates=chosen['updates'],readout=readout,solver=cfg,
                head_parameters=0,training_seconds=st['seconds'],selection_policy=policy)
            r['development']=evaluate_state(ctx,r,ctx['packs']['select'],False);result[name]=r
    s.save('native_rows_s'+str(seed),result);return result

def checked_native_evaluate(ctx,m,pack,compact):
    # Exact existing object metrics; joint reconstruction is explicit for both native readouts.
    return evaluate_model(m,pack,fixed_targets(ctx),ctx['cfg'],ctx['device'],compact=compact,reconstruction=False)

def pack_variance(pack):
    if '_global_variance' in pack:return pack['_global_variance']
    a=pack['x'].reshape(-1,pack['x'].shape[-1]);total=np.zeros(a.shape[1]);energy=0.
    for lo in range(0,len(a),512):
        chunk=np.asarray(a[lo:lo+512],np.float64);total+=chunk.sum(0);energy+=float(np.square(chunk).sum())
    result=max(energy-float(total@total)/len(a),1e-12);pack['_global_variance']=result;return result

@torch.no_grad()
def evaluate_model(m,pack,targets,cfg,device,compact=False,reconstruction=True):
    """Existing metric arithmetic, one bounded patch pass and streaming global variance."""
    from scale_metrics import ReadoutView,checked
    class Meter:
        def __init__(self,parent):
            self.parent=parent;self.n_groups=parent.n_groups;self.group_size=parent.group_size;self.d=parent.d
            self.sse=0.;self.n=0;self.active=0.;self.alive=np.zeros(parent.n_groups,bool)
        def eval(self):self.parent.eval();return self
        def atoms(self):return self.parent.atoms()
        def encode(self,x):
            z=self.parent.encode(x)
            self.n+=len(x)
            if not reconstruction:return z
            joint=self.parent.solve(x).reshape(-1,self.n_groups,self.group_size) if isinstance(self.parent,NativeSLiCS) and self.parent.readout=='partial' else z
            prediction=joint.flatten(1)@self.parent.atoms().reshape(-1,self.d)
            self.sse+=float((prediction-x).double().square().sum())
            active=joint.norm(dim=-1)>1e-6;self.active+=float(active.sum());self.alive|=active.any(0).cpu().numpy()
            return z
    meter=Meter(m);ids=sorted({t[k] for t in targets for k in ['contrast_group','detector_group']});index={g:i for i,g in enumerate(ids)}
    mapped=[dict(t,contrast_group=index[t['contrast_group']],detector_group=index[t['detector_group']]) for t in targets]
    arrays=metrics.audit_arrays(ReadoutView(meter,ids),pack['x'],pack['masks'],pack['ids'],cfg,device)
    result=core.category_scores(arrays,mapped,cfg)
    if meter.n!=len(pack['ids'])*196:raise ValueError('Every held-out patch must be scored')
    result['reconstruction']=(dict(status='computed',mse=meter.sse/(meter.n*m.d),ve=1-meter.sse/pack_variance(pack),active_groups=meter.active/meter.n,
        dead_groups=int((~meter.alive).sum()),patches=meter.n) if reconstruction else
        dict(status='deferred',mse=None,ve=None,active_groups=None,dead_groups=None,patches=meter.n))
    for r,t in zip(result['categories'],targets):r.update(contrast_group=t['contrast_group'],detector_group=t['detector_group'])
    checked(result,cfg['category_indices'])
    if compact:result['_compact']=study.compact_audit(arrays,matched.mapping(mapped),cfg)
    del arrays;gc.collect();return result

def evaluate_state(ctx,row,pack,compact):
    st=memory.required(ctx['store'],row['key'])
    if row['method']=='native_slics':
        if training.weight_sha({'D':st['D']})!=row['weights_sha']:raise ValueError('Native selected weights changed')
        m=NativeSLiCS(st['D'].to(ctx['device']),row['solver'],row['readout'])
    else:
        if training.weight_sha(st['weights'])!=row['weights_sha']:raise ValueError('Selected BSF weights changed')
        m=model('released_vanilla',row['groups'],12,32,pack['x'].shape[-1]).to(ctx['device']);m.load_state_dict(st['weights'])
    del st
    start=time.monotonic()
    # A completed object pass survives an interruption in the later joint pass.
    signature=identity(dict(weights=row['weights_sha'],targets=row['targets_sha'],ids=pack['ids'],
        cohort=pack.get('identity'),normalization=identity(ctx.get('norm',{})),cfg=ctx['cfg'],readout=row.get('readout','encoder')))
    cache_key='final_objects_'+signature[:32]
    report=ctx['store'].load(cache_key) if compact else None
    if report is None:
        with matched.progress(ctx,m,'audit_'+row['arm']+'_s'+str(row['seed'])):
            report=evaluate_model(m,pack,row['targets'],ctx['cfg'],ctx['device'],compact=compact,
                reconstruction=not isinstance(m,NativeSLiCS))
        if compact:ctx['store'].save(cache_key,report)
    if compact and isinstance(m,NativeSLiCS):
        report['reconstruction']=final_reconstruction(ctx,m,pack,row['weights_sha'])

    report.update(seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters())+row.get('head_parameters',0),
        dictionary_parameters=int(m.atoms().numel()),solver_stats=copy.deepcopy(getattr(m,'stats',None)))
    # D is a buffer in native SLiCS, but is learned and must count as a model parameter.
    if row['method']=='native_slics':report['parameters']=int(m.D.numel())
    del m;gc.collect();torch.cuda.empty_cache() if str(ctx['device']).startswith('cuda') else None
    return finite_report(report)

def trim_local(ctx,keys):
    """Remove only verified redundant LOCAL slots; Drive checkpoints are never removed."""
    s=ctx['store'].job;local=Path(ctx['local']).resolve();base=Path(s.rescue).resolve();remote=Path(s.root).resolve()
    if not base.is_relative_to(local) or base==local or remote.is_relative_to(local):raise ValueError('Unsafe cache location')
    for key in keys:
        folder=(base/key).resolve()
        if folder.parent!=base:raise ValueError('Invalid local cache key')
        if not folder.exists():continue
        local_records=s._records(base,key);drive_records=s._records(remote,key)
        if not local_records or not drive_records:continue
        metadata=json.loads(local_records[0][1].read_text())
        if not any(json.loads(p.read_text())['sha256']==metadata['sha256'] for _,p,_ in drive_records):continue
        for p in list(folder.glob('slot*.pt'))+list(folder.glob('slot*.json')):
            if p.resolve().parent!=folder:raise ValueError('Unsafe local cache member')
            p.unlink()
        if not any(folder.iterdir()):folder.rmdir()

def selection(ctx):
    s=ctx['store'];old=s.load('supervised_settings_frozen')
    if old is not None:return old
    native=[];by_lr={lr:[] for lr in LRS}
    for seed in SCREEN:
        print(f'INTERLEAVED SCREEN: SLiCS seed {seed}, then all GGF learning rates; object metrics only select models.',flush=True)
        records=native_fit(ctx,seed);native.extend(records.values())
        for lr in LRS:
            row=small_fit(ctx,seed,lr);by_lr[lr].append(row)
            s.write('development_progress.json',dict(seed=seed,lr=lr,models=by_lr[lr],holdout_features_opened=False))
            a=records['slics_object_partial']['development'];b=row['development']
            print(f"DEVELOPMENT seed {seed}: SLiCS Capture {100*a['macro_capture']:.2f}%, AP {100*a['macro_ap']:.2f}%; GGF LR {lr}: Capture {100*b['macro_capture']:.2f}%, AP {100*b['macro_ap']:.2f}%; SLiCS joint VE deferred to final holdout.",flush=True)
            trim_local(ctx,[row['key'],row['fitted_key'],f'small75_s{seed}_warm'])
        trim_local(ctx,[f'native_slics12_s{seed}',f'native_slics12_s{seed}_object'])
    trials=[dict(lr=lr,models=by_lr[lr]) for lr in LRS];chosen=select_trials(trials)
    out=dict(complete=True,ggf75_lr=chosen['lr'],ggf75_policy='object',ggf_trials=trials,native_models=native,
        native_checkpoint_policies=['object'],native_readouts=['partial'],independent_holdout_opened=False,selection=protocol()['selection'])
    s.save('supervised_settings_frozen',out);s.write('supervised_settings_frozen.json',out)
    print('SUPERVISED DEVELOPMENT COMPLETE; fresh holdout still closed; selected GGF75 LR:',out['ggf75_lr'],flush=True)
    return out

def final_rows(ctx,seed,choice):
    rows=[small_fit(ctx,seed,choice['ggf75_lr'])]
    small=rows[0];st=memory.required(ctx['store'],small['fitted_key'])
    final=dict(small,key=small['fitted_key'],arm='matched_ggf_final',weights_sha=training.weight_sha(st['weights']),selected_total_updates=TOTAL,selection_policy='fixed_budget')
    final['development']=evaluate_state(ctx,final,ctx['packs']['select'],False);rows.append(final)
    # Verify identical clean training images and presentation counts for the matched pair.
    native=native_fit(ctx,seed);rows.extend(native.values())
    native_final=memory.required(ctx['store'],f'native_slics12_s{seed}')
    warm=memory.required(ctx['store'],f'small75_s{seed}_warm')
    if native_final['phase_batch_shas']['warm']!=warm['sampled_batches_sha'] or native_final['sampled_batches_sha']!=st['sampled_batches_sha']:
        raise ValueError('Matched supervised methods saw different training image schedules')
    nf=dict(native['slics_object_partial'],key=f'native_slics12_s{seed}',arm='slics_final_partial',
        weights_sha=training.weight_sha({'D':native_final['D']}),selected_total_updates=TOTAL,selection_policy='fixed_budget')
    nf['development']=evaluate_state(ctx,nf,ctx['packs']['select'],False);rows.append(nf)
    # The current full GGF is a separate capacity anchor. It is never part of the matched primary contrast.
    if seed in SCREEN:rows.append(current(ctx,seed,'agf_g'))
    for r in rows:r['capacity_scope']='matched_75x12' if r['groups']==75 else 'current_4096x12_capacity_anchor'
    del st,warm,native_final;gc.collect();return rows

ARMS=['matched_ggf','matched_ggf_final','slics_object_partial','slics_final_partial']

def expected_pairs():return {(s,a) for s in SEEDS for a in ARMS}|{(s,'current_ggf') for s in SCREEN}

def freeze(ctx,choice):
    s=ctx['store'];old=s.load('all_models_frozen')
    if old is not None:return verify_frozen(ctx)
    rows=[]
    for seed in SEEDS:
        rows.extend(final_rows(ctx,seed,choice))
        s.write('supervised_final_progress.json',dict(models=rows,complete=False,expected=len(expected_pairs())))
        trim_local(ctx,list({r['key'] for r in rows if r['seed']==seed}|{f'small75_s{seed}_warm',f'small75_s{seed}_lr{int(choice["ggf75_lr"]*1e7)}_ggf'}))
    out=dict(complete=True,models=rows,protocol=protocol(),settings_sha=identity(choice),holdout_features_opened=False,
             holdout_plan_sha=identity(memory.required(s,'fresh_holdout_plan')),
             expected_pairs=[list(p) for p in sorted(expected_pairs())])
    s.save('all_models_frozen',out);s.write('all_models_frozen.json',out)
    return verify_frozen(ctx)

def verify_frozen(ctx):
    frozen=memory.required(ctx['store'],'all_models_frozen');rows=frozen['models']
    if not frozen.get('complete') or frozen['protocol']!=protocol() or len(rows)!=len(expected_pairs()):raise ValueError('Incomplete supervised freeze')
    if {(r['seed'],r['arm']) for r in rows}!=expected_pairs():raise ValueError('Every declared seed/arm must freeze')
    if frozen['settings_sha']!=identity(memory.required(ctx['store'],'supervised_settings_frozen')):raise ValueError('Frozen development choices changed')
    if frozen['holdout_plan_sha']!=identity(memory.required(ctx['store'],'fresh_holdout_plan')):raise ValueError('Frozen holdout cohort changed')
    for r in rows:
        if identity(r['targets'])!=r['targets_sha'] or r['total_training_updates']!=TOTAL:raise ValueError('Frozen readout/budget changed')
        st=memory.required(ctx['store'],r['key']);w={'D':st['D']} if r['method']=='native_slics' else st['weights']
        if training.weight_sha(w)!=r['weights_sha']:raise ValueError('Frozen weights changed')
        if r['arm'].endswith('_final') or r['arm'].startswith('slics_final'):
            expected=TOTAL if r['method']=='native_slics' else GUIDED
            if st['updates']!=expected:raise ValueError('Fixed-budget endpoint incomplete')
        del st;gc.collect()
    matched.assert_sources(ctx);return frozen

def plan_fresh(ctx,coco):
    return reserve_predecessor_holdout(ctx,coco)

def extract_after_freeze(ctx,coco):
    verify_frozen(ctx)
    if ctx['store'].load('supervised_settings_frozen') is None:raise ValueError('Freeze tuning before any test extraction')
    return load_holdout(ctx,coco)

def seed_stat(values):
    a=np.asarray(values,float)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('At least two finite paired starts required')
    mean=float(a.mean());sd=float(a.std(ddof=1));se=sd/math.sqrt(len(a));half=float(student_t.ppf(.975,len(a)-1)*se)
    p=float(2*student_t.sf(abs(mean)/se,len(a)-1)) if se>0 else (1. if mean==0 else 0.)
    return dict(mean=mean,sd=sd,ci_low=mean-half,ci_high=mean+half,p_two_sided=p,n=len(a))

def holm(pvalues):
    p=np.asarray(pvalues,float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid p-values')
    order=np.argsort(p,kind='stable');adjusted=np.zeros(len(p));running=0.
    for i,j in enumerate(order):running=max(running,(len(p)-i)*p[j]);adjusted[j]=min(1.,running)
    return adjusted.tolist()

def summarize(ctx,models,pairs):
    from coco_ggf.matched_baselines import csv_write
    s=ctx['store'];table=[]
    for r in models:
        v=r['report'];row=r['row']
        table.append(dict(seed=row['seed'],arm=row['arm'],groups=row['groups'],width=12,
            capture_pct=100*v['macro_capture'],patch_AP_pct=100*v['macro_ap'],VE_pct=100*v['reconstruction']['ve'],
            reconstruction_active_groups=v['reconstruction']['active_groups'],parameters=v['parameters'],
            selected_updates=row['selected_total_updates'],total_fitted_updates=TOTAL,audit_seconds=v['seconds'],
            capacity_scope=row['capacity_scope']))
    csv_write(s.results/'supervised_models.csv',table);means=[]
    for arm in ARMS+['current_ggf']:
        groups=[('capacity_anchor_three',SCREEN)] if arm=='current_ggf' else [('all_seven',SEEDS),('development_tuning_orders',SCREEN),('additional_orders',SEEDS[3:])]
        for label,seeds in groups:
            rr=[r for r in table if r['arm']==arm and r['seed'] in seeds]
            if sorted(r['seed'] for r in rr)!=seeds:raise ValueError('Do not drop missing seeds')
            out=dict(arm=arm,seed_group=label,seeds=len(rr))
            for metric in ['capture_pct','patch_AP_pct','VE_pct']:
                st=seed_stat([r[metric] for r in rr]);out[metric]=st['mean'];out[metric+'_sd']=st['sd'];out[metric+'_ci_low']=st['ci_low'];out[metric+'_ci_high']=st['ci_high']
            out['mean_active_groups']=float(np.mean([r['reconstruction_active_groups'] for r in rr]));means.append(out)
    csv_write(s.results/'supervised_means.csv',means)
    paired=[]
    for a,b in [('matched_ggf','slics_object_partial'),('matched_ggf_final','slics_final_partial')]:
        for metric in ['capture_pct','patch_AP_pct','VE_pct']:
            values=[]
            for seed in SEEDS:
                x=next(r for r in table if r['seed']==seed and r['arm']==a);y=next(r for r in table if r['seed']==seed and r['arm']==b)
                values.append(x[metric]-y[metric])
            paired.append(dict(treatment=a,control=b,metric=metric,**seed_stat(values),
                primary=(a,b)==('matched_ggf','slics_object_partial') and metric!='VE_pct',
                scope='Paired training-order variation with common SVD initialization, conditional on this cohort; no dataset/researcher-selection uncertainty'))
    primary=[r for r in paired if r['primary']]
    for row,p in zip(primary,holm([r['p_two_sided'] for r in primary])):row['holm_p_two_sided']=p
    for r in paired:r.setdefault('holm_p_two_sided',None)
    csv_write(s.results/'supervised_paired_seeds.csv',paired)
    if pairs:csv_write(s.results/'supervised_paired_images.csv',pairs)
    cats=[]
    for r in models:
        for c in r['report']['categories']:cats.append(dict(seed=r['row']['seed'],arm=r['row']['arm'],**c))
    csv_write(s.results/'supervised_categories.csv',cats)
    result=dict(complete=True,means=means,seed_intervals=paired,image_intervals=pairs,fresh=True,images=1500,
        primary='Matched75x12 GGF vs native SLiCS object-selected partial concept projections; capture and AP are co-primary, Holm corrected.',
        reconstruction='Native SLiCS VE always uses the joint NNLS reconstruction; partial projections are never summed.',
        current_model='4096-group GGF is a capacity anchor, excluded from capacity-matched significance claims.',
        limits='Same available annotations and decoder capacity; GGF has an untied encoder and detection heads, SLiCS does not. Actual activity, parameter counts, selected training lengths and inference costs differ and remain visible. No universal superiority/acceptance guarantee.')
    result['evaluation_protocol']=protocol()
    result['baseline_scope']='SLiCS Algorithm 1 adapted to patch-level inputs with fixed category supports; not original image-level SLiCS evaluation.'
    s.save('supervised_test_complete',result);s.write('supervised_summary.json',result);plot(ctx,means);return result

def audit(ctx,pack):
    frozen=verify_frozen(ctx);s=ctx['store'];models=[];pairs=[]
    stamp=identity(dict(ids=pack['ids'],normalization=identity(ctx['norm']),cfg=ctx['cfg']))
    signatures=s.load('audit_signature_index') or {}
    for row in frozen['models']:
        key=f'audit_s{row["seed"]}_{row["arm"]}';sig=identity(dict(weights=row['weights_sha'],targets=row['targets_sha'],cohort=stamp,readout=row.get('readout','native_bsf')))
        saved=s.load(key)
        if saved is None:
            if sig in signatures:
                origin=memory.required(s,signatures[sig]);saved=dict(origin,row=row,signature=sig,reused_inference=True)
            else:
                report=evaluate_state(ctx,row,pack,True);compact=report.pop('_compact')
                saved=dict(row=row,signature=sig,report=report,compact=compact);signatures[sig]=key
            s.save(key,saved);s.save('audit_signature_index',signatures);s.write('reports/'+key+'.json',dict(row=row,report=saved['report']))
        if saved['signature']!=sig:raise ValueError('Audit signature changed')
        models.append(dict(row=row,report=saved['report']));del saved;gc.collect()
        s.event('supervised_test_model_complete',seed=row['seed'],arm=row['arm'])
    for seed in SEEDS:
        for a,b in [('matched_ggf','slics_object_partial'),('matched_ggf_final','slics_final_partial')]:
            key=f'image_ci_s{seed}_{a}_{b}';saved=s.load(key)
            if saved is None:
                x=memory.required(s,f'audit_s{seed}_{a}');y=memory.required(s,f'audit_s{seed}_{b}')
                ci=study.paired(x['compact'],y['compact'],ctx['cfg'])
                saved=dict(seed=seed,treatment=a,control=b,capture_ci_low_pp=100*ci['capture_gain_ci95'][0],
                    capture_ci_high_pp=100*ci['capture_gain_ci95'][1],AP_ci_low_pp=100*ci['ap_gain_ci95'][0],
                    AP_ci_high_pp=100*ci['ap_gain_ci95'][1],valid_bootstraps=ci['valid_bootstrap_replicates'],
                    invalid_bootstraps=ci['invalid_full_cohort_replicates'],scope='Paired whole-image bootstrap conditional on these fitted models')
                s.save(key,saved);del x,y;gc.collect()
            pairs.append(saved)
    return summarize(ctx,models,pairs)

def plot(ctx,means):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    rr=[r for r in means if r['seed_group'] in ['all_seven','capacity_anchor_three'] and r['arm'] in ['matched_ggf','slics_object_partial','current_ggf']]
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained',sharey=True)
    names=dict(matched_ggf='GGF75x12 (7 seeds)',slics_object_partial='SLiCS patch adaptation (7 seeds)',slics_object_joint='SLiCS75x12 joint (7 seeds)',current_ggf='GGF4096x12 anchor (3 seeds)')
    for ax,metric,title in zip(axs,['capture_pct','patch_AP_pct','VE_pct'],['Capture','Patch AP','Reconstruction VE']):
        bars=ax.barh(range(len(rr)),[r[metric] for r in rr],color=['#077f83' if r['arm'].startswith(('matched','current')) else '#9473ac' for r in rr],
            xerr=[r[metric+'_sd'] for r in rr],capsize=3)
        ax.bar_label(bars,labels=[f'{r[metric]:.2f}%' for r in rr],padding=4,fontsize=9)
        ax.set_yticks(range(len(rr)),[names[r['arm']] for r in rr]);ax.set_title(title);ax.set_xlabel('Percent; error bars = seed SD')
        lo=min(0.,min(r[metric] for r in rr)-8);hi=max(105.,max(r[metric] for r in rr)+15);ax.set_xlim(lo,hi);ax.grid(axis='x',alpha=.15)
    axs[0].invert_yaxis();fig.suptitle('Prospective supervised comparison | seven seeds | new holdout')
    folder=ctx['store'].results/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/('supervised_comparison.'+ext),dpi=200)
    plt.close(fig)

def export(ctx,source_zip,complete=False):
    s=ctx['store'];s.write('completion_status.json',dict(complete=complete,settings_frozen=s.load('supervised_settings_frozen') is not None,
        all_models_frozen=s.load('all_models_frozen') is not None,test_complete=s.load('supervised_test_complete') is not None,
        bsfs_trained=0,modified_bsf_control=False))
    path=s.output/f'ggf_supervised_{s.run_id[:12]}_results.zip';hashes={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(s.results.rglob('*')):
            if p.is_file():
                n='results/'+p.relative_to(s.results).as_posix();z.write(p,n);hashes[n]=sha_file(p)
        z.write(source_zip,'source.zip');hashes['source.zip']=sha_file(source_zip)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in hashes.items()):raise ValueError('Export checksum mismatch')
    return path
