"""New category-restricted fits. Original supervised checkpoints are never reused."""
import copy,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core
from coco_coordinated import run as memory
from coco_ggf.slics_closeout import support,atom_sweep
from coco_ggf.slics_repair import Solver
from training_precision import JointPrecisionRetry,precision_atom_sweep
from scale_fit import fit
from scale_metrics import train_targets
from scale_models import model
from scale_data import mask_dictionary
from cd_common import SPEC,make_store,store_path,fit_payload,finish_model,flush
from cd_data import training_context

def get_store(ctx,fold,seed,kind):
    payload=fit_payload(ctx,fold,seed,kind)
    if kind=='concept_sae':payload.update(cohort='coco',task='category_disjoint_guidance_only')
    if kind=='vanilla':payload.update(fold='label_free_shared',guide=[],excluded=ctx['eligible'])
    s=make_store(store_path(ctx,fold,seed,kind),Path(ctx['options']['local'])/'fits'/f'{fold}_{seed}_{kind}',payload,ctx['store'].require_drive)
    ctx.setdefault('writable_stores',[]).append(s);return s

def frozen_row(s):
    state=s.load('frozen')
    if state is None:return None
    weights=state['weights'];digest=training.weight_sha({'D':weights} if state['kind']=='slics' else weights)
    if state['weights_sha']!=digest:raise ValueError('New frozen checkpoint checksum differs')
    return dict(kind=state['kind'],fold=state['fold'],seed=state['seed'],weights_sha=digest,output=str(s.output),run_id=s.run_id,guide=state['guide'],excluded=state['excluded'])

def open_frozen(row,local):
    s=memory.FrozenStore(Path(row['output'])/'checkpoints'/row['run_id'][:20],Path(local)/'read_fitted')
    state=memory.required(s,'frozen');w=state['weights']
    if training.weight_sha({'D':w} if row['kind']=='slics' else w)!=row['weights_sha']:raise ValueError('Frozen fitted weights differ')
    return state

def fit_bsf(ctx,fold,seed,kind):
    if kind=='vanilla':fold=0
    s=get_store(ctx,fold,seed,kind);old=frozen_row(s)
    if old is not None:return old
    sub=training_context(ctx,fold,seed,kind,s);sub['local'].mkdir(parents=True,exist_ok=True)
    g,b,k=SPEC['groups'],SPEC['width'],SPEC['top_k'];d=ctx['packs']['train']['x'].shape[-1]
    if kind=='vanilla':
        # The label-free baseline does not receive any annotation arrays.
        sub['packs']={name:dict(x=p['x'],ids=p['ids'],identity=identity(dict(ids=p['ids'],norm=identity(ctx['norm'])))) for name,p in sub['packs'].items()}
    torch.manual_seed(seed);m=model('released_vanilla',g,b,k,d)
    if kind=='ggf':
        D=mask_dictionary(sub)
        with torch.no_grad():m.W_dec[:len(D)].copy_(D);m.W_enc.copy_(m.W_dec.T);m.b_enc.zero_()
    start=training.weights(m);del m
    if kind=='supervised_vanilla':
        vanilla=get_store(ctx,0,seed,'vanilla');warm=memory.required(vanilla,'warm')
        if warm['updates']!=SPEC['warm_updates']:raise ValueError('Incomplete label-free warm-up')
    else:warm=fit(sub,'warm','released_vanilla',g,k,seed,start,SPEC['warm_updates'],SPEC['warm_updates'],stream=0)
    targets=None
    if kind!='vanilla':
        m=model('released_vanilla',g,b,k,d).to(ctx['device']);m.load_state_dict(warm['weights'])
        targets=train_targets(sub,m,'guided_training_targets');del m;gc.collect();torch.cuda.empty_cache()
        if {r['category_index'] for r in targets}!=set(ctx['folds'][fold]['guide']):raise ValueError('Training target categories leaked or disappeared')
    if kind=='supervised_vanilla':
        if seed==21:
            selection=s.load('fixed_weight_selection')
            if selection is None:
                trials=[]
                for j,pair in enumerate(SPEC['fixed_weight_candidates']):
                    trial=dict(sub,rule=dict(sub['rule'],fixed_weights=[1.,*pair]))
                    state=fit(trial,'fixed_candidate_'+str(j),'released_vanilla',g,k,seed,warm['weights'],SPEC['guided_updates'],SPEC['guided_updates'],targets=targets,fixed=True,stream=1,cap=SPEC['cap'],learning_rate=SPEC['continuation_lr'])
                    trials.append(dict(candidate=j,weights=[1.,*pair],development=state['history'][-1]['validation']))
                    del state;gc.collect();torch.cuda.empty_cache()
                reference=warm['history'][-1]['validation']['reconstruction']['mse']
                feasible=[r for r in trials if r['development']['reconstruction']['mse']<=1.02*reference]
                winner=max(feasible,key=lambda r:(r['development']['macro_capture']+.2*r['development']['macro_ap'],-r['candidate'])) if feasible else min(trials,key=lambda r:(r['development']['reconstruction']['mse'],r['candidate']))
                selection=dict(candidate=winner['candidate'],weights=winner['weights'],trials=trials,reconstruction_guard_satisfied=bool(feasible),
                    rule='Guided development only. Max final Capture+0.2AP within warm MSE+2%. If none qualifies retain minimum MSE and disclose. Lock one pair per fold for remaining runs.')
                s.save('fixed_weight_selection',selection);s.write('fixed_weight_selection.json',selection)
            sub['rule']=dict(sub['rule'],fixed_weights=selection['weights'])
            # Candidate identities do not include the key, so the selected full
            # state is exactly the continuation state. No duplicate fitting.
            if s.load('continuation') is None:s.save('continuation',memory.required(s,'fixed_candidate_'+str(selection['candidate'])))
        else:
            selection=memory.required(get_store(ctx,fold,21,kind),'fixed_weight_selection')
            sub['rule']=dict(sub['rule'],fixed_weights=selection['weights'])
    final=fit(sub,'continuation','released_vanilla',g,k,seed,warm['weights'],SPEC['guided_updates'],SPEC['guided_updates'],
        targets=targets,cap=SPEC['cap'],fixed=(kind=='supervised_vanilla'),stream=1,learning_rate=SPEC['continuation_lr'])
    row=finish_model(s,final,kind,fold,seed)
    s.write('training_lineage.json',dict(guide=[] if kind=='vanilla' else ctx['folds'][fold]['guide'],
        excluded=ctx['folds'][fold]['excluded'],warm_weights_sha=training.weight_sha(warm['weights']),
        stream_batch_sha=final['sampled_batches_sha'],stream_noise_sha=final['sampled_noise_sha'],fixed_endpoint=True,
        fixed_weights=sub['rule']['fixed_weights'] if kind=='supervised_vanilla' else None))
    del warm,final;gc.collect();torch.cuda.empty_cache();return row

def oriented_initializer(sub):
    D=mask_dictionary(sub).clone();b=SPEC['width']
    for j,c in enumerate(sub['cfg']['category_indices']):
        block=D[j*b:(j+1)*b].double().numpy();pos=np.zeros(b);neg=np.zeros(b)
        for x,m in zip(sub['packs']['train']['x'],sub['packs']['train']['masks']):
            keep=(m['crowd']<=.1)&(m['fraction'][:,c]/255>=.5)
            a=np.asarray(x[keep],np.float64)@block.T
            pos+=np.square(a.clip(min=0)).sum(0);neg+=np.square(a.clip(max=0)).sum(0)
        D[j*b:(j+1)*b]*=torch.tensor(np.where(pos>=neg,1.,-1.),dtype=D.dtype)[:,None]
    return D

def fit_slics(ctx,fold,seed):
    s=get_store(ctx,fold,seed,'slics');old=frozen_row(s)
    if old is not None:return old
    sub=training_context(ctx,fold,seed,'slics',s);guide=sub['cfg']['category_indices'];width=SPEC['width']
    D0=oriented_initializer(sub);total=SPEC['warm_updates']+SPEC['guided_updates'];solver_cfg=dict(nnls_iterations=2000,nnls_tolerance=1e-5,nnls_fallback_iterations=20000)
    signature=identity(dict(initial=training.weight_sha({'D':D0}),guide=guide,seed=seed,spec=SPEC,train=sub['packs']['train']['identity']))
    st=s.load('native_training')
    if st is None:st=dict(fingerprint=signature,updates=0,target=total,D=D0,seconds=0.,sample_rng=None,permutation=None)
    if st['fingerprint']!=signature:raise ValueError('SLiCS resume identity differs')
    rng=np.random.default_rng(126000+seed)
    if st['sample_rng'] is not None:rng.bit_generator.state=st['sample_rng']
    n=len(sub['packs']['train']['ids']);nb=n//SPEC['images_per_update']
    retry=JointPrecisionRetry(Solver.solve,emit=lambda e:s.event('nnls_precision',**e))
    def save(force=False):
        st['sample_rng']=copy.deepcopy(rng.bit_generator.state)
        (s._width_drive_recovery_original_save if force else s.save)('native_training',st,drive=True)
    save(True)
    try:
        while st['updates']<total:
            u=st['updates'];offset=u if u<SPEC['warm_updates'] else u-SPEC['warm_updates']
            if u==SPEC['warm_updates'] and not st.get('phase_reset'):
                rng=np.random.default_rng(226000+seed);st['phase_reset']=True
            if offset%nb==0 or st['permutation'] is None:st['permutation']=rng.permutation(n)
            ids=st['permutation'][(offset%nb)*SPEC['images_per_update']:((offset%nb)+1)*SPEC['images_per_update']]
            pack=sub['packs']['train'];x=torch.tensor(np.array(pack['x'][ids],copy=True),device=ctx['device']).reshape(-1,D0.shape[1])
            allow,known=support([pack['masks'][i] for i in ids],guide,dict(rank=width),'mask',ctx['device'])
            keep=torch.as_tensor(known,device=ctx['device']);x=x[keep];allow=allow[keep]
            D=st['D'].to(ctx['device']).clone();started=time.monotonic()
            columns=allow.any(0);z=torch.zeros((len(x),len(D)),device=ctx['device']);info=dict(max_relative_kkt=0.,cpu_fallback_rows=0)
            if bool(columns.any()):
                zz,info=retry(Solver(D[columns],solver_cfg),x,allow[:,columns]);z=z.to(zz.dtype);z[:,columns]=zz
            changed,atom=precision_atom_sweep(atom_sweep,x,D,z,allow,False)
            if not torch.isfinite(changed).all():raise FloatingPointError('Nonfinite SLiCS update')
            if str(ctx['device']).startswith('cuda'):torch.cuda.synchronize()
            st.update(D=changed.cpu(),updates=u+1,seconds=st['seconds']+time.monotonic()-started,last_solver=info)
            if st['updates']%50==0 or st['updates']==total:
                save(force=st['updates']==total)
                s.event('SLiCS_training',fold=fold,seed=seed,update=st['updates'],total=total,seconds_per_update=st['seconds']/st['updates'],solver=info)
            del x,D,z,allow,changed
        save(True);row=finish_model(s,st,'slics',fold,seed)
        s.write('training_lineage.json',dict(guide=guide,groups=len(guide),width=width,endpoint=total,selection='Fixed endpoint, no excluded-category monitoring',seconds=st['seconds']))
        return row
    except BaseException:
        # Only the last committed generation is flushed, never a partial sweep.
        flush(s);raise
    finally:gc.collect();torch.cuda.empty_cache()

