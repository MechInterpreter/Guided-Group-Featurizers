"""Random warm-up + both object objectives + scalar heads, fixed weights only.

Three seed-21 full-budget candidates. Development scores select one pair;
all seeds then use that same pair. No test observations select anything.
"""
import copy,gc
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from scale_fit import fit
from scale_models import model
from scale_metrics import train_targets
import scale_run

CANDIDATES=((.05,.01),(.20,.20),(.40,.40))

def choose(trials):
    # Same historical capture+.2AP utility, with a common random warm-up MSE guard.
    feasible=[r for r in trials if r['development']['reconstruction']['mse']<=1.02*r['reference_warm_mse']]
    pool=feasible or trials
    if feasible:
        best=max(pool,key=lambda r:(r['development']['macro_capture']+.2*r['development']['macro_ap'],-r['candidate']))
    else:
        best=min(pool,key=lambda r:(r['development']['reconstruction']['mse'],r['candidate']))
    return dict(candidate=best['candidate'],weights=[1.,*CANDIDATES[best['candidate']]],
        reconstruction_guard_satisfied=bool(feasible),
        rule='Max development capture+0.2AP within random-warm MSE+2%; if none eligible, minimum development MSE; retain and disclose failure',
        selection_seed=21,full_candidate_updates=3000,candidates=[list(x) for x in CANDIDATES],
        prior_GGF_search_budget_matched=False,scope='Three-point fixed-weight calibration, not exhaustive optimization')

def trial(ctx,seed,j):
    G,K,B=1024,16,12;s=ctx['store'];base=f'g1024_k16_s{seed}_released_vanilla';warmkey=base+'_warm'
    initial=scale_run.initial(ctx,'released_vanilla',G,K,seed,False)
    warm=fit(ctx,warmkey,'released_vanilla',G,K,seed,initial,7640,7640,learning_rate=.0003,stream=0)
    keytargets=base+'_targets';targets=s.load(keytargets)
    if targets is None:
        m=model('released_vanilla',G,B,K,ctx['packs']['train']['x'].shape[-1]).to(ctx['device']);m.load_state_dict(warm['weights'])
        targets=train_targets(ctx,m,keytargets);del m;gc.collect();torch.cuda.empty_cache()
    cc=dict(ctx,rule=copy.deepcopy(ctx['rule']));cc['rule']['fixed_weights']=[1.,*CANDIDATES[j]]
    key=base+f'_fixed_supervision_candidate{j}'
    ctx['active_key']=key
    try:
        st=fit(cc,key,'released_vanilla',G,K,seed,warm['weights'],3000,3000,targets=targets,
            cap=.4,learning_rate=.0003,fixed=True,stream=1)
    finally:
        # The fit uses a copied rule. Propagate its active key so outer cleanup
        # can flush the last valid LOCAL generation if Adam was interrupted.
        ctx['active_key']=cc.get('active_key')
    if st['balance'] is not None or tuple(st['head'].shape)!=(len(targets),2):raise ValueError('Fixed supervision acquired controller or wrong heads')
    row=dict(key=key,kind='supervised_vanilla',method='released_vanilla',seed=seed,groups=G,rank=B,active_groups=K,
        cap=None,lr=.0003,coefficient=.001,masked=False,targets=targets,targets_sha=identity(targets),
        weights_sha=training.weight_sha(st['weights']),additional_updates=st['updates'],total_training_updates=10640,
        reference_warm_mse=warm['history'][-1]['validation']['reconstruction']['mse'],
        parameter_count=sum(v.numel() for v in st['weights'].values()),head_parameters=2*len(targets),
        history_key=key,warm_weights_sha=training.weight_sha(warm['weights']),warm_key=warmkey,
        candidate=j,fixed_weights=[1.,*CANDIDATES[j]],controller_parameters=0)
    del st,warm,initial;gc.collect()
    row['development']=scale_run.development(cc,row)['report'];return row

def select(ctx):
    s=ctx['store'];saved=s.load('supervised_fixed_weight_selection')
    if saved is not None:
        if saved['candidates']!=[list(x) for x in CANDIDATES]:raise ValueError('Fixed-weight search changed')
        return saved
    trials=[]
    for j in range(len(CANDIDATES)):
        key=f'supervised_fixed_candidate_{j}';r=s.load(key)
        if r is None:r=trial(ctx,21,j);s.save(key,r)
        trials.append(r);s.write('fixed_weight_tuning_progress.json',dict(trials=trials,complete=False))
    choice=choose(trials);choice['trials']=trials
    s.save('supervised_fixed_weight_selection',choice);s.write('fixed_weight_selection.json',choice)
    print('FIXED SUPERVISION WEIGHTS LOCKED',choice['weights'],flush=True);return choice

def endpoint(ctx,seed):
    choice=select(ctx)
    return choice['trials'][choice['candidate']] if seed==21 else trial(ctx,seed,choice['candidate'])
