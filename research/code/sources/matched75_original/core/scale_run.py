"""Development factorial, cap screen, frozen seven-seed suite and exchanges."""
import copy,gc,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity,sha_file
from coco_ggf.capacity_fit import state_identity
from coco_coordinated import run as memory
from coco_ggf import five_exchanges as exchange,exchange5_ops as ops,matched_baselines as matched
from scale_plan import SCREEN_SEEDS,FINAL_SEEDS,GRID,CAPS,group_tag,cap_tag,select
from scale_models import model,BASELINE_AUDIT
from scale_fit import fit
from scale_metrics import train_targets,evaluate
from scale_data import mask_dictionary


def initial(ctx,method,g,k,seed,masked=False,coefficient=.001):
    torch.manual_seed(seed);d=ctx['packs']['train']['x'].shape[-1]
    m=model(method,g,ctx['rule']['width'],k,d,coefficient)
    if masked:
        if method!='released_vanilla':raise ValueError('Mask initialization only enters AGF/free-map ablation')
        D=mask_dictionary(ctx);n=len(D)
        with torch.no_grad():m.W_dec[:n].copy_(D);m.W_enc.copy_(m.W_dec.T);m.b_enc.zero_()
    if method=='paper_grassmannian':
        with torch.no_grad():
            x=torch.as_tensor(np.array(ctx['packs']['train']['x'][0],copy=True)).reshape(-1,d)
            prediction,_=m(x)
            gamma=(x*prediction).sum()/prediction.square().sum().clamp_min(1e-12)
            m.log_gamma.copy_(gamma.clamp_min(1e-6).log())
    w=training.weights(m);del m;return w


def lr_tag(lr):return f'lr{int(round(lr*1e7))}'


def run_branch(ctx,g,k,seed,kind='agf_g',cap=.4,limit=None,lr=None,coefficient=.001):
    """The 3000-update screen is an exact prefix of the 12000 final phase."""
    rule=ctx['rule'];lr=rule['lr'] if lr is None else lr;limit=rule['screen_updates'] if limit is None else limit
    method='released_vanilla' if kind in ['agf_g','mask_only','fixed_guided'] else kind
    masked=kind in ['agf_g','mask_only','fixed_guided'];sup=kind in ['agf_g','fixed_guided']
    warmbase=f'{group_tag(g,k)}_s{seed}_{"mask" if masked else method}'
    base=warmbase+f'_{lr_tag(lr)}_p{int(round(coefficient*1e7))}'
    warmkey=warmbase+'_warm'
    start=initial(ctx,method,g,k,seed,masked,.001)
    warm=fit(ctx,warmkey,method,g,k,seed,start,rule['warm_updates'],rule['warm_updates'],
             learning_rate=rule['warm_lr'],coefficient=.001,stream=0)
    reference_warm=warm['history'][-1]['validation']['reconstruction']['mse']
    targetskey=warmbase+'_targets';targets=ctx['store'].load(targetskey)
    if targets is None:
        m=model(method,g,rule['width'],k,ctx['packs']['train']['x'].shape[-1],coefficient).to(ctx['device']);m.load_state_dict(warm['weights'])
        targets=train_targets(ctx,m,targetskey);del m;gc.collect();torch.cuda.empty_cache()
    # Reconstruction-only controls use exactly the same continuation stream.
    key=base+'_'+kind+'_'+cap_tag(cap) if kind=='agf_g' else base+'_'+kind
    continuation_initial=dict(warm['weights'])
    if method=='paper_group_lasso':continuation_initial['coefficient']=torch.tensor(float(coefficient))
    st=fit(ctx,key,method,g,k,seed,continuation_initial,rule['final_guided_updates'],limit,
           targets=targets if sup else None,cap=cap if kind=='agf_g' else .4,learning_rate=lr,fixed=kind=='fixed_guided',coefficient=coefficient,stream=1)
    if st['updates']<limit:raise ValueError('Incomplete stage')
    row=dict(key=key,kind=kind,method=method,seed=seed,groups=g,rank=rule['width'],active_groups=k,cap=cap if kind=='agf_g' else None,
        lr=lr,coefficient=coefficient,masked=masked,targets=targets,targets_sha=identity(targets),
        weights_sha=training.weight_sha(st['weights']),additional_updates=st['updates'],
        total_training_updates=rule['warm_updates']+st['updates'],reference_warm_mse=reference_warm,
        parameter_count=sum(v.numel() for v in st['weights'].values()),head_parameters=150 if sup else 0,
        model_parameters=None,maximum_active_coordinates=k*rule['width'],history_key=key)
    del warm,start;gc.collect()
    return row


def development(ctx,row):
    key='dev_'+row['key']+'_'+str(row['additional_updates']);old=ctx['store'].load(key)
    if old is not None:
        if old['weights_sha']!=row['weights_sha']:raise ValueError('Development audit identity changed')
        return old
    state=memory.required(ctx['store'],row['key'])
    m=model(row['method'],row['groups'],row['rank'],row['active_groups'],ctx['packs']['train']['x'].shape[-1],row['coefficient']).to(ctx['device'])
    m.load_state_dict(state['weights']);row['model_parameters']=sum(p.numel() for p in m.parameters());del state
    v=evaluate(m,ctx['packs']['select'],row['targets'],ctx['cfg'],ctx['device'])
    out=dict(weights_sha=row['weights_sha'],report=v);ctx['store'].save(key,out)
    ctx['store'].write('development/'+key+'.json',out);del m;gc.collect();torch.cuda.empty_cache();return out


def means(rows,ident,cost):
    if sorted(r['seed'] for r in rows)!=SCREEN_SEEDS:raise ValueError('Screen must contain three paired seeds')
    reports=[r['development'] for r in rows]
    return dict(id=ident,seeds=SCREEN_SEEDS,cost=cost,capture=float(np.mean([r['macro_capture'] for r in reports])),
                ap=float(np.mean([r['macro_ap'] for r in reports])),ve=float(np.mean([r['reconstruction']['ve'] for r in reports])))


def factorial(ctx):
    s=ctx['store'];saved=s.load('configuration_selected')
    if saved is not None:return saved
    allrows=[];records=[]
    for g,k in GRID:
        guided=[]
        for seed in SCREEN_SEEDS:
            for kind in ['agf_g','released_vanilla']:
                row=run_branch(ctx,g,k,seed,kind);row['development']=development(ctx,row)['report'];allrows.append(row)
                if kind=='agf_g':guided.append(row)
                s.write('factorial_progress.json',dict(models=allrows,complete=False))
        records.append(means(guided,group_tag(g,k),g*ctx['rule']['width']*k))
    choice=select(records,group_tag(*GRID[0]),ctx['rule']);g,k=next((g,k) for g,k in GRID if group_tag(g,k)==choice['chosen'])
    choice.update(groups=g,active_groups=k,models=allrows,complete=True)
    s.save('configuration_selected',choice);s.write('configuration_selected.json',choice)
    print('Development-selected group count / Top-k:',g,k,flush=True);return choice


def caps(ctx,configuration):
    s=ctx['store'];saved=s.load('cap_selected')
    if saved is not None:return saved
    g=configuration['groups'];k=configuration['active_groups'];records=[];allrows=[]
    for cap in CAPS:
        rows=[]
        for seed in SCREEN_SEEDS:
            row=run_branch(ctx,g,k,seed,'agf_g',cap);row['development']=development(ctx,row)['report'];rows.append(row);allrows.append(row)
        records.append(means(rows,cap_tag(cap),cap))
    choice=select(records,cap_tag(.4),ctx['rule']);cap=next(c for c in CAPS if cap_tag(c)==choice['chosen'])
    choice.update(cap=cap,models=allrows,complete=True);s.save('cap_selected',choice);s.write('cap_selected.json',choice)
    print('Development-selected adaptive cap:',cap,flush=True);return choice


def tune_baselines(ctx,g,k):
    s=ctx['store'];saved=s.load('baseline_settings_selected')
    if saved is not None:return saved
    choices={};details=[]
    for method in ctx['rule']['methods']:
        trials=[]
        # The main released Vanilla LR .0003 is already fitted by the factorial.
        for j,lr in enumerate(ctx['rule']['baseline_learning_rates']):
            coef=ctx['rule']['paper_lasso_coefficients'][j] if method=='paper_group_lasso' else .001
            rows=[run_branch(ctx,g,k,seed,method,lr=lr,coefficient=coef) for seed in SCREEN_SEEDS]
            rec=[];activity=[]
            for row in rows:
                v=development(ctx,row)['report'];rec.append(v['reconstruction']['mse']);activity.append(v['reconstruction']['active_groups'])
            eligible=all(abs(a-k)<=.2*k for a in activity) if method.endswith('group_lasso') else True
            trial=dict(lr=lr,coefficient=coef,mse=float(np.mean(rec)),activity=activity,sparsity_eligible=eligible,models=rows)
            trials.append(trial);s.write('baseline_tuning_progress.json',dict(completed=details,current_method=method,trials=trials))
        feasible=[t for t in trials if t['sparsity_eligible']]
        best=min(feasible or trials,key=lambda r:(r['mse'],r['lr']))
        choices[method]=dict(lr=best['lr'],coefficient=best['coefficient'],sparsity_matched=bool(feasible),selection='minimum three-seed development MSE among sparsity-eligible trials; explicit mismatch retained if none eligible')
        details.append(dict(method=method,trials=trials,chosen=choices[method]))
    out=dict(choices=choices,details=details,complete=True);s.save('baseline_settings_selected',out);s.write('baseline_settings_selected.json',out)
    return out


def exchanges(ctx,row,native_reference):
    """Reuse checked width-aware dependency/exchange algebra, not old decisions."""
    key='exchange_'+row['key'];cached=ctx['store'].load(key)
    if cached is not None:return cached
    st=memory.required(ctx['store'],row['key']);root=dict(weights=st['weights'],spec=dict(kind='native',groups=row['groups'],rank=row['rank'],active=row['active_groups'],targets=row['targets'],
        atom_ids=list(range(row['groups']*row['rank']))),head=st['head'],source_weights_sha=row['weights_sha'],
        source_head_sha=training.weight_sha({'head':st['head']}),source_balance_sha=identity(state_identity(st['balance'])),
        source_optimizer_sha=identity(state_identity(st['optimizer'])),reference_mse=native_reference)
    s=exchange.SeedStore(ctx['store'],row['seed'])
    rule=exchange.settings();rule.update(groups=row['groups'],block_rank=row['rank'],active_groups=row['active_groups'])
    dev=development(ctx,row)['report']
    arm=dict(ctx,store=s,seed=row['seed'],initial=root,source_row=dict(development=dev),
             source_head_sha=training.weight_sha({'head':st['head']}),protocol=rule)
    if dev['reconstruction']['mse']>root['reference_mse']*1.02:
        s.save('final_agfx',root)
        selected=dict(accepted_proposals=[],attempted=0,reason='AGF-G exceeds common native-warm development MSE limit; retain unchanged parent',
                      final_sha=training.weight_sha(root['weights']),new_gradient_updates=0)
    else:selected=exchange.select_arm(arm)
    final=memory.required(s,'final_agfx');exchange.assert_permutation(root,final)
    newkey=row['key']+'_exchanged';ctx['store'].save(newkey,dict(weights=final['weights'],head=st['head'],updates=row['additional_updates'],target=ctx['rule']['final_guided_updates']))
    selected['total_training_updates']=row['total_training_updates']
    out=dict(row,key=newkey,kind='agf_x',weights_sha=training.weight_sha(final['weights']),exchange_decisions=selected,new_gradient_updates=0,
             common_native_warm_mse=native_reference)
    if 'development' in selected:out['development']=selected['development']['select']
    ctx['store'].save(key,out);ctx['store'].write('exchanges/'+key+'.json',out);del st,final,root;gc.collect();return out


def final_suite(ctx,configuration,capchoice,tuning):
    s=ctx['store'];old=s.load('all_models_frozen')
    if old is not None:return old
    g=configuration['groups'];k=configuration['active_groups'];cap=capchoice['cap'];rows=[]
    # Preserve the priority: all AGF-G endpoints first; all matched baselines;
    # ablations; all exchange descendants. No holdout features open in this phase.
    for kind in ['agf_g',*ctx['rule']['methods'],'mask_only','fixed_guided']:
        for seed in FINAL_SEEDS:
            choice=tuning['choices'].get(kind,dict(lr=ctx['rule']['lr'],coefficient=.001))
            row=run_branch(ctx,g,k,seed,kind,cap=cap,limit=ctx['rule']['final_guided_updates'],lr=choice['lr'],coefficient=choice['coefficient'])
            row['development']=development(ctx,row)['report'];rows.append(row)
            s.write('final_suite_progress.json',dict(models=rows,complete=False))
            del row;gc.collect()
    for parent in [r for r in rows if r['kind']=='agf_g']:
        native=next(r for r in rows if r['kind']=='released_vanilla' and r['seed']==parent['seed'])
        parent['common_native_warm_mse']=native['reference_warm_mse']
        rows.append(exchanges(ctx,parent,native['reference_warm_mse']))
    expected=len(FINAL_SEEDS)*(len(ctx['rule']['methods'])+4)
    if len(rows)!=expected or len({(r['seed'],r['kind']) for r in rows})!=expected:raise ValueError('Final catalog incomplete')
    for row in rows:
        st=memory.required(s,row['key'])
        if st['updates']!=ctx['rule']['final_guided_updates'] or training.weight_sha(st['weights'])!=row['weights_sha']:
            raise ValueError('Final endpoint changed or incomplete')
        del st
    out=dict(complete=True,models=rows,configuration=configuration['chosen'],cap=cap,test_opened=False,
             final_seeds=FINAL_SEEDS,all_models_frozen_before_test=True)
    s.save('all_models_frozen',out);s.write('all_models_frozen.json',out);matched.assert_sources(ctx);return out


def preflight(ctx):
    """Disposable largest-config GPU smoke checks for every requested method."""
    old=ctx['store'].load('study_preflight')
    if old is not None:return old
    timings=[];g=4096;k=32;b=ctx['rule']['width'];x=torch.as_tensor(np.array(ctx['packs']['train']['x'][:ctx['rule']['images_per_update']],copy=True),device=ctx['device']).reshape(-1,ctx['packs']['train']['x'].shape[-1])
    for method in ctx['rule']['methods']:
        torch.manual_seed(126401);m=model(method,g,b,k,x.shape[-1]).to(ctx['device']);opt=torch.optim.Adam(m.parameters(),lr=.0003)
        torch.cuda.reset_peak_memory_stats();began=time.monotonic();m.train()
        for _ in range(3):
            loss,_=m.loss(x);opt.zero_grad();loss.backward();opt.step();m.normalize_decoder()
        torch.cuda.synchronize();m.eval()
        with torch.no_grad():pred,z=m(x)
        if not torch.isfinite(pred).all() or not torch.isfinite(loss):raise ValueError('T4 numerical preflight failed')
        if not method.endswith('group_lasso') and int(z.ne(0).any(-1).sum(-1).max())>k:raise ValueError('Top-k violated')
        timings.append(dict(method=method,seconds_per_patch_update=(time.monotonic()-began)/3,
                            peak_GiB=torch.cuda.max_memory_allocated()/2**30,parameters=sum(p.numel() for p in m.parameters())))
        del m,opt;gc.collect();torch.cuda.empty_cache()
    # Also exercise the expensive three-loss shared-encoder gradient controller
    # on the largest dictionary and the ACTUAL eight-image batch size.
    from coco_adaptation import core
    from coco_ggf.capacity_fit import Balancer
    from scale_models import encoder_parameters
    m=model('released_vanilla',g,b,k,x.shape[-1]).to(ctx['device']);m.load_state_dict(initial(ctx,'released_vanilla',g,k,21,True))
    targets=[dict(category_index=c,category=ctx['provenance']['categories'][c]['name'],contrast_group=j,detector_group=j,heat_mean=0.,heat_scale=1.) for j,c in enumerate(ctx['cfg']['category_indices'])]
    head=torch.nn.Parameter(torch.zeros(len(targets),2,device=ctx['device']));opt=torch.optim.Adam(list(m.parameters())+[head],lr=.0003)
    recipe=copy.deepcopy(ctx['balance_recipe']);recipe['bounds']['capture'][1]=1.;recipe['bounds']['detection'][1]=1.;controller=Balancer(recipe,ctx['device'])
    images=x.reshape(-1,196,x.shape[-1]);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for _ in range(3):
        m.train();rec,_=m.loss(x);cap,det,_=core.supervised_loss(m,images,ctx['packs']['train']['masks'][:len(images)],targets,head,ctx['cfg']);terms=[rec,cap,det]
        for __ in range(recipe['reference_batches']):controller.observe(terms)
        norms=[]
        for term in terms:
            gradients=torch.autograd.grad(term,encoder_parameters(m),retain_graph=True,allow_unused=True)
            norms.append(torch.sqrt(sum((q.detach().square().sum() for q in gradients if q is not None),rec.detach()*0)))
        controller.adapt(terms,torch.stack(norms));w=controller.weights().detach();loss=rec+w[1]*cap+w[2]*det
        opt.zero_grad();loss.backward();opt.step();m.normalize_decoder()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):raise ValueError('AGF T4 controller preflight failed')
    timings.append(dict(method='agf_g',seconds_per_patch_update=(time.monotonic()-start)/3,
                        peak_GiB=torch.cuda.max_memory_allocated()/2**30,parameters=sum(p.numel() for p in m.parameters())+head.numel()))
    del m,opt,head,controller;gc.collect();torch.cuda.empty_cache()
    out=dict(complete=True,gpu=torch.cuda.get_device_name(),timings=timings,
             excludes='Mapping, development/test audits, analysis and verified Drive saves; three disposable full-batch updates are not a full-run ETA')
    ctx['store'].save('study_preflight',out);ctx['store'].write('study_preflight.json',out)
    return out


def export(ctx,source_zip,complete=False):
    s=ctx['store'];s.write('completion_status.json',dict(complete=complete,configuration_selected=s.load('configuration_selected') is not None,
        cap_selected=s.load('cap_selected') is not None,models_frozen=s.load('all_models_frozen') is not None,
        test_complete=s.load('test_complete') is not None,analysis_complete=s.load('analysis_complete') is not None))
    out=s.output/f'agf_scale_study_{s.run_id[:12]}_results.zip';hashes={}
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(s.results.rglob('*')):
            if p.is_file() and 'accumulators' not in p.relative_to(s.results).parts:
                name='results/'+p.relative_to(s.results).as_posix();z.write(p,name);hashes[name]=sha_file(p)
        z.write(source_zip,'source.zip');hashes['source.zip']=sha_file(source_zip)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        import hashlib
        if any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in hashes.items()):raise ValueError('ZIP checksum failed')
    return out


def flush(ctx):
    key=ctx.get('active_key')
    if key:
        st=ctx['store'].load(key)
        if st is not None:
            save=getattr(ctx['store'],'_width_drive_recovery_original_save',ctx['store'].save)
            save(key,st,drive=True);print('Verified last complete checkpoint on Drive:',key,st['updates'],'/',st['target'],flush=True)
