import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation.inputs import JournalStore
from coco_ggf.width_balance import settings as old_settings
from scale_models import model,PaperGrassmannian,PaperGroupLasso,PaperVanilla
from scale_plan import settings,select,SCREEN_SEEDS,FINAL_SEEDS,METHODS
from scale_fit import fit
from scale_metrics import evaluate,train_targets,ReadoutView
from scale_analysis import ranks,gaussian_proxy,coding_proxy
from scale_evaluate import seed_interval


def context(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(9);x=rng.normal(size=(6,196,4)).astype(np.float32)
    masks=[]
    for i in range(len(x)):
        frac=np.zeros((196,2),np.uint8);frac[:80,0]=255;frac[85:165,1]=255
        masks.append(dict(fraction=frac,crowd=np.zeros(196,np.float32),instances=[
            dict(category=c,annotation_id=100*i+c,fraction=(frac[:,c]/255).astype(np.float32)) for c in [0,1]]))
    cfg=dict(category_indices=[0,1],negative_fraction=.1,positive_fraction=.5,min_object_patch_mass=2,
        min_background_patches=4,decision_min_images=1,target_min_images=1,patch_sample_seed=61002,
        detection_patches=32,preview_images=0,audit_image_batch=1,batch=128,score_ap_weight=.2)
    p=dict(x=x,ids=list(range(1,len(x)+1)),masks=masks,identity=identity(dict(ids=list(range(1,len(x)+1)))))
    rule=settings();rule.update(width=2,images_per_update=2,evaluate_every=2,local_save_every=1)
    s=JournalStore(tmp_path/'drive',dict(test=True),tmp_path/'local',False)
    return dict(store=s,local=tmp_path/'local',rule=rule,packs=dict(train=p,select=p,confirm=p),cfg=cfg,device='cpu',
                balance_recipe=dict(old_settings()['balance'],reference_batches=1,every=1),
                provenance=dict(categories=[dict(name='a'),dict(name='b')]),coding_reference_variance=1.)


def targets():return [dict(category_index=c,category=chr(97+c),contrast_group=c,detector_group=c,heat_mean=0.,heat_scale=1.) for c in [0,1]]


@pytest.mark.parametrize('method',METHODS)
def test_all_models_forward_backward(method):
    torch.manual_seed(2);m=model(method,6,2,2,4);x=torch.randn(12,4)
    m.train();loss,_=m.loss(x);loss.backward();torch.optim.Adam(m.parameters(),lr=.001).step();m.normalize_decoder()
    assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in m.parameters())
    m.eval();y,z=m(x);assert y.shape==x.shape and z.shape==(12,6,2)
    if not method.endswith('group_lasso'):assert (z.ne(0).any(-1).sum(-1)<=2).all()


def test_paper_soft_shrinkage_and_release_difference():
    m=PaperGroupLasso(4,6,2,2);m.eval();m.threshold_initialized.fill_(True)
    m.raw_theta.data.fill_(0);x=torch.randn(20,4);a=m.preact(x);z=m.encode(x)
    assert torch.allclose(z.norm(dim=-1),(a.norm(dim=-1)-np.log(2)).clamp_min(0),atol=1e-6)
    n=model('released_group_lasso',6,2,2,4);assert type(n).__module__=='bsf.group_lasso'


def test_paper_constraints_and_shared_gain():
    m=PaperVanilla(4,6,2,2);m.W_dec.data.mul_(10);m.normalize_decoder()
    assert (m.W_dec.reshape(6,2,4).norm(dim=(1,2))<=1.000001).all()
    g=PaperGrassmannian(4,6,2,2);assert g.log_gamma.numel()==1
    g.W_dec.data.add_(.1)
    for _ in range(20):g.normalize_decoder()
    w=g.W_dec.reshape(6,2,4);assert torch.allclose(w@w.transpose(1,2),torch.eye(2).expand(6,2,2),atol=1e-6)


def test_auxiliary_uses_unselected_blocks():
    m=PaperVanilla(4,6,2,2);x=torch.randn(5,4);z=m.encode(x)
    m.auxiliary(x,x,z).backward();assert torch.isfinite(m.W_enc.grad).all()


@pytest.mark.parametrize('supervised',[False,True])
def test_real_fit_resume_exact(tmp_path,supervised):
    ctx=context(tmp_path/'a');torch.manual_seed(8);w=training.weights(model('released_vanilla',6,2,2,4));t=targets() if supervised else None
    a=fit(ctx,'fit','released_vanilla',6,2,21,w,4,2,t,stream=1)
    assert a['updates']==2
    b=fit(ctx,'fit','released_vanilla',6,2,21,w,4,4,t,stream=1)
    full=context(tmp_path/'b');c=fit(full,'fit','released_vanilla',6,2,21,w,4,4,t,stream=1)
    assert training.weight_sha(b['weights'])==training.weight_sha(c['weights'])
    assert b['sampled_batches_sha']==c['sampled_batches_sha'] and b['sampled_noise_sha']==c['sampled_noise_sha']
    if supervised:
        assert torch.equal(b['head'],c['head'])
        assert b['balance']['adaptations']==c['balance']['adaptations']


def test_matched_stream_independent_of_supervision(tmp_path):
    ctx=context(tmp_path/'a');torch.manual_seed(8);w=training.weights(model('released_vanilla',6,2,2,4))
    a=fit(ctx,'agf','released_vanilla',6,2,21,w,2,2,targets(),stream=1)
    b=fit(ctx,'vanilla','released_vanilla',6,2,21,w,2,2,stream=1)
    assert a['sampled_batches_sha']==b['sampled_batches_sha'] and a['sampled_noise_sha']==b['sampled_noise_sha']


def test_resume_rejects_changed_cap(tmp_path):
    ctx=context(tmp_path);w=training.weights(model('released_vanilla',6,2,2,4))
    fit(ctx,'fit','released_vanilla',6,2,21,w,2,1,targets(),cap=.4)
    with pytest.raises(ValueError,match='identity'):fit(ctx,'fit','released_vanilla',6,2,21,w,2,2,targets(),cap=1.)


def test_partial_optimizer_update_not_committed(tmp_path,monkeypatch):
    ctx=context(tmp_path);w=training.weights(model('released_vanilla',6,2,2,4))
    old=torch.optim.Adam.step
    def bad(self,*a,**k):old(self,*a,**k);raise KeyboardInterrupt()
    monkeypatch.setattr(torch.optim.Adam,'step',bad)
    with pytest.raises(KeyboardInterrupt):fit(ctx,'fit','released_vanilla',6,2,21,w,2,2)
    saved=ctx['store'].load('fit');assert saved['updates']==0 and training.weight_sha(saved['weights'])==training.weight_sha(w)


def test_subset_evaluation_exact(tmp_path):
    ctx=context(tmp_path);m=model('released_vanilla',6,2,2,4)
    from coco_adaptation import core
    a=core.validation(m,ctx['packs']['select'],targets(),ctx['cfg'],'cpu')
    b=evaluate(m,ctx['packs']['select'],targets(),ctx['cfg'],'cpu',compact=True)
    for k in ['macro_capture','macro_ap']:assert a[k]==pytest.approx(b[k],abs=1e-8)
    assert b['reconstruction']['mse']==pytest.approx(a['reconstruction']['mse'],abs=1e-7)
    assert b['_compact']['category_indices']==[0,1]


def test_disk_mapping_exact_original_policy(tmp_path):
    ctx=context(tmp_path);m=model('released_vanilla',6,2,2,4)
    from coco_adaptation import core
    from coco_baselines import metrics
    p=ctx['packs']['train'];a=metrics.audit_arrays(m,p['x'],p['masks'],p['ids'],ctx['cfg'],'cpu')
    old=core.choose_targets(a,ctx['cfg'],ctx['provenance']['categories']);new=train_targets(ctx,m,'targets')
    assert [(t['contrast_group'],t['detector_group']) for t in old]==[(t['contrast_group'],t['detector_group']) for t in new]
    for a,b in zip(old,new):assert a['training_capture']==pytest.approx(b['training_capture'],abs=1e-6)


def test_development_selector_preserves_AP_and_prefers_cost():
    r=settings();base=dict(id='ref',seeds=SCREEN_SEEDS,cost=1,capture=.4,ap=.8,ve=.7)
    bad=dict(base,id='bad',cost=2,capture=.5,ap=.7)
    near=dict(base,id='near',cost=2,capture=.4005)
    assert select([base,bad,near],'ref',r)['chosen']=='ref'
    good=dict(base,id='good',cost=2,capture=.42)
    assert select([base,good],'ref',r)['chosen']=='good'


def test_selector_rejects_missing_seed():
    r=dict(id='ref',seeds=[21],cost=1,capture=.4,ap=.8,ve=.7)
    with pytest.raises(ValueError):select([r],'ref',settings())


@pytest.mark.parametrize('method',['released_vanilla','released_group_lasso','paper_group_lasso'])
def test_rank_inference_and_resume_for_variable_activity(tmp_path,method):
    m=model(method,6,2,2,4);m.train();m.loss(torch.randn(20,4));m.eval();x=np.random.default_rng(3).normal(size=(512,4)).astype(np.float32)
    a=ranks(m,x,tmp_path/'rank','abc','cpu',not method.endswith('group_lasso'))
    b=ranks(m,x,tmp_path/'rank','abc','cpu',not method.endswith('group_lasso'))
    assert a.patches==len(x) and np.array_equal(a.n,b.n)


def test_gaussian_proxy_charges_capacity_and_distortion():
    spec=np.ones((3,2));p=np.full(3,.5);res=np.ones(4)
    a=gaussian_proxy(spec,p,res,.1,2,1000,100,100)
    b=gaussian_proxy(spec,p,res,.1,2,2000,100,100)
    assert b['total_bits_per_patch']-a['total_bits_per_patch']==pytest.approx(10)
    c=gaussian_proxy(spec,p,res,.2,2,1000,100,100);assert c['total_bits_per_patch']<a['total_bits_per_patch']


def test_real_coding_accumulation_reuses_completed_result(tmp_path):
    ctx=context(tmp_path);m=model('released_vanilla',6,2,2,4)
    row=dict(key='toy',kind='agf_g',seed=21,weights_sha=training.weight_sha(training.weights(m)),head_parameters=4)
    a=coding_proxy(ctx,m,row,tmp_path);b=coding_proxy(ctx,m,row,tmp_path)
    assert a==b and len(a['records'])==4
    assert all(np.isfinite(r['total_bits_per_patch']) for r in a['records'])


def test_seed_uncertainty_separate_from_image_uncertainty():
    r=seed_interval([1,2,3,4,5,6,7]);assert r['mean']==4 and r['seeds']==7 and r['ci95'][0]<4<r['ci95'][1]


def test_seven_seed_catalog_and_freeze_barrier(tmp_path,monkeypatch):
    import scale_evaluate as e
    ctx=context(tmp_path);ctx['rule']['methods']=['released_vanilla']
    ctx['store'].save('all_models_frozen',dict(complete=True,models=[]))
    with pytest.raises(ValueError,match='seven'):e.verify_freeze(ctx)


def test_fixed_budget_and_baseline_inventory():
    r=settings();assert len(FINAL_SEEDS)==7 and len(r['methods'])==6
    assert r['warm_updates']+r['final_guided_updates']==19640 and r['screen_updates']<r['final_guided_updates']
    assert len(r['configurations'])==4 and r['caps']==[.4,.6,1.]


def test_real_staged_scheduler_reuses_prefixes_and_freezes_seven_seeds(tmp_path,monkeypatch):
    import scale_run as run
    import scale_evaluate as ev
    ctx=context(tmp_path);ctx['rule'].update(warm_updates=2,screen_updates=2,final_guided_updates=4,
        baseline_learning_rates=[.0003],paper_lasso_coefficients=[.001],methods=['released_vanilla','paper_vanilla'])
    monkeypatch.setattr(run,'GRID',[(6,2),(8,2)])
    D=torch.linalg.qr(torch.randn(4,2)).Q.T.repeat(2,1)
    ctx['store'].save('mask_dictionary12',dict(D=D,sha=training.weight_sha({'D':D})))
    ctx['stamps']=[];ctx['checkpoint_stamps']=[]
    monkeypatch.setattr(run.matched,'assert_sources',lambda c:None)
    monkeypatch.setattr(ev.matched,'assert_sources',lambda c:None)
    def no_exchange(c,row,reference):
        state=c['store'].load(row['key']);key=row['key']+'_exchanged';c['store'].save(key,state)
        return dict(row,key=key,kind='agf_x',exchange_decisions=dict(accepted_proposals=[]),new_gradient_updates=0)
    monkeypatch.setattr(run,'exchanges',no_exchange)
    configuration=run.factorial(ctx);cap=run.caps(ctx,configuration)
    tuning=run.tune_baselines(ctx,configuration['groups'],configuration['active_groups'])
    frozen=run.final_suite(ctx,configuration,cap,tuning)
    assert len(frozen['models'])==42
    assert all(r['total_training_updates']==6 for r in frozen['models'])
    assert len(ev.verify_freeze(ctx)['models'])==42
    def no_fit(*a,**k):raise AssertionError('Completed phases must not fit again')
    monkeypatch.setattr(run,'fit',no_fit)
    assert run.factorial(ctx)['chosen']==configuration['chosen']
    assert run.caps(ctx,configuration)['cap']==cap['cap']
    assert run.final_suite(ctx,configuration,cap,tuning)==frozen


def test_baseline_resume_unaffected_by_selected_cap_and_lr_trials_share_warm(tmp_path):
    import scale_run as r
    ctx=context(tmp_path);ctx['rule'].update(warm_updates=2,screen_updates=2,final_guided_updates=4)
    a=r.run_branch(ctx,6,2,21,'released_vanilla',cap=.4,lr=.0003)
    b=r.run_branch(ctx,6,2,21,'released_vanilla',cap=1.,lr=.0003)
    assert a['weights_sha']==b['weights_sha'] and a['key']==b['key']
    r.run_branch(ctx,6,2,21,'released_vanilla',cap=1.,lr=.0001)
    assert len(list(ctx['store'].root.glob('*_warm')))==1


def test_lasso_penalty_tuning_changes_only_additional_phase(tmp_path):
    import scale_run as r
    ctx=context(tmp_path);ctx['rule'].update(warm_updates=2,screen_updates=2,final_guided_updates=4)
    a=r.run_branch(ctx,6,2,21,'paper_group_lasso',coefficient=.0001)
    b=r.run_branch(ctx,6,2,21,'paper_group_lasso',coefficient=.01)
    assert ctx['store'].load(a['key'])['weights']['coefficient'].item()==pytest.approx(.0001)
    assert ctx['store'].load(b['key'])['weights']['coefficient'].item()==pytest.approx(.01)
    assert len(list(ctx['store'].root.glob('*_warm')))==1


def test_actual_seven_seed_evaluation_and_reuse(tmp_path,monkeypatch):
    import scale_evaluate as e
    ctx=context(tmp_path);ctx['rule']['methods']=['released_vanilla'];ctx['rule']['fresh']=False
    ctx['cfg'].update(bootstrap=8,fresh_test_seed=12);ctx['norm']={'scale':1.,'mean':np.zeros(4)}
    ctx['stamps']=[];ctx['checkpoint_stamps']=[];monkeypatch.setattr(e.matched,'assert_sources',lambda c:None)
    rows=[]
    for seed in FINAL_SEEDS:
        torch.manual_seed(seed);w=training.weights(model('released_vanilla',6,2,2,4))
        for kind in ['agf_g','agf_x','mask_only','fixed_guided','released_vanilla']:
            key=f'{kind}_{seed}';ctx['store'].save(key,dict(weights=w))
            rows.append(dict(key=key,seed=seed,kind=kind,method='released_vanilla',groups=6,rank=2,active_groups=2,coefficient=.001,
                targets=targets(),targets_sha=identity(targets()),weights_sha=training.weight_sha(w),head_parameters=4 if kind.startswith('agf') else 0,
                cap=.4,total_training_updates=6))
    ctx['store'].save('all_models_frozen',dict(complete=True,models=rows))
    report=e.run(ctx,ctx['packs']['select'])
    assert report['complete'] and len(report['models'])==35 and len(report['paired_comparisons'])==28
    assert all(abs(r['capture_gain_pp'])<1e-10 and abs(r['AP_gain_pp'])<1e-10 for r in report['paired_comparisons'])
    monkeypatch.setattr(e,'evaluate',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Must reuse audits')))
    assert e.run(ctx,ctx['packs']['select'])==report


@pytest.mark.parametrize('eligible',[False,True])
def test_exchange_wrapper_preserves_heads_inventory_and_source_state(tmp_path,monkeypatch,eligible):
    import scale_run as r
    ctx=context(tmp_path);ctx['base']=dict(groups=6,block_rank=2,active_groups=2)
    w=training.weights(model('released_vanilla',6,2,2,4));head=torch.zeros(2,2)
    state=dict(weights=w,head=head,optimizer={},balance={},updates=4,target=4)
    ctx['store'].save('parent',state);ctx['rule']['final_guided_updates']=4
    row=dict(key='parent',kind='agf_g',method='released_vanilla',seed=21,groups=6,rank=2,active_groups=2,
        targets=targets(),targets_sha=identity(targets()),weights_sha=training.weight_sha(w),additional_updates=4,total_training_updates=6)
    dev=evaluate(model('released_vanilla',6,2,2,4),ctx['packs']['select'],targets(),ctx['cfg'],'cpu')
    # Recompute with the actual parent's weights.
    m=model('released_vanilla',6,2,2,4);m.load_state_dict(w);dev=evaluate(m,ctx['packs']['select'],targets(),ctx['cfg'],'cpu')
    monkeypatch.setattr(r,'development',lambda c,x:dict(report=dev))
    monkeypatch.setattr(r.exchange.coverage,'assignment_indices',lambda p,*a:(np.arange(len(p['ids'])),dict(ids=p['ids'])))
    monkeypatch.setattr(r.ops,'candidate_pool',lambda *a:[])
    monkeypatch.setattr(r.exchange.cache.bound,'validate_metrics',lambda *a:None)
    ref=dev['reconstruction']['mse']*(2 if eligible else .1)
    out=r.exchanges(ctx,row,ref)
    assert out['weights_sha']==row['weights_sha'] and out['new_gradient_updates']==0
    assert torch.equal(ctx['store'].load(out['key'])['head'],head)
    assert training.weight_sha(ctx['store'].load('parent')['weights'])==row['weights_sha']


def test_fresh_holdout_excludes_all_prior_splits_and_reuses_plan(tmp_path,monkeypatch):
    import scale_data as d
    ctx=context(tmp_path);ctx['rule']['final_holdout_images']=24
    ctx['provenance']['splits']=dict(train=list(range(1,7)),select=list(range(7,13)),confirm=list(range(13,19)),mapping=list(range(19,25)),test=list(range(25,29)))
    class Source:
        def load(self,key):return dict(splits={'test':[29,30]})
    ctx['sources']={'strong':Source()}
    class Coco:
        cats={0:{},1:{}}
        imgs={i:dict(width=224,height=224) for i in range(1,81)}
    monkeypatch.setattr(d.dataset,'candidate_ids',lambda *a:list(range(1,81)))
    monkeypatch.setattr(d.data,'masks_for_image',lambda *a:ctx['packs']['train']['masks'][0])
    p=d.holdout_plan(ctx,Coco())
    assert len(p['splits']['test'])==24 and min(p['splits']['test'])>30
    assert all(x>=10 for x in p['coverage']['split_counts']['test'])
    monkeypatch.setattr(d.dataset,'candidate_ids',lambda *a:(_ for _ in ()).throw(AssertionError('Must reuse reserved plan')))
    assert d.holdout_plan(ctx,Coco())==p


def test_final_holdout_guard_before_extraction(tmp_path):
    from scale_data import load_holdout
    ctx=context(tmp_path)
    with pytest.raises(ValueError,match='Freeze'):load_holdout(ctx,None)
