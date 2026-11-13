"""Synthetic tests exercise real BSF gates, Adam surgery and continuation RNGs."""
import copy,json
import numpy as np
import pytest
import torch
from coco_adaptation import core,inputs
from coco_baselines import training
from coco_baselines.storage import identity
from coco_ggf import warm_capacity as exp,capacity_fit as engine


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):yield
    torch.set_num_threads(old)


def fixture(tmp_path,supervised=True,d=12):
    groups=32;base=dict(groups=groups,block_rank=5,active_groups=16)
    torch.manual_seed(37);m=training.make_model('vanilla',base,d)
    head=torch.nn.Parameter(torch.zeros(2,2));opt=torch.optim.Adam(list(m.parameters())+([head] if supervised else []),lr=.0003)
    x=torch.randn(16,d);rec=m.loss(x)[0];loss=rec+(.01*head.sum() if supervised else 0)
    loss.backward();opt.step();m.normalize_decoder()
    balance=engine.Balancer(exp.settings()['balance'],'cpu') if supervised else None
    if balance is not None:
        balance.reference=np.array([1.,1.,1.]);balance.reference_count=20;balance.reference_sum=20*balance.reference
    st=dict(weights=training.weights(m),optimizer=copy.deepcopy(opt.state_dict()),head=head.detach().clone(),
        balance=None if balance is None else balance.state_dict())
    targets=[dict(category_index=c,category=f'c{c}',contrast_group=c,detector_group=c,heat_mean=0.,heat_scale=1.) for c in [0,1]]
    cfg=dict(category_indices=[0,1],patch_sample_seed=61002,detection_patches=32,positive_fraction=.5,negative_fraction=.1,
        min_object_patch_mass=2.,min_background_patches=4,decision_min_images=1,batch=128,device='cpu',target_min_images=1,
        audit_image_batch=2,fresh_test_seed=99,bootstrap=10,score_ap_weight=.2,preview_images=0,
        images_per_update=2,lr=.0003,snr=.1,grad_clip=5.,evaluate_every=2,local_save_every=1)
    masks=[];rng=np.random.default_rng(17);xx=rng.normal(size=(4,196,d)).astype(np.float32)
    for i in range(4):
        fraction=np.zeros((196,2),np.uint8);fraction[:65,0]=255;fraction[70:140,1]=255
        masks.append(dict(fraction=fraction,crowd=np.zeros(196,np.float32),instances=[dict(category=c,annotation_id=2*i+c,fraction=fraction[:,c].astype(np.float32)/255.) for c in [0,1]]))
    pack=dict(x=xx,ids=[101,102,103,104],masks=masks)
    store=inputs.JournalStore(tmp_path/'output',dict(test='capacity'),tmp_path/'local',False)
    ctx=dict(base=base,cfg=cfg,device='cpu',recipe=exp.settings(),packs=dict(train=pack,select=pack,confirm=pack),store=store)
    return ctx,st,targets


@pytest.mark.parametrize('rank',[6,8])
def test_growth_preserves_gated_and_ungated_prediction(tmp_path,rank):
    ctx,st,_=fixture(tmp_path);sha=training.weight_sha(st['weights'])
    w=exp.widen_weights(st['weights'],32,rank,21)
    m=core.new_model('vanilla',ctx['base'],st['weights'],'cpu')
    n=core.new_model('vanilla',dict(ctx['base'],block_rank=rank),w,'cpu')
    x=torch.from_numpy(ctx['packs']['train']['x'][0])
    with torch.no_grad():
        a,z=m(x);b,zz=n(x)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5);torch.testing.assert_close(z,zz[:,:,:5])
        assert torch.count_nonzero(zz[:,:,5:])==0
        torch.testing.assert_close((x@m.W_enc+m.b_enc)@m.W_dec,(x@n.W_enc+n.b_enc)@n.W_dec,atol=1e-6,rtol=1e-5)
    dec=w['W_dec'].reshape(32,rank,12);old=st['weights']['W_dec'].reshape(32,5,12)
    assert torch.equal(dec[:,:5],old)
    torch.testing.assert_close(dec[:,5:].norm(dim=-1),torch.ones(32,rank-5))
    assert (dec[:,5:]@old.transpose(1,2)).abs().max()<2e-6
    assert training.weight_sha(st['weights'])==sha


def test_growth_nested_seeded_and_group_offsets(tmp_path):
    _,st,_=fixture(tmp_path);a=exp.widen_weights(st['weights'],32,6,22);b=exp.widen_weights(st['weights'],32,8,22)
    assert torch.equal(a['W_dec'].reshape(32,6,12),b['W_dec'].reshape(32,8,12)[:,:6])
    assert training.weight_sha(a)==training.weight_sha(exp.widen_weights(st['weights'],32,6,22))
    assert training.weight_sha(a)!=training.weight_sha(exp.widen_weights(st['weights'],32,6,23))
    for key,axis in [('W_enc',1),('W_dec',0),('b_enc',0)]:
        x=st['weights'][key];y=exp.expand_tensor(x,key,32,5,8)
        old=x.reshape(12,32,5) if axis==1 else x.reshape(32,5,*x.shape[1:])
        expanded=y.reshape(12,32,8) if axis==1 else y.reshape(32,8,*y.shape[1:])
        assert torch.equal(expanded[:,:,:5] if axis==1 else expanded[:,:5],old)


@pytest.mark.parametrize('supervised',[True,False])
@pytest.mark.parametrize('rank',[5,6,8])
def test_adam_moments_step_and_heads_preserved(tmp_path,supervised,rank):
    ctx,st,_=fixture(tmp_path,supervised);original=identity(engine.state_identity(st['optimizer']))
    opt=exp.widen_optimizer(st['optimizer'],st['weights'],32,rank,supervised)
    ids=opt['param_groups'][0]['params']
    for pid,key in zip(ids,exp.MODEL_ORDER):
        for moment in ['exp_avg','exp_avg_sq']:
            expected=exp.expand_tensor(st['optimizer']['state'][pid][moment],key,32,5,rank)
            assert torch.equal(opt['state'][pid][moment],expected)
        assert torch.equal(opt['state'][pid]['step'],st['optimizer']['state'][pid]['step'])
    if supervised:
        assert identity(engine.state_identity(opt['state'][ids[-1]]))==identity(engine.state_identity(st['optimizer']['state'][ids[-1]]))
    w=exp.widen_weights(st['weights'],32,rank,21);m=core.new_model('vanilla',dict(ctx['base'],block_rank=rank),w,'cpu')
    h=torch.nn.Parameter(st['head'].clone());loaded=torch.optim.Adam(list(m.parameters())+([h] if supervised else []),lr=.0003)
    loaded.load_state_dict(opt);m.loss(torch.randn(16,12))[0].backward();loaded.step()
    assert identity(engine.state_identity(st['optimizer']))==original


def test_rejects_wrong_adam_schema_and_width(tmp_path):
    _,st,_=fixture(tmp_path)
    bad=copy.deepcopy(st['optimizer']);pid=bad['param_groups'][0]['params'][0];bad['state'][pid]['exp_avg']=torch.zeros(3)
    with pytest.raises(ValueError,match='Adam shape'):exp.widen_optimizer(bad,st['weights'],32,8,True)
    with pytest.raises(ValueError):exp.widen_weights(st['weights'],32,4,21)
    with pytest.raises(ValueError):exp.arm_spec('agf_16d32')


def run_fit(ctx,st,targets,key,rank,active,supervised,maximum=None):
    w=exp.widen_weights(st['weights'],32,rank,21);opt=exp.widen_optimizer(st['optimizer'],st['weights'],32,rank,supervised)
    return engine.fit(exp.subcontext(ctx,rank,active),key,w,targets,4,97021,supervised,supervised,10.,
        initial_head=st['head'] if supervised else None,initial_balance=copy.deepcopy(st['balance']) if supervised else None,
        initial_optimizer=opt,max_updates=maximum)


@pytest.mark.parametrize('rank,active',[(6,16),(8,16),(5,24)])
def test_real_training_activates_new_coordinates_and_exact_resume(tmp_path,rank,active):
    ctx,st,targets=fixture(tmp_path);before=identity(engine.state_identity(st))
    whole=run_fit(ctx,st,targets,'whole',rank,active,True)
    interrupted=run_fit(ctx,st,targets,'resumed',rank,active,True,2)
    assert interrupted['updates']==2
    resumed=run_fit(ctx,st,targets,'resumed',rank,active,True)
    for field in ['weights','optimizer','balance','head','sample_rng','torch_rng']:
        assert identity(engine.state_identity(whole[field]))==identity(engine.state_identity(resumed[field]))
    assert whole['sampled_batches_sha']==resumed['sampled_batches_sha']
    assert whole['sampled_noise_sha']==resumed['sampled_noise_sha']
    assert identity(engine.state_identity(st))==before
    if rank>5:assert torch.count_nonzero(whole['weights']['W_enc'].reshape(12,32,rank)[:,:,5:])>0
    assert whole['balance']['log_weights'].exp().max()<=.4+1e-7


def test_all_capacity_arms_share_images_noise_and_have_matched_controls(tmp_path):
    runs=[]
    for arm in exp.ARMS:
        rank,active,sup=exp.arm_spec(arm);ctx,st,targets=fixture(tmp_path/arm,sup)
        out=run_fit(ctx,st,targets,arm,rank,active,sup);runs.append(out)
        assert out['updates']==4
        assert out['balance'] is not None if sup else out['balance'] is None
    assert len({r['sampled_batches_sha'] for r in runs})==1
    assert len({r['sampled_noise_sha'] for r in runs})==1
    for name in exp.CONFIGURATIONS:assert ('agf_'+name,'vanilla_'+name) in exp.evaluation_pairs()


def test_controller_updates_on_carried_tenth_step_and_clamps(tmp_path):
    ctx,st,targets=fixture(tmp_path);w=exp.widen_weights(st['weights'],32,8,21)
    opt=exp.widen_optimizer(st['optimizer'],st['weights'],32,8,True)
    out=engine.fit(exp.subcontext(ctx,8,16),'adapt',w,targets,10,97021,True,True,10.,
        initial_head=st['head'],initial_balance=copy.deepcopy(st['balance']),initial_optimizer=opt)
    assert out['balance']['adaptations']==st['balance']['adaptations']+1
    np.testing.assert_array_equal(out['balance']['reference'],st['balance']['reference'])
    assert torch.isfinite(out['balance']['log_weights']).all()
    assert out['balance']['log_weights'].exp().max()<=.4+1e-7


def test_partial_optimizer_failure_replays_complete_checkpoint(tmp_path,monkeypatch):
    ctx,st,targets=fixture(tmp_path);whole=run_fit(ctx,st,targets,'whole',8,16,True)
    original=torch.optim.Adam.step;calls=[0]
    def fail_after_partial(self,*args,**kw):
        result=original(self,*args,**kw);calls[0]+=1
        if calls[0]==2:raise RuntimeError('synthetic partial update')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam,'step',fail_after_partial)
        with pytest.raises(RuntimeError,match='synthetic partial'):run_fit(ctx,st,targets,'broken',8,16,True)
    assert ctx['store'].load('broken')['updates']==1
    resumed=run_fit(ctx,st,targets,'broken',8,16,True)
    for field in ['weights','optimizer','balance','head']:
        assert identity(engine.state_identity(whole[field]))==identity(engine.state_identity(resumed[field]))
    assert whole['sampled_noise_sha']==resumed['sampled_noise_sha']


def test_completed_fit_skips_gradient_updates_and_changed_gate_rejected(tmp_path,monkeypatch):
    ctx,st,targets=fixture(tmp_path);run_fit(ctx,st,targets,'done',5,24,True)
    def forbidden(*args,**kw):raise AssertionError('completed fit retrained')
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    out=run_fit(ctx,st,targets,'done',5,24,True);assert out['updates']==4
    with pytest.raises(ValueError,match='identity changed'):run_fit(ctx,st,targets,'done',5,16,True)


def test_audit_identity_contains_topk(tmp_path):
    ctx,_,_=fixture(tmp_path);ctx['test_stamp']={'ids':[1,2]}
    row=dict(weights_sha='same',targets_sha='same',rank=5,active_groups=16)
    assert exp.audit_signature(ctx,row)!=exp.audit_signature(ctx,dict(row,active_groups=24))


def test_test_barrier_rejects_incomplete_catalog_before_reading_features(tmp_path):
    ctx,_,_=fixture(tmp_path);ctx['store'].save('capacity_frozen',dict(complete=True,models=[]))
    with pytest.raises(ValueError,match='24 capacity'):exp.load_test(ctx)


def test_force_flush_newest_local_generation_on_graceful_stop(tmp_path):
    ctx,st,_=fixture(tmp_path);s=ctx['store'];s.save('active',dict(updates=0,target=3000,weights=st['weights']))
    s.save('active',dict(updates=51,target=3000,weights=st['weights']),drive=False);s._capacity_active_key='active'
    exp.flush_active_checkpoint(ctx)
    # Read the Drive-side files with a separate local rescue, as after reset.
    other=inputs.ReadStore(s.root,tmp_path/'fresh');assert other.load('active')['updates']==51


def test_protocol_budget_and_caps():
    recipe=exp.settings();assert recipe['new_planned_updates']==72000
    assert len(exp.ARMS)==8 and exp.SEEDS==[21,22,23]
    assert recipe['balance']['bounds']['capture']==[.0125,.4]
    assert recipe['balance']['bounds']['detection']==[.0025,.4]
    assert exp.CONFIGURATIONS['6d16']==(6,16) and exp.CONFIGURATIONS['8d16']==(8,16) and exp.CONFIGURATIONS['5d24']==(5,24)


def test_all_seed_freeze_endpoint_tables_and_resume(tmp_path,monkeypatch):
    """Run the actual 24-model scheduler at 768 features with tiny CPU packs."""
    ctx,_,targets=fixture(tmp_path/'main',d=768)
    ctx.update(stamps=[],checkpoint_stamps=[],roots=[])
    saved={}
    for seed in exp.SEEDS:
        for sup in [True,False]:
            _,state,_=fixture(tmp_path/f'source_{seed}_{sup}',sup,d=768)
            arm='guided5_continue' if sup else 'vanilla5_continue'
            row=dict(seed=seed,arm=arm,supervised=sup,weights_sha=training.weight_sha(state['weights']),
                optimizer_sha=identity(engine.state_identity(state['optimizer'])),balance_sha=identity(engine.state_identity(state['balance'])),
                targets=targets,targets_sha=identity(targets),reference_mse=10.,rank=5,total_training_updates=13640,
                core_parameters=sum(v.numel() for v in state['weights'].values()),maximum_active_coordinates=80)
            model=core.new_model('vanilla',ctx['base'],state['weights'],'cpu')
            with torch.no_grad():row['development']=core.validation(model,ctx['packs']['select'],targets,ctx['cfg'],'cpu')
            ctx['roots'].append(row);saved[(seed,arm)]=state
    monkeypatch.setattr(exp,'UPDATES',2);monkeypatch.setattr(exp,'TOTAL',13642)
    monkeypatch.setattr(exp,'source_state',lambda ctx,row:copy.deepcopy(saved[(row['seed'],row['arm'])]))
    frozen=exp.run(ctx)
    assert len(frozen['models'])==24 and all(r['checkpoint_updates']==2 for r in frozen['models'])
    assert all(r['rank']*r['active_groups']==r['maximum_active_coordinates'] for r in frozen['models'])
    exp.verify_freeze(ctx)
    rows=list(__import__('csv').DictReader((ctx['store'].results/'development_select_means.csv').open()))
    assert len(rows)==10
    def forbidden(*a,**kw):raise AssertionError('frozen run retrained')
    monkeypatch.setattr(exp,'fit_branch',forbidden)
    assert identity(exp.run(ctx))==identity(frozen)
    # Run the real inference/compact-audit/paired-interval/export-table path.
    parent=inputs.JournalStore(tmp_path/'parent',dict(fit_config=ctx['cfg']),tmp_path/'parent_local',False)
    ctx.update(sources=dict(continued5=parent),test_ids=ctx['packs']['train']['ids'],test_stamp=dict(ids=ctx['packs']['train']['ids']))
    for root in ctx['roots']:
        state=saved[(root['seed'],root['arm'])];model=core.new_model('vanilla',ctx['base'],state['weights'],'cpu')
        with torch.no_grad():r=core.validation(model,ctx['packs']['train'],targets,ctx['cfg'],'cpu',with_arrays=True)
        compact=exp.cache.study.compact_audit(r.pop('_audit'),exp.matched.mapping(targets),ctx['cfg'])
        r.update({k:root[k] for k in ['seed','arm','rank','core_parameters','maximum_active_coordinates','weights_sha','targets_sha','total_training_updates']})
        r.update(label=exp.label(root['arm']),origin='saved',head_parameters=4 if root['supervised'] else 0,
            legacy_native_eligible=True,development_reconstruction=root['development']['reconstruction'],common_native_MSE_limit=10.2)
        tag=f"s{root['seed']}_{root['arm']}"
        parent.save('audit_'+tag,dict(identity=exp.cache.report_signature(root,ctx['test_stamp'],ctx['cfg']),report=r))
        parent.save('compact_'+tag,compact)
    def load_synthetic_test(actual):
        exp.verify_freeze(actual);actual['packs']['test']=actual['packs']['train'];return actual['packs']['test']
    monkeypatch.setattr(exp,'load_test',load_synthetic_test)
    result=exp.evaluate(ctx)
    assert result['complete'] and len(result['models'])==30 and len(result['comparisons'])==36
    assert len(list(__import__('csv').DictReader((ctx['store'].results/'test_means.csv').open())))==10
    assert (ctx['store'].results/'figures/test_means.png').exists()
    monkeypatch.setattr(exp,'audit',forbidden)
    assert identity(exp.evaluate(ctx))==identity(result)
