import copy,inspect,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from coco_adaptation import core
from coco_baselines import training
from coco_ggf import next_fit as engine,next_experiments as exp,pcgrad_fit as old_engine
from test_width_balance import fixture,cheap_validation,store
from coco_ggf.next_warm import fit_warm


def test_decoder_projection_preserves_encoder_and_private_heads():
    def run(both):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));d=torch.nn.Parameter(torch.tensor([3.,4.]));h=torch.nn.Parameter(torch.zeros(1))
        losses=[p[0]+d[0],-p[0]-d[0],p[1]+2*h.sum()]
        if both:engine.pcgrad_both_backward(losses,torch.ones(3),[p,d],h,np.random.default_rng(5),np.random.default_rng(9))
        else:old_engine.pcgrad_backward(losses,torch.ones(3),[p,d],h,np.random.default_rng(5))
        return p.grad,d.grad,h.grad
    a=run(False);b=run(True)
    assert torch.equal(a[0],b[0]) and torch.equal(a[2],b[2])
    assert torch.equal(b[1],torch.zeros(2))
    p=torch.nn.Parameter(torch.ones(2));d=torch.nn.Parameter(torch.ones(2));h=torch.nn.Parameter(torch.zeros(1))
    losses=[p.sum()+d[0],p.sum()-d[0]+d[1],p.sum()+h.sum()]
    engine.pcgrad_both_backward(losses,torch.ones(3),[p,d],h,np.random.default_rng(5),np.random.default_rng(6))
    assert torch.allclose(d.grad,torch.tensor([.5,1.5])) and h.grad.item()==1


def test_aligned_projection_equals_ordinary_sum():
    p=torch.nn.Parameter(torch.ones(2));d=torch.nn.Parameter(torch.ones(2));h=torch.nn.Parameter(torch.zeros(1))
    losses=[p.sum()+d.sum(),2*p.sum()+3*d.sum(),4*p.sum()+h.sum()];w=torch.tensor([1.,.4,.4])
    expected=torch.autograd.grad(sum(a*b for a,b in zip(w,losses)),[p,d,h],retain_graph=True)
    engine.pcgrad_both_backward(losses,w,[p,d],h,np.random.default_rng(4),np.random.default_rng(5))
    for param,grad in zip([p,d,h],expected):assert torch.allclose(param.grad,grad)


@pytest.mark.parametrize('rank,supervised,both',[(4,True,False),(4,True,True),(4,False,False),(5,True,False),(5,False,False)])
def test_all_new_arms_resume_model_adam_controller_rngs_exactly(tmp_path,monkeypatch,rank,supervised,both):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'a',rank);b,_=fixture(tmp_path,'b',rank)
    for c in [a,b]:c['recipe']=exp.settings();c['cfg']['evaluate_every']=5
    initial=training.weights(training.make_model('vanilla',a['base'],8))
    start=engine.fit(a,'root',initial,t,10,44,supervised,supervised,1.)
    kwargs=dict(reference_mse=1.,initial_head=start['head'] if supervised else None,
        initial_balance=start['balance'] if supervised else None,initial_optimizer=start['optimizer'],pcgrad=both,decoder_pcgrad=both)
    before=engine.state_identity(start)
    full=engine.fit(a,'trial',start['weights'],t,20,93021,supervised,supervised,**kwargs)
    engine.fit(b,'trial',start['weights'],t,20,93021,supervised,supervised,max_updates=13,**kwargs)
    resumed=engine.fit(b,'trial',start['weights'],t,20,93021,supervised,supervised,**kwargs)
    for field in ['weights','head','optimizer','balance','sample_rng','projection_rng','decoder_rng','torch_rng','sampled_batches_sha','sampled_noise_sha']:
        assert engine.state_identity(full[field])==engine.state_identity(resumed[field])
    assert before==engine.state_identity(start)
    if supervised:
        with pytest.raises(ValueError,match='identity changed'):
            engine.fit(b,'trial',start['weights'],t,20,93021,supervised,supervised,**dict(kwargs,pcgrad=True,decoder_pcgrad=not both))


def test_ordinary_new_engine_matches_saved_engine(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'new',4);b,_=fixture(tmp_path,'old',4)
    for c in [a,b]:c['recipe']=exp.settings()
    w=training.weights(training.make_model('vanilla',a['base'],8))
    aa=engine.fit(a,'fit',w,t,20,93021,True,True,1.)
    bb=old_engine.fit(b,'fit',w,t,20,93021,True,True,1.)
    for field in ['weights','head','optimizer','balance','sampled_batches_sha','sampled_noise_sha']:
        assert engine.state_identity(aa[field])==engine.state_identity(bb[field])


def test_decoder_projection_cannot_change_image_and_noise_stream(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'ordinary',4);b,_=fixture(tmp_path,'both',4)
    for c in [a,b]:c['recipe']=exp.settings()
    w=training.weights(training.make_model('vanilla',a['base'],8))
    ordinary=engine.fit(a,'fit',w,t,20,93021,True,True,1.)
    both=engine.fit(b,'fit',w,t,20,93021,True,True,1.,pcgrad=True,decoder_pcgrad=True)
    for field in ['sampled_batches_sha','sampled_noise_sha']:assert ordinary[field]==both[field]
    assert training.weight_sha(ordinary['weights'])!=training.weight_sha(both['weights'])


def test_fifth_axis_preserves_four_and_rejects_invalid_energy():
    Q,_=np.linalg.qr(np.random.default_rng(7).normal(size=(8,8)))
    gram=Q@np.diag([8,7,6,5,4,3,2,1])@Q.T;saved=Q[:,:4].T
    d,e,a=exp.orthogonal_fifth(gram,saved)
    assert np.linalg.norm(saved@d)<1e-8 and abs(abs(d@Q[:,4])-1)<1e-8
    with pytest.raises(ValueError,match='disagrees'):exp.orthogonal_fifth(gram,Q[:,4:].T)
    with pytest.raises(ValueError,match='no measured'):exp.orthogonal_fifth(np.diag([8,7,6,5,0,0,0,0]),np.eye(8)[:4])


def test_width5_warmup_resume_and_matching_width4_permutations(tmp_path):
    a,t=fixture(tmp_path,'warm_full',5);b,_=fixture(tmp_path,'warm_resume',5);c,_=fixture(tmp_path,'warm4',4)
    x=a['packs']['train']['x'].reshape(-1,8)[:32];val=a['packs']['select']['x'].reshape(-1,8)[:32]
    for ctx in [a,b,c]:ctx['base'].update(epochs=2,save_updates=4,local_save_updates=1)
    paired=exp.base.width.paired_warm_rng(dict(a,base=dict(a['base'],block_rank=3)),21)
    initial=training.weights(training.make_model('vanilla',a['base'],8))
    kw=dict(initial=initial,pair_rng=paired,drive_every=2)
    full=fit_warm(a['store'],'warm','vanilla',x,val,a['base'],21,.001,'cpu',**kw)
    part=fit_warm(b['store'],'warm','vanilla',x,val,b['base'],21,.001,'cpu',max_updates=5,**kw)
    assert part['updates']==5
    resumed=fit_warm(b['store'],'warm','vanilla',x,val,b['base'],21,.001,'cpu',**kw)
    for field in ['weights','best','opt','scheduler','rng','sampled_batches_sha','sampled_noise_sha','best_update']:
        assert engine.state_identity(full[field])==engine.state_identity(resumed[field])
    w4=training.weights(training.make_model('vanilla',c['base'],8))
    control=fit_warm(c['store'],'warm','vanilla',x,val,c['base'],21,.001,'cpu',initial=w4,pair_rng=paired)
    assert full['sampled_batches_sha']==control['sampled_batches_sha']


def test_category_slots_are_distinct_from_coco_indices():
    source=inspect.getsource(exp.dictionary5)
    assert "for slot,ci in enumerate(ctx['cfg']['category_indices'])" in source
    assert 'D4[ci]' not in source and "m['fraction'][:,ci]" in source


def test_priority_and_controls_are_fixed():
    assert exp.PRIORITY==['width5','pcgrad_both','longer']
    assert exp.SEEDS==[21,22,23]
    assert exp.settings()['balance']['bounds']==dict(capture=[.0125,.4],detection=[.0025,.4])
    assert sum(3*3000*len(v) for v in exp.STAGE_ARMS.values())+6*7640==90840
    assert exp.STAGE_ARMS['pcgrad_both']==['pcgrad_both']
    assert 'ggf_continue' in exp.STAGE_CONTROLS['pcgrad_both'] and 'vanilla_continue' in exp.STAGE_CONTROLS['pcgrad_both']


def test_incomplete_stage_cannot_open_test_or_skip_priority(tmp_path,monkeypatch):
    ctx=dict(store=store(tmp_path,'barrier'),packs={})
    monkeypatch.setattr(exp.base,'load_test',lambda *a:pytest.fail('Test must remain closed'))
    with pytest.raises(FileNotFoundError):exp.run_stage(ctx,'pcgrad_both')
    ctx['store'].save('frozen_width5',dict(complete=True,models=[]))
    with pytest.raises(ValueError,match='Freeze every'):exp.run_stage(ctx,'width5')


def test_freeze_checks_controller_as_well_as_model(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'freeze',4);ctx['recipe']=exp.settings()
    monkeypatch.setattr(core,'validation',cheap_validation)
    w=training.weights(training.make_model('vanilla',ctx['base'],8))
    st=engine.fit(ctx,'fit',w,t,4,1,True,True,1.)
    monkeypatch.setattr(exp.matched,'valid_weights',lambda *a:None)
    rows=[exp.endpoint_row(ctx,s,'pcgrad_both',4,st,t,1.,key='fit',checkpoint_updates=4) for s in exp.SEEDS]
    ctx['store'].save('frozen_pcgrad_both',dict(complete=True,models=rows))
    exp.verify_freeze(ctx,'pcgrad_both')
    st['balance']['adaptations']+=1;ctx['store'].save('fit',st)
    with pytest.raises(ValueError,match='Frozen checkpoint'):exp.verify_freeze(ctx,'pcgrad_both')


def test_real_three_stage_inference_reuses_completed_controls_and_endpoints(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'pipeline',4);ctx['recipe']=exp.settings()
    ctx['cfg'].update(audit_image_batch=2,patch_sample_seed=4,detection_patches=196,preview_images=0,
        bootstrap=5,fresh_test_seed=77,evaluate_every=2)
    ctx.update(roots=[],test_stamp=dict(normalized_sha='synthetic'),test_ids=ctx['packs']['train']['ids'])
    monkeypatch.setattr(exp.matched,'valid_weights',lambda *a:None)
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    states={};original_validation=core.validation
    monkeypatch.setattr(core,'validation',cheap_validation)
    for stage in exp.PRIORITY:
        rows=[]
        for arm in exp.STAGE_ARMS[stage]:
            rank=5 if stage=='width5' else 4;sub=exp.subcontext(ctx,rank)
            initial=training.weights(training.make_model('vanilla',sub['base'],8));sup=arm not in ['vanilla5','longer_vanilla']
            st=engine.fit(sub,arm,initial,t,4,21,sup,sup,10.);states[arm]=st
            for seed in exp.SEEDS:
                rows.append(exp.endpoint_row(ctx,seed,arm,rank,st,t,10.,key=arm,checkpoint_updates=4,total_training_updates=10640))
        ctx['store'].save('frozen_'+stage,dict(complete=True,models=rows))
    monkeypatch.setattr(core,'validation',original_validation)
    for arm in ['cap040','vanilla4','ggf_continue','ggf_pcgrad','vanilla_continue']:
        st=states['longer_guided'];model=core.new_model('vanilla',ctx['base'],st['weights'],'cpu')
        r=core.validation(model,ctx['packs']['train'],t,ctx['cfg'],'cpu',with_arrays=True)
        compact=exp.base.study.compact_audit(r.pop('_audit'),exp.matched.mapping(t),ctx['cfg'])
        for seed in exp.SEEDS:
            row=dict(seed=seed,arm=arm,rank=4,targets=t,targets_sha=exp.identity(t),weights_sha=training.weight_sha(st['weights']))
            ctx['roots'].append(row);report=dict(r,**{k:row[k] for k in ['seed','arm','rank','targets_sha','weights_sha']},legacy_native_eligible=True)
            tag=f's{seed}_{arm}';ctx['store'].save('audit_'+tag,dict(identity=exp.base.report_signature(row,ctx['test_stamp'],ctx['cfg']),report=report))
            ctx['store'].save('compact_'+tag,compact)
    def forbidden(*a,**k):raise AssertionError('No training or control rerun during cached evaluation')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    monkeypatch.setattr(exp,'run_width5',forbidden);monkeypatch.setattr(exp,'run_continuation',forbidden)
    monkeypatch.setattr(exp.base,'load_test',lambda c:c['packs'].update(test=c['packs']['train']))
    for stage in exp.PRIORITY:
        result=exp.run_stage(ctx,stage)
        assert result['complete'] and not result['controls_retrained'] and result['new_test_gradient_updates']==0
        assert len(result['comparisons'])==3*len(exp.PAIRS[stage])
        # Completed endpoints and cached audits are both reused on repeat.
        assert exp.identity(exp.run_stage(ctx,stage))==exp.identity(result)


def test_previous_source_lineage_and_optimizer_are_checked_before_new_work(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'source',4)
    cfg=dict(continuation_recipe=exp.previous.settings(),base=ctx['base'],fit_config=ctx['cfg'],versions={'synthetic':'1'},
        cap040_run='cap',parent_run='parent')
    ctx.update(roots=[],sources=dict(cap040=SimpleNamespace(run_id='cap')),parent=SimpleNamespace(run_id='parent'),
        test_ids=ctx['packs']['train']['ids'],test_stamp={'x':'synthetic'},checkpoint_stamps=[])
    monkeypatch.setattr(exp.matched,'versions',lambda:cfg['versions'])
    monkeypatch.setattr(exp.matched,'source_key_stamp',lambda s,k:dict(key=k))
    monkeypatch.setattr(exp.base,'check_report',lambda r,row,c:None)
    w=training.weights(training.make_model('vanilla',ctx['base'],8));head=torch.zeros(1,2);states={};rows=[];reports=[];saved={}
    for seed in exp.SEEDS:
        for arm in ['cap040','vanilla4']:
            ctx['roots'].append(dict(seed=seed,arm=arm,weights_sha='root_'+arm,targets_sha=exp.identity(t)))
        for arm in exp.previous.ARMS:
            sup=arm!='vanilla_continue';key=f'continuation_{arm}_s{seed}'
            st=dict(updates=3000,target=3000,weights=w,head=head,optimizer={'saved_step':3000},sampled_batches_sha='batches',sampled_noise_sha='noise')
            states[key]=st;root='cap040' if sup else 'vanilla4'
            row=dict(seed=seed,arm=arm,key=key,rank=4,total_training_updates=13640,supervised=sup,targets=t,targets_sha=exp.identity(t),
                weights_sha=training.weight_sha(w),head_sha=training.weight_sha({'head':head}) if sup else None,
                optimizer_sha=exp.identity(engine.state_identity(st['optimizer'])),starting_arm=root,starting_weights_sha='root_'+root,
                phase_batch_sha='batches',phase_noise_sha='noise')
            rows.append(row);report=dict(seed=seed,arm=arm,rank=4,weights_sha=row['weights_sha'],targets_sha=row['targets_sha'])
            reports.append(report);tag=f's{seed}_{arm}'
            saved['audit_'+tag]=dict(identity=exp.base.report_signature(row,ctx['test_stamp'],ctx['cfg']),report=report)
            saved['compact_'+tag]=dict(image_ids=ctx['test_ids'],category_indices=ctx['cfg']['category_indices'])
    saved.update(states);saved['continuation_models_frozen']=dict(complete=True,models=rows)
    saved['experiment_complete']=dict(complete=True,models=reports)
    source=SimpleNamespace(config=cfg,run_id=exp.PREVIOUS_RUN,load=lambda key:saved.get(key))
    exp.validate_previous(ctx,source)
    assert len([r for r in ctx['roots'] if r.get('source_name')=='previous'])==9
    states[rows[0]['key']]['optimizer']['saved_step']=2999
    with pytest.raises(ValueError,match='identity changed'):exp.validate_previous(ctx,source)
    states[rows[0]['key']]['optimizer']['saved_step']=3000
    states[rows[0]['key']]['sampled_noise_sha']='changed'
    with pytest.raises(ValueError,match='batch/noise hashes'):exp.validate_previous(ctx,source)
