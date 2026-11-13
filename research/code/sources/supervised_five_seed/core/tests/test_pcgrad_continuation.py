"""Synthetic CPU checks; full saved-data T4 experiment runs in Colab."""
import copy,csv
import numpy as np
import pytest
import torch
from coco_baselines import training
from coco_adaptation import core
from coco_ggf import pcgrad_fit as fit,pcgrad_continuation as exp,completed_caps_test as completed
from test_width_balance import fixture,cheap_validation,store
from test_caps_pipeline import metric

def test_pcgrad_conflict_projection_and_private_gradients():
    encoder=torch.nn.Parameter(torch.tensor([1.,2.]));decoder=torch.nn.Parameter(torch.tensor([3.]));head=torch.nn.Parameter(torch.tensor([0.]))
    losses=[encoder[0]+2*decoder.sum(),-encoder[0]+3*decoder.sum(),encoder[1]+4*head.sum()]
    fit.pcgrad_backward(losses,torch.ones(3),[encoder,decoder],head,np.random.default_rng(5))
    assert torch.equal(encoder.grad,torch.tensor([0.,1.]))
    assert torch.equal(decoder.grad,torch.tensor([5.])) and torch.equal(head.grad,torch.tensor([4.]))

def test_pcgrad_aligned_tasks_equal_weighted_sum():
    p=torch.nn.Parameter(torch.ones(2));head=torch.nn.Parameter(torch.zeros(1))
    losses=[p.sum(),2*p.sum(),3*p.sum()+head.sum()];weights=torch.tensor([1.,.4,.2])
    expected=torch.autograd.grad(sum(w*l for w,l in zip(weights,losses)),p,retain_graph=True)[0]
    fit.pcgrad_backward(losses,weights,[p],head,np.random.default_rng(2))
    assert torch.allclose(p.grad,expected)

@pytest.mark.parametrize('arm',exp.ARMS)
def test_carried_optimizer_and_controller_resume_exactly(tmp_path,monkeypatch,arm):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'a',4);b,_=fixture(tmp_path,'b',4)
    for c in [a,b]:c['recipe']=exp.settings();c['cfg']['evaluate_every']=5
    supervised=arm!='vanilla_continue';pcgrad=arm=='ggf_pcgrad'
    initial=training.weights(training.make_model('vanilla',a['base'],8))
    warm=fit.fit(a,'start',initial,t,10,44,supervised,supervised,1.)
    kwargs=dict(reference_mse=1.,initial_head=warm['head'] if supervised else None,
        initial_balance=warm['balance'] if supervised else None,initial_optimizer=warm['optimizer'],pcgrad=pcgrad)
    before=fit.state_identity(warm)
    full=fit.fit(a,'trial',warm['weights'],t,20,93021,supervised,supervised,**kwargs)
    part=fit.fit(b,'trial',warm['weights'],t,20,93021,supervised,supervised,max_updates=13,**kwargs)
    assert part['updates']==13
    resumed=fit.fit(b,'trial',warm['weights'],t,20,93021,supervised,supervised,**kwargs)
    for field in ['weights','head','optimizer','balance','sample_rng','projection_rng','torch_rng','sampled_batches_sha','sampled_noise_sha']:
        assert fit.state_identity(full[field])==fit.state_identity(resumed[field])
    assert before==fit.state_identity(warm)
    changed=dict(kwargs,pcgrad=not pcgrad) if supervised else dict(kwargs,initial_optimizer=None)
    with pytest.raises(ValueError,match='identity changed'):fit.fit(b,'trial',warm['weights'],t,20,93021,supervised,supervised,**changed)

def test_pcgrad_projection_rng_does_not_change_batch_or_noise_schedule(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'ordinary',4);b,_=fixture(tmp_path,'projected',4)
    for c in [a,b]:c['recipe']=exp.settings()
    initial=training.weights(training.make_model('vanilla',a['base'],8))
    ordinary=fit.fit(a,'trial',initial,t,20,93021,True,True,1.,pcgrad=False)
    projected=fit.fit(b,'trial',initial,t,20,93021,True,True,1.,pcgrad=True)
    assert ordinary['sampled_batches_sha']==projected['sampled_batches_sha']
    assert ordinary['sampled_noise_sha']==projected['sampled_noise_sha']
    assert training.weight_sha(ordinary['weights'])!=training.weight_sha(projected['weights'])

def test_incomplete_continuation_cannot_open_test(tmp_path,monkeypatch):
    ctx=dict(store=store(tmp_path,'barrier'))
    monkeypatch.setattr(exp.base,'load_test',lambda *a:pytest.fail('Test must stay closed'))
    with pytest.raises(FileNotFoundError):exp.load_test(ctx)
    ctx['store'].save('continuation_models_frozen',dict(complete=True,models=[]))
    with pytest.raises(ValueError,match='all nine'):exp.load_test(ctx)

def test_no_training_in_completed_cap_evaluation_module():
    import inspect
    source=inspect.getsource(completed)
    assert 'fit(' not in source and 'run_sweep(' not in source
    assert 'for cap in [.6]:' in source and "len(frozen['models'])!=6" in source


def test_three_arm_freeze_and_real_inference_keep_all_models(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from coco_ggf import cap_followup as base
    ctx,t=fixture(tmp_path,'pipeline',4)
    ctx['recipe']=exp.settings();ctx['cfg'].update(audit_image_batch=2,patch_sample_seed=4,detection_patches=196,
        preview_images=0,bootstrap=5,fresh_test_seed=77,evaluate_every=2)
    ctx.update(roots=[],cap040_rows=[],test_stamp=dict(normalized_sha='synthetic'),rows=[])
    initial=training.weights(training.make_model('vanilla',ctx['base'],8))
    original_fit=fit.fit;states={}
    for arm in ['cap040','vanilla4']:
        supervised=arm=='cap040'
        warm=original_fit(ctx,'start_'+arm,initial,t,2,55,supervised,supervised,10.)
        warm['updates']=warm['target']=500;states[arm]=warm
        for seed in base.SEEDS:
            row=dict(seed=seed,arm=arm,key=arm,source_name=arm,targets=t,targets_sha=base.identity(t),reference_mse=10.,
                weights_sha=training.weight_sha(warm['weights']),head_sha=training.weight_sha({'head':warm['head']}) if supervised else None,
                rank=4,core_parameters=100,head_parameters=2 if supervised else 0,maximum_active_coordinates=8,
                label=arm,origin='synthetic',supervised=supervised,balanced=supervised)
            ctx['roots'].append(row)
            if supervised:ctx['cap040_rows'].append(row)
    ctx['sources']={a:SimpleNamespace(load=lambda key,a=a:states[a]) for a in states}
    def short_fit(c,key,w,targets,updates,seed,*a,**k):
        st=original_fit(c,key,w,targets,4,seed,*a,**k)
        st['updates']=st['target']=3000;st['history'][-1]['update']=3000;c['store'].save(key,st);return st
    monkeypatch.setattr(exp.engine,'fit',short_fit)
    monkeypatch.setattr(exp.matched,'valid_weights',lambda *a:None)
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    frozen=exp.run(ctx);assert len(frozen['models'])==9
    exp.verify_freeze(ctx)
    ctx['packs']['test']=ctx['packs']['train'];ctx['test_ids']=ctx['packs']['test']['ids']
    # Root reports are genuinely scored, then reused during evaluation.
    for row in ctx['roots']:
        model=core.new_model('vanilla',ctx['base'],states[row['arm']]['weights'],'cpu')
        report=core.validation(model,ctx['packs']['test'],t,ctx['cfg'],'cpu',with_arrays=True)
        compact=base.study.compact_audit(report.pop('_audit'),exp.matched.mapping(t),ctx['cfg'])
        report.update({k:row[k] for k in ['seed','arm','rank','weights_sha','targets_sha']});report['legacy_native_eligible']=True
        tag=f"s{row['seed']}_{row['arm']}";ctx['store'].save('audit_'+tag,dict(identity=base.report_signature(row,ctx['test_stamp'],ctx['cfg']),report=report));ctx['store'].save('compact_'+tag,compact)
    def forbidden(*a,**k):raise AssertionError('Evaluation cannot fit models')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    out=exp.evaluate(ctx)
    assert out['complete'] and len(out['models'])==15 and len(out['comparisons'])==18
    assert out['new_test_gradient_updates']==0
    st=ctx['store'].load(ctx['rows'][0]['key']);st['head']=st['head']+1;ctx['store'].save(ctx['rows'][0]['key'],st)
    with pytest.raises(ValueError,match='Frozen checkpoint changed'):exp.verify_freeze(ctx)


def test_real_completed_cap_inference_reuses_controls_and_cached_audits(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'test',4)
    ctx['cfg'].update(audit_image_batch=2,patch_sample_seed=4,detection_patches=196,preview_images=0,bootstrap=10,fresh_test_seed=77)
    model=training.make_model('vanilla',ctx['base'],8);weights=training.weights(model)
    pack=ctx['packs']['train'];ctx['packs']['test']=pack
    report=core.validation(model,pack,t,ctx['cfg'],'cpu',with_arrays=True)
    compact=completed.base.study.compact_audit(report.pop('_audit'),completed.matched.mapping(t),ctx['cfg'])
    dev=training.measure(model,ctx['packs']['select']['x'].reshape(-1,8),'cpu',ctx['cfg']['batch'])
    ctx.update(test_stamp=dict(normalized_sha='synthetic'),pipeline_rows=[],development_selection=dict(selected_cap=.6,test_access=False))
    for seed in completed.base.SEEDS:
        for cap in [.4]+[.6]:
            arm=f'cap{int(cap*100):03d}'
            row=dict(seed=seed,arm=arm,label=arm,rank=4,origin='saved',core_parameters=100,
                head_parameters=150,maximum_active_coordinates=64,weights_sha=training.weight_sha(weights),
                targets_sha=completed.base.identity(t),targets=t,upper_bound=cap,development=dict(reconstruction=dev),reference_mse=10.)
            ctx['pipeline_rows'].append(row)
        for arm in completed.base.CONTROLS+['cap040']:
            row=dict(report,seed=seed,arm=arm,rank=4,weights_sha=training.weight_sha(weights),targets_sha=completed.base.identity(t),legacy_native_eligible=True)
            tag=f's{seed}_{arm}'
            ctx['store'].save('audit_'+tag,dict(identity='synthetic_'+tag,report=row));ctx['store'].save('compact_'+tag,compact)
    monkeypatch.setattr(completed,'verify_all',lambda *a:None)
    monkeypatch.setattr(completed.caps,'frozen_weights',lambda *a:weights)
    monkeypatch.setattr(completed.matched,'assert_sources',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    def forbidden(*a,**k):raise AssertionError('No model fitting or control optimization is permitted')
    monkeypatch.setattr(torch.optim,'Adam',forbidden);monkeypatch.setattr(core,'supervised_loss',forbidden)
    original_audit=completed.caps.audit_one
    # 0.40 reports are reused; nine new higher-cap inference calls use real scoring.
    monkeypatch.setattr(completed.base,'audit_treatment',lambda c,r:c['store'].load(f"audit_s{r['seed']}_cap040")['report'])
    out=completed.evaluate(ctx)
    assert out['complete'] and len(out['models'])==15 and len(out['comparisons'])==21
    assert out['new_gradient_updates']==0 and not out['controls_retrained']
    assert out['development_selection']==dict(selected_cap=.6,test_access=False)
    assert len(list(csv.DictReader((ctx['store'].results/'three_seed_test_means.csv').open())))==5
    monkeypatch.setattr(core,'validation',forbidden)
    assert completed.evaluate(ctx)['complete']
    saved=ctx['store'].load('audit_s21_cap060');saved['identity']='different';ctx['store'].save('audit_s21_cap060',saved)
    with pytest.raises(ValueError,match='identity changed'):completed.caps.audit_one(ctx,ctx['pipeline_rows'][1])

def test_freeze_checks_each_saved_phase_and_checks_only_completed_scope(tmp_path,monkeypatch):
    train,t=fixture(tmp_path,'freeze_training',4)
    train['provenance']=dict(splits=dict(train=list(range(5)),select=list(range(5,10))))
    weights=training.weights(training.make_model('vanilla',train['base'],8));head=torch.zeros(1,2)
    warm=dict(best=weights);dev=metric();balance=dict(counter=1)
    row=dict(targets=t,targets_sha=completed.base.identity(t),reference_mse=.35,rank=4)
    old1=dict(sampled_batches_sha='b1',sampled_noise_sha='n1')
    old2=dict(sampled_batches_sha='b2',sampled_noise_sha='n2',history=[dict(validation=dev)])
    train['cap040_rows']=[dict(row,seed=s,arm='cap040',development=dev) for s in completed.base.SEEDS]
    monkeypatch.setattr(completed.base.bound,'verify_seed',lambda c,s:(dict(row,seed=s),warm,old1,old2))
    monkeypatch.setattr(completed.matched,'valid_weights',lambda *a:None)
    monkeypatch.setattr(completed.matched,'assert_sources',lambda *a:None)
    summary=dict(complete=True,study_complete=True,caps=[.6],test_access=False,results=[])
    for seed in completed.base.SEEDS:
        for cap in [.6]:
            arm=f'cap{int(cap*100):03d}'
            for n,rng,previous,initial_head,initial_balance,phase in [
                (2500,71000+seed,old1,None,None,'phase1'),
                (500,82000+seed,old2,head,balance,'phase2')]:
                st=dict(weights=weights,head=head,balance=balance,updates=n,target=n,
                    history=[dict(update=n,validation=dev)],
                    sampled_batches_sha=previous['sampled_batches_sha'],sampled_noise_sha=previous['sampled_noise_sha'])
                st['fingerprint']=completed.base.phase_identity(train,weights,t,n,rng,.35,
                    completed.base.cap_recipe(cap)['balance'],initial_head,initial_balance)
                train['store'].save(f'{phase}_{arm}_s{seed}',st)
            summary['results'].append(dict(seed=seed,upper_bound=cap,development=dev,
                baselines={.2:dev,.4:dev},reconstruction_limit=.35*1.02,
                weights_sha=training.weight_sha(weights),head_sha=training.weight_sha({'head':head}),
                targets_sha=completed.base.identity(t),warm_weights_sha=training.weight_sha(weights),test_access=False,new_updates=3000))
    ctx=train;ctx['sources']={'sweep':train['store']}
    frozen=completed.freeze_endpoints(ctx,summary)
    assert len(frozen['models'])==6 and frozen['development_selection']['selected_cap'] is None
    assert completed.base.identity(completed.freeze_endpoints(ctx,summary))==completed.base.identity(frozen)
    st=train['store'].load('phase1_cap060_s21');st['fingerprint']='changed';train['store'].save('phase1_cap060_s21',st)
    with pytest.raises(ValueError,match='phase lineage'):completed.freeze_endpoints(ctx,summary)
