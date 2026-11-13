"""Synthetic numerical and protocol checks for the saved 5D continuation study."""
import copy,inspect,json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from coco_baselines import training
from coco_adaptation import core,inputs
from coco_ggf import five_continuation as exp
from test_width_balance import fixture,cheap_validation


def source_row(ctx,targets,state,seed,arm):
    sup=arm=='mask5'
    return dict(seed=seed,arm=arm,rank=5,supervised=sup,balanced=sup,
        key=f'phase2_{arm}_s{seed}',checkpoint_updates=500,total_training_updates=10640,
        targets=targets,targets_sha=exp.identity(targets),reference_mse=10.,
        weights_sha=training.weight_sha(state['weights']),
        head_sha=training.weight_sha({'head':state['head']}) if sup else None,
        optimizer_sha=exp.identity(exp.engine.state_identity(state['optimizer'])),
        balance_sha=exp.identity(exp.engine.state_identity(state['balance'])),
        phase_batch_sha=state['sampled_batches_sha'],phase_noise_sha=state['sampled_noise_sha'],
        warm_batch_sha='warm_b',warm_noise_sha='warm_n',phase1_batch_sha='p1b',phase1_noise_sha='p1n',
        warm_selected_update=7640,development=state['history'][-1]['validation'])


def test_plan_cap_order_budget_and_comparisons():
    assert exp.ARMS==['guided5_continue','vanilla5_continue','pcgrad5_both']
    assert exp.SEEDS==[21,22,23] and exp.UPDATES==3000
    assert exp.settings()['balance']['bounds']==dict(capture=[.0125,.4],detection=[.0025,.4])
    assert exp.settings()['new_planned_updates']==3*3*3000
    assert ('pcgrad5_both','guided5_continue') in exp.PAIRS[exp.STAGE]
    assert ('guided5_continue','ggf_continue') in exp.PAIRS[exp.STAGE]
    assert ('vanilla5_continue','vanilla_continue') in exp.PAIRS[exp.STAGE]
    assert 'fit_warm' not in inspect.getsource(exp) and 'dictionary5(' not in inspect.getsource(exp)


def test_incomplete_source_catalog_cannot_start():
    with pytest.raises(ValueError,match='Finish Section 4'):
        exp.validate_catalog(dict(complete=True,models=[]),dict(complete=True,stage='width5'))
    rows=[dict(seed=s,arm=a,rank=5,total_training_updates=10640,checkpoint_updates=500,
        key=f'phase2_{a}_s{s}',supervised=a=='mask5',balanced=a=='mask5')
        for s in exp.SEEDS for a in ['mask5','vanilla5']]
    exp.validate_catalog(dict(complete=True,models=rows),dict(complete=True,stage='width5'))
    rows[-1]['total_training_updates']=13640
    with pytest.raises(ValueError,match='Wrong 5D'):exp.validate_catalog(dict(complete=True,models=rows),dict(complete=True,stage='width5'))


@pytest.mark.parametrize('supervised,both',[(True,False),(False,False),(True,True)])
def test_five_dimensional_exact_resume_with_carried_adam_and_controller(tmp_path,monkeypatch,supervised,both):
    monkeypatch.setattr(core,'validation',cheap_validation)
    monkeypatch.setattr(exp,'UPDATES',20)
    a,t=fixture(tmp_path,'full',5);b,_=fixture(tmp_path,'resumed',5)
    for c in [a,b]:c['recipe']=exp.settings();c['cfg']['evaluate_every']=5
    w=training.weights(training.make_model('vanilla',a['base'],8))
    root=exp.engine.fit(a,'root',w,t,20,1,supervised,supervised,10.)
    before=exp.identity(exp.engine.state_identity(root))
    row=dict(targets=t,reference_mse=10.)
    full=exp.fit_branch(a,'trial',root,row,21,supervised,both)
    exp.fit_branch(b,'trial',root,row,21,supervised,both,max_updates=13)
    # Reopen the same Drive directory with an empty local rescue, as after T4 loss.
    b['store']=inputs.JournalStore(b['store'].output,b['store'].config,tmp_path/'fresh_local',require_drive=False)
    resumed=exp.fit_branch(b,'trial',root,row,21,supervised,both)
    for field in ['weights','head','optimizer','balance','sample_rng','projection_rng','decoder_rng','torch_rng',
        'sampled_batches_sha','sampled_noise_sha']:
        assert exp.engine.state_identity(full[field])==exp.engine.state_identity(resumed[field])
    assert exp.identity(exp.engine.state_identity(root))==before
    assert all(r['step']>=40 for r in resumed['optimizer']['state'].values())
    if supervised:
        weights=exp.engine.Balancer(exp.settings()['balance'],'cpu');weights.load_state_dict(resumed['balance'])
        assert (weights.weights()[1:]<=.4000001).all()


def test_source_rejects_mutated_adam_controller_head_and_noise(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'checks',5);ctx['recipe']=exp.settings()
    monkeypatch.setattr(core,'validation',cheap_validation);monkeypatch.setattr(exp.matched,'valid_weights',lambda *a:None)
    w=training.weights(training.make_model('vanilla',ctx['base'],8))
    st=exp.engine.fit(ctx,'root',w,t,2,1,True,True,10.);row=source_row(ctx,t,st,21,'mask5')
    exp.check_state(ctx,row,st)
    for field in ['optimizer','balance','head','sampled_noise_sha']:
        changed=copy.deepcopy(st)
        if field=='optimizer':next(iter(changed[field]['state'].values()))['step']+=1
        elif field=='balance':changed[field]['adaptations']+=1
        elif field=='head':changed[field][0,0]+=1
        else:changed[field]='wrong'
        with pytest.raises(ValueError,match='identity changed'):exp.check_state(ctx,row,changed)


def test_missing_five_source_fails_before_loading_caches(tmp_path,monkeypatch):
    monkeypatch.setattr(exp.matched,'discover',lambda *a,**kw:SimpleNamespace(load=lambda key:None))
    monkeypatch.setattr(exp.prior,'prepare',lambda *a,**kw:pytest.fail('Must not load cached features'))
    with pytest.raises(FileNotFoundError,match='Section 4'):
        exp.prepare({},tmp_path/'out',tmp_path/'local',tmp_path/'cache','sha',tmp_path/'previous',tmp_path/'width5',require_drive=False)


def test_full_pipeline_branches_from_same_roots_and_reuses_control_audits(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'pipeline',5);ctx['recipe']=exp.settings();ctx['roots']=[]
    ctx['cfg'].update(evaluate_every=2,audit_image_batch=2,patch_sample_seed=4,detection_patches=196,
        preview_images=0,bootstrap=5,fresh_test_seed=77)
    ctx.update(test_ids=ctx['packs']['train']['ids'],test_stamp=dict(normalized_sha='synthetic'))
    monkeypatch.setattr(exp.matched,'valid_weights',lambda *a:None)
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    monkeypatch.setattr(exp,'UPDATES',4)
    validation=core.validation;monkeypatch.setattr(core,'validation',cheap_validation)
    states={};calls=[]
    w=training.weights(training.make_model('vanilla',ctx['base'],8))
    for seed in exp.SEEDS:
        for arm in ['mask5','vanilla5']:
            sup=arm=='mask5';st=exp.engine.fit(ctx,f'root_{arm}_{seed}',w,t,2,seed,sup,sup,10.)
            st.update(updates=500,target=500);row=source_row(ctx,t,st,seed,arm)
            ctx['roots'].append(row);states[row['key']]=st
    ctx['sources']=dict(width5=SimpleNamespace(load=lambda key:states.get(key)))
    original_fit=exp.engine.fit
    # Small synthetic streams stand in for the precomputed 4D hash records.
    for seed in exp.SEEDS:
        root=exp.root_for(ctx,seed,'mask5');st=states[root['key']]
        expected=original_fit(ctx,f'expected_{seed}',st['weights'],t,4,93000+seed,True,True,10.,
            initial_head=st['head'],initial_balance=st['balance'],initial_optimizer=st['optimizer'])
        for arm in ['ggf_continue','ggf_pcgrad','vanilla_continue','cap040','vanilla4']:
            ctx['roots'].append(dict(root,arm=arm,rank=4,total_training_updates=13640 if 'continue' in arm or arm=='ggf_pcgrad' else 10640,
                phase_batch_sha=expected['sampled_batches_sha'],phase_noise_sha=expected['sampled_noise_sha']))
    root_hash=exp.identity(exp.engine.state_identity(states))
    def fit(*args,**kw):
        calls.append((args[1],training.weight_sha(args[2]),kw['pcgrad'],kw['decoder_pcgrad']))
        return original_fit(*args,**kw)
    monkeypatch.setattr(exp.engine,'fit',fit)
    frozen=exp.run(ctx)
    assert [k.rsplit('_s',1)[0] for k,_,_,_ in calls]==[a for a in exp.ARMS for _ in exp.SEEDS]
    assert len(frozen['models'])==9 and not frozen['test_features_opened']
    assert exp.identity(exp.engine.state_identity(states))==root_hash
    for seed in exp.SEEDS:
        a=next(r for r in frozen['models'] if r['seed']==seed and r['arm']=='guided5_continue')
        b=next(r for r in frozen['models'] if r['seed']==seed and r['arm']=='pcgrad5_both')
        assert a['starting_weights_sha']==b['starting_weights_sha']
        assert a['starting_optimizer_sha']==b['starting_optimizer_sha']
        assert a['starting_balance_sha']==b['starting_balance_sha']
    exp.verify_freeze(ctx)
    monkeypatch.setattr(core,'validation',validation)
    # Cache real synthetic audits for every control; evaluation must not train them.
    for row in ctx['roots']:
        model=core.new_model('vanilla',ctx['base'],w,'cpu')
        r=core.validation(model,ctx['packs']['train'],t,ctx['cfg'],'cpu',with_arrays=True)
        compact=exp.base.study.compact_audit(r.pop('_audit'),exp.matched.mapping(t),ctx['cfg'])
        report=dict(r,seed=row['seed'],arm=row['arm'],rank=row['rank'],weights_sha=row['weights_sha'],
            targets_sha=row['targets_sha'],legacy_native_eligible=True)
        tag=f"s{row['seed']}_{row['arm']}"
        ctx['store'].save('audit_'+tag,dict(identity=exp.base.report_signature(row,ctx['test_stamp'],ctx['cfg']),report=report))
        ctx['store'].save('compact_'+tag,compact)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**kw:pytest.fail('Inference cannot create an optimizer'))
    monkeypatch.setattr(exp.base,'load_test',lambda c:c['packs'].update(test=c['packs']['train']))
    result=exp.evaluate(ctx)
    assert result['complete'] and not result['controls_retrained']
    assert len(result['models'])==30 and len(result['comparisons'])==27
    assert exp.identity(exp.evaluate(ctx))==exp.identity(result)


def test_freeze_barrier_cannot_open_test_with_missing_endpoints(tmp_path,monkeypatch):
    ctx,_=fixture(tmp_path,'barrier',5)
    monkeypatch.setattr(exp.base,'load_test',lambda *a:pytest.fail('No premature test access'))
    ctx['store'].save('frozen_'+exp.STAGE,dict(complete=True,models=[]))
    with pytest.raises(ValueError,match='Freeze all nine'):exp.evaluate(ctx)


def test_bad_width5_protocol_rejected_before_state_reads(tmp_path,monkeypatch):
    ctx,_=fixture(tmp_path,'protocol',5)
    source=SimpleNamespace(run_id='wrong',config={})
    with pytest.raises(ValueError,match='protocol'):exp.validate_width5(ctx,source,{}, {})


def test_width5_validation_checks_exact_lineage_and_reuses_six_reports(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'lineage',5);ctx['recipe']=exp.settings();ctx['roots']=[]
    ctx.update(sources={'cap040':SimpleNamespace(run_id='cap')},parent=SimpleNamespace(run_id='parent'),
        test_ids=ctx['packs']['train']['ids'],test_stamp={'test':'synthetic'},checkpoint_stamps=[])
    monkeypatch.setattr(core,'validation',cheap_validation)
    monkeypatch.setattr(exp.matched,'valid_weights',lambda *a:None)
    monkeypatch.setattr(exp.matched,'check_targets',lambda *a:None)
    monkeypatch.setattr(exp.matched,'versions',lambda:{'synthetic':'1'})
    monkeypatch.setattr(exp.matched,'source_key_stamp',lambda s,k:dict(key=k))
    cfg=dict(source_sha=exp.WIDTH5_SOURCE_SHA,continuation_recipe=exp.prior.settings(),base=ctx['base'],fit_config=ctx['cfg'],
        versions={'synthetic':'1'},parent_run='parent',cap040_run='cap',previous_run=exp.prior.PREVIOUS_RUN)
    saved={};rows=[];reports=[]
    w=training.weights(training.make_model('vanilla',ctx['base'],8))
    for seed in exp.SEEDS:
        ctx['roots'].append(dict(seed=seed,arm='cap040',reference_mse=10.))
        for arm in ['mask5','vanilla5']:
            sup=arm=='mask5';st=exp.engine.fit(ctx,f'root_{arm}_{seed}',w,t,2,seed,sup,sup,10.)
            st.update(updates=500,target=500);row=source_row(ctx,t,st,seed,arm);rows.append(row);saved[row['key']]=st
            report=dict(cheap_validation(core.new_model('vanilla',ctx['base'],st['weights'],'cpu'),ctx['packs']['select'],t,ctx['cfg'],'cpu'),
                seed=seed,arm=arm,rank=5,weights_sha=row['weights_sha'],targets_sha=row['targets_sha'],
                categories=[dict(category_index=0,capture=.2,ap=.6)])
            reports.append(report);tag=f's{seed}_{arm}'
            saved['audit_'+tag]=dict(identity=exp.base.report_signature(row,ctx['test_stamp'],ctx['cfg']),report=report)
            saved['compact_'+tag]=dict(image_ids=ctx['test_ids'],category_indices=ctx['cfg']['category_indices'])
    frozen=dict(complete=True,models=rows);result=dict(complete=True,stage='width5',models=reports)
    source=SimpleNamespace(run_id=exp.WIDTH5_RUN,config=cfg,load=lambda key:saved.get(key))
    exp.validate_width5(ctx,source,frozen,result)
    assert len([r for r in ctx['roots'] if r.get('source_name')=='width5'])==6
    assert all(ctx['store'].load(f'audit_s{s}_{a}') is not None for s in exp.SEEDS for a in ['mask5','vanilla5'])
    source.config['source_sha']='changed'
    with pytest.raises(ValueError,match='protocol'):exp.validate_width5(ctx,source,frozen,result)
