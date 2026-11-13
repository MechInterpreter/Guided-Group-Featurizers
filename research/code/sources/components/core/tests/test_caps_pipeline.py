"""Synthetic ordering, selection, frozen inference and package checks."""
import copy
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from coco_baselines import training
from coco_adaptation import core,inputs
from coco_ggf import caps_pipeline as exp
from test_width_balance import fixture,store


def metric(cap=.4,ap=.8,ve=.65,mse=.35):
    return dict(macro_capture=cap,macro_ap=ap,reconstruction=dict(ve=ve,mse=mse))


def development_summary():
    results=[]
    for seed in exp.base.SEEDS:
        for cap in exp.base.CAPS:
            results.append(dict(seed=seed,upper_bound=cap,development=metric(cap=.41+cap/100,mse=.35 if cap==.6 else .36),
                baselines={.2:metric(.38),.4:metric(.40)},reconstruction_limit=.357))
    return dict(complete=True,study_complete=True,caps=exp.base.CAPS,results=results,test_access=False)


def test_development_decision_uses_all_three_seeds_and_native_guard():
    summary=development_summary();choice=exp.development_decision(summary,.2)
    assert choice['selected_cap']==.6 and choice['split']=='development' and not choice['test_access']
    assert len(choice['candidates'])==5
    # A high-scoring cap that fails even one original guard cannot win.
    summary['results'][0]['development']['reconstruction']['mse']=.36
    assert exp.development_decision(summary,.2)['selected_cap']==.4
    summary['results'].pop()
    with pytest.raises(ValueError,match='Complete all three'):exp.development_decision(summary,.2)


def test_training_completes_before_freeze_and_test_is_not_called(tmp_path,monkeypatch):
    train,_=fixture(tmp_path,'train',4);ctx=dict(training=train,store=store(tmp_path,'pipeline'))
    calls=[];summary=development_summary()
    monkeypatch.setattr(exp.base,'run_sweep',lambda c:(calls.append('all_training') or summary))
    monkeypatch.setattr(exp,'freeze_endpoints',lambda c,s:calls.append('freeze_all_endpoints'))
    def forbidden(*a,**k):raise AssertionError('Test cannot start in the training stage')
    monkeypatch.setattr(exp,'load_test',forbidden)
    assert exp.train_and_freeze(ctx) is summary
    assert calls==['all_training','freeze_all_endpoints']


def test_missing_or_incomplete_catalog_stops_before_test_features(tmp_path,monkeypatch):
    s=store(tmp_path,'barrier');ctx=dict(store=s)
    def forbidden(*a,**k):raise AssertionError('Experimental test data must stay closed')
    monkeypatch.setattr(exp.base,'prepare_test',forbidden)
    monkeypatch.setattr(exp.base,'load_test',forbidden)
    with pytest.raises(FileNotFoundError):exp.prepare_evaluation(ctx)
    s.save('all_caps_frozen',dict(complete=True,models=[]))
    with pytest.raises(ValueError,match='Freeze all twelve'):exp.prepare_evaluation(ctx)
    s.save('combined_models_frozen',dict(complete=False,models=[],development_selection={}))
    ctx.update(pipeline_rows=[],development_selection={})
    with pytest.raises(ValueError,match='Frozen catalog'):exp.load_test(ctx)


def test_frozen_checkpoint_cannot_change_weights_or_head(tmp_path):
    ctx,_=fixture(tmp_path,'weights',4)
    weights=training.weights(training.make_model('vanilla',ctx['base'],8));head=torch.zeros(1,2)
    state=dict(weights=weights,head=head,updates=500,target=500)
    ctx['sources']=dict(sweep=SimpleNamespace(load=lambda key:state))
    row=dict(source_name='sweep',key='phase2',weights_sha=training.weight_sha(weights),head_sha=training.weight_sha({'head':head}))
    assert exp.frozen_weights(ctx,row) is weights
    state['head']=torch.ones_like(head)
    with pytest.raises(ValueError,match='checkpoint changed'):exp.frozen_weights(ctx,row)


def test_real_all_cap_inference_reuses_controls_and_cached_audits(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'test',4)
    ctx['cfg'].update(audit_image_batch=2,patch_sample_seed=4,detection_patches=196,preview_images=0,bootstrap=10,fresh_test_seed=77)
    model=training.make_model('vanilla',ctx['base'],8);weights=training.weights(model)
    pack=ctx['packs']['train'];ctx['packs']['test']=pack
    report=core.validation(model,pack,t,ctx['cfg'],'cpu',with_arrays=True)
    compact=exp.base.study.compact_audit(report.pop('_audit'),exp.matched.mapping(t),ctx['cfg'])
    dev=training.measure(model,ctx['packs']['select']['x'].reshape(-1,8),'cpu',ctx['cfg']['batch'])
    ctx.update(test_stamp=dict(normalized_sha='synthetic'),pipeline_rows=[],development_selection=dict(selected_cap=.6,test_access=False))
    for seed in exp.base.SEEDS:
        for cap in [.4]+exp.base.CAPS:
            arm=f'cap{int(cap*100):03d}'
            row=dict(seed=seed,arm=arm,label=arm,rank=4,origin='saved',core_parameters=100,
                head_parameters=150,maximum_active_coordinates=64,weights_sha=training.weight_sha(weights),
                targets_sha=exp.base.identity(t),targets=t,upper_bound=cap,development=dict(reconstruction=dev),reference_mse=10.)
            ctx['pipeline_rows'].append(row)
        for arm in exp.base.CONTROLS+['cap040']:
            row=dict(report,seed=seed,arm=arm,rank=4,weights_sha=training.weight_sha(weights),targets_sha=exp.base.identity(t),legacy_native_eligible=True)
            tag=f's{seed}_{arm}'
            ctx['store'].save('audit_'+tag,dict(identity='synthetic_'+tag,report=row));ctx['store'].save('compact_'+tag,compact)
    monkeypatch.setattr(exp,'verify_all',lambda *a:None)
    monkeypatch.setattr(exp,'frozen_weights',lambda *a:weights)
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    def forbidden(*a,**k):raise AssertionError('No model fitting or control optimization is permitted')
    monkeypatch.setattr(torch.optim,'Adam',forbidden);monkeypatch.setattr(core,'supervised_loss',forbidden)
    original_audit=exp.audit_one
    # 0.40 reports are reused; nine new higher-cap inference calls use real scoring.
    monkeypatch.setattr(exp.base,'audit_treatment',lambda c,r:c['store'].load(f"audit_s{r['seed']}_cap040")['report'])
    out=exp.evaluate(ctx)
    assert out['complete'] and len(out['models'])==21 and len(out['comparisons'])==45
    assert out['new_gradient_updates']==0 and not out['controls_retrained']
    assert out['development_selection']==dict(selected_cap=.6,test_access=False)
    assert len(list(csv.DictReader((ctx['store'].results/'three_seed_test_means.csv').open())))==7
    monkeypatch.setattr(core,'validation',forbidden)
    assert exp.evaluate(ctx)['complete']
    saved=ctx['store'].load('audit_s21_cap060');saved['identity']='different';ctx['store'].save('audit_s21_cap060',saved)
    with pytest.raises(ValueError,match='identity changed'):exp.audit_one(ctx,ctx['pipeline_rows'][1])


def test_compact_combined_export_contains_both_stages_and_hashes(tmp_path):
    pipeline=store(tmp_path,'pipeline');train=store(tmp_path,'training');evaluation=store(tmp_path,'eval')
    for s in [pipeline,train,evaluation]:s.write('summary.json',dict(complete=True))
    ctx=dict(store=pipeline,training=dict(store=train),evaluation=dict(store=evaluation),sources={})
    source=tmp_path/'source.zip';source.write_bytes(b'source bytes')
    result=exp.export(ctx,source)
    with zipfile.ZipFile(result) as z:
        assert {'pipeline/summary.json','development/summary.json','test/summary.json','source.zip'}<=set(z.namelist())
        assert not any(n.endswith('.pt') for n in z.namelist())
        for name,digest in json.loads(z.read('artifact_hashes.json')).items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest


def test_resume_engine_really_is_byte_identical():
    assert exp.verify_unchanged_engine()['verified_files']==76


def test_freeze_checks_each_saved_phase_and_preserves_development_choice(tmp_path,monkeypatch):
    train,t=fixture(tmp_path,'freeze_training',4)
    train['provenance']=dict(splits=dict(train=list(range(5)),select=list(range(5,10))))
    weights=training.weights(training.make_model('vanilla',train['base'],8));head=torch.zeros(1,2)
    warm=dict(best=weights);dev=metric();balance=dict(counter=1)
    row=dict(targets=t,targets_sha=exp.base.identity(t),reference_mse=.35,rank=4)
    old1=dict(sampled_batches_sha='b1',sampled_noise_sha='n1')
    old2=dict(sampled_batches_sha='b2',sampled_noise_sha='n2',history=[dict(validation=dev)])
    train['cap040_rows']=[dict(row,seed=s,arm='cap040',development=dev) for s in exp.base.SEEDS]
    monkeypatch.setattr(exp.base.bound,'verify_seed',lambda c,s:(dict(row,seed=s),warm,old1,old2))
    monkeypatch.setattr(exp.matched,'valid_weights',lambda *a:None)
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    summary=dict(complete=True,study_complete=True,caps=exp.base.CAPS,test_access=False,results=[])
    for seed in exp.base.SEEDS:
        for cap in exp.base.CAPS:
            arm=f'cap{int(cap*100):03d}'
            for n,rng,previous,initial_head,initial_balance,phase in [
                (2500,71000+seed,old1,None,None,'phase1'),
                (500,82000+seed,old2,head,balance,'phase2')]:
                st=dict(weights=weights,head=head,balance=balance,updates=n,target=n,
                    history=[dict(update=n,validation=dev)],
                    sampled_batches_sha=previous['sampled_batches_sha'],sampled_noise_sha=previous['sampled_noise_sha'])
                st['fingerprint']=exp.base.phase_identity(train,weights,t,n,rng,.35,
                    exp.base.cap_recipe(cap)['balance'],initial_head,initial_balance)
                train['store'].save(f'{phase}_{arm}_s{seed}',st)
            summary['results'].append(dict(seed=seed,upper_bound=cap,development=dev,
                baselines={.2:dev,.4:dev},reconstruction_limit=.35*1.02,
                weights_sha=training.weight_sha(weights),head_sha=training.weight_sha({'head':head}),
                targets_sha=exp.base.identity(t),warm_weights_sha=training.weight_sha(weights),test_access=False,new_updates=3000))
    ctx=dict(training=train,store=store(tmp_path,'freeze_pipeline'))
    frozen=exp.freeze_endpoints(ctx,summary)
    assert len(frozen['models'])==12 and frozen['development_selection']['selected_cap']==.2
    assert exp.base.identity(exp.freeze_endpoints(ctx,summary))==exp.base.identity(frozen)
    st=train['store'].load('phase1_cap060_s21');st['fingerprint']='changed';train['store'].save('phase1_cap060_s21',st)
    with pytest.raises(ValueError,match='phase lineage'):exp.freeze_endpoints(ctx,summary)


def test_prior_cap040_audits_reused_only_for_exact_identity(tmp_path,monkeypatch):
    prior_output=tmp_path/'prior';config=dict(study='test',source_sha='new')
    prior=inputs.JournalStore(prior_output,dict(config,source_sha=exp.LEGACY_SWEEP_SOURCE_SHA),tmp_path/'prior_local',False)
    current=store(tmp_path,'current')
    row=dict(seed=21,rank=4,weights_sha='weights',targets_sha='targets')
    cfg=dict(category_indices=[0]);stamp=dict(ids=[1,2])
    report=dict(row,**metric(),categories=[dict(category_index=0,capture=.4,ap=.8)])
    saved=dict(identity=exp.base.report_signature(row,stamp,cfg),report=report)
    compact=dict(image_ids=[1,2],category_indices=[0])
    prior.save('audit_s21_cap040',saved);prior.save('compact_s21_cap040',compact)
    current.config=config
    ctx=dict(store=current,local=tmp_path/'read',cap040_rows=[row],test_stamp=stamp,cfg=cfg,
        test_ids=[1,2],sources={},stamps=[],checkpoint_stamps=[])
    assert exp.reuse_previous_cap040_audits(ctx,prior_output)==[21]
    assert current.load('audit_s21_cap040')['identity']==saved['identity']
    saved['identity']='wrong';prior.save('audit_s21_cap040',saved)
    with pytest.raises(ValueError,match='different checkpoint'):exp.reuse_previous_cap040_audits(ctx,prior_output)
