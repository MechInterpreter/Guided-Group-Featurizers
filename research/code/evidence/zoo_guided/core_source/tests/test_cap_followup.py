"""CPU checks for inference-only reuse, cap isolation, lineage and resume."""
import copy
import csv
import hashlib
import inspect
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core
from coco_ggf import cap_followup as exp, width_balance_fit as engine
from test_width_balance import fixture, cheap_validation


def metric(cap=.3,ap=.7,ve=.65,mse=.35):
    return dict(macro_capture=cap,macro_ap=ap,reconstruction=dict(ve=ve,mse=mse))


def test_caps_change_only_the_two_upper_bounds_and_preserve_capacity():
    for cap in exp.CAPS:
        recipe=exp.cap_recipe(cap)
        expected=exp.bound.settings()['balance']
        assert recipe['balance']['bounds']==dict(capture=[.0125,cap],detection=[.0025,cap])
        recipe['balance']['bounds']=expected['bounds']
        assert recipe['balance']==expected
        assert recipe['groups']==1024 and recipe['active_groups']==16 and not recipe['test_access']
    for cap in [False,.4,0.,float('nan'),2.]:
        with pytest.raises(ValueError):exp.cap_recipe(cap)
    assert exp.caps_checked([1.,.6,.8])==exp.CAPS
    for caps in [[],[.6,.6],[.4],[True]]:
        with pytest.raises(ValueError):exp.caps_checked(caps)
    for method in [exp.prepare_test,exp.load_test,exp.audit_treatment,exp.evaluate]:
        assert 'fit(' not in inspect.getsource(method)


def test_output_overlap_and_missing_freeze_stop_before_test_loading(tmp_path):
    with pytest.raises(ValueError,match='overlaps'):exp.safe_destination(tmp_path/'source'/'new',[tmp_path/'source'])
    ctx,_=fixture(tmp_path,'barrier',4)
    with pytest.raises(FileNotFoundError,match='all_models_frozen'):exp.load_test(ctx)
    ctx['store'].save('all_models_frozen',dict(complete=False,models=[]))
    with pytest.raises(ValueError,match='Freeze'):exp.load_test(ctx)


@pytest.mark.parametrize('cap',exp.CAPS)
def test_real_higher_cap_fit_resume_and_noise_match(tmp_path,monkeypatch,cap):
    monkeypatch.setattr(core,'validation',cheap_validation)
    full,t=fixture(tmp_path,'full',4);part,_=fixture(tmp_path,'part',4);old,_=fixture(tmp_path,'old',4)
    full['recipe']=exp.cap_recipe(cap);part['recipe']=exp.cap_recipe(cap)
    torch.manual_seed(21);weights=training.weights(training.make_model('vanilla',full['base'],8))
    reference=engine.fit(old,'first',weights,t,24,71021,True,True,reference_mse=1.)
    first=engine.fit(full,'first',weights,t,24,71021,True,True,reference_mse=1.)
    engine.fit(part,'first',weights,t,24,71021,True,True,reference_mse=1.,max_updates=13)
    resumed=engine.fit(part,'first',weights,t,24,71021,True,True,reference_mse=1.)
    assert first['fingerprint']==resumed['fingerprint']
    assert engine.state_identity(first['weights'])==engine.state_identity(resumed['weights'])
    assert engine.state_identity(first['balance'])==engine.state_identity(resumed['balance'])
    assert first['sampled_batches_sha']==reference['sampled_batches_sha']==resumed['sampled_batches_sha']
    assert first['sampled_noise_sha']==reference['sampled_noise_sha']==resumed['sampled_noise_sha']
    part['recipe']=exp.cap_recipe(.8 if cap!=.8 else .6)
    with pytest.raises(ValueError,match='identity changed'):engine.fit(part,'first',weights,t,24,71021,True,True,reference_mse=1.)


def test_saved_cap040_phase_and_summary_lineage_rejected(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'lineage',4)
    weights=training.weights(training.make_model('vanilla',ctx['base'],8));head=torch.zeros(75,2)
    ctx['provenance']=dict(splits=dict(train=ctx['packs']['train']['ids'],select=ctx['packs']['select']['ids']))
    row=dict(seed=21,targets=t,targets_sha=identity(t),weights_sha=training.weight_sha(weights),reference_mse=.35,head_sha=training.weight_sha({'head':head}),rank=4)
    warm=dict(best=weights)
    old1=dict(sampled_batches_sha='b1',sampled_noise_sha='n1')
    old2=dict(sampled_batches_sha='b2',sampled_noise_sha='n2',history=[dict(validation=metric())])
    phase1=dict(old1,updates=2500,target=2500,weights=weights,head=head,balance=None,history=[dict(update=2500,validation=metric())])
    phase2=dict(old2,updates=500,target=500,weights=weights,head=head,balance=None,history=[dict(update=500,validation=metric())])
    for st,initial,n,rng,h,b in [(phase1,weights,2500,71021,None,None),(phase2,weights,500,82021,head,None)]:
        st['fingerprint']=exp.phase_identity(ctx,initial,t,n,rng,.35,exp.bound.settings()['balance'],h,b)
    result=dict(seed=21,new_updates=3000,test_access=False,baseline_weights_sha=row['weights_sha'],
        treatment_weights_sha=row['weights_sha'],targets_sha=row['targets_sha'],warm_weights_sha=row['weights_sha'],baseline=metric(),treatment=metric())
    payloads=dict(phase1_cap040_s21=phase1,phase2_cap040_s21=phase2,result_s21=result)
    ctx.update(sources=dict(cap040=SimpleNamespace(load=lambda key:payloads[key])),checkpoint_stamps=[],completed_cap040=dict(results=[copy.deepcopy(result)]))
    monkeypatch.setattr(exp.bound,'verify_seed',lambda *a:(row,warm,old1,old2))
    monkeypatch.setattr(exp.matched,'valid_weights',lambda *a:None)
    monkeypatch.setattr(exp.matched,'source_key_stamp',lambda *a:dict(key=a[1]))
    assert exp.verify_cap040(ctx,21)['key']=='phase2_cap040_s21'
    phase1['fingerprint']='wrong'
    with pytest.raises(ValueError,match='lineage'):exp.verify_cap040(ctx,21)
    phase1['fingerprint']=exp.phase_identity(ctx,weights,t,2500,71021,.35,exp.bound.settings()['balance'])
    ctx['completed_cap040']['results'][0]['treatment']['macro_ap']=.9
    with pytest.raises(ValueError,match='summary differ'):exp.verify_cap040(ctx,21)


def test_real_test_inference_reuses_controls_and_resumes_without_gradients(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'test_eval',4)
    ctx['cfg'].update(audit_image_batch=2,patch_sample_seed=4,detection_patches=196,preview_images=0,bootstrap=10,fresh_test_seed=77)
    model=training.make_model('vanilla',ctx['base'],8);weights=training.weights(model)
    pack=ctx['packs']['train'];ctx['packs']['test']=pack
    report=core.validation(model,pack,t,ctx['cfg'],'cpu',with_arrays=True)
    compact=exp.study.compact_audit(report.pop('_audit'),exp.matched.mapping(t),ctx['cfg'])
    dev=training.measure(model,ctx['packs']['select']['x'].reshape(-1,8),'cpu',ctx['cfg']['batch'])
    ctx['test_stamp']=dict(normalized_sha='synthetic');ctx['cap040_rows']=[]
    for seed in exp.SEEDS:
        row=dict(seed=seed,arm='cap040',label='Cap 0.40',rank=4,origin='saved',core_parameters=100,
            head_parameters=150,maximum_active_coordinates=64,weights_sha=training.weight_sha(weights),targets_sha=identity(t),targets=t,
            development=dict(reconstruction=dev),reference_mse=10.)
        ctx['cap040_rows'].append(row)
        for arm in exp.CONTROLS:
            control=dict(report,**{k:row[k] for k in ['seed','label','rank','core_parameters','head_parameters','maximum_active_coordinates','weights_sha','targets_sha']},arm=arm,legacy_native_eligible=True)
            tag=f's{seed}_{arm}'
            ctx['store'].save('audit_'+tag,dict(identity='synthetic_'+tag,report=control))
            ctx['store'].save('compact_'+tag,compact)
    ctx['store'].save('all_models_frozen',dict(models=ctx['cap040_rows']))
    monkeypatch.setattr(exp,'verify_test_freeze',lambda *a:None)
    monkeypatch.setattr(exp,'get_test_weights',lambda *a:weights)
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    def forbidden(*a,**k):raise AssertionError('No training or optimizer is permitted')
    monkeypatch.setattr(exp,'fit',forbidden)
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    monkeypatch.setattr(core,'supervised_loss',forbidden)
    out=exp.evaluate(ctx)
    assert out['complete'] and out['new_gradient_updates']==0 and not out['controls_retrained']
    assert len(out['models'])==12 and len(out['comparisons'])==9
    assert all(abs(r['capture_gain_pp'])<1e-5 and abs(r['AP_gain_pp'])<1e-5 for r in out['comparisons'])
    # Completed scoring resumes directly from the cached reports, never inference.
    monkeypatch.setattr(core,'validation',forbidden)
    assert exp.evaluate(ctx)['complete']
    source=tmp_path/'source.zip';source.write_bytes(b'code')
    ctx['sources']={}
    result=exp.export(ctx,source)
    with zipfile.ZipFile(result) as z:
        assert not any(n.endswith('.pt') for n in z.namelist())
        for name,digest in json.loads(z.read('artifact_hashes.json')).items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest


def test_sweep_all_nine_trials_same_warmup_and_no_test(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'sweep',4);ctx['cap040_rows']=[]
    weights=training.weights(training.make_model('vanilla',ctx['base'],8));head=torch.zeros(1,2);balance=dict(marker='carry')
    warm=dict(best=weights);row=dict(targets=t,targets_sha=identity(t),reference_mse=.35)
    old1=dict(sampled_batches_sha='b1',sampled_noise_sha='n1')
    old2=dict(sampled_batches_sha='b2',sampled_noise_sha='n2',history=[dict(validation=metric())])
    ctx['cap040_rows']=[dict(seed=s,development=metric(.4,.8,.64,.351)) for s in exp.SEEDS]
    monkeypatch.setattr(exp.bound,'verify_seed',lambda *a:(row,warm,old1,old2))
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    calls=[]
    def fitted(sub,key,initial,targets,n,rng,balanced,supervised,**kwargs):
        assert initial is weights and targets is t and balanced and supervised
        cap=sub['recipe']['upper_bound'];assert cap in exp.CAPS
        if n==500:assert kwargs['initial_head'] is head and kwargs['initial_balance'] is balance
        calls.append((cap,n,rng))
        return dict(old1 if n==2500 else old2,updates=n,target=n,weights=weights,head=head,balance=balance,
            history=[dict(validation=metric(.4,.8,.64,.36),weights_next=[1.,cap,cap])])
    monkeypatch.setattr(exp,'fit',fitted)
    out=exp.run_sweep(ctx)
    assert len(calls)==18 and len(out['results'])==9 and out['new_gradient_updates']==27000
    assert not out['test_access'] and not out['controls_retrained'] and out['new_warmups']==0
    assert all(not r['comparisons']['0.2']['reconstruction_guard_passed'] for r in out['results'])
    rows=list(csv.DictReader((ctx['store'].results/'development_comparison.csv').open()))
    assert len(rows)==15 and sum(r['origin']=='reused_control' for r in rows)==6
    ctx['caps']=[.6];calls.clear()
    first_cap=exp.run_sweep(ctx)
    assert len(calls)==6 and len(first_cap['results'])==3
    assert first_cap['new_gradient_updates']==9000 and not first_cap['study_complete']
    assert len(list(csv.DictReader((ctx['store'].results/'development_comparison.csv').open())))==9
    ctx['packs']['test']={}
    with pytest.raises(ValueError,match='cannot access test'):exp.run_sweep(ctx)


def test_cached_control_report_cannot_change_readouts_or_categories():
    row=dict(weights_sha='w',targets_sha='t',rank=4)
    r=dict(metric(),**row,categories=[dict(category_index=0,capture=.3,ap=.7)])
    exp.check_report(r,row,dict(category_indices=[0]))
    r['targets_sha']='different'
    with pytest.raises(ValueError,match='readouts'):exp.check_report(r,row,dict(category_indices=[0]))
    r['targets_sha']='t';r['categories']=[]
    with pytest.raises(ValueError,match='categories'):exp.check_report(r,row,dict(category_indices=[0]))


def test_test_prepare_reuses_nine_exact_audits_without_opening_test_features(tmp_path,monkeypatch):
    ctx,t=fixture(tmp_path,'test_prepare',4)
    ids=list(range(2400,3900));norm=dict(mean=np.zeros((196,768)),scale=1.)
    stamp=dict(ids=ids,normalization_sha=identity(norm),normalized_sha='saved')
    ctx.update(local=tmp_path,norm=norm,provenance=dict(splits=dict(train=list(range(2000)),select=list(range(2000,2400)))),
        stamps=[],checkpoint_stamps=[],catalog={},cap040_rows=[dict(seed=s) for s in exp.SEEDS])
    saved={};opened=[]
    for seed in exp.SEEDS:
        for arm in exp.CONTROLS:
            row=dict(seed=seed,arm=arm,rank=4,weights_sha='w',targets_sha=identity(t),targets=t)
            ctx['catalog'][(seed,arm)]=row
            tag=f's{seed}_{arm}'
            saved['audit_'+tag]=dict(identity=exp.report_signature(row,stamp,ctx['cfg']),
                report=dict(metric(),weights_sha='w',targets_sha=identity(t),rank=4,
                categories=[dict(category_index=0,capture=.3,ap=.7)]))
            saved['compact_'+tag]=dict(image_ids=ids,category_indices=[0])
    def parent_load(key):
        opened.append(key);return saved[key]
    parent=SimpleNamespace(config=dict(source_runs=dict(strong='strong_run'),test_identity=stamp),load=parent_load)
    strong=SimpleNamespace(config=dict(config=dict(extract_chunk=20)),load=lambda k:dict(splits=dict(test=ids)) if k=='fresh_plan' else stamp)
    ctx['parent']=parent;ctx['sources']=dict(original=object())
    monkeypatch.setattr(exp,'open_sources',lambda *a,**kw:ctx)
    monkeypatch.setattr(exp.matched,'discover',lambda *a,**kw:strong)
    monkeypatch.setattr(exp.memory,'source_snapshot',lambda *a:[])
    monkeypatch.setattr(exp.matched,'source_key_stamp',lambda source,key:dict(key=key))
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    monkeypatch.setattr(exp.matched,'check_targets',lambda *a:None)
    loaded=[]
    monkeypatch.setattr(exp,'cached_split',lambda c,split,cache:(loaded.append(split) or {}))
    assert exp.prepare_test(dict(strong='strong'),tmp_path/'new',tmp_path,tmp_path/'cache','sha',False) is ctx
    assert loaded==['select'] and 'test' not in ctx['packs']
    assert len(opened)==18 and all(k.startswith(('audit_','compact_')) for k in opened)
    assert len(ctx['store'].load('all_models_frozen')['reused_controls'])==9
    first=saved['audit_s21_mask4_balanced'];first['identity']='different'
    with pytest.raises(ValueError,match='different test'):exp.prepare_test(dict(strong='strong'),tmp_path/'new',tmp_path,tmp_path/'cache','sha',False)


def test_sweep_prepare_loads_only_train_and_development(tmp_path,monkeypatch):
    ctx,_=fixture(tmp_path,'sweep_prepare',4);ctx.update(checkpoint_stamps=[])
    monkeypatch.setattr(exp,'open_sources',lambda *a,**kw:ctx)
    loaded=[]
    monkeypatch.setattr(exp,'cached_split',lambda c,split,cache:(loaded.append(split) or {}))
    monkeypatch.setattr(exp.matched,'assert_sources',lambda *a:None)
    exp.prepare_sweep({},tmp_path/'new',tmp_path,tmp_path/'cache','sha',False)
    assert loaded==['train','select'] and 'test' not in ctx['packs']


def test_source_open_requires_all_seeds_and_exact_protocol(tmp_path,monkeypatch):
    ctx,_=fixture(tmp_path,'source_fixture',4)
    versions=exp.matched.versions()
    norm=dict(mean=np.zeros((196,768)),scale=1.)
    provenance=dict(splits=dict(train=list(range(2000)),select=list(range(2000,2400))))
    original_results=tmp_path/'original_results';original_results.mkdir()
    for name in ['provenance.json','coverage_plan.json']:
        (original_results/name).write_text(json.dumps(provenance))
    original=SimpleNamespace(run_id=exp.memory.EXPECTED_PARENT,results=original_results,config={},load=lambda k:norm)
    pc=dict(recipe=exp.width.settings(),versions=versions,base=dict(ctx['base'],block_rank=3),fit_config=ctx['cfg'],
        source_runs=dict(original=original.run_id))
    frozen=dict(complete=True,models=[dict(seed=s,arm=a) for s in exp.SEEDS for a in exp.width.ARMS])
    parent=SimpleNamespace(run_id=exp.bound.PARENT_RUN,config=pc,load=lambda k:frozen)
    bc=dict(recipe=exp.bound.settings(),parent_run=parent.run_id,activation_run=original.run_id,
        fit_config=pc['fit_config'],base=dict(pc['base'],block_rank=4,local_save_updates=50),versions=versions,
        normalization_sha=identity(norm),train_ids=provenance['splits']['train'],development_ids=provenance['splits']['select'],gpu='Tesla T4')
    completed=dict(complete=True,study_complete=True,requested_seeds=exp.SEEDS,test_access=False,results=[dict(seed=s) for s in exp.SEEDS])
    results=tmp_path/'cap_results';results.mkdir();(results/'summary.json').write_text(json.dumps(completed))
    cap=SimpleNamespace(run_id=exp.BOUND_RUN,config=bc,results=results,load=lambda k:completed)
    stores=dict(parent=parent,cap040=cap,original=original)
    monkeypatch.setattr(exp.matched,'discover',lambda path,*a,**kw:stores[Path(path).name])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'Tesla T4')
    monkeypatch.setattr(exp.memory,'source_snapshot',lambda *a:[])
    monkeypatch.setattr(exp.matched,'source_key_stamp',lambda s,k:dict(key=k))
    monkeypatch.setattr(exp,'verify_cap040',lambda c,s:dict(seed=s))
    paths={k:tmp_path/k for k in stores}
    opened=exp.open_sources(paths,tmp_path/'new',tmp_path/'runtime','source','sweep',False)
    assert len(opened['cap040_rows'])==3 and not opened['packs']
    bc['recipe']=copy.deepcopy(bc['recipe']);bc['recipe']['upper_bound']=.8
    with pytest.raises(ValueError,match='protocol'):exp.open_sources(paths,tmp_path/'new',tmp_path/'runtime','source','sweep',False)
    bc['recipe']=exp.bound.settings()
    completed['study_complete']=False;(results/'summary.json').write_text(json.dumps(completed))
    with pytest.raises(ValueError,match='All three'):exp.open_sources(paths,tmp_path/'new',tmp_path/'runtime','source','sweep',False)


def test_preflight_counts_only_requested_unfinished_cap_updates(tmp_path,monkeypatch):
    ctx,_=fixture(tmp_path,'preflight',4);ctx['caps']=[.6]
    ctx['store'].save('phase1_cap060_s21',dict(updates=1000,target=2500))
    monkeypatch.setattr(exp.bound,'preflight',lambda c:dict(seconds_per_update=.1,peak_GiB=1.))
    result=exp.sweep_preflight(ctx)
    assert result['pending_updates']==dict(cap060_s21=2000,cap060_s22=3000,cap060_s23=3000)
    assert result['estimated_gradient_hours']==pytest.approx(800/3600)
