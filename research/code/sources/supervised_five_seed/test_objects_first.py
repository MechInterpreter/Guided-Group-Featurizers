import copy,json,inspect
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.optimize import nnls
import objects_first as task
import supervised_study as s
import object_schedule as schedule
from coco_baselines import training
from coco_baselines.storage import identity
from test_supervised import Store
from test_five_seed import fixture as catalog_fixture
from test_precision_retry import rounding_fixture

def test_existing_manifest_is_pinned_and_never_replaced(tmp_path):
    expected=json.loads((task.HERE/'FIVE_MANIFEST.json').read_text())
    p=tmp_path/'results'/task.FIVE_RUN[:20]/'manifest.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(expected));assert task.pinned_manifest(tmp_path)==expected
    expected['config']['settings_sha']='changed';p.write_text(json.dumps(expected))
    with pytest.raises(ValueError,match='exact active'):task.pinned_manifest(tmp_path)

def test_object_evaluator_uses_real_cached_partial_metrics_without_joint_nnls(tmp_path,monkeypatch):
    from test_object_schedule import fixture
    ctx,pack,model,targets=fixture();store=Store(tmp_path/'r')
    D=model.D.clone();store.save('native',dict(D=D))
    ctx.update(store=store,norm=dict(scale=1),device='cpu')
    row=dict(method='native_slics',key='native',weights_sha=training.weight_sha({'D':D}),targets=targets,
        targets_sha=identity(targets),solver=s.native_solver_config(),readout='partial',groups=75,arm='slics_final_partial',seed=21)
    monkeypatch.setattr(s,'final_reconstruction',lambda *a:(_ for _ in ()).throw(AssertionError('blocked on joint reconstruction')))
    monkeypatch.setattr(s.NativeSLiCS,'solve',lambda *a:(_ for _ in ()).throw(AssertionError('joint solve used')))
    direct=s.evaluate_model(model,pack,targets,ctx['cfg'],'cpu',compact=True,reconstruction=False)
    evaluate=task.object_evaluator();actual=evaluate(ctx,row,pack,True)
    assert actual['macro_capture']==direct['macro_capture'] and actual['macro_ap']==direct['macro_ap']
    assert actual['reconstruction']['ve'] is None and len(actual['categories'])==75
    monkeypatch.setattr(s,'partial_nnls',lambda *a:(_ for _ in ()).throw(AssertionError('cached object pass rerun')))
    cached=evaluate(ctx,row,pack,True)
    assert cached['macro_capture']==actual['macro_capture'] and np.array_equal(cached['_compact']['error'],actual['_compact']['error'])

def object_context(tmp_path,monkeypatch):
    ctx,rows,_,_=catalog_fixture(tmp_path);store=ctx['store'];ctx['norm']={};ctx['cfg']={}
    monkeypatch.setattr(s,'verify_frozen',lambda ctx:dict(models=rows))
    monkeypatch.setattr(s,'SEEDS',[21,22,23,24,25])
    monkeypatch.setattr(task,'object_plot',lambda *a:None)
    pack=dict(ids=[1,2],identity='same-test')
    def evaluator(ctx,row,pack,compact):
        rec=dict(status='deferred',ve=None) if row['method']=='native_slics' else dict(status='computed',ve=.8)
        return dict(macro_capture=.3+row['seed']*.001,macro_ap=.7,reconstruction=rec,_compact={'marker':'same'},categories=[])
    return ctx,rows,pack,evaluator

def test_objects_pass_all23_reports_before_any_reconstruction_and_reuses_on_resume(tmp_path,monkeypatch):
    ctx,rows,pack,evaluate=object_context(tmp_path,monkeypatch);calls=[]
    def observed(*args):calls.append(args[1]['arm']);return evaluate(*args)
    summary=task.objects_pass(ctx,pack,observed)
    assert summary['object_metrics_complete'] and summary['seed_intervals_complete']
    assert not summary['reconstruction_complete'] and not summary['whole_image_intervals_complete']
    assert all(r['n']==5 for r in summary['seed_intervals'])
    assert len([k for k in ctx['store'].values if k.startswith(tuple('objects_s'+str(seed)+'_' for seed in task.SEEDS))])==23
    assert all(r['VE_pct'] is None for r in summary['means'] if r['arm'].startswith('slics'))
    def forbidden(*args):raise AssertionError('completed metric recomputed')
    again=task.objects_pass(ctx,pack,forbidden)
    assert again['means']==summary['means'] and calls

def test_completed_previous_audit_and_its_alias_are_reused(tmp_path,monkeypatch):
    ctx,rows,pack,evaluate=object_context(tmp_path,monkeypatch);first=rows[0]
    sig=task.audit_signature(ctx,first,pack);old=evaluate(ctx,first,pack,True);compact=old.pop('_compact')
    key='audit_s21_matched_ggf';ctx['store'].save(key,dict(row=first,signature=sig,report=old,compact=compact))
    ctx['store'].save('audit_signature_index',{sig:key})
    def fail_for_first(ctx,row,*args):
        if row['weights_sha']==first['weights_sha']:raise AssertionError('existing audit repeated')
        return evaluate(ctx,row,*args)
    task.objects_pass(ctx,pack,fail_for_first)
    assert ctx['store'].load(task.object_key(first))['reused_inference']

def test_bad_cached_audit_signature_stops(tmp_path,monkeypatch):
    ctx,rows,pack,evaluate=object_context(tmp_path,monkeypatch)
    ctx['store'].save(task.object_key(rows[0]),dict(signature='wrong'))
    with pytest.raises(ValueError,match='audit changed'):task.objects_pass(ctx,pack,evaluate)

def test_pending_ve_is_not_a_partial_seed_mean_or_zero(tmp_path,monkeypatch):
    ctx,rows,pack,evaluate=object_context(tmp_path,monkeypatch);models=[]
    for r in rows:
        report=evaluate(ctx,r,pack,True);report.pop('_compact')
        if r['seed']==21:report['reconstruction']=dict(status='computed',ve=.9)
        models.append(dict(row=r,report=report))
    result=task.object_summary(ctx,models)
    assert next(r for r in result['means'] if r['arm']=='slics_final_partial' and r['seed_group']=='all_five')['VE_pct'] is None
    assert all(r['seeds']==5 for r in result['means'] if r['seed_group']=='all_five')

def test_CPU_intervals_are_computed_before_reconstruction_and_are_cached(tmp_path,monkeypatch):
    ctx,rows,pack,evaluate=object_context(tmp_path,monkeypatch);task.objects_pass(ctx,pack,evaluate)
    calls=[]
    def paired(a,b,cfg):
        calls.append(1);return dict(capture_gain_ci95=[.1,.2],ap_gain_ci95=[.2,.3],valid_bootstrap_replicates=1000,invalid_full_cohort_replicates=0)
    monkeypatch.setattr(s.study,'paired',paired)
    models=[dict(row=r,report=ctx['store'].load(task.object_key(r))['report']) for r in rows]
    result=task.cpu_intervals(ctx,models)
    assert len(calls)==10 and result['whole_image_intervals_complete'] and not result['reconstruction_complete']
    assert len(result['image_intervals'])==10
    monkeypatch.setattr(s.study,'paired',lambda *a:(_ for _ in ()).throw(AssertionError('bootstrap repeated')))
    assert task.cpu_intervals(ctx,models)['image_intervals']==result['image_intervals']

def test_CPU_direct_NNLS_matches_scipy_and_keeps_original_threshold(monkeypatch):
    x,D,cfg=rounding_fixture();before=D.clone();calls=[]
    from coco_ggf.slics_repair import Solver
    original=Solver.solve
    def check(self,x,allowed=None,force_fallback=False):
        calls.append((x.device.type,x.dtype,force_fallback));return original(self,x,allowed,force_fallback)
    monkeypatch.setattr(Solver,'solve',check)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a:(_ for _ in ()).throw(AssertionError('CUDA called')))
    model=task.CPUJointModel(D,cfg);z=model.solve(x)
    assert calls==[('cpu',torch.float64,True)] and torch.equal(D,before)
    assert float(model.solver._errors(z,x,torch.ones_like(z,dtype=torch.bool)).max())<=cfg['nnls_tolerance']
    for i in range(len(x)):
        expected,_=nnls(D.double().numpy().T,x[i].double().numpy(),maxiter=20000)
        np.testing.assert_allclose(z[i].numpy(),expected,atol=1e-10,rtol=1e-10)

def test_CPU_resumes_existing_verified_image_prefix_with_identical_cache_key(tmp_path):
    D=torch.eye(12,14);cfg=s.native_solver_config();rng=np.random.default_rng(31)
    x=rng.normal(size=(3,196,14)).astype(np.float32);pack=dict(x=x,ids=[1,2,3],identity='same-cohort')
    ctx=dict(store=Store(tmp_path/'r'),device='cpu',norm={});model=task.CPUJointModel(D,cfg);count=[0]
    solve=model.solve
    def interrupt(batch):
        count[0]+=1
        if count[0]==2:raise KeyboardInterrupt()
        return solve(batch)
    model.solve=interrupt
    with pytest.raises(KeyboardInterrupt):schedule.final_reconstruction(ctx,model,pack,'same-frozen-weights')
    keys=[k for k in ctx['store'].values if k.startswith('final_joint_')]
    assert len(keys)==1 and ctx['store'].load(keys[0])['completed_images']==1
    resumed=task.CPUJointModel(D,cfg);calls=[];original=resumed.solve
    def observe(batch):calls.append(len(batch));return original(batch)
    resumed.solve=observe;out=schedule.final_reconstruction(ctx,resumed,pack,'same-frozen-weights')
    assert calls==[196,196] and ctx['store'].load(keys[0])['completed_images']==3
    pred=np.zeros_like(x,dtype=float);pred[:,:,:12]=np.maximum(x[:,:,:12],0)
    assert out['mse']==pytest.approx(np.square(x.astype(float)-pred).mean(),abs=1e-12)
    resumed.solve=lambda *a:(_ for _ in ()).throw(AssertionError('finished reconstruction rerun'))
    assert schedule.final_reconstruction(ctx,resumed,pack,'same-frozen-weights')==out

def test_separate_workers_have_no_training_or_extraction_calls_and_CPU_order_is_correct():
    gpu=(task.HERE/'objects_first_worker.py').read_text();cpu=(task.HERE/'cpu_finish_worker.py').read_text()
    for text in [gpu,cpu]:
        assert 'study.final_rows(' not in text and 'study.selection(' not in text and 'backbone_preflight(' not in text
        assert 'study.extract_after_freeze(' not in text
    assert 'task.cpu_reconstruction(' not in gpu and 'task.objects_pass(' in gpu
    assert cpu.index('task.cpu_intervals(')<cpu.index("'objects_with_intervals'")<cpu.index('task.cpu_reconstruction(')
    assert 'cuda.synchronize(' not in inspect.getsource(task.CPUJointModel)

def test_CPU_requires_complete_object_handoff_not_a_new_training_run(tmp_path,monkeypatch):
    manifest=json.loads((task.HERE/'FIVE_MANIFEST.json').read_text())
    changed=copy.deepcopy(manifest);changed['config']['evaluation_protocol']['seeds']=[21,22,23]
    fake=Store(tmp_path/'r');fake.save('all_models_frozen',dict(complete=True,models=[]))
    with pytest.raises(ValueError):task.check_frozen(fake,manifest)
    text=(task.HERE/'cpu_finish_worker.py').read_text()
    assert "required(ctx['store'],'objects_phase_complete')" in text

def test_cached_test_loader_has_no_extraction_fallback(tmp_path,monkeypatch):
    manifest=json.loads((task.HERE/'FIVE_MANIFEST.json').read_text())
    store=Store(tmp_path/'r');ctx=dict(store=store,local=tmp_path,norm={},cfg=dict(extract_chunk=20))
    plan=json.loads((Path(s.__file__).parent/'predecessor_holdout_fixture.json').read_text())['plan'];store.save('fresh_holdout_plan',plan)
    from coco_stratified import dataset
    monkeypatch.setattr(dataset,'extract',lambda *a:(_ for _ in ()).throw(AssertionError('reextracted features')))
    monkeypatch.setattr(task.memory,'normalized_pack',lambda *a:(_ for _ in ()).throw(FileNotFoundError('Missing cached shard')))
    with pytest.raises(FileNotFoundError,match='cached shard'):task.cached_test_pack(ctx)

def test_cached_feature_normalization_and_cohort_identity_match_existing_audit(tmp_path,monkeypatch):
    from test_object_schedule import fixture
    from coco_adaptation.inputs import JournalStore
    from coco_adaptation import coverage
    from coco_baselines import data
    cfg_ctx,small,_,_=fixture();cfg=dict(cfg_ctx['cfg'],extract_chunk=1)
    rng=np.random.default_rng(31);raw=rng.normal(size=(2,196,768)).astype(np.float16)
    norm=dict(mean=rng.normal(size=(196,768)).astype(np.float32)*.01,scale=.9,train_sha='unchanged')
    expected=data.normalize(raw.astype(np.float32),norm)
    plan=dict(splits=dict(test=small['ids']),coverage=dict(split_counts=dict(test=coverage.evidence(small,cfg).all(2).sum(0).tolist())))
    folder=tmp_path/'revision';folder.mkdir()
    (folder/'FIVE_MANIFEST.json').write_text(json.dumps(dict(config=dict(holdout_plan_sha=identity(plan)))))
    monkeypatch.setattr(task,'HERE',folder)
    store=JournalStore(tmp_path/'output',dict(test='cached-normalization'),tmp_path/'rescue',False)
    store.save('fresh_holdout_plan',plan)
    for i in range(2):store.save(f'features_test_{i:06d}',dict(ids=small['ids'][i:i+1],masks=small['masks'][i:i+1],x=torch.from_numpy(raw[i:i+1])))
    ctx=dict(store=store,local=tmp_path/'local',cfg=cfg,norm=norm)
    pack=task.cached_test_pack(ctx)
    assert np.array_equal(pack['x'],expected)
    assert pack['identity']==identity(dict(ids=small['ids'],normalization=identity(norm)))
    before=task.audit_signature(ctx,dict(weights_sha='w',targets_sha='t'),pack)
    again=task.cached_test_pack(ctx)
    assert task.audit_signature(ctx,dict(weights_sha='w',targets_sha='t'),again)==before
    assert np.array_equal(again['x'],expected)
