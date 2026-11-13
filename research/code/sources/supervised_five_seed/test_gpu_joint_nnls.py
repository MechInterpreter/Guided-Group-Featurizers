import inspect
import numpy as np
import pytest
import torch
from scipy.optimize import nnls
import gpu_joint_nnls as fast

CFG=dict(nnls_tolerance=1e-5,nnls_fallback_iterations=20000,nnls_iterations=2000)
torch.set_num_threads(2)


@pytest.mark.parametrize('shape,seed',[( (12,18),1),((24,32),2),((36,24),3)])
def test_active_set_matches_independent_full_nnls(shape,seed):
    rng=np.random.default_rng(seed);D=rng.normal(size=shape)
    solver=fast.SharedActiveSet(D)
    for _ in range(5):
        x=rng.normal(size=shape[1]);start=np.maximum(rng.normal(size=shape[0]),0)
        z,_=solver.solve(x,start);reference,_=nnls(D.T,x,maxiter=20000)
        assert (z>=0).all() and solver.error(z,x)<=1e-5
        assert abs(np.square(z@D-x).sum()-np.square(reference@D-x).sum())<1e-7*max(x@x,1)


def test_working_set_can_add_every_excluded_direction():
    D=np.eye(12);x=np.arange(1,13,dtype=float);solver=fast.SharedActiveSet(D)
    start=np.zeros(12);start[0]=1
    z,_=solver.solve(x,start)
    np.testing.assert_allclose(z,x,atol=1e-10)


@pytest.mark.parametrize('case',['duplicate','zero_atoms','near_duplicate','zero_input'])
def test_degenerate_and_inactive_inputs(case):
    rng=np.random.default_rng(9);D=rng.normal(size=(24,16));x=rng.normal(size=16)
    if case=='duplicate':D[12:]=D[:12]
    if case=='zero_atoms':D[:8]=0
    if case=='near_duplicate':D[12:]=D[:12]+rng.normal(size=(12,16))*1e-8
    if case=='zero_input':x[:]=0
    solver=fast.SharedActiveSet(D);z,_=solver.solve(x,np.ones(24))
    ref,_=nnls(D.T,x,maxiter=20000)
    assert solver.error(z,x)<=1e-5
    assert abs(np.square(z@D-x).sum()-np.square(ref@D-x).sum())<1e-7*max(x@x,1)


def test_batched_pipeline_original_space_certificate_and_inputs_immutable():
    rng=np.random.default_rng(12);D=torch.tensor(rng.normal(size=(24,32)),dtype=torch.float32)
    x=torch.tensor(rng.normal(size=(9,32)),dtype=torch.float32);before=D.clone();xb=x.clone()
    model=fast.GPUJointModel(D,CFG,device='cpu',iterations=120,check_every=20)
    z=model.solve(x)
    assert z.dtype==torch.float64 and torch.equal(D,before) and torch.equal(x,xb)
    assert float(fast.certificate(z,x,model.D).max())<=1e-5
    for a,b in zip(z.numpy(),x.double().numpy()):
        ref,_=nnls(D.double().numpy().T,b,maxiter=20000)
        assert abs(np.square(a@model.active.D-b).sum()-np.square(ref@model.active.D-b).sum())<1e-7*max(b@b,1)


def test_certified_rows_are_compacted_without_changing_order():
    D=torch.eye(12,16);x=torch.zeros((3,16));x[1,0]=3;x[2,2]=4
    model=fast.GPUJointModel(D,CFG,device='cpu',iterations=100,check_every=10)
    z,stats=model.warm_start(x)
    assert stats['gpu_certified']==3 and stats['iterations']==10
    torch.testing.assert_close(z,x[:,:12].double(),atol=1e-5,rtol=0)


def test_failed_cpu_refinement_cannot_commit_uncertified_solution(monkeypatch):
    D=torch.eye(12,14);model=fast.GPUJointModel(D,CFG,device='cpu',iterations=0)
    monkeypatch.setattr(model.active,'solve',lambda b,start:(np.zeros(12),False))
    with pytest.raises(RuntimeError,match='original per-row'):model.solve(torch.ones((2,14)))


def test_fixed_benchmark_checks_objective_with_independent_scipy():
    model=fast.GPUJointModel(torch.eye(12,14),CFG,device='cpu',iterations=30,check_every=10)
    out=fast.benchmark(model,torch.randn(8,14))
    assert out['patches']==8 and out['max_scaled_SSE_difference']<=1e-7 and out['max_relative_kkt']<=1e-5


def test_unchanged_threshold_is_mandatory():
    with pytest.raises(ValueError,match='tolerance'):fast.GPUJointModel(torch.eye(12),dict(CFG,nnls_tolerance=1e-3),device='cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime required; automatically executed in Colab')
def test_real_cuda_pipeline_matches_reference():
    torch.backends.cuda.matmul.allow_tf32=False
    rng=np.random.default_rng(123);D=torch.tensor(rng.normal(size=(36,24)),dtype=torch.float32)
    x=torch.tensor(rng.normal(size=(8,24)),dtype=torch.float32,device='cuda')
    model=fast.GPUJointModel(D,CFG,iterations=100,check_every=20)
    result=fast.benchmark(model,x)
    assert result['max_relative_kkt']<=1e-5 and result['max_scaled_SSE_difference']<=1e-7


def test_streamed_reconstruction_resumes_existing_prefix(tmp_path):
    from test_supervised import Store
    import object_schedule as schedule
    rng=np.random.default_rng(41);pack=dict(x=rng.normal(size=(3,196,14)).astype(np.float32),ids=[1,2,3],identity='pinned')
    ctx=dict(store=Store(tmp_path/'r'),norm={},device='cpu')
    model=fast.GPUJointModel(torch.eye(12,14),CFG,device='cpu',iterations=20,check_every=10)
    original=model.solve;calls=[0]
    def stop(x):
        calls[0]+=1
        if calls[0]==2:raise KeyboardInterrupt()
        return original(x)
    model.solve=stop
    with pytest.raises(KeyboardInterrupt):schedule.final_reconstruction(ctx,model,pack,'frozen')
    key=next(k for k in ctx['store'].values if k.startswith('final_joint_'))
    assert ctx['store'].load(key)['completed_images']==1
    model.solve=original;answer=schedule.final_reconstruction(ctx,model,pack,'frozen')
    assert answer['patches']==588 and ctx['store'].load(key)['complete']
    assert len([k for k in ctx['store'].values if k.startswith('final_joint_')])==1


def test_saved_native_dictionary_numerical_smoke():
    from pathlib import Path
    checkpoint=Path(__file__).resolve().parent/'native24_saved.pt'
    # Optional large integration fixture is local-only, never embedded in Colab.
    if not checkpoint.exists():pytest.skip('Local saved-dictionary fixture is not embedded')
    D=torch.load(checkpoint,map_location='cpu',weights_only=False)['D']
    rng=np.random.default_rng(81);z=np.zeros((2,900));z[:,rng.choice(900,24,replace=False)]=rng.random((2,24))
    x=torch.tensor(z@D.double().numpy()+rng.normal(size=(2,768))*.02,dtype=torch.float32)
    model=fast.GPUJointModel(D,CFG,device='cpu',iterations=100,check_every=50)
    result=model.solve(x)
    assert float(fast.certificate(result,x,model.D).max())<=1e-5


def test_worker_has_no_training_or_extraction_entrypoints():
    import gpu_finish_worker
    source=inspect.getsource(gpu_finish_worker)
    assert 'task.resume_cpu(options)' in source and 'task.cached_test_pack(ctx)' in source
    assert 'study.prepare(' not in source and 'study.native_fit(' not in source
    assert "object_schedule.final_reconstruction(ctx,model,pack,row['weights_sha'])" in source


@pytest.mark.parametrize('already_complete',[True,False])
def test_worker_reuses_full_results_or_resumes_saved_images(tmp_path,monkeypatch,already_complete):
    import json,sys
    from types import SimpleNamespace
    from pathlib import Path
    import objects_first as task
    import supervised_study as study
    import supervised_live
    import object_schedule as schedule
    import gpu_finish_worker as worker
    from test_supervised import Store
    from coco_baselines import training
    store=Store(tmp_path/'results');ctx=dict(store=store,cfg={},norm={},device='cpu',packs={})
    D=torch.eye(12,14);store.save('native',dict(D=D))
    row=dict(method='native_slics',seed=21,arm='slics_final_partial',key='native',solver=CFG,weights_sha=training.weight_sha({'D':D}))
    models=[dict(row=row,report=dict(reconstruction=dict(status='deferred',ve=None)))]
    rng=np.random.default_rng(42);pack=dict(x=rng.normal(size=(3,196,14)).astype(np.float32),ids=[4,5,6],identity='same')
    key='final_joint_'+task.identity(dict(weights=row['weights_sha'],ids=pack['ids'],cohort=pack['identity'],norm=task.identity({}),solver=CFG,arithmetic='image-streamed-float64-SSE-v1'))[:32]
    m=fast.GPUJointModel(D,CFG,device='cpu',iterations=20,check_every=10)
    prefix_x=torch.tensor(pack['x'][:2].reshape(-1,14));prefix_z=m.solve(prefix_x)
    active=prefix_z.reshape(-1,1,12).norm(dim=-1)>1e-6
    state=dict(completed_images=2,total_images=3,sse=float((prefix_z@m.D-prefix_x).square().sum()),patches=392,
        active=float(active.sum()),alive=active.any(0).numpy(),seconds=1.)
    if already_complete:state.update(complete=True,report=dict(status='computed',ve=.123,patches=588))
    store.save(key,state);store.save('objects_phase_complete',dict(complete=True))
    for seed in task.SEEDS:
        for a,b in [('matched_ggf','slics_object_partial'),('matched_ggf_final','slics_final_partial')]:
            store.save(f'image_ci_s{seed}_{a}_{b}',dict(saved=True))
    (tmp_path/'REVISION_HASHES.json').write_text('{}');revision=task.identity({})
    benchmark_key='GPU_benchmark_'+task.identity(dict(weights=row['weights_sha'],revision=revision,gpu='test device'))[:24]
    store.save(benchmark_key,dict(complete=True))
    monkeypatch.setattr(worker,'__file__',str(tmp_path/'gpu_finish_worker.py'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'test device')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(supervised_live,'install_observers',lambda:SimpleNamespace(monitor=supervised_live.Monitor()))
    monkeypatch.setattr(task,'resume_cpu',lambda options:ctx)
    monkeypatch.setattr(task,'load_object_models',lambda ctx:models)
    monkeypatch.setattr(task,'cpu_intervals',lambda *args:dict(reused=True))
    monkeypatch.setattr(task,'cached_test_pack',lambda ctx:pack)
    monkeypatch.setattr(task,'export_phase',lambda *args:tmp_path/'result.zip')
    seen=[];monkeypatch.setattr(study,'summarize',lambda ctx,completed,pairs:seen.extend(completed))
    constructors=[];original_model=fast.GPUJointModel
    def constructor(*args,**kwargs):
        constructors.append(1);return original_model(*args,device='cpu',iterations=20,check_every=10)
    monkeypatch.setattr(fast,'GPUJointModel',constructor)
    original=schedule.final_reconstruction;calls=[]
    def streamed(ctx,model,pack,sha):
        solve=model.solve
        def counted(x):calls.append(len(x));return solve(x)
        model.solve=counted
        return original(dict(ctx,device='cpu'),model,pack,sha)
    monkeypatch.setattr(schedule,'final_reconstruction',streamed)
    config=tmp_path/'config.json';receipt=tmp_path/'receipt.json'
    config.write_text(json.dumps(dict(expected_versions={},receipt=str(receipt),wrapper_zip='unchanged')))
    monkeypatch.setattr(sys,'argv',['worker','--config',str(config)])
    worker.main()
    assert json.loads(receipt.read_text())['complete']
    assert calls==([] if already_complete else [196])
    assert len(constructors)==(0 if already_complete else 1)
    assert len(seen)==1 and seen[0]['report']['reconstruction']['status']=='computed'
    assert store.load(key)['complete'] and len([k for k in store.values if k.startswith('final_joint_')])==1
