import copy,inspect
import numpy as np
import pytest
import torch
from scipy.optimize import nnls
import supervised_study as s
import object_schedule as o
from coco_ggf.slics_repair import Solver
from coco_baselines import training
from training_precision import JointPrecisionRetry,JOINT_FAILURE,precision_atom_sweep,patch_paths,revise
from test_precision_retry import rounding_fixture

torch.set_num_threads(2)

def patched(monkeypatch):
    for obj,key in [(s,'native_fit'),(s,'evaluate_model'),(s,'final_reconstruction'),
                    (s.NativeSLiCS,'forward'),(o,'final_reconstruction')]:
        monkeypatch.setattr(obj,key,getattr(obj,key))
    monkeypatch.setitem(s.__dict__,'_ggf_precision_atom_sweep',None)
    originals=patch_paths(s,o)
    retry=JointPrecisionRetry(Solver.solve,emit=lambda e:None)
    monkeypatch.setattr(Solver,'solve',lambda self,x,allowed=None,force_fallback=False:retry(self,x,allowed,force_fallback))
    return originals,retry

@pytest.mark.parametrize('force',[False,True])
def test_joint_rounding_failure_recovery_matches_scipy_and_support(force):
    x,D,cfg=rounding_fixture();allowed=torch.ones(len(x),len(D),dtype=torch.bool)
    allowed[-1]=False;original=Solver.solve;before=D.clone()
    solver=Solver(D,cfg)
    with pytest.raises(RuntimeError,match='float64 KKT'):original(solver,x,allowed,force_fallback=force)
    events=[];retry=JointPrecisionRetry(original,events.append)
    z,info=retry(solver,x,allowed,force)
    assert z.dtype==torch.float64 and torch.count_nonzero(z[-1])==0
    assert torch.equal(D,before) and torch.equal(solver.D,before)
    assert info['max_relative_kkt']<=cfg['nnls_tolerance'] and retry.retries==1
    for i in range(len(x)-1):
        answer,_=nnls(D.double().numpy().T,x[i].double().numpy(),maxiter=20000)
        np.testing.assert_allclose(z[i].numpy(),answer,atol=1e-10,rtol=1e-10)
    assert events[-1]['status']=='verified'

@pytest.mark.parametrize('force',[False,True])
def test_successful_joint_call_is_bitwise_unchanged(force):
    D=torch.eye(12,14);x=torch.randn(9,14,generator=torch.Generator().manual_seed(9))
    allowed=torch.rand((9,12),generator=torch.Generator().manual_seed(3))>.3
    solver=Solver(D,s.native_solver_config());expected,info=solver.solve(x,allowed,force_fallback=force)
    rng=torch.get_rng_state();retry=JointPrecisionRetry(Solver.solve)
    actual,new=retry(solver,x,allowed,force)
    assert torch.equal(actual,expected) and new==info and retry.retries==0
    assert torch.equal(torch.get_rng_state(),rng)

@pytest.mark.parametrize('message',['CUDA out of memory','NNLS active-set fallback failed; no unconverged result accepted','other'])
def test_joint_never_hides_other_failures(message):
    def fail(*args,**kw):raise RuntimeError(message)
    x,D,cfg=rounding_fixture()
    with pytest.raises(RuntimeError,match=message):JointPrecisionRetry(fail)(Solver(D,cfg),x)

def test_failed_double_joint_retry_still_stops():
    calls=[]
    def fail(self,*args,**kwargs):calls.append(self.D.dtype);raise RuntimeError(JOINT_FAILURE)
    x,D,cfg=rounding_fixture()
    with pytest.raises(RuntimeError,match='float64 KKT'):JointPrecisionRetry(fail)(Solver(D,cfg),x)
    assert calls==[torch.float32,torch.float64]
    calls.clear()
    with pytest.raises(RuntimeError):JointPrecisionRetry(fail)(Solver(D.double(),cfg),x.double())
    assert calls==[torch.float64]

def test_joint_certificate_is_recomputed_and_forbidden_support_rejected():
    x,D,cfg=rounding_fixture()
    def bad(self,x,allowed=None,force_fallback=False):
        if self.D.dtype==torch.float32:raise RuntimeError(JOINT_FAILURE)
        return torch.zeros(len(x),len(D),dtype=torch.float64),dict(max_relative_kkt=0.,cpu_fallback_rows=0)
    with pytest.raises(RuntimeError,match='threshold'):JointPrecisionRetry(bad)(Solver(D,cfg),x)
    def forbidden(self,x,allowed=None,force_fallback=False):
        if self.D.dtype==torch.float32:raise RuntimeError(JOINT_FAILURE)
        return torch.ones(len(x),len(D),dtype=torch.float64),dict(max_relative_kkt=0.,cpu_fallback_rows=0)
    with pytest.raises(RuntimeError,match='support'):
        JointPrecisionRetry(forbidden)(Solver(D,cfg),x,torch.zeros(len(x),len(D),dtype=torch.bool))

def test_atom_sweep_passes_original_tensors_or_uses_double_then_stores_float32():
    calls=[];x,D,cfg=rounding_fixture();allowed=torch.ones(len(x),len(D),dtype=torch.bool)
    def spy(a,b,c,d,e):calls.append((a,b,c,d,e));return b.clone(),dict(ok=True)
    z=torch.ones(len(x),len(D));a,_=precision_atom_sweep(spy,x,D,z,allowed,False)
    assert calls[0][0] is x and calls[0][1] is D and calls[0][2] is z
    a,_=precision_atom_sweep(spy,x,D,z.double(),allowed,False)
    assert all(v.dtype==torch.float64 for v in calls[1][:3]) and a.dtype==D.dtype
    assert torch.equal(a,D) and calls[1][3] is allowed and calls[1][4] is False

def setup_native(tmp_path,monkeypatch,stress):
    from test_supervised import fit_context,report
    ctx,_,_=fit_context(tmp_path)
    ctx['cfg']['category_indices']=list(range(75));ctx['provenance']=dict(categories=[dict(name=str(i)) for i in range(75)])
    x,D,cfg=rounding_fixture()
    if not stress:D=torch.eye(12,14)
    ctx['packs']['train']['x']=np.tile(x[2:3].numpy(),(16,196,1))
    fullD=D.repeat(75,1)
    monkeypatch.setattr(s,'native_initializer',lambda ctx:fullD.clone())
    monkeypatch.setattr(s,'native_solver_config',lambda:cfg)
    monkeypatch.setattr(s,'WARM',1);monkeypatch.setattr(s,'GUIDED',1);monkeypatch.setattr(s,'TOTAL',2)
    monkeypatch.setattr(s,'checked_native_evaluate',lambda *a:report())
    monkeypatch.setattr(s,'evaluate_state',lambda *a:report())
    return ctx

def test_native_fast_path_and_checkpoint_fingerprint_match_original(tmp_path,monkeypatch):
    a=setup_native(tmp_path/'old',monkeypatch,False)
    old=s.native_fit;old(a,21);expected=a['store'].load('native_slics12_s21')
    b=setup_native(tmp_path/'new',monkeypatch,False)
    _,retry=patched(monkeypatch);s.native_fit(b,21);actual=b['store'].load('native_slics12_s21')
    assert retry.retries==0 and torch.equal(expected['D'],actual['D'])
    for k in ['fingerprint','sampled_batches_sha','phase_batch_shas','sample_rng','updates','last_evaluated']:
        assert expected[k]==actual[k]

def test_actual_supported_training_recovery_and_interrupt_resume(tmp_path,monkeypatch):
    a=setup_native(tmp_path/'full',monkeypatch,True);_,retry=patched(monkeypatch)
    real_sweep=s.atom_sweep;seen=[]
    def record(x,D,z,allowed,guard):
        seen.append((x.dtype,D.dtype,z.dtype));return real_sweep(x,D,z,allowed,guard)
    monkeypatch.setattr(s,'atom_sweep',record)
    s.native_fit(a,21);full=a['store'].load('native_slics12_s21')
    assert retry.retries>=1 and seen[0]==(torch.float64,)*3 and full['D'].dtype==torch.float32
    b=setup_native(tmp_path/'resume',monkeypatch,True);initial=s.native_initializer(b)
    def stop(x,D,z,allowed,guard):D[0].fill_(999.);raise KeyboardInterrupt()
    monkeypatch.setattr(s,'atom_sweep',stop)
    with pytest.raises(KeyboardInterrupt):s.native_fit(b,21)
    saved=b['store'].load('native_slics12_s21')
    assert saved['updates']==0 and torch.equal(saved['D'],initial)
    monkeypatch.setattr(s,'atom_sweep',real_sweep);s.native_fit(b,21)
    resumed=b['store'].load('native_slics12_s21')
    assert torch.equal(full['D'],resumed['D'])
    for k in ['fingerprint','sampled_batches_sha','phase_batch_shas','sample_rng','updates','last_evaluated']:
        assert full[k]==resumed[k]

def test_double_joint_forward_and_complete_metric_pipeline(monkeypatch):
    from test_object_schedule import fixture
    patched(monkeypatch);ctx,pack,_,targets=fixture()
    # Orthogonal repeated dictionaries: force a verified double output to exercise
    # every dtype-sensitive consumer independently from the numerical fixture.
    model=s.NativeSLiCS(torch.eye(12).repeat(75,1),s.native_solver_config(),'partial')
    pack['x']=np.tile(pack['x'][:,:,:1],(1,1,12))
    def solve(x):
        z=torch.zeros(len(x),900,dtype=torch.float64);z[:,:12]=x.double().clamp_min(0);return z
    monkeypatch.setattr(model,'solve',solve)
    x=torch.as_tensor(pack['x'][0]);pred,z=model(x)
    assert pred.dtype==torch.float64 and torch.equal(pred,x.double().clamp_min(0))
    out=s.evaluate_model(model,pack,targets,ctx['cfg'],'cpu',reconstruction=True)
    assert len(out['categories'])==75 and np.isfinite(out['reconstruction']['ve'])

def test_double_joint_final_reconstruction_resumes_and_reuses_cache(tmp_path,monkeypatch):
    from test_supervised import Store
    patched(monkeypatch)
    rng=np.random.default_rng(5);x=rng.normal(size=(5,196,12)).astype(np.float32)
    pack=dict(x=x,ids=list(range(5)),identity='fixed');store=Store(tmp_path/'r')
    ctx=dict(store=store,device='cpu');model=s.NativeSLiCS(torch.eye(12),s.native_solver_config(),'joint')
    count=[0]
    def solve(batch):
        count[0]+=1
        if count[0]==3:raise KeyboardInterrupt()
        return batch.double().clamp_min(0)
    monkeypatch.setattr(model,'solve',solve)
    with pytest.raises(KeyboardInterrupt):o.final_reconstruction(ctx,model,pack,'fixed')
    state=next(v for k,v in store.values.items() if k.startswith('final_joint_'))
    assert state['completed_images']==2
    monkeypatch.setattr(model,'solve',lambda batch:batch.double().clamp_min(0))
    out=o.final_reconstruction(ctx,model,pack,'fixed')
    expected=np.square(x.astype(float)-np.maximum(x,0).astype(float)).sum()/x.size
    assert out['mse']==pytest.approx(expected,abs=1e-12) and out['patches']==980
    monkeypatch.setattr(model,'solve',lambda *a:(_ for _ in ()).throw(AssertionError('cached result rerun')))
    assert s.final_reconstruction(ctx,model,pack,'fixed')==out

def test_revision_rejects_unexpected_source_instead_of_guessing():
    with pytest.raises(ValueError,match='Unexpected source'):revise(s.native_fit,[('missing statement','')])

def test_install_guards_manifest_and_preserves_observers(tmp_path,monkeypatch):
    import training_precision as p
    from test_supervised import Store
    from supervised_live import Monitor,Observers
    from precision_retry import EXPECTED_RUN,BASE_SHA
    import json,os
    # Register changes for cleanup, then exercise the actual production installer.
    for obj,key in [(s,'native_fit'),(s,'evaluate_model'),(s,'final_reconstruction'),(s,'partial_nnls'),
                    (s,'prepare'),(s.NativeSLiCS,'forward'),(o,'final_reconstruction'),(Solver,'solve')]:
        monkeypatch.setattr(obj,key,getattr(obj,key))
    monkeypatch.setitem(s.__dict__,'_ggf_precision_atom_sweep',None)
    expected=json.loads((p.HERE/'ACTIVE_MANIFEST.json').read_text());output=tmp_path/'output'
    manifest=output/'results'/EXPECTED_RUN[:20]/'manifest.json';manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(expected));store=Store(tmp_path/'evidence');store.run_id=EXPECTED_RUN
    monkeypatch.setattr(s,'prepare',lambda options,require_drive=True:dict(store=store))
    p.install();observer=Observers(Monitor(emit=lambda message:None)).install()
    try:
        s.prepare(dict(output=str(output),wrapper_sha=BASE_SHA,wrapper_zip=os.environ['PRECISION_ORIGINAL_ZIP']))
        record=json.loads((store.results/'execution_training_precision_revision.json').read_text())
        assert record['original_run_id']==EXPECTED_RUN and record['tolerance_changed'] is False
        assert record['training_arithmetic_changed_on_precision_recovery_batches'] is True
        assert list((store.results/'execution_training_precision').rglob('training_precision.py'))
        x,D,cfg=rounding_fixture();z,info=Solver(D,cfg).solve(x)
        assert z.dtype==torch.float64 and info['max_relative_kkt']<=1e-5
    finally:observer.close()
