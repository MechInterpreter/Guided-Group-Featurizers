import copy,json,os
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.optimize import nnls
import supervised_study as s
from precision_retry import PrecisionRetry,FAILURE,BASE_SHA,EXPECTED_RUN,check_run,verify_source

torch.set_num_threads(2)

def rounding_fixture():
    # Opposing, nearly collinear atoms need two large positive coefficients.
    # Rounding their small difference to float32 violates the 1e-5 KKT check.
    D=torch.eye(12,14);D[1].zero_();D[1,0]=-1.;D[1,1]=.05;D[1]/=D[1].norm()
    x=torch.zeros(8,14);x[:,0]=torch.tensor([.01,.0123,.0234,.0345,.0456,.0567,.0678,.0789]);x[:,1]=1
    return x,D,dict(s.native_solver_config(),nnls_iterations=1)

def test_reproduces_failure_then_meets_same_strict_threshold_in_float64():
    x,D,cfg=rounding_fixture();old_x=x.clone();old_D=D.clone();old_cfg=copy.deepcopy(cfg)
    original=s.partial_nnls
    with pytest.raises(RuntimeError,match='strict KKT'):original(x,D,cfg,12)
    events=[];retry=PrecisionRetry(original,emit=events.append)
    z,info=retry(x,D,cfg,12)
    assert z.dtype==torch.float64 and info['max_relative_kkt']<=cfg['nnls_tolerance']
    assert len(events)==2 and events[-1]['status']=='verified'
    assert torch.equal(x,old_x) and torch.equal(D,old_D) and cfg==old_cfg
    for i in range(len(x)):
        expected,_=nnls(D.double().numpy().T,x[i].double().numpy(),maxiter=20000)
        np.testing.assert_allclose(z[i,0].numpy(),expected,rtol=1e-10,atol=1e-10)
        residual=(z[i,0]@D.double()-x[i].double());grad=residual@D.double().T
        error=(z[i,0]-(z[i,0]-grad).clamp_min(0)).abs().max()/(x[i].double()@D.double().T).abs().max()
        assert float(error)<=cfg['nnls_tolerance']

def test_passing_float32_call_is_bitwise_identical_and_does_not_advance_rng():
    torch.manual_seed(1);D=torch.randn(24,32);D/=D.norm(dim=1,keepdim=True);x=torch.randn(11,32)
    cfg=s.native_solver_config();original=s.partial_nnls
    before=torch.get_rng_state();expected,details=original(x,D,cfg,12)
    calls=[]
    def spy(*args):calls.append(args[0].dtype);return original(*args)
    retry=PrecisionRetry(spy);actual,info=retry(x,D,cfg,12)
    assert calls==[torch.float32] and retry.retries==0
    assert torch.equal(actual,expected) and actual.dtype==torch.float32 and info==details
    assert torch.equal(before,torch.get_rng_state())

@pytest.mark.parametrize('message',['CUDA out of memory','NNLS active-set fallback failed','unrelated runtime failure'])
def test_unrelated_errors_are_never_hidden(message):
    calls=[]
    def fail(*args):calls.append(1);raise RuntimeError(message)
    x,D,cfg=rounding_fixture()
    with pytest.raises(RuntimeError,match=message):PrecisionRetry(fail)(x,D,cfg,12)
    assert calls==[1]

def test_a_failed_float64_retry_still_stops_instead_of_accepting_bad_results():
    def fail(*args):raise RuntimeError(FAILURE)
    x,D,cfg=rounding_fixture()
    with pytest.raises(RuntimeError,match='strict KKT'):PrecisionRetry(fail)(x,D,cfg,12)

def test_existing_double_failure_is_not_retried_forever():
    calls=[]
    def fail(*args):calls.append(1);raise RuntimeError(FAILURE)
    x,D,cfg=rounding_fixture()
    with pytest.raises(RuntimeError):PrecisionRetry(fail)(x.double(),D.double(),cfg,12)
    assert calls==[1]

def test_retry_uses_identical_solver_settings_and_retains_double_output():
    x,D,cfg=rounding_fixture();calls=[]
    def original(a,b,c,width):
        calls.append((a.dtype,b.dtype,id(c),copy.deepcopy(c),width))
        if a.dtype==torch.float32:raise RuntimeError(FAILURE)
        return torch.zeros(len(a),1,width,dtype=torch.float64),dict(max_relative_kkt=0.,cpu_fallback_rows=len(a))
    z,info=PrecisionRetry(original)(x,D,cfg,12)
    assert z.dtype==torch.float64 and len(calls)==2
    assert calls[0][2:]==calls[1][2:]

@pytest.mark.parametrize('kind',['nonfinite','negative','threshold','wrong_precision'])
def test_invalid_retry_results_remain_rejected(kind):
    x,D,cfg=rounding_fixture()
    def original(a,b,c,width):
        if a.dtype==torch.float32:raise RuntimeError(FAILURE)
        z=torch.zeros(len(a),1,width,dtype=torch.float64)
        info=dict(max_relative_kkt=0.,cpu_fallback_rows=len(a))
        if kind=='nonfinite':z[0,0,0]=float('nan')
        elif kind=='negative':z[0,0,0]=-1
        elif kind=='threshold':info['max_relative_kkt']=c['nnls_tolerance']*2
        else:z=z.float()
        return z,info
    with pytest.raises(RuntimeError):PrecisionRetry(original)(x,D,cfg,12)

def test_full75_category_metric_pipeline_accepts_verified_double_coefficients(monkeypatch):
    from test_object_schedule import fixture
    ctx,pack,_,targets=fixture();x,D,cfg=rounding_fixture()
    # Same fixed complete category cohort, with deliberate numerical stress.
    pack['x']=np.zeros((2,196,14),np.float32);pack['x'][:,:,0]=.0123;pack['x'][:,:,1]=1
    pack['x'][:,:10,0]=.0234;pack['x'][:,:10,1]=2
    model=s.NativeSLiCS(D.repeat(75,1),cfg,'partial')
    retry=PrecisionRetry(s.partial_nnls,emit=lambda x:None)
    monkeypatch.setattr(s,'partial_nnls',retry)
    monkeypatch.setattr(model,'solve',lambda *a:(_ for _ in ()).throw(AssertionError('joint solve used')))
    report=s.evaluate_model(model,pack,targets,ctx['cfg'],'cpu',reconstruction=False)
    assert retry.retries>0 and len(report['categories'])==75
    assert np.isfinite([report['macro_capture'],report['macro_ap']]).all()
    assert report['reconstruction']['status']=='deferred'

def test_existing_run_guard_requires_exact_manifest_and_original_archive(tmp_path):
    # FILE_HASHES proves original arithmetic files were not edited in place.
    source=Path(s.__file__).parent;verify_source(source)
    original_zip=Path(os.environ['PRECISION_ORIGINAL_ZIP'])
    options=dict(wrapper_sha=BASE_SHA,wrapper_zip=str(original_zip),output=str(tmp_path))
    expected=json.loads((Path(__file__).parent/'ACTIVE_MANIFEST.json').read_text())
    target=tmp_path/'results'/EXPECTED_RUN[:20]/'manifest.json';target.parent.mkdir(parents=True)
    target.write_text(json.dumps(expected));check_run(options)
    changed=copy.deepcopy(expected);changed['run_id']='changed';target.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='differs'):check_run(options)
