import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
torch.set_num_threads(1)
import voc_pair as study
from compute_probe import benchmark

def test_frozen_transfer_recipe():
    p=study.protocol()
    assert p['groups']==19 and p['width']==12 and p['top_k']==round(19*32/75)==8
    assert p['GGF_LR']==.001 and p['cap']==.6
    assert p['warm_updates']+p['guided_updates']==p['SLiCS_updates']==10640
    assert p['selection'].startswith('Primary: completed endpoints')
    assert 'registered' in p['compute']

def test_no_training_masks_in_inference_and_native_partial_is_not_joint():
    torch.manual_seed(1);D=torch.randn(24,5,dtype=torch.float64);D/=D.norm(dim=1,keepdim=True)
    x=torch.randn(4,5,dtype=torch.float64)
    partial=study.NativeSLiCS(D,study.SOLVER,'partial')
    z=partial.encode(x)
    assert z.shape==(4,2,12)
    pred,joint=partial(x)
    assert pred.shape==x.shape
    assert not torch.allclose(pred,z.flatten(1)@D)

def test_batched_partial_matches_scipy_and_certificate():
    from scipy.optimize import nnls
    torch.manual_seed(4);D=torch.randn(24,16,dtype=torch.float64);x=torch.randn(3,16,dtype=torch.float64)
    z,info=study.partial_nnls(x,D,study.SOLVER,12)
    assert info['max_relative_kkt']<=1e-5
    for i in range(3):
        for g in range(2):
            a=D[g*12:(g+1)*12].numpy();expected=nnls(a.T,x[i].numpy(),maxiter=20000)[0]
            assert np.linalg.norm(z[i,g].numpy()@a-x[i].numpy())==pytest.approx(np.linalg.norm(expected@a-x[i].numpy()),abs=1e-8)

def test_profile_counts_backward_and_does_not_mutate_saved_weights():
    original=torch.arange(12,dtype=torch.float32).reshape(3,4);before=original.clone()
    def factory():
        w=torch.nn.Parameter(original.clone());x=torch.ones(2,3)
        def call():
            (x@w).sum().backward();return w.grad
        return call,lambda:None
    r=benchmark(factory,2)
    assert torch.equal(original,before)
    f=r['flop_probes'][0]
    assert f['registered_operator_FLOPs']>=2*2*3*4*2
    assert f['full_FLOPs_available'] is False
    assert f['operators_without_registered_formula']
    assert r['seconds_SD']>=0 and len(r['seconds'])==3

def test_initialization_and_target_shapes_for_19_groups(monkeypatch):
    ctx=dict(cfg=dict(category_indices=list(range(19))),provenance=dict(categories=[dict(name=str(i)) for i in range(80)]))
    targets=study.fixed_targets(ctx)
    assert len(targets)==19
    assert [t['contrast_group'] for t in targets]==list(range(19))

def test_completed_endpoint_not_development_initializer(monkeypatch):
    class Store:
        def load(self,k):return dict(updates=10640,D=torch.ones(228,768),seconds=2.)
    monkeypatch.setattr(study,'native_fit',lambda *a:None)
    ctx=dict(store=Store(),cfg=dict(category_indices=list(range(19))),provenance=dict(categories=[dict(name=str(i)) for i in range(80)]))
    row=study.final_row(ctx,21,'SLiCS')
    assert row['key']=='native_slics12_s21' and row['selected_total_updates']==10640

def test_epoch_pairing_matches_both_methods():
    from supervised_fit import epoch_indices
    for u in [0,150,7639,7640,10639]:
        phase=int(u>=study.WARM);offset=u-phase*study.WARM
        np.testing.assert_array_equal(study.batch_indices(1208,21,u),epoch_indices(1208,21,offset,phase))

def test_invalid_source_manifest_fails_closed(tmp_path):
    from voc_inputs import unique_manifest
    p=tmp_path/'results/a';p.mkdir(parents=True)
    (p/'manifest.json').write_text(json.dumps(dict(run_id='invalid',config={})))
    with pytest.raises(ValueError,match='Invalid manifest'):unique_manifest(tmp_path,lambda c:True)

def test_native_interruption_preserves_and_resumes_committed_dictionary(tmp_path,monkeypatch):
    from test_supervised import fit_context,report
    ctx,_,_=fit_context(tmp_path)
    ctx['cfg']['category_indices']=list(range(19));ctx['provenance']=dict(categories=[dict(name=str(i)) for i in range(80)])
    torch.manual_seed(7);D=torch.randn(228,4);D/=D.norm(dim=1,keepdim=True);initial=D.clone()
    monkeypatch.setattr(study,'native_initializer',lambda ctx:D.clone())
    monkeypatch.setattr(study,'WARM',1);monkeypatch.setattr(study,'GUIDED',1);monkeypatch.setattr(study,'TOTAL',2)
    class Dummy:
        def __init__(self,*a):self.stats={}
    monkeypatch.setattr(study,'NativeSLiCS',Dummy)
    monkeypatch.setattr(study,'checked_native_evaluate',lambda *a:report())
    monkeypatch.setattr(study,'evaluate_state',lambda *a:report())
    real=study.atom_sweep
    def interrupted(x,D,z,allowed,guard):D[0].fill_(900.);raise KeyboardInterrupt()
    monkeypatch.setattr(study,'atom_sweep',interrupted)
    with pytest.raises(KeyboardInterrupt):study.native_fit(ctx,21)
    state=ctx['store'].load('native_slics12_s21')
    assert state['updates']==0 and torch.equal(state['D'],initial)
    monkeypatch.setattr(study,'atom_sweep',real)
    out=study.native_fit(ctx,21)
    assert ctx['store'].load('native_slics12_s21')['updates']==2
    assert out['slics_object_partial']['selected_total_updates']==0

def test_real_19_group_object_metrics_and_compact_audit(tmp_path):
    from test_supervised import fit_context
    ctx,_,_=fit_context(tmp_path)
    from coco_baselines import training
    from coco_ggf import matched_baselines as matched
    from scale_models import model
    cats=list(range(2));ctx['cfg']=study.memory.config('cpu')
    ctx['cfg'].update(category_indices=cats,detection_patches=196,preview_images=0,min_category_images=1,decision_min_images=1,bootstrap=0,fresh_test_seed=71002,audit_image_batch=1,max_audit_groups=4,batch=128)
    ctx['provenance']=dict(categories=[dict(name=str(i)) for i in range(80)])
    # The shared metric arithmetic must retain both categories and all 196 patches.
    for pack in ctx['packs'].values():
        pack['x']=np.pad(pack['x'],((0,0),(0,0),(0,12)))
        for i,mm in enumerate(pack['masks']):
            mm['fraction']=np.zeros((196,80),np.uint8);mm['fraction'][:10,:2]=255
            mm['instances']=[dict(category=c,annotation_id=i*100+c,fraction=np.r_[np.ones(10),np.zeros(186)]) for c in cats]
    m=model('released_vanilla',19,12,8,16)
    targets=[dict(category_index=c,category=str(c),contrast_group=c,detector_group=c,heat_mean=0.,heat_scale=1.) for c in cats]
    report=study.evaluate_model(m,ctx['packs']['select'],targets,ctx['cfg'],'cpu',compact=True)
    assert len(report['categories'])==2
    assert report['_compact']['category_indices']==cats
    assert report['reconstruction']['patches']==len(ctx['packs']['select']['ids'])*196
    assert np.isfinite([report['macro_capture'],report['macro_ap'],report['reconstruction']['ve']]).all()
