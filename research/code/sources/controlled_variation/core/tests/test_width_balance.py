"""CPU protocol checks. The full cached-data CUDA study is not run here."""
import copy
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent/'source'))
from coco_baselines import training
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from coco_ggf import width_balance as experiment
from coco_ggf import width_balance_fit as engine
from coco_ggf import width_balance_warm as warm
from coco_ggf import large_mask_fit

torch.set_num_threads(1)


def store(tmp,name):
    return inputs.JournalStore(tmp/name,dict(test=name),tmp/(name+'_local'),require_drive=False)


def fixture(tmp,name,rank=3):
    rng=np.random.default_rng(5)
    masks=[]
    for i in range(5):
        f=np.zeros(196,np.float32);f[:12]=1
        masks.append(dict(crowd=np.zeros(196,np.float32),fraction=(f[:,None]*255).astype(np.uint8),
            instances=[dict(category=0,fraction=f,annotation_id=i)]))
    packs={k:dict(x=rng.normal(size=(5,196,8)).astype(np.float32),ids=list(range(5)),masks=masks) for k in ['train','select']}
    base=dict(groups=5,block_rank=rank,active_groups=2,sparsity_tolerance=.2,batch=8,epochs=2,warmup=.25,snr=.1,save_updates=4,local_save_updates=1)
    cfg=dict(lr=.001,snr=.1,grad_clip=5.,images_per_update=2,batch=16,evaluate_every=10,local_save_every=3,
        positive_fraction=.5,negative_fraction=.1,min_object_patch_mass=2,min_background_patches=4,device='cpu',
        score_ap_weight=.2,decision_min_images=1,category_indices=[0],updates=30,lambda_capture=.05,lambda_detection=.01,
        fit_reconstruction_tolerance=.02,diagnostic_grad_every=10000)
    targets=[dict(category='object',category_index=0,detector_group=0,contrast_group=0,heat_mean=0.,heat_scale=1.)]
    return dict(store=store(tmp,name),base=base,cfg=cfg,device='cpu',packs=packs,recipe=experiment.settings()),targets


def cheap_validation(model,pack,targets,cfg,device):
    r=training.measure(model,pack['x'].reshape(-1,model.d),device,cfg['batch'])
    return dict(reconstruction=r,macro_capture=.2,macro_ap=.6,score=.32)


@pytest.mark.parametrize('rank,balanced,supervised',[(3,False,True),(4,False,True),(3,True,True),(4,True,True),(4,False,False)])
def test_five_arms_resume_exactly(tmp_path,monkeypatch,rank,balanced,supervised):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'full',rank);b,_=fixture(tmp_path,'resume',rank)
    torch.manual_seed(21);initial=training.weights(training.make_model('vanilla',a['base'],8))
    full=engine.fit(a,'fit',initial,t,30,71021,balanced,supervised,reference_mse=1.)
    part=engine.fit(b,'fit',initial,t,30,71021,balanced,supervised,reference_mse=1.,max_updates=23)
    assert part['updates']==23
    resumed=engine.fit(b,'fit',initial,t,30,71021,balanced,supervised,reference_mse=1.)
    for field in ['weights','optimizer','balance']:
        assert engine.state_identity(full[field])==engine.state_identity(resumed[field])
    assert torch.equal(full['head'],resumed['head'])
    assert torch.equal(full['torch_rng'],resumed['torch_rng'])
    assert full['sampled_batches_sha']==resumed['sampled_batches_sha']
    assert full['sampled_noise_sha']==resumed['sampled_noise_sha']
    if balanced:assert full['balance']['adaptations']==2
    with pytest.raises(ValueError,match='identity changed'):
        engine.fit(b,'fit',initial,t,31,71021,balanced,supervised,reference_mse=1.)


@pytest.mark.parametrize('supervised',[True,False])
def test_fixed_loss_matches_original_engine(tmp_path,monkeypatch,supervised):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'new');b,_=fixture(tmp_path,'old')
    a['cfg']['updates']=b['cfg']['updates']=6
    torch.manual_seed(21);initial=training.weights(training.make_model('vanilla',a['base'],8))
    new=engine.fit(a,'fit',initial,t,6,77,False,supervised,reference_mse=1.)
    old=large_mask_fit.fit_guided(b,'fit',initial,t,supervised,77,reference_mse=1.)
    assert training.weight_sha(new['weights'])==training.weight_sha(old['weights'])
    assert torch.equal(new['head'],old['head'])
    assert new['sampled_batches_sha']==old['sampled_batches_sha']
    assert torch.equal(new['torch_rng'],old['torch_rng'])


def test_balancer_anchor_limits_reference_and_zero_gradient():
    recipe=experiment.settings()['balance'];b=engine.Balancer(recipe,'cpu')
    losses=[torch.tensor(1.),torch.tensor(2.),torch.tensor(3.)]
    for _ in range(20):b.observe(losses)
    assert b.reference_count==20 and np.array_equal(b.reference,[1.,2.,3.])
    assert not b.adapt(losses,torch.zeros(3))['updated']
    for _ in range(600):b.adapt(losses,torch.tensor([.1,.001,.0001]))
    w=b.weights().detach().numpy()
    assert w[0]==1 and .0125-1e-8<=w[1]<=.2+1e-8 and .0025-1e-8<=w[2]<=.2+1e-8
    assert b.adaptations==600
    with pytest.raises(FloatingPointError):b.adapt(losses,torch.tensor([1.,float('nan'),1.]))


def test_balancer_can_increase_weights_from_the_lower_boundary():
    recipe=experiment.settings()['balance'];b=engine.Balancer(recipe,'cpu')
    with torch.no_grad():b.log_weights.copy_(torch.tensor(np.log([.0125,.0025]),dtype=torch.float32))
    b.reference=np.ones(3)
    before=b.weights().detach().clone()
    b.adapt([torch.tensor(1.)]*3,torch.tensor([1.,.001,.001]))
    assert (b.weights().detach()[1:]>before[1:]).all()


def test_shared_gradient_norm_excludes_decoder_and_head(tmp_path):
    ctx,_=fixture(tmp_path,'grad');m=training.make_model('vanilla',ctx['base'],8)
    head=torch.nn.Parameter(torch.ones(2))
    l1=m.W_enc.sum()+m.b_enc.sum()
    l2=l1+100*m.W_dec.sum()+10000*head.sum()
    norms,_=engine.encoder_diagnostics(m,[l1,l2,l1*0])
    assert norms[0]==norms[1] and norms[2]==0
    assert m.W_dec.grad is None and head.grad is None


def test_midstep_balancer_failure_replays_complete_slot(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'validation',cheap_validation)
    a,t=fixture(tmp_path,'interrupted');b,_=fixture(tmp_path,'reference')
    for ctx in [a,b]:ctx['recipe']['balance'].update(reference_batches=1,every=1)
    torch.manual_seed(21);initial=training.weights(training.make_model('vanilla',a['base'],8))
    adapt=engine.Balancer.adapt
    def interrupt(self,*args):
        adapt(self,*args)
        raise KeyboardInterrupt('after weight mutation, before model update')
    monkeypatch.setattr(engine.Balancer,'adapt',interrupt)
    with pytest.raises(KeyboardInterrupt):engine.fit(a,'fit',initial,t,6,77,True,True,reference_mse=1.)
    saved=a['store'].load('fit')
    assert saved['updates']==0 and saved['balance']['adaptations']==0
    assert training.weight_sha(saved['weights'])==training.weight_sha(initial)
    monkeypatch.setattr(engine.Balancer,'adapt',adapt)
    resumed=engine.fit(a,'fit',initial,t,6,77,True,True,reference_mse=1.)
    full=engine.fit(b,'fit',initial,t,6,77,True,True,reference_mse=1.)
    assert engine.state_identity(resumed['balance'])==engine.state_identity(full['balance'])
    assert training.weight_sha(resumed['weights'])==training.weight_sha(full['weights'])


def test_second_phase_keeps_head_and_balancer_and_guards_entire_state(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'validation',cheap_validation)
    ctx,t=fixture(tmp_path,'phases')
    torch.manual_seed(21);initial=training.weights(training.make_model('vanilla',ctx['base'],8))
    first=engine.fit(ctx,'phase1',initial,t,30,71021,True,True,reference_mse=1.)
    second=engine.fit(ctx,'phase2',first['weights'],t,5,82021,True,True,reference_mse=1.,initial_head=first['head'],initial_balance=first['balance'])
    assert second['balance']['adaptations']==first['balance']['adaptations']
    assert second['balance']['reference_count']==20
    assert not torch.equal(second['head'],torch.zeros_like(second['head']))
    changed=copy.deepcopy(first['balance']);changed['reference_count']+=1
    with pytest.raises(ValueError,match='identity changed'):
        engine.fit(ctx,'phase2',first['weights'],t,5,82021,True,True,reference_mse=1.,initial_head=first['head'],initial_balance=changed)


def test_width_four_warm_pairing_and_resume(tmp_path):
    a,_=fixture(tmp_path,'unused',3);b,_=fixture(tmp_path,'unused4',4)
    # CPU noise consumes this RNG too; pairing must survive different model widths.
    rng=np.random.default_rng(5);x=rng.normal(size=(32,8)).astype(np.float32)
    pair=experiment.paired_warm_rng(a,21)
    old=large_mask_fit.fit_warm(store(tmp_path,'warm3'),'fit','vanilla',x,x[:16],a['base'],21,.003,'cpu')
    full=warm.fit_warm(store(tmp_path,'warm4'),'fit','vanilla',x,x[:16],b['base'],21,.003,'cpu',pair_rng=pair)
    resumed_store=store(tmp_path,'warm4_resume')
    warm.fit_warm(resumed_store,'fit','vanilla',x,x[:16],b['base'],21,.003,'cpu',pair_rng=pair,max_updates=3)
    resumed=warm.fit_warm(resumed_store,'fit','vanilla',x,x[:16],b['base'],21,.003,'cpu',pair_rng=pair)
    assert old['sampled_batches_sha']==full['sampled_batches_sha']==resumed['sampled_batches_sha']
    assert training.weight_sha(full['weights'])==training.weight_sha(resumed['weights'])
    assert full['scheduler']==resumed['scheduler']


def test_vanilla_initialization_and_mask_width_capacity():
    base=dict(groups=5,block_rank=4,active_groups=2)
    D=torch.eye(8)[:4]
    mask,vanilla=experiment.initialize(base,D,21)
    assert torch.equal(mask['W_dec'][:4],D)
    assert torch.equal(mask['W_dec'][4:],vanilla['W_dec'][4:])
    assert torch.equal(mask['W_enc'],mask['W_dec'].T)
    assert mask['W_dec'].shape==(20,8) and mask['b_enc'].eq(0).all()


def test_fourth_direction_preserves_saved_axes_and_rejects_wrong_subspace():
    gram=np.diag([8.,7.,6.,5.,4.,3.,2.,1.])
    saved=np.eye(8)[:3].copy();before=saved.copy()
    direction,eig,alignment=experiment.orthogonal_fourth(gram,saved)
    assert np.array_equal(saved,before)
    assert np.allclose(saved@direction,0.) and np.linalg.norm(direction)==pytest.approx(1.)
    assert abs(direction[3])==pytest.approx(1.) and alignment.min()==pytest.approx(1.)
    with pytest.raises(ValueError,match='eigenspace disagrees'):experiment.orthogonal_fourth(gram,np.eye(8)[4:7])


@pytest.mark.parametrize('rank',[3,4])
def test_real_loss_validation_and_paired_audit_are_consistent(tmp_path,rank):
    from coco_baselines import metrics
    from coco_ggf import study, matched_baselines
    ctx,t=fixture(tmp_path,'audit',rank)
    ctx['cfg'].update(audit_image_batch=2,patch_sample_seed=4,detection_patches=196,preview_images=0,bootstrap=10,fresh_test_seed=77)
    m=training.make_model('vanilla',ctx['base'],8)
    pack=ctx['packs']['train'];x=torch.as_tensor(pack['x']);head=torch.nn.Parameter(torch.zeros(1,2))
    cap,det,_=core.supervised_loss(m,x,pack['masks'],t,head,ctx['cfg'])
    assert torch.isfinite(cap+det)
    cap.backward(retain_graph=True)
    assert m.W_dec.grad is not None and m.W_dec.grad.norm()>0
    m.zero_grad();det.backward()
    assert m.W_enc.grad is not None and head.grad is not None
    result=core.validation(m,pack,t,ctx['cfg'],'cpu',with_arrays=True)
    compact=study.compact_audit(result['_audit'],matched_baselines.mapping(t),ctx['cfg'])
    reconstructed_capture=1-compact['error'].sum()/compact['denominator'].sum()
    assert result['macro_capture']==pytest.approx(reconstructed_capture,abs=1e-6)
    ci=study.paired(compact,compact,ctx['cfg'])
    assert ci['capture_gain_ci95']==pytest.approx([0.,0.]) and ci['ap_gain_ci95']==pytest.approx([0.,0.])


def test_freeze_barrier_and_catalog_required_control(tmp_path):
    s=store(tmp_path,'barrier')
    with pytest.raises(FileNotFoundError,match='all_models_frozen'):experiment.load_test(dict(store=s))
    with pytest.raises(ValueError,match='fifteen'):experiment.verify_catalog({},dict(complete=True,models=[]))
    fake=[dict(seed=seed,arm=arm) for seed in experiment.SEEDS for arm in experiment.ARMS]
    fake[0]['arm']='wrong'
    with pytest.raises(ValueError,match='fifteen'):experiment.verify_catalog({},dict(complete=True,models=fake))


def test_reports_export_all_arms_seeds_and_failures(tmp_path):
    s=store(tmp_path,'report');reports=[]
    for seed in experiment.SEEDS:
        for arm,(rank,balanced,supervised,label) in experiment.ARMS.items():
            reports.append(dict(seed=seed,arm=arm,label=label,rank=rank,origin='new',macro_capture=-.05 if arm=='vanilla4' else .3,
                macro_ap=.6,reconstruction=dict(ve=.65,mse=.35),core_parameters=100,head_parameters=150 if supervised else 0,
                maximum_active_coordinates=rank*16,legacy_native_eligible=False,activity=dict(mean_active_groups=16),
                categories=[dict(category_index=0,category='object',capture=.2,ap=.6)]))
    ci=dict(capture_gain_ci95=[-.01,.02],ap_gain_ci95=[-.01,.02])
    comparisons=[dict(seed=21,treatment='mask4_balanced',control='vanilla4',capture_gain=.35,ap_gain=0.,ve_gain=0.,
        paired_intervals=ci,all_three_point_estimates_improve=False,reconstruction_guard_passed=False)]
    experiment.tables_and_plots(dict(store=s),dict(models=reports,comparisons=comparisons))
    with (s.results/'comparison.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==15 and {r['seed'] for r in rows}=={'21','22','23'}
    assert all(float(r['capture_pct'])==-5 for r in rows if r['arm']=='vanilla4')
    assert (s.results/'figures/width_balance.pdf').stat().st_size>1000
    s.write('finite.json',experiment.json_safe(dict(v=np.array([1.,float('nan')]),t=torch.tensor(float('inf')))))
    assert json.loads((s.results/'finite.json').read_text())==dict(v=[1.,None],t=None)
    package=tmp_path/'source.zip';package.write_bytes(b'source bytes')
    path=experiment.export(dict(store=s,sources={}),package)
    assert path.name.startswith('ggf_width_balance_')
    with zipfile.ZipFile(path) as z:
        assert not any(n.endswith('.pt') for n in z.namelist())
        for name,digest in json.loads(z.read('artifact_hashes.json')).items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
