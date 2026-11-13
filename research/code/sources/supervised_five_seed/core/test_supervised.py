import copy,hashlib,inspect,json
from pathlib import Path
import numpy as np
import pytest
import torch
torch.set_num_threads(2)
from scipy.optimize import nnls
import supervised_study as s
import supervised_fit as f
from coco_baselines import training
from coco_baselines.storage import identity
from coco_ggf import width_balance
from coco_adaptation.inputs import JournalStore

class Store:
    def __init__(self,path):
        self.values={};self.results=path;path.mkdir(parents=True,exist_ok=True);self.run_id='f'*64
        self.output=path.parent;self._width_drive_recovery_original_save=self.save
    def load(self,key):return copy.deepcopy(self.values.get(key))
    def save(self,key,v,*args,**kw):self.values[key]=copy.deepcopy(v)
    def write(self,key,v):
        p=self.results/key;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return p
    def event(self,*a,**kw):pass

def report(capture=.3,ap=.5,mse=.2):
    return dict(macro_capture=capture,macro_ap=ap,reconstruction=dict(mse=mse,ve=.7,active_groups=2.))

def trial(lr,cap,ap,mse):
    return dict(lr=lr,models=[dict(seed=z,development=report(cap,ap,mse)) for z in s.SCREEN])

def masks(category=0):
    frac=np.zeros((196,75),np.uint8);frac[:98,category]=255
    return dict(fraction=frac,crowd=np.zeros(196),instances=[dict(category=category,annotation_id=4,fraction=np.r_[np.ones(98),np.zeros(98)])])

def test_only_supervised_default_scope_and_fresh_protocol():
    p=s.protocol();assert p['new_bsf_controls'] is False and p['new_detection_only_bsf'] is False
    assert p['seeds']==list(range(21,28)) and p['current_capacity_anchor_seeds']==[21,22,23]
    assert len(s.expected_pairs())==31
    assert 'released_vanilla' not in s.ARMS and 'supervised_bsf' not in inspect.getsource(s.selection)
    assert p['primary_contrasts']==[['matched_ggf','slics_object_partial']]

def test_development_selector_not_mse_or_test_and_does_not_drop_starts():
    a=trial(.0001,.3,.5,.1);b=trial(.0003,.4,.6,.3)
    assert s.select_trials([a,b])['lr']==.0003
    with pytest.raises(ValueError,match='removed'):s.select_trials([a,b],'mse')
    bad=trial(.001,.9,.9,.01);bad['models'].pop()
    with pytest.raises(ValueError,match='three'):s.select_trials([a,bad])
    bad=trial(.001,.9,float('nan'),.01)
    with pytest.raises(ValueError,match='Nonfinite'):s.select_trials([bad])

def test_deterministic_disjoint_epochs_and_phase_alignment():
    n=32;epoch=np.concatenate([f.epoch_indices(n,21,u,0) for u in range(4)])
    assert sorted(epoch)==list(range(n))
    assert np.array_equal(f.epoch_indices(n,21,3,0),s.batch_indices(n,21,3))
    assert np.array_equal(f.epoch_indices(n,21,0,1),s.batch_indices(n,21,s.WARM))
    assert not np.array_equal(f.epoch_indices(n,21,0,0),f.epoch_indices(n,22,0,0))
    with pytest.raises(ValueError):f.epoch_indices(3,21,0,0)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_vectorized_partial_nnls_matches_independent_scipy(dtype):
    torch.manual_seed(16);D=torch.randn(24,32,dtype=dtype);D/=D.norm(dim=1,keepdim=True)
    x=torch.randn(6,32,dtype=dtype);cfg=s.native_solver_config()
    z,info=s.partial_nnls(x,D,cfg,12)
    assert info['max_relative_kkt']<=1e-5 and (z>=0).all()
    for i in range(len(x)):
        for g in range(2):
            expected,_=nnls(D[g*12:(g+1)*12].double().numpy().T,x[i].double().numpy())
            np.testing.assert_allclose(z[i,g].double().numpy(),expected,atol=3e-5,rtol=3e-5)

def test_partial_fallback_is_verified_and_zero_input_exact():
    torch.manual_seed(19);D=torch.randn(24,30);x=torch.randn(3,30)
    cfg=dict(s.native_solver_config(),nnls_iterations=1)
    z,info=s.partial_nnls(x,D,cfg,12)
    assert info['cpu_fallback_rows']>0 and info['max_relative_kkt']<=1e-5
    zero,_=s.partial_nnls(torch.zeros_like(x),D,cfg,12);assert torch.count_nonzero(zero)==0

def test_native_partial_projections_not_summed_for_reconstruction():
    torch.manual_seed(11);D=torch.randn(24,28);D/=D.norm(dim=1,keepdim=True);x=torch.randn(3,28)
    m=s.NativeSLiCS(D,s.native_solver_config(),'partial');partial=m.encode(x);pred,joint=m(x)
    assert not torch.allclose(partial.flatten(1)@D,pred,atol=1e-4)
    for i in range(len(x)):
        expected,_=nnls(D.double().numpy().T,x[i].double().numpy())
        np.testing.assert_allclose(pred[i].numpy(),expected@D.numpy(),atol=4e-5,rtol=4e-5)
    assert m.stats['readout_rows']==3 and m.stats['readout_active_group_sum']>0

def test_streamed_global_variance_matches_direct_computation():
    rng=np.random.default_rng(33);x=rng.normal(size=(4,196,7)).astype('float32');p=dict(x=x)
    flat=x.reshape(-1,7).astype(float);expected=np.square(flat-flat.mean(0)).sum()
    assert s.pack_variance(p)==pytest.approx(expected,rel=1e-12)
    assert '_global_variance' in p

def test_holdout_gate_requires_complete_freeze(tmp_path,monkeypatch):
    store=Store(tmp_path/'results');ctx=dict(store=store)
    monkeypatch.setattr(s,'load_holdout',lambda *a:(_ for _ in ()).throw(AssertionError('test opened')))
    with pytest.raises(FileNotFoundError):s.extract_after_freeze(ctx,None)

def test_freeze_rejects_missing_seeds_and_changed_protocol(tmp_path):
    store=Store(tmp_path/'results');store.save('all_models_frozen',dict(complete=True,protocol=s.protocol(),models=[]))
    with pytest.raises(ValueError,match='Incomplete'):s.verify_frozen(dict(store=store))

def test_holm_adjustment_and_paired_start_statistics():
    assert s.holm([.01,.04])==pytest.approx([.02,.04])
    assert s.holm([.6,.2,.1])==pytest.approx([.6,.4,.3])
    stat=s.seed_stat([1.,2.,3.,4.,5.,6.,7.]);assert stat['mean']==4 and stat['ci_low']<4<stat['ci_high']
    assert s.seed_stat([0.,0.])['p_two_sided']==1.
    with pytest.raises(ValueError):s.seed_stat([1.])

def test_no_large_anchor_training_fallback(tmp_path):
    with pytest.raises(ValueError,match='no new large'):s.current(dict(pilot_rows=[]),24,'agf_g')
    with pytest.raises(ValueError,match='only supervised'):s.current(dict(pilot_rows=[]),21,'released_vanilla')

def test_fixed_readouts_cannot_relabel_categories():
    ctx=dict(cfg=dict(category_indices=list(range(75))),provenance=dict(categories=[dict(name=str(i)) for i in range(75)]))
    t=s.fixed_targets(ctx);assert len(t)==75
    assert all(x['category_index']==x['contrast_group']==x['detector_group'] for x in t)

def fit_context(tmp_path,seed=21):
    torch.manual_seed(seed);rng=np.random.default_rng(62)
    x=rng.normal(size=(16,196,4)).astype(np.float32);p=dict(x=x,ids=list(range(16)),masks=[masks() for _ in range(16)],identity='train')
    balance=width_balance.settings()['balance'];balance.update(reference_batches=1,every=1)
    store=Store(tmp_path/'results')
    cfg=dict(positive_fraction=.5,negative_fraction=.1,min_object_patch_mass=1.,min_background_patches=2)
    ctx=dict(store=store,device='cpu',packs=dict(train=p,select=dict(p,identity='select')),cfg=cfg,
        balance_recipe=balance,rule=dict(lr=.0003,width=2,snr=0.,clip=5.,images_per_update=8,evaluate_every=1,local_save_every=1,sampling='epochs'))
    m=s.model('released_vanilla',4,2,2,4);weights=training.weights(m)
    targets=[dict(category_index=0,category='a',contrast_group=0,detector_group=0,heat_mean=0.,heat_scale=1.)]
    return ctx,weights,targets

def test_real_adam_controller_resume_equals_uninterrupted(tmp_path,monkeypatch):
    # Numerical training is real; metric evaluation is replaced to isolate exact state replay.
    monkeypatch.setattr(f,'evaluate',lambda *a,**k:report())
    a,w,t=fit_context(tmp_path/'a');b,_,_=fit_context(tmp_path/'b')
    full=f.fit(a,'g','released_vanilla',4,2,21,w,4,4,targets=t,cap=.6,stream=1,keep_candidates=True)
    f.fit(b,'g','released_vanilla',4,2,21,w,4,2,targets=t,cap=.6,stream=1,keep_candidates=True)
    resumed=f.fit(b,'g','released_vanilla',4,2,21,w,4,4,targets=t,cap=.6,stream=1,keep_candidates=True)
    assert training.weight_sha(full['weights'])==training.weight_sha(resumed['weights'])
    assert full['sampled_batches_sha']==resumed['sampled_batches_sha']
    assert full['sampled_noise_sha']==resumed['sampled_noise_sha']
    assert torch.equal(full['head'],resumed['head']) and torch.equal(full['balance']['log_weights'],resumed['balance']['log_weights'])
    assert full['last_evaluated']==4 and b['store'].load('g_object')['updates']==0

def test_real_complete75_category_audit_matches_existing_metrics(tmp_path):
    from scale_metrics import evaluate as original_evaluate
    from coco_ggf.slics_repair import Model
    # Configuration is the same small complete-cohort fixture as numerical repair tests.
    cfg=s.memory.config('cpu');cfg.update(category_indices=list(range(75)),detection_patches=196,
        preview_images=0,min_category_images=1,decision_min_images=1,bootstrap=0,fresh_test_seed=71002,
        audit_image_batch=1,max_audit_groups=4,batch=128)
    rng=np.random.default_rng(27);x=rng.normal(size=(3,196,75)).astype(np.float32);mm=[]
    for image in range(3):
        frac=np.zeros((196,75),np.uint8);frac[:10]=255
        instances=[dict(category=c,annotation_id=image*100+c,fraction=np.r_[np.ones(10),np.zeros(186)]) for c in range(75)]
        mm.append(dict(fraction=frac,crowd=np.zeros(196),instances=instances))
    pack=dict(x=x,masks=mm,ids=[10,11,12]);recipe=dict(nnls_iterations=100,nnls_tolerance=1e-5,nnls_fallback_iterations=20000,rank=1,active_groups=16)
    m=Model(torch.eye(75),recipe,'dense')
    targets=[dict(category_index=c,category=str(c),contrast_group=c,detector_group=c,heat_mean=0.,heat_scale=1.) for c in range(75)]
    a=s.evaluate_model(m,pack,targets,cfg,'cpu',compact=True);b=original_evaluate(m,pack,targets,cfg,'cpu',compact=True)
    assert a['macro_capture']==pytest.approx(b['macro_capture'],abs=1e-10)
    assert a['macro_ap']==pytest.approx(b['macro_ap'],abs=1e-10)
    assert a['reconstruction']['ve']==pytest.approx(b['reconstruction']['ve'],abs=1e-10)
    assert len(a['categories'])==75 and a['_compact']['category_indices']==list(range(75))

def test_native_atom_interrupt_keeps_committed_dictionary_and_resumes(tmp_path,monkeypatch):
    ctx,_,_=fit_context(tmp_path)
    ctx['cfg']['category_indices']=list(range(75));ctx['provenance']=dict(categories=[dict(name=str(i)) for i in range(75)])
    torch.manual_seed(7);D=torch.randn(900,4);D/=D.norm(dim=1,keepdim=True);initial=D.clone()
    monkeypatch.setattr(s,'native_initializer',lambda ctx:D.clone())
    monkeypatch.setattr(s,'WARM',1);monkeypatch.setattr(s,'GUIDED',1);monkeypatch.setattr(s,'TOTAL',2)
    class Dummy:
        def __init__(self,*a):self.stats={}
    monkeypatch.setattr(s,'NativeSLiCS',Dummy)
    monkeypatch.setattr(s,'checked_native_evaluate',lambda *a:report())
    monkeypatch.setattr(s,'evaluate_state',lambda *a:report())
    real=s.atom_sweep
    def interrupted(x,D,z,allowed,guard):D[0].fill_(900.);raise KeyboardInterrupt()
    monkeypatch.setattr(s,'atom_sweep',interrupted)
    with pytest.raises(KeyboardInterrupt):s.native_fit(ctx,21)
    state=ctx['store'].load('native_slics12_s21')
    assert state['updates']==0 and torch.equal(state['D'],initial)
    monkeypatch.setattr(s,'atom_sweep',real);out=s.native_fit(ctx,21)
    assert len(out)==1 and ctx['store'].load('native_slics12_s21')['updates']==2
    assert out['slics_object_partial']['selected_total_updates']==0
    done=ctx['store'].load('native_slics12_s21');empty=hashlib.sha256().hexdigest()
    warm_ids=np.random.default_rng(np.random.SeedSequence([126000,21,0,0])).permutation(16)[:8]
    guided_ids=np.random.default_rng(np.random.SeedSequence([126000,21,1,0])).permutation(16)[:8]
    assert done['phase_batch_shas']['warm']==hashlib.sha256(empty.encode()+warm_ids.tobytes()).hexdigest()
    assert done['sampled_batches_sha']==hashlib.sha256(empty.encode()+guided_ids.tobytes()).hexdigest()

def test_stage_order_freezes_before_extraction_and_only_supervised():
    import supervised_worker
    src=inspect.getsource(supervised_worker.main)
    assert src.index('study.selection(ctx)')<src.index('study.freeze(ctx,choice)')<src.index('study.extract_after_freeze(ctx,coco)')
    assert 'tune_baselines' not in src and 'focused_run' not in src

def test_independent_plan_retains_the_verified_unopened_reservation(monkeypatch):
    marker=object();ctx=dict()
    monkeypatch.setattr(s,'reserve_predecessor_holdout',lambda supplied,coco:marker if supplied is ctx else None)
    assert s.plan_fresh(ctx,None) is marker

def test_current_anchor_is_saved_fixed_budget_and_does_not_train():
    row=dict(seed=21,kind='agf_g',total_training_updates=10640,development=report())
    out=s.current(dict(pilot_rows=[row]),21,'agf_g')
    assert out['arm']=='current_ggf' and out['selected_total_updates']==10640
    assert 'arm' not in row

def test_full_current_catalog_is_checked_against_minimal_weight_pins(tmp_path):
    rows=[];pins={};targets=[]
    for seed in s.SCREEN:
        for kind in ['agf_g','released_vanilla']:
            row=dict(key=f'g4096_k32_s{seed}_{kind}',seed=seed,kind=kind,groups=4096,rank=12,active_groups=32,
                additional_updates=3000,total_training_updates=10640,method='released_vanilla',coefficient=.001,lr=.0003,
                cap=.6 if kind=='agf_g' else None,targets=targets,targets_sha=identity(targets),weights_sha=str(seed))
            rows.append(row)
            if kind=='agf_g':pins[row['key']]={k:row[k] for k in ['weights_sha','targets_sha','seed','kind']}
    p=tmp_path/'priority_pilot_summary.json';p.write_text(json.dumps(dict(complete=True,models=rows,final_test_opened=False,seven_seed_started=False)))
    selected=s.current_catalog(p,pins)
    assert len(selected)==3 and all('key' in r and 'targets' in r for r in selected)
    assert all('key' not in pin for pin in pins.values())
    rows[0]['weights_sha']='changed';p.write_text(json.dumps(dict(complete=True,models=rows,final_test_opened=False,seven_seed_started=False)))
    with pytest.raises(ValueError,match='differs'):s.current_catalog(p,pins)

def test_freeze_binds_development_choices_cohort_and_weights(tmp_path,monkeypatch):
    store=Store(tmp_path/'r');targets=[];weights={'w':torch.tensor([1.])}
    row=dict(seed=21,arm='matched_ggf',targets=targets,targets_sha=identity(targets),total_training_updates=s.TOTAL,
        key='model',method='released_vanilla',weights_sha=training.weight_sha(weights))
    settings=dict(lr=.0003);plan=dict(splits=dict(test=[9]))
    store.save('supervised_settings_frozen',settings);store.save('fresh_holdout_plan',plan);store.save('model',dict(weights=weights))
    store.save('all_models_frozen',dict(complete=True,protocol=s.protocol(),models=[row],settings_sha=identity(settings),holdout_plan_sha=identity(plan)))
    monkeypatch.setattr(s,'expected_pairs',lambda:{(21,'matched_ggf')})
    ctx=dict(store=store,stamps=[],checkpoint_stamps=[]);assert s.verify_frozen(ctx)['complete']
    store.save('fresh_holdout_plan',dict(splits=dict(test=[10])))
    with pytest.raises(ValueError,match='cohort'):s.verify_frozen(ctx)
    store.save('fresh_holdout_plan',plan);store.save('supervised_settings_frozen',dict(lr=.001))
    with pytest.raises(ValueError,match='choices'):s.verify_frozen(ctx)
    store.save('supervised_settings_frozen',settings);store.save('model',dict(weights={'w':torch.tensor([2.])}))
    with pytest.raises(ValueError,match='weights'):s.verify_frozen(ctx)

def test_changing_unknown_patches_cannot_change_matched_ggf_training(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'evaluate',lambda *a,**k:report())
    a,w,t=fit_context(tmp_path/'a');b,_,_=fit_context(tmp_path/'b')
    for ctx in [a,b]:
        ctx['rule']['known_only']=True
        for mm in ctx['packs']['train']['masks']:mm['crowd'][:10]=1.
    b['packs']['train']['x'][:,:10]+=1000.
    x=f.fit(a,'g','released_vanilla',4,2,21,w,2,2,targets=t,cap=.6,stream=1)
    y=f.fit(b,'g','released_vanilla',4,2,21,w,2,2,targets=t,cap=.6,stream=1)
    assert training.weight_sha(x['weights'])==training.weight_sha(y['weights'])
