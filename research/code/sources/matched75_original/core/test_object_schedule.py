import copy,hashlib,inspect,json
from pathlib import Path
import numpy as np
import pytest
import torch
import supervised_study as s
import supervised_fit as f
import object_schedule as o
from coco_baselines import training
from coco_baselines.storage import identity
from test_supervised import Store,report,trial,fit_context

def fixture():
    cfg=s.memory.config('cpu');cfg.update(category_indices=list(range(75)),detection_patches=196,
        preview_images=0,min_category_images=1,decision_min_images=1,bootstrap=0,fresh_test_seed=71002,
        audit_image_batch=1,max_audit_groups=4,batch=128)
    rng=np.random.default_rng(27);x=rng.normal(size=(2,196,12)).astype(np.float32);mm=[]
    for i in range(2):
        frac=np.zeros((196,75),np.uint8);frac[:10]=255
        instances=[dict(category=c,annotation_id=i*100+c,fraction=np.r_[np.ones(10),np.zeros(186)]) for c in range(75)]
        mm.append(dict(fraction=frac,crowd=np.zeros(196),instances=instances))
    pack=dict(x=x,masks=mm,ids=[10,11],identity='fixture')
    # A small native model with repeated orthonormal category blocks gives exact
    # partial NNLS, while the joint solve remains an independent test operation.
    model=s.NativeSLiCS(torch.eye(12).repeat(75,1),s.native_solver_config(),'partial')
    ctx=dict(cfg=cfg,provenance=dict(categories=[dict(name=str(i)) for i in range(75)]))
    return ctx,pack,model,s.fixed_targets(ctx)

def test_object_only_has_identical_capture_ap_without_joint_call(monkeypatch):
    ctx,p,m,targets=fixture()
    # Full evaluator normally calls encode(partial), then joint solve. Supply a
    # valid explicit joint solution for repeated identical blocks.
    def joint(x):return torch.cat([x.clamp_min(0),torch.zeros(len(x),888)],1)
    monkeypatch.setattr(m,'solve',joint)
    full=s.evaluate_model(m,p,targets,ctx['cfg'],'cpu',True)
    monkeypatch.setattr(m,'solve',lambda *a:(_ for _ in ()).throw(AssertionError('joint solve called')))
    cheap=s.evaluate_model(m,p,targets,ctx['cfg'],'cpu',True,reconstruction=False)
    assert full['macro_capture']==cheap['macro_capture'] and full['macro_ap']==cheap['macro_ap']
    assert full['categories']==cheap['categories']
    assert cheap['reconstruction']['status']=='deferred' and cheap['reconstruction']['ve'] is None
    s.finite_report(cheap)

def test_grid_tie_ignores_mse_and_preserves_grid_order():
    a=trial(.0001,.3,.5,.8);b=trial(.0003,.3,.5,.01)
    assert s.select_trials([a,b])['lr']==.0001
    assert s.select_trials([b,a])['lr']==.0003
    assert 'mean_mse' not in a
    with pytest.raises(ValueError):s.select_trials([a,b],'mse')

def test_revision_freeze_has_no_mse_or_joint_object_rows():
    assert len(s.expected_pairs())==31
    assert s.protocol()['version']==3 and not s.protocol()['training_arithmetic_changed']
    assert all('mse' not in a and 'joint' not in a for a in s.ARMS)
    assert 'MSE_selector' not in inspect.getsource(s.selection)

@pytest.mark.parametrize('key',['native_rows_s21','row_small75_s21_lr2999_ggf_object','small75_s21_lr2999_ggf_mse',
    'native_slics12_s21_mse','supervised_settings_frozen','all_models_frozen','fresh_holdout_plan','audit_s21_matched_ggf'])
def test_never_import_old_decisions_or_test_reports(key):assert not o.PrefixReuse.reusable(key)

@pytest.mark.parametrize('key',['native_svd12','native_slics12_s21','native_slics12_s21_object',
    'small75_s21_warm','small75_s21_targets','small75_s27_lr10000_ggf','small75_s22_lr2999_ggf_object'])
def test_only_compatible_training_prefix_keys_are_importable(key):assert o.PrefixReuse.reusable(key)

def test_every_actual_declared_learning_rate_key_is_reused():
    for seed in s.SEEDS:
        for lr in s.LRS:
            # Preserve the predecessor's integer conversion, including 2999.
            key=f'small75_s{seed}_lr{int(lr*1e7)}_ggf'
            assert o.PrefixReuse.reusable(key) and o.PrefixReuse.reusable(key+'_object')

def config():
    old=json.loads((Path(o.__file__).parent/'legacy_manifest.json').read_text())['config']
    return copy.deepcopy(old)

def test_migration_preserves_adam_rng_and_dictionary_and_rejects_changed_inputs(tmp_path,monkeypatch):
    a,w,t=fit_context(tmp_path/'a');monkeypatch.setattr(f,'evaluate',lambda *a,**k:report())
    st=f.fit(a,'small75_s21_lr2999_ggf','released_vanilla',4,2,21,w,4,2,targets=t,cap=.6,stream=1,keep_candidates=True)
    old=a['store'];old.config=config();old.run_id=o.LEGACY_RUN
    now=Store(tmp_path/'new');now.config=config();now.config['protocol']=s.protocol();now.config['source_sha']='new'
    # Minimal stand-in for the existing Reuse wrapper's forced save interface.
    ctx=dict(checkpoint_stamps=[])
    monkeypatch.setattr(o.matched,'source_key_stamp',lambda *a:dict(payload_sha='checked'))
    migrate=o.PrefixReuse(now,old,ctx)
    imported=migrate.load('small75_s21_lr2999_ggf')
    assert training.weight_sha(st['weights'])==training.weight_sha(imported['weights'])
    assert torch.equal(st['head'],imported['head']) and torch.equal(st['noise_rng'],imported['noise_rng'])
    assert identity(st['sample_rng'])==identity(imported['sample_rng'])
    assert 'mse' not in imported['selected'] and st['fingerprint']==imported['fingerprint']
    assert training.weight_sha(old.values['small75_s21_lr2999_ggf']['weights'])==training.weight_sha(st['weights'])
    now.config['train_identity']='changed'
    with pytest.raises(ValueError,match='training inputs'):o.PrefixReuse(now,old,ctx)

def test_original_adam_prefix_continues_to_identical_final_weights(tmp_path,monkeypatch):
    import importlib.util
    original=Path(o.__file__).parent/'legacy_supervised_fit.py'
    spec=importlib.util.spec_from_file_location('legacy_fit_reference',original)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    monkeypatch.setattr(legacy,'evaluate',lambda *a,**k:report())
    monkeypatch.setattr(f,'evaluate',lambda *a,**k:report())
    full,w,targets=fit_context(tmp_path/'full');old,_,_=fit_context(tmp_path/'old');key='small75_s21_lr2999_ggf'
    expected=legacy.fit(full,key,'released_vanilla',4,2,21,w,4,4,targets=targets,cap=.6,stream=1,keep_candidates=True)
    prefix=legacy.fit(old,key,'released_vanilla',4,2,21,w,4,2,targets=targets,cap=.6,stream=1,keep_candidates=True)
    before_hash=training.weight_sha(prefix['weights'])
    old['store'].config=config();old['store'].run_id=o.LEGACY_RUN
    resumed,_,_=fit_context(tmp_path/'resumed');resumed['store'].config=config();resumed['store'].config['protocol']=s.protocol()
    resumed['checkpoint_stamps']=[]
    monkeypatch.setattr(o.matched,'source_key_stamp',lambda *a:dict(payload_sha='verified'))
    resumed['store']=o.PrefixReuse(resumed['store'],old['store'],resumed)
    actual=f.fit(resumed,key,'released_vanilla',4,2,21,w,4,4,targets=targets,cap=.6,stream=1,keep_candidates=True)
    assert training.weight_sha(actual['weights'])==training.weight_sha(expected['weights'])
    assert torch.equal(actual['head'],expected['head']) and torch.equal(actual['noise_rng'],expected['noise_rng'])
    assert torch.equal(actual['balance']['log_weights'],expected['balance']['log_weights'])
    assert actual['sampled_batches_sha']==expected['sampled_batches_sha']
    assert actual['sampled_noise_sha']==expected['sampled_noise_sha']
    for param,state in expected['optimizer']['state'].items():
        for name,value in state.items():assert torch.equal(value,actual['optimizer']['state'][param][name])
    assert training.weight_sha(old['store'].load(key)['weights'])==before_hash

def test_native_prefix_rebinds_only_verified_protocol_fingerprint(tmp_path,monkeypatch):
    old=Store(tmp_path/'old');old.config=config();old.run_id=o.LEGACY_RUN
    new=Store(tmp_path/'new');new.config=config();new.config['protocol']=s.protocol()
    D=torch.eye(12).repeat(75,1);packs=dict(train=dict(identity='t',x=np.zeros((1,196,12))),select=dict(identity='s'))
    ctx=dict(packs=packs,checkpoint_stamps=[])
    monkeypatch.setattr(s,'native_initializer',lambda ctx:D)
    monkeypatch.setattr(o.matched,'source_key_stamp',lambda *a:dict(payload_sha='checked'))
    args=dict(source=training.weight_sha({'D':D}),train='t',select='s',seed=21,solver=s.native_solver_config(),total=s.TOTAL)
    old.save('native_slics12_s21',dict(D=D,updates=500,fingerprint=identity(dict(args,protocol=old.config['protocol'])),selected={}))
    migration=o.PrefixReuse(new,old,ctx);ctx['store']=migration
    v=migration.load('native_slics12_s21')
    assert v['updates']==500 and torch.equal(v['D'],D)
    assert v['fingerprint']==identity(dict(args,protocol=s.protocol()))
    assert v['fingerprint']!=old.load('native_slics12_s21')['fingerprint']
    other=Store(tmp_path/'bad');other.config=copy.deepcopy(new.config)
    state=old.load('native_slics12_s21');state['fingerprint']='bad';old.save('native_slics12_s21',state)
    with pytest.raises(ValueError,match='fingerprint'):o.PrefixReuse(other,old,ctx).load('native_slics12_s21')

def test_reconstruction_resumes_per_image_and_matches_direct_joint(tmp_path):
    rng=np.random.default_rng(31);x=rng.normal(size=(5,196,12)).astype(np.float32)
    pack=dict(x=x,ids=list(range(5)),identity='cohort');store=Store(tmp_path/'r');ctx=dict(store=store,device='cpu',norm={})
    m=s.NativeSLiCS(torch.eye(12),s.native_solver_config(),'partial');real=m.solve;count=[0]
    def stop(x):
        count[0]+=1
        if count[0]==3:raise KeyboardInterrupt()
        return real(x)
    m.solve=stop
    with pytest.raises(KeyboardInterrupt):o.final_reconstruction(ctx,m,pack,'weights')
    st=next(v for k,v in store.values.items() if k.startswith('final_joint_'))
    assert st['completed_images']==2
    count[0]=0
    def resume(x):count[0]+=1;return real(x)
    m.solve=resume;out=o.final_reconstruction(ctx,m,pack,'weights')
    assert count[0]==3
    direct=float(np.square(x.astype(float)-np.maximum(x,0).astype(float)).sum())
    assert out['mse']==pytest.approx(direct/x.size,abs=1e-12)
    assert out['ve']==pytest.approx(1-direct/s.pack_variance(pack),abs=1e-12)
    before=count[0];again=o.final_reconstruction(ctx,m,pack,'weights');assert again==out and count[0]==before

def test_native_pending_validation_resumes_without_joint(tmp_path,monkeypatch):
    # The inherited real atom-update test covers transactional training. Here a
    # pending final validation must perform only the object evaluator.
    store=Store(tmp_path/'r');D=torch.eye(12).repeat(75,1)
    ctx=dict(store=store,device='cpu',packs=dict(train=dict(identity='t'),select=dict(identity='s')),
        cfg=dict(category_indices=list(range(75))),provenance=dict(categories=[dict(name=str(i)) for i in range(75)]))
    monkeypatch.setattr(s,'native_initializer',lambda ctx:D)
    fingerprint=identity(dict(source=training.weight_sha({'D':D}),train='t',select='s',seed=21,
        solver=s.native_solver_config(),total=s.TOTAL,protocol=s.protocol()))
    store.save('native_slics12_s21',dict(D=D,updates=s.TOTAL,fingerprint=fingerprint,selected={},history=[],seconds=0.,sample_rng=np.random.default_rng(1).bit_generator.state))
    monkeypatch.setattr(s.NativeSLiCS,'solve',lambda *a:(_ for _ in ()).throw(AssertionError('joint validation called')))
    calls=[]
    def evaluate(*args,**kwargs):calls.append(1);return o.object_report(report())
    monkeypatch.setattr(s,'evaluate_model',evaluate);monkeypatch.setattr(s,'evaluate_state',lambda *a:o.object_report(report()))
    rows=s.native_fit(ctx,21)
    assert set(rows)=={'slics_object_partial'} and len(calls)==1
    assert store.load('native_slics12_s21')['last_evaluated']==s.TOTAL

def test_live_handles_deferred_reconstruction(tmp_path,monkeypatch):
    from supervised_live import Monitor,Observers
    ctx,p,m,t=fixture();messages=[]
    monitor=Monitor(emit=messages.append)
    monkeypatch.setattr(s,'evaluate_model',lambda *a,**k:o.object_report(report()))
    observer=Observers(monitor).install()
    try:
        s.evaluate_model(m,p,t,ctx['cfg'],'cpu',reconstruction=False)
        assert any('VE deferred' in message for message in messages)
    finally:observer.close()

def reservation_context(tmp_path,monkeypatch):
    fixture=json.loads((Path(o.__file__).parent/'predecessor_holdout_fixture.json').read_text())
    previous=Store(tmp_path/'previous');previous.run_id=o.LEGACY_RUN
    previous.root=tmp_path/'previous_checkpoints';previous.root.mkdir()
    previous.save('fresh_holdout_plan',fixture['plan'])
    previous.write('other_study_holdout_excluded.json',fixture['excluded'])
    previous.write('completion_status.json',fixture['status'])
    current=Store(tmp_path/'current');current.previous=previous
    current.root=tmp_path/'current_checkpoints';current.root.mkdir()
    ctx=dict(store=current,local=tmp_path,checkpoint_stamps=[],source_stores=[],
        cfg=dict(category_indices=fixture['plan']['category_indices']),rule=dict(final_holdout_images=1500),
        provenance=dict(splits=dict(train=[-1],select=[-2])))
    monkeypatch.setattr(o.matched,'source_key_stamp',lambda *a:dict(payload_sha='verified'))
    return ctx,previous,fixture

def test_actual_drive_reservation_preserves_all_75_quotas_without_rescan(tmp_path,monkeypatch):
    ctx,old,fixture=reservation_context(tmp_path,monkeypatch)
    monkeypatch.setattr(s,'holdout_plan',lambda *a:(_ for _ in ()).throw(AssertionError('unnecessary scan')))
    before=copy.deepcopy(old.values)
    plan=o.reserve_predecessor_holdout(ctx)
    assert plan==fixture['plan'] and identity(plan)==o.RESERVED_PLAN_SHA
    assert len(plan['splits']['test'])==1500 and len(plan['category_indices'])==75
    assert min(plan['coverage']['split_counts']['test'][c] for c in plan['category_indices'])>=10
    assert not set(plan['splits']['test'])&set(fixture['excluded']['other_study_ids'])
    assert before==old.values
    assert o.reserve_predecessor_holdout(ctx)==plan
    assert ctx['store'].load('fresh_holdout_plan')==plan

@pytest.mark.parametrize('marker',['frozen_key','test_complete_key','audit_index','frozen_file','test_shard','audit_report','test_event','complete_status'])
def test_reservation_rejects_any_evidence_of_test_access(tmp_path,monkeypatch,marker):
    ctx,old,_=reservation_context(tmp_path,monkeypatch)
    if marker=='frozen_key':old.save('all_models_frozen',dict(complete=True))
    elif marker=='test_complete_key':old.save('supervised_test_complete',dict(complete=True))
    elif marker=='audit_index':old.save('audit_signature_index',{})
    elif marker=='frozen_file':old.write('all_models_frozen.json',{})
    elif marker=='test_shard':(old.root/'features_test_000000').mkdir()
    elif marker=='audit_report':old.write('reports/audit_s21_slics_object_partial.json',{})
    elif marker=='test_event':(old.results/'events.jsonl').write_text(json.dumps(dict(stage='extraction',split='test'))+'\n')
    else:old.write('completion_status.json',dict(test_complete=True))
    with pytest.raises(ValueError,match='holdout'):o.reserve_predecessor_holdout(ctx)
    assert ctx['store'].load('fresh_holdout_plan') is None

def test_reservation_checks_prior_split_overlap(tmp_path,monkeypatch):
    ctx,old,fixture=reservation_context(tmp_path,monkeypatch)
    ctx['provenance']['splits']['train']=[fixture['plan']['splits']['test'][0]]
    with pytest.raises(ValueError,match='overlaps'):o.reserve_predecessor_holdout(ctx)

def test_reservation_checks_exact_plan_and_category_identity(tmp_path,monkeypatch):
    ctx,old,fixture=reservation_context(tmp_path,monkeypatch)
    plan=old.load('fresh_holdout_plan');plan['splits']['test'][0]=-99;old.save('fresh_holdout_plan',plan)
    with pytest.raises(ValueError,match='Pinned'):o.reserve_predecessor_holdout(ctx)
    old.save('fresh_holdout_plan',fixture['plan']);ctx['cfg']['category_indices']=ctx['cfg']['category_indices'][:-1]
    with pytest.raises(ValueError,match='75-category'):o.reserve_predecessor_holdout(ctx)

def test_reservation_checks_other_study_exclusion_identity(tmp_path,monkeypatch):
    ctx,old,_=reservation_context(tmp_path,monkeypatch)
    old.write('other_study_holdout_excluded.json',dict(ids=[],other_study_ids=[]))
    with pytest.raises(ValueError,match='exclusions'):o.reserve_predecessor_holdout(ctx)

def test_reservation_rejects_sibling_that_opened_the_same_cohort(tmp_path,monkeypatch):
    from coco_adaptation import inputs
    ctx,old,fixture=reservation_context(tmp_path,monkeypatch)
    sibling=Store(tmp_path/'sibling');sibling.root=tmp_path/'sibling_checkpoints';sibling.root.mkdir()
    sibling.write('fresh_holdout_plan.json',fixture['plan']);sibling.save('all_models_frozen',{})
    monkeypatch.setattr(inputs,'ReadStore',lambda *a:sibling)
    with pytest.raises(ValueError,match='unopened'):o.reserve_predecessor_holdout(ctx)
