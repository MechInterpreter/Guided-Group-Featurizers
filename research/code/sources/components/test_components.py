import copy,inspect,json
import numpy as np
import pytest
import torch
import components as c
from coco_adaptation.inputs import JournalStore
from coco_baselines import training
from coco_ggf import width_balance
from scale_models import model

def history():
    return [dict(update=u,weights=[1.,.05+u/10000,.1+u/10000],validation=dict(macro_capture=999)) for u in range(500,3001,500)]

def test_fixed_policy_uses_only_prespecified_training_weight_records():
    a=history();p=c.fixed_policy(a);b=copy.deepcopy(a)
    for r in b:r['validation']={'test_AP':-999};r['gradient_norm']=-99
    assert c.fixed_policy(b)==p
    assert np.allclose(p['fixed_weights'],np.mean([r['weights'] for r in a],axis=0))
    assert not p['validation_or_test_scores_used'] and 'not a full-step' in p['rule']

@pytest.mark.parametrize('change',[lambda a:a.pop(),lambda a:a.append(a[-1]),lambda a:a[0].update(weights=[1.,2.,.2]),lambda a:a[0].update(weights=[1.,float('nan'),.2])])
def test_invalid_or_incomplete_training_weights_reject(change):
    a=history();change(a)
    with pytest.raises(ValueError):c.fixed_policy(a)

def context(tmp_path):
    torch.set_num_threads(1);rng=np.random.default_rng(7);x=rng.normal(size=(6,196,4)).astype(np.float32);masks=[]
    for i in range(6):
        f=np.zeros((196,2),np.uint8);f[:80,0]=255;f[85:165,1]=255
        masks.append(dict(fraction=f,crowd=np.zeros(196,np.float32),instances=[dict(category=j,annotation_id=100*i+j,fraction=f[:,j]/255.) for j in [0,1]]))
    pack=dict(x=x,ids=list(range(6)),masks=masks,identity='toy-fixed')
    cfg=dict(category_indices=[0,1],negative_fraction=.1,positive_fraction=.5,min_object_patch_mass=2,min_background_patches=4,decision_min_images=1,target_min_images=1,patch_sample_seed=61002,detection_patches=32,preview_images=0,audit_image_batch=1,batch=128,score_ap_weight=.2,bootstrap=8,fresh_test_seed=12)
    rule=dict(width=2,lr=.0003,images_per_update=2,snr=.1,clip=5.,fixed_weights=[1.,.2,.3],local_save_every=1,drive_save_every=2,evaluate_every=2)
    s=JournalStore(tmp_path/'job',dict(test=True),tmp_path/'local',False)
    return dict(store=s,device='cpu',rule=rule,balance_recipe=copy.deepcopy(width_balance.settings()['balance']),packs=dict(train=pack,select=pack),cfg=cfg,active_key=None)

@pytest.mark.parametrize('kind',['mask_only','fixed_guided','adaptive'])
def test_real_pinned_fit_resume_and_shared_training_stream(tmp_path,kind):
    ctx=context(tmp_path);torch.manual_seed(21);w=training.weights(model('released_vanilla',6,2,2,4))
    targets=[dict(category_index=i,category=str(i),contrast_group=i,detector_group=i,heat_mean=0.,heat_scale=1.) for i in range(2)]
    kw=dict(targets=None if kind=='mask_only' else targets,fixed=kind=='fixed_guided',cap=.6,learning_rate=.0003,stream=1)
    a=c.fit(ctx,'whole','released_vanilla',6,2,21,w,4,4,**kw)
    b=c.fit(ctx,'resume','released_vanilla',6,2,21,w,4,2,**kw)
    b=c.fit(ctx,'resume','released_vanilla',6,2,21,w,4,4,**kw)
    assert training.weight_sha(a['weights'])==training.weight_sha(b['weights'])
    assert a['sampled_batches_sha']==b['sampled_batches_sha'] and a['sampled_noise_sha']==b['sampled_noise_sha']
    assert (a['head'] is None)==(kind=='mask_only') and (a['balance'] is None)==(kind!='adaptive')
    native=c.fit(ctx,'native','released_vanilla',6,2,21,w,4,4,targets=None,cap=.6,learning_rate=.0003,stream=1)
    assert a['sampled_batches_sha']==native['sampled_batches_sha'] and a['sampled_noise_sha']==native['sampled_noise_sha']
    assert c.fit(ctx,'whole','released_vanilla',6,2,21,w,4,4,**kw)['updates']==4

def test_exact_six_new_fits_and_freeze_before_test():
    t=inspect.getsource(c.train_components);m=inspect.getsource(c.main);e=inspect.getsource(c.evaluate_all)
    assert "for kind in ['mask_only','fixed_guided']" in t and c.SEEDS==[21,22,23]
    assert '3000,3000' in t and '10640' in t and 'expected_streams' in t
    assert 'run_branch(' not in t and 'train_targets(' not in t
    assert m.index('train_components(ctx)')<m.index('evaluate_all(ctx)')
    assert e.index('verify_freeze(ctx)')<e.index('source.load_cached_test')
    assert "if row['kind'] not in source.KINDS:continue" in e and 'fresh_holdout_opened=False' in e
    assert "s.load(key)" in e and "if saved is None:" in e

def test_transport_checkpoint_and_forced_boundary_save(tmp_path):
    ctx=context(tmp_path);s=ctx['store'];c.install_transport(s)
    report=json.loads((s.results/'drive_recovery.json').read_text())
    assert report['local_checkpoint_updates']==100 and report['drive_checkpoint_updates']==500
    assert report['force_save_at_stop_and_boundaries'] and report['source_stores_read_only']
    s.save('toy',dict(updates=100,target=3000));assert s.load('toy')['updates']==100
    s._width_drive_recovery_original_save('toy',dict(updates=123,target=3000),drive=True)
    assert s.load('toy')['updates']==123

def test_archive_manifest_no_checkpoints(tmp_path):
    ctx=context(tmp_path);a=tmp_path/'a.zip';a.write_bytes(b'core');b=tmp_path/'b.zip';b.write_bytes(b'wrapper')
    zpath=c.export(ctx,dict(source_zip=str(a),wrapper_zip=str(b)),False)
    import hashlib,zipfile
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None and not any(n.endswith('.pt') for n in z.namelist())
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in json.loads(z.read('artifact_hashes.json')).items())

def test_contribution_projection_and_matching_color_function():
    from figure_current import projection,rgb,field
    rng=np.random.default_rng(8);code=rng.normal(size=(80,12));decoder=rng.normal(size=(12,20))
    transform,center,u,fractions=projection(code,decoder)
    _,s,_=np.linalg.svd(code@decoder-(code@decoder).mean(0),full_matrices=False)
    assert np.allclose(fractions,s[:3]**2/np.sum(s**2))
    assert np.allclose(u,code@transform-center)
    lo,hi=u.min(0),u.max(0)
    assert np.array_equal(rgb(u[:5],lo,hi),rgb(code[:5]@transform-center,lo,hi))
    v=np.arange(196.);sm=field(v,100);assert sm.shape==(100,100) and sm.min()>=0 and sm.max()<=195

def test_real_evaluation_reuses_six_audits_and_resumes_all_intervals(tmp_path,monkeypatch):
    from coco_baselines.storage import identity
    ctx=context(tmp_path);s=ctx['store'];pack=ctx['packs']['select'];norm=dict(mean=np.zeros((196,4),np.float32),scale=1.)
    ctx.update(norm=norm,test_stamp=dict(ids=pack['ids'],normalized_sha='toy'),stamps=[],checkpoint_stamps=[])
    controls=JournalStore(tmp_path/'controls',dict(test=True),tmp_path/'control_local',False);ctx['controls']=controls
    cohort=identity(dict(test=ctx['test_stamp'],cfg=ctx['cfg'],normalization=identity(norm)))
    targets=[dict(category_index=i,category=str(i),contrast_group=i,detector_group=i,heat_mean=0.,heat_scale=1.) for i in range(2)]
    rows=[]
    for seed in c.SEEDS:
        torch.manual_seed(seed);m=model('released_vanilla',6,2,2,4);w=training.weights(m)
        report=c.evaluate(m,pack,targets,ctx['cfg'],'cpu',compact=True);compact=report.pop('_compact')
        for kind in c.KINDS:
            r=dict(seed=seed,kind=kind,key=f'{kind}_s{seed}',groups=6,rank=2,active_groups=2,weights_sha=training.weight_sha(w),targets=targets,targets_sha=identity(targets),reference_warm_mse=1.,development=dict(reconstruction=dict(mse=.5)))
            rows.append(r)
            if kind in ['mask_only','fixed_guided']:s.save(r['key'],dict(weights=w,updates=3000))
            else:
                rr=dict(report,seed=seed,kind=kind,weights_sha=r['weights_sha'],targets_sha=r['targets_sha'],native_warm_eligible=True,total_training_updates=10640)
                controls.save(f'audit_s{seed}_{kind}',dict(signature=identity(dict(weights=r['weights_sha'],targets=r['targets_sha'],cohort=cohort)),report=rr,compact=compact))
    s.save('components_frozen',dict(complete=True,models=rows));ctx['base_context']={}
    monkeypatch.setattr(c.source,'load_cached_test',lambda _:pack)
    ctx['base_context']['packs']={}
    before={str(p):c.sha_file(p) for p in controls.output.rglob('*') if p.is_file()}
    real=c.evaluate;calls=[]
    def score(*a,**kw):calls.append(1);return real(*a,**kw)
    monkeypatch.setattr(c,'evaluate',score);out=c.evaluate_all(ctx)
    assert len(calls)==6 and len(out['models'])==12 and len(out['means'])==4 and len(out['comparisons'])==9
    assert all(abs(r['capture_gain_pp'])<1e-10 and abs(r['AP_gain_pp'])<1e-10 for r in out['comparisons'])
    assert all(c.sha_file(__import__('pathlib').Path(p))==sha for p,sha in before.items())
    monkeypatch.setattr(c,'evaluate',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No repeated inference')))
    monkeypatch.setattr(c.study,'paired',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No repeated CIs')))
    assert c.evaluate_all(ctx)['complete']

def test_original_figure_only_export_is_verified_and_reused(tmp_path):
    from figure_current import cached_export_valid
    import hashlib,zipfile
    out=tmp_path/'figure';out.mkdir();record=dict(signature='correct')
    (out/'figure1_current.json').write_text(json.dumps(record));names=['figure1_current.pdf','figure1_current.png','figure1_current.npz']
    for n in names:(out/n).write_bytes(('measured '+n).encode())
    hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in names};archive=tmp_path/'figure.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for n in names:z.write(out/n,n)
        z.writestr('SHA256.json',json.dumps(hashes))
    assert cached_export_valid(out,archive,'correct')
    assert not cached_export_valid(out,archive,'other checkpoint')
    (out/'figure1_current.png').write_bytes(b'changed')
    assert not cached_export_valid(out,archive,'correct')
