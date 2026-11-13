"""Meaningful contract, metric-parity, real optimizer/resume and leakage checks."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
import all75_run as run
import dashboard as score
import original_concentration as original
from coco_baselines import training
from coco_baselines.storage import identity
from coco_ggf.width_balance import settings
from scale_models import model
from scale_fit import fit
import cd_common,cd_data,cd_train

@pytest.fixture(autouse=True)
def small_threads():torch.set_num_threads(1)

def pack(start=0,d=16):
    rng=np.random.default_rng(start+12);x=rng.normal(size=(8,196,d)).astype(np.float32);masks=[]
    for i in range(8):
        frac=np.zeros((196,80),np.uint8);frac[:49,0]=255;frac[49:98,1]=255
        x[i,:49,0]+=2;x[i,49:98,1]+=2
        masks.append(dict(fraction=frac,crowd=np.zeros(196,np.float32),instances=[dict(category=c,annotation_id=i*2+c,fraction=(frac[:,c]/255).astype(np.float32)) for c in [0,1]]))
    return dict(x=x,ids=list(range(start,start+8)),masks=masks,identity=identity(dict(start=start,d=d)))

def fixture_context(tmp_path,monkeypatch):
    run.configure('category_split')
    for k,v in dict(warm_updates=2,guided_updates=2,images_per_update=2,evaluate_every=2,local_save_every=1).items():monkeypatch.setitem(cd_common.SPEC,k,v)
    train,dev=pack(),pack(100);guidance={}
    for name,p in [('train',train),('select',dev)]:
        p=copy.deepcopy(p)
        for m in p['masks']:m['fraction'][:,1:]=0;m['instances']=[o for o in m['instances'] if o['category']==0]
        guidance[name]=p
    cfg=dict(category_indices=[0],target_min_images=3,decision_min_images=3,negative_fraction=.1,positive_fraction=.5,
        min_object_patch_mass=2,min_background_patches=4,patch_sample_seed=61002,detection_patches=32,audit_image_batch=1,batch=512,score_ap_weight=.2,preview_images=0)
    opts=dict(regime='category_split',output=str(tmp_path/'new'),local=str(tmp_path/'local'),experiment_sha='synthetic',core_sha='synthetic',concept_source=str(Path(__file__).parent/'concept_sae_native.py'))
    store=cd_common.make_store(opts['output'],tmp_path/'journal',dict(fixture=True),False)
    return dict(options=opts,store=store,versions={},packs=dict(train=train,select=dev),
        folds=[dict(guide=[0],excluded=[1]),dict(guide=[1],excluded=[0])],eligible=[0,1],cfg=cfg,device='cpu',
        norm=dict(mean=np.zeros(16),scale=1.),provenance=dict(categories=[dict(id=i+1,name='c'+str(i)) for i in range(80)]),_guidance_cache={0:guidance})

def test_genuine75_rejects_slice_and_wrong_width():
    run.check_weights(training.weights(model('released_vanilla',75,12,32,16)),d=16)
    with pytest.raises(ValueError):run.check_weights(training.weights(model('released_vanilla',1024,12,32,16)),d=16)
    with pytest.raises(ValueError):run.check_weights(training.weights(model('released_vanilla',75,3,32,16)),d=16)

def test_exact_old_concentration_parity_and_strict_ties():
    rng=np.random.default_rng(17);x=rng.normal(size=(40,75));x[:,0]=0;ids=list(range(40));cal=ids[:20];audit=ids[20:]
    labels={i:('yes' if i%20<7 else 'no') for i in ids}
    lock=dict(presence_complete=True,presence_sha256='fixture-presence',protocol_sha256='fixture-protocol')
    new=score.calibration_selection(x,ids,cal,labels,lock)
    old=original.calibrate_description(x,ids,cal,labels)
    assert new['quantiles']==old['quantiles']
    rows=score.audit_concentration(new,x,ids,audit,labels,dict(selection_sha256=new['selection_sha256'],locked_before_audit=True))
    oldrows=original.evaluate_category_synthetic_or_after_gates(old,x,ids,audit,labels)
    assert [r['metrics'] for r in rows]==[r['metrics'] for r in oldrows]
    assert new['quantiles'][0]['thresholds'][0]==0
    assert ((x[:,0]>new['quantiles'][0]['thresholds'][0])==False).all()

def test_signed_pooling_and_native_reference():
    x=np.arange(2*196*75,dtype=float).reshape(2,196,75)*-1-1
    assert np.array_equal(score.pooled_native(x),original.mean_top4(x))
    assert (score.pooled_native(x)<0).all()
    for n in [37,38]:assert score.pooled_native(x[:,:,:n],n).shape==(2,n)
    with pytest.raises(ValueError):score.pooled_native(x[:,:,:40],40)

def test_low_support_missing_and_lock_guards():
    x=np.zeros((40,75));ids=list(range(40));labels={i:'no' for i in ids};lock=dict(presence_complete=True,presence_sha256='p',protocol_sha256='f')
    with pytest.raises(ValueError):score.calibration_selection(x,ids,ids[:20],labels,{})
    sel=score.calibration_selection(x,ids,ids[:20],labels,lock)
    with pytest.raises(ValueError):score.audit_concentration(sel,x,ids,ids[20:],labels,{})
    with pytest.raises(ValueError):score.validate_splits(ids[:20],ids[19:])
    rows=score.audit_concentration(sel,x,ids,ids[20:],labels,dict(selection_sha256=sel['selection_sha256'],locked_before_audit=True))
    assert len(rows)==8 and all(r['metrics'] is None and r['status']!='complete' for r in rows)
    sel['quantiles'][0]['ranked_units']=[0]
    with pytest.raises(ValueError):score.audit_concentration(sel,x,ids,ids[20:],labels,dict(selection_sha256=sel['selection_sha256'],locked_before_audit=True))

def test_whole_bank_coherence_and_alignment_denominators():
    records=[dict(unit=u,pairs=[dict(semantic='yes',visual='unclear') for _ in range(3)]) for u in range(75)]
    out=score.coherence_counts(records);assert out['semantic']['confirmed']==1 and out['visual']['confirmed']==0 and out['visual']['unresolved_upper']==1
    with pytest.raises(ValueError):score.coherence_counts(records[:-1])
    a=score.alignment_counts(['target','other','mixed']);assert a['confirmed']==1/6 and a['counts']['missing']==3 and a['unresolved_upper']==5/6

def test_native_category_partition_is_unchanged():
    cats=[dict(id=i+1) for i in range(80)];folds=cd_common.category_split(cats,list(range(75)))
    assert sorted(len(f['guide']) for f in folds)==[37,38]
    assert folds[0]['guide']==folds[1]['excluded'] and folds[1]['guide']==folds[0]['excluded']
    assert not set(folds[0]['guide'])&set(folds[0]['excluded'])

def test_vanilla_never_builds_annotation_guidance(tmp_path,monkeypatch):
    ctx=fixture_context(tmp_path,monkeypatch)
    def forbidden(*a,**k):raise AssertionError('Label-free baseline accessed guidance annotations')
    monkeypatch.setattr(cd_data,'guidance_packs',forbidden)
    sub=run.training_context(ctx,0,21,'vanilla',ctx['store'])
    assert all('masks' not in p for p in sub['packs'].values())
    assert not sub['rule'].get('known_only',False)

def test_full_category_epoch_sampling_and_cap_resume_identity(tmp_path,monkeypatch):
    ctx=fixture_context(tmp_path,monkeypatch)
    ctx['options']['regime']='full_category';monkeypatch.setitem(cd_common.SPEC,'cap',.6)
    monkeypatch.setitem(cd_common.SPEC,'continuation_lr',.001);monkeypatch.setitem(cd_common.SPEC,'snr',0.)
    monkeypatch.setitem(cd_common.SPEC,'sampling','epochs')
    monkeypatch.setattr(cd_train,'training_context',run.training_context)
    cd_train.fit_bsf(ctx,0,21,'vanilla')
    row=cd_train.fit_bsf(ctx,0,21,'supervised_vanilla')
    assert cd_train.fit_bsf(ctx,0,21,'supervised_vanilla')['weights_sha']==row['weights_sha']

def test_real75_updates_share_random_warm_and_resume(tmp_path,monkeypatch):
    ctx=fixture_context(tmp_path,monkeypatch);monkeypatch.setattr(cd_train,'training_context',run.training_context)
    rows=[cd_train.fit_bsf(ctx,0,21,k) for k in ['vanilla','supervised_vanilla','ggf']]
    lineages=[json.loads((Path(r['output'])/'results'/r['run_id'][:20]/'training_lineage.json').read_text()) for r in rows]
    assert lineages[0]['warm_weights_sha']==lineages[1]['warm_weights_sha']
    assert len({r['stream_batch_sha'] for r in lineages})==len({r['stream_noise_sha'] for r in lineages})==1
    for r in rows:
        state=cd_train.open_frozen(r,ctx['options']['local']);run.check_weights(state['weights'],d=16)
    assert cd_train.fit_bsf(ctx,1,21,'vanilla')['weights_sha']==rows[0]['weights_sha']
    # Resume the actual optimizer/heads/controller, not an implementation-mirroring mock.
    recipe=copy.deepcopy(settings()['balance']);recipe.update(reference_batches=1,every=1)
    torch.manual_seed(21);initial=training.weights(model('released_vanilla',75,12,32,16))
    targets=[dict(category_index=0,category='c0',contrast_group=0,detector_group=0,heat_mean=0.,heat_scale=1.)]
    sub=dict(packs=ctx['_guidance_cache'][0],cfg=ctx['cfg'],device='cpu',balance_recipe=recipe,rule=dict(cd_common.SPEC),active_key=None)
    a=cd_common.make_store(tmp_path/'uninterrupted',tmp_path/'u_local',dict(test='u'),False)
    b=cd_common.make_store(tmp_path/'resumed',tmp_path/'r_local',dict(test='r'),False)
    expected=fit(dict(sub,store=a),'fit','released_vanilla',75,32,21,initial,4,4,targets=targets)
    fit(dict(sub,store=b),'fit','released_vanilla',75,32,21,initial,4,2,targets=targets)
    actual=fit(dict(sub,store=b),'fit','released_vanilla',75,32,21,initial,4,4,targets=targets)
    assert training.weight_sha(expected['weights'])==training.weight_sha(actual['weights'])
    assert torch.equal(expected['head'],actual['head'])
    assert expected['sampled_batches_sha']==actual['sampled_batches_sha'] and expected['sampled_noise_sha']==actual['sampled_noise_sha']

def test_excluded_crowd_annotation_changes_do_not_change_training_masks(tmp_path,monkeypatch):
    from pycocotools import mask as cm
    def rle(a):
        x=cm.encode(np.asfortranarray(a.astype(np.uint8)));x['counts']=x['counts'].decode();return x
    obj=np.zeros((224,224),np.uint8);obj[:100,:100]=1;crowd=np.zeros_like(obj);crowd[100:]=1
    ann=dict(images=[dict(id=1,width=224,height=224),dict(id=2,width=224,height=224)],categories=[dict(id=i+1,name=str(i)) for i in range(80)],annotations=[])
    for i in [1,2]:
        for c,m,ic in [(1,obj,0),(2,crowd,1)]:ann['annotations'].append(dict(id=i*10+c,image_id=i,category_id=c,segmentation=rle(m),area=int(m.sum()),bbox=[0,0,224,224],iscrowd=ic))
    p=tmp_path/'annotations.json';p.write_text(json.dumps(ann));monkeypatch.setattr(cd_data,'annotation_file',lambda _:p)
    def build(name):
        s=cd_common.make_store(tmp_path/name,tmp_path/(name+'_local'),dict(fixture=name),False)
        return cd_data.guidance_packs(dict(store=s,folds=[dict(guide=[0],excluded=[1])],provenance=dict(categories=ann['categories']),packs={k:dict(x=np.zeros((1,196,16),np.float32),ids=[i],identity=str(i)) for k,i in [('train',1),('select',2)]}),0)
    a=build('first')
    for row in ann['annotations']:
        if row['category_id']==2:row['segmentation']=rle(np.ones_like(obj));row['area']=224**2
    p.write_text(json.dumps(ann));b=build('second')
    for split in ['train','select']:
        assert a[split]['identity']==b[split]['identity']
        assert not a[split]['masks'][0]['crowd'].any() and not a[split]['masks'][0]['fraction'][:,1:].any()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU runtime unavailable')
def test_production_dimensions_supervised_GPU_updates(tmp_path,monkeypatch):
    ctx=fixture_context(tmp_path,monkeypatch)
    train,dev=pack(d=768),pack(100,d=768)
    recipe=copy.deepcopy(settings()['balance']);recipe.update(reference_batches=1,every=1)
    torch.manual_seed(21);torch.cuda.reset_peak_memory_stats()
    initial=training.weights(model('released_vanilla',75,12,32,768))
    sub=dict(packs=dict(train=train,select=dev),cfg=ctx['cfg'],device='cuda',balance_recipe=recipe,
        rule=dict(cd_common.SPEC),active_key=None,store=ctx['store'])
    targets=[dict(category_index=0,category='c0',contrast_group=0,detector_group=0,heat_mean=0.,heat_scale=1.)]
    st=fit(sub,'gpu_smoke','released_vanilla',75,32,21,initial,2,2,targets=targets)
    run.check_weights(st['weights']);assert st['updates']==2 and st['balance']['adaptations']>=1
    assert training.weight_sha(initial)!=training.weight_sha(st['weights'])
    assert torch.isfinite(st['head']).all()
    print('GPU preflight',json.dumps(dict(device=torch.cuda.get_device_name(),updates=2,groups=75,width=12,dimensions=768,peak_bytes=torch.cuda.max_memory_allocated())))
