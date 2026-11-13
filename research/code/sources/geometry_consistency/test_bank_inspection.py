import ast,json
from pathlib import Path
import numpy as np
import study
import bank_inspection as bank
from study_math import geometry

def test_sparse_reference_prior_cannot_inflate_conditional_geometry():
    y=np.r_[np.ones(100,bool),np.zeros(100,bool)]
    dense=np.ones((200,3));sparse=dense.copy();sparse[111:]=0
    a=geometry(dense,dense,y,y);b=geometry(sparse,sparse,y,y)
    assert np.isclose(a['active_balanced_neighbor_accuracy'],.5)
    assert np.isclose(b['active_balanced_neighbor_accuracy'],.5)
    # Real selectivity from silence may improve the all-patch diagnostic, but
    # must not be mislabeled improved geometry among the active contributions.
    assert b['balanced_neighbor_accuracy']>a['balanced_neighbor_accuracy']
    assert b['chance_baseline']==a['chance_baseline']==.5

def test_no_active_negatives_is_missing_not_perfect_geometry():
    y=np.r_[np.ones(20,bool),np.zeros(20,bool)]
    r=np.ones((40,3));r[~y]=0
    q=geometry(r,r,y,y)
    assert q['balanced_neighbor_accuracy']==1
    assert q['active_balanced_neighbor_accuracy'] is None
    assert q['active_geometry_status']=='insufficient_active_class_support'

def test_ties_ignore_order_and_unbalanced_input_prevalence():
    y=np.r_[np.ones(80,bool),np.zeros(20,bool)]
    x=np.ones((100,3));a=geometry(x,x,y,y,k=30)
    order=np.random.default_rng(4).permutation(len(y));b=geometry(x[order],x[order],y[order],y[order],k=30)
    assert a['balanced_neighbor_accuracy']==b['balanced_neighbor_accuracy']==.5
    assert a['active_balanced_neighbor_accuracy']==b['active_balanced_neighbor_accuracy']==.5

def test_controls_precede_every_bank_and_resume_skips_completed(tmp_path,monkeypatch):
    for name in ['controlled_consistency','response_scales','discovery']:(tmp_path/name).mkdir()
    ctx=dict(out=tmp_path,signature='original-input-identity');rows=[dict(kind='agf_g',seed=21),dict(kind='slics',seed=21)]
    calls=[]
    monkeypatch.setattr(study,'RUNS',[21]);monkeypatch.setattr(study,'METHODS',['agf_g','slics'])
    monkeypatch.setattr(study,'syn_pairs',lambda ctx:('root',['pair']))
    monkeypatch.setattr(study,'extract_pairs',lambda *args:calls.append('features') or 'features')
    monkeypatch.setattr(bank,'prepare_bank',lambda ctx,row,*args:calls.append('prepare_'+row['kind']) or (None,None,None,None,None,None,None))
    def controls(ctx,row,*args):
        calls.append('control_'+row['kind']);study.done_json(tmp_path/'controlled_consistency'/f'{row["kind"]}_21.json',[])
    monkeypatch.setattr(study,'run_controls',controls)
    monkeypatch.setattr(study,'summarize',lambda *args:None)
    monkeypatch.setattr(bank,'run_bank_v2',lambda ctx,row,*args:calls.append('bank_'+row['kind']))
    bank.run_stages(ctx,rows,None,None,None,None)
    assert calls==['features','prepare_agf_g','control_agf_g','prepare_slics','control_slics','bank_agf_g','bank_slics']
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file() and 'controlled_consistency' in str(p)}
    calls.clear();bank.run_stages(ctx,rows,None,None,None,None)
    assert calls==['bank_agf_g','bank_slics']
    assert all(p.read_bytes()==raw for p,raw in before.items())

def test_all_unit_gallery_retains_inactive_and_unassigned(tmp_path,monkeypatch):
    import sys,types
    # Fake data module suffices: inactive example units must not download images.
    package=types.ModuleType('coco_baselines');package.data=types.SimpleNamespace()
    monkeypatch.setitem(sys.modules,'coco_baselines',package)
    pn=dict(ids=np.array([4,4]),patch=np.array([0,1]),images=np.zeros(2,int),y=np.zeros((2,1),bool))
    ref=dict(score=np.zeros((2,3)),z=np.zeros((2,3,2)))
    ctx=dict(out=tmp_path,signature='legacy',options=dict(local=str(tmp_path/'local')),packs=dict(confirm=dict(ids=[4])))
    row=dict(kind='agf_g',seed=21,weights_sha='frozen')
    entries=[dict(unit=g,category=None,scope='not_directly_guided',calibration_activity=0.) for g in range(3)]
    bank.gallery(ctx,row,None,np.zeros((3,2,4)),None,[],ref,ref,pn,pn,entries)
    folder=tmp_path/'whole_bank_atlas_v2/agf_g/21'
    complete=study.read_done(folder/'COMPLETE.json')
    assert complete['units']==3
    page=(folder/'page_1.html').read_text(encoding='utf8')
    assert all(f'id="unit-{g}"' in page for g in range(3))
    assert 'No positive annotated-category association' in page and 'No nonzero response' in page
    raw=(folder/'inventory.csv').read_bytes()
    bank.gallery(ctx,row,None,np.zeros((3,2,4)),None,[],ref,ref,pn,pn,entries)
    assert (folder/'inventory.csv').read_bytes()==raw

def test_patch_overlay_only_uses_supplied_coordinates():
    html=bank.patch_overlay('x.jpg',[0,195],[0.,1.],scale=1)
    assert html.count('<rect ')==1 and 'x="208" y="208"' in html

def test_active_unassigned_unit_gets_pca_and_image_geometry(tmp_path,monkeypatch):
    import sys,types
    from PIL import Image
    source=tmp_path/'source.jpg';Image.new('RGB',(224,224),(160,130,110)).save(source)
    package=types.ModuleType('coco_baselines')
    package.data=types.SimpleNamespace(COCO_BUCKET='',download=lambda *args:source,geometry=lambda *args:None,transform_image=lambda im,geom:im.resize((224,224)))
    monkeypatch.setitem(sys.modules,'coco_baselines',package)
    rng=np.random.default_rng(12);z=rng.normal(size=(16,1,3));score=np.linalg.norm(z,axis=-1)
    pn=dict(ids=np.repeat([4,5],8),patch=np.tile(np.arange(8),2),images=np.repeat([0,1],8),y=np.zeros((16,1),bool))
    ref=dict(score=score,z=z)
    ctx=dict(out=tmp_path,signature='legacy',options=dict(local=str(tmp_path/'local')),packs=dict(confirm=dict(ids=[4,5])))
    row=dict(kind='released_vanilla',seed=22,weights_sha='frozen')
    entries=[dict(unit=0,category=None,scope='other_groups',calibration_activity=1.)]
    bank.gallery(ctx,row,None,rng.normal(size=(1,3,7)),None,[],ref,ref,pn,pn,entries)
    folder=tmp_path/'whole_bank_atlas_v2/released_vanilla/22'
    card=study.read_done(folder/'cards/0.json')
    assert card['info']['pca_status']=='available' and card['info']['category'] is None
    assert '<circle ' in card['html'] and 'PCA coordinates' in card['html']
    assert all((tmp_path/'whole_bank_atlas_v2/images'/f'{i}.jpg').exists() for i in [4,5])

def test_pca_cache_reuses_compatible_legacy_and_all_labels(tmp_path,monkeypatch):
    rng=np.random.default_rng(2);x=rng.normal(size=(20,4));pn=dict(ids=np.arange(20),patch=np.zeros(20,int))
    ref=dict(score=np.ones((20,1)));ctx=dict(out=tmp_path,signature='legacy',options=dict(local=str(tmp_path/'local')))
    row=dict(kind='agf_g',seed=21,weights_sha='frozen')
    from study_math import pca_fit
    pc=pca_fit(x);path=tmp_path/'pca/agf_g/21/unit_0.npz';path.parent.mkdir(parents=True)
    np.savez_compressed(path,**pc,reference_ids=pn['ids'],reference_patch=pn['patch'])
    before=path.read_bytes();monkeypatch.setattr(study,'embed',lambda *args:x)
    monkeypatch.setattr(bank,'pca_fit',lambda *args:(_ for _ in ()).throw(AssertionError('Should reuse old PCA')))
    actual=bank.unit_pca(ctx,row,0,ref,ref,pn,pn,None,None)
    assert np.array_equal(actual['axes'],pc['axes']) and path.read_bytes()==before
    again=bank.unit_pca(ctx,row,0,ref,ref,pn,pn,None,None)
    assert np.array_equal(again['axes'],actual['axes'])

def test_cached_geometry_encodings_and_controls_keep_legacy_identity():
    legacy=Path(__file__).with_name('legacy_study.py')
    # A local baseline copy is included by the build process for this audit.
    baseline=Path(__file__).with_name('legacy_study.py')
    if baseline.exists():legacy=baseline
    old=ast.parse(legacy.read_text(encoding='utf8'));new=ast.parse(Path(study.__file__).read_text(encoding='utf8'))
    functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    oldf=functions(old);newf=functions(new)
    for name in ['observe','encode_panel','embed','extract_pairs','run_controls']:
        assert ast.dump(oldf[name])==ast.dump(newf[name]),name
    protocol=lambda fn:next(n.value for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in n.targets))
    assert ast.dump(protocol(oldf['main']))==ast.dump(protocol(newf['main']))
