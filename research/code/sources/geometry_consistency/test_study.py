import numpy as np
from study_math import geometry,balanced_indices,pca_fit,pca_rgb

def test_separated_and_rotation_invariant():
    rng=np.random.default_rng(4);y=np.r_[np.ones(30,bool),np.zeros(30,bool)]
    r=rng.normal(0,.01,(60,4));r[y,0]+=4;r[~y,0]-=4
    q=r+rng.normal(0,.001,r.shape)
    a=geometry(r,q,y,y);u,_=np.linalg.qr(rng.normal(size=(4,4)))
    b=geometry(r@u,q@u,y,y)
    assert a['balanced_neighbor_accuracy']==1 and abs(a['balanced_neighbor_accuracy']-b['balanced_neighbor_accuracy'])<1e-9

def test_collapse_and_inactivity_not_rewarded():
    y=np.r_[np.ones(10,bool),np.zeros(10,bool)]
    a=geometry(np.ones((20,3)),np.ones((20,3)),y,y)
    assert abs(a['balanced_neighbor_accuracy']-.5)<1e-8
    a=geometry(np.ones((20,3)),np.zeros((20,3)),y,y)
    assert a['target_nonzero_rate']==0 and a['active_balanced_neighbor_accuracy'] is None
    assert abs(a['balanced_neighbor_accuracy']-.5)<1e-8

def test_normalization_does_not_convert_magnitude_into_geometry():
    y=np.r_[np.ones(20,bool),np.zeros(20,bool)]
    r=np.ones((40,3));r[y]*=10
    assert geometry(r,r,y,y)['balanced_neighbor_accuracy']>.99
    assert abs(geometry(r,r,y,y,True)['balanced_neighbor_accuracy']-.5)<1e-8

def test_sampling_and_pca_frozen():
    y=np.r_[np.ones(40,bool),np.zeros(60,bool)]
    a=balanced_indices(y,np.ones(100,bool),20,3)
    assert y[a].sum()==10 and np.array_equal(a,balanced_indices(y,np.ones(100,bool),20,3))
    r=np.random.default_rng(2).normal(size=(30,8));pc=pca_fit(r);before=pc['lo'].copy()
    assert pca_rgb(r*1e6,pc).shape==(30,3) and np.array_equal(before,pc['lo'])

def test_pixel_context_preserves_exact_target_and_backgrounds():
    from study_math import background_composites
    fg=np.full((5,5,3),127,np.uint8);a=np.zeros_like(fg);b=np.full_like(fg,255)
    mask=np.zeros((5,5),bool);mask[1:3,1:3]=True
    x,y=background_composites(fg,mask,a,b)
    assert np.array_equal(x[mask],fg[mask]) and np.array_equal(y[mask],fg[mask])
    assert np.array_equal(x[~mask],a[~mask]) and np.array_equal(y[~mask],b[~mask])

def test_pair_category_weighting_and_dataset_separation(tmp_path):
    import study,pandas as pd
    folder=tmp_path/'controlled_consistency';folder.mkdir()
    for seed in [21,22,23]:
        for method in ['agf_g','slics']:
            rows=[]
            for dataset in ['synthetic','pixel']:
                for category,n,value in [(0,2,0.),(1,8,10.)]:
                    for i in range(n):rows.append(dict(method=method,seed=seed,scope='fixed_category_readout',dataset=dataset,category=category,unit=category,target_drop=value+(1 if method=='agf_g' else 0)))
            study.done_json(folder/f'{method}_{seed}.json',rows)
    study.summarize(dict(out=tmp_path),'controlled_consistency')
    per=pd.read_csv(tmp_path/'controlled_consistency_per_run.csv')
    assert np.allclose(per[per.method=='agf_g'].target_drop,6.)
    paired=pd.read_csv(tmp_path/'controlled_consistency_paired_run_intervals.csv')
    assert len(paired)==2 and np.all(paired.paired_runs==3) and np.allclose(paired.GGF_minus_comparator,1.)

def test_committed_cache_corruption_rejected(tmp_path):
    import pytest
    from cv_math import Cache
    c=Cache(tmp_path/'local',tmp_path/'drive','p');c.put('x',v=np.eye(3))
    assert np.array_equal(c.get('x')['v'],np.eye(3))
    (tmp_path/'local/x.npz').write_bytes(b'broken')
    with pytest.raises(ValueError):c.get('x')
