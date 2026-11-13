import csv,json,zipfile
from pathlib import Path
import numpy as np
import pytest
from semantic_math import *

def test_AP_ties_and_permutation():
    y=np.array([1,0,1,0],bool);s=np.array([1.,1.,0.,0.]);p=[3,0,2,1]
    assert average_precision(y,s)==pytest.approx(.5)
    assert average_precision(y[p],s[p])==average_precision(y,s)
    assert average_precision(y,y)==1
    with pytest.raises(ValueError):average_precision([1,1],[0,1])

def test_consistency_rotation_invariance_and_no_zero_omission():
    rng=np.random.default_rng(5);z=rng.normal(size=(12,4,3));D=rng.normal(size=(3,8));Q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    a=trajectory_consistency(z,D);b=trajectory_consistency(z@Q,Q.T@D)
    assert a['consistency']==pytest.approx(b['consistency'])
    same=np.ones((4,2,3));assert trajectory_consistency(same,D)['consistency']==pytest.approx(1)
    same[0]=0;zero=trajectory_consistency(same,D)
    assert zero['consistency']==pytest.approx(.5) and zero['zero_change_fraction']==.25
    assert trajectory_consistency(same*0,D)['consistency']==0

def test_repeated_image_is_not_a_new_pair():
    z=np.ones((2,4,3));D=np.eye(3)
    assert trajectory_consistency(z,D,ids=[5,5])['consistency'] is None

def test_paired_and_multiplicity():
    q=paired_summary([2,3,4],[1,1,1]);assert q['difference']==2
    assert q['paired_t95_low']<2<q['paired_t95_high']
    np.testing.assert_allclose(holm([.01,.04,.03,.2]),[.04,.09,.09,.2])
    with pytest.raises(ValueError):paired_summary([1],[1])

def test_review_agreement():
    assert quadratic_kappa([1,2,3,4,5],[1,2,3,4,5])==1
    assert quadratic_kappa([1,2,3,4,5],[5,4,3,2,1])<0

def fixture():
    from cv_math import pca_from_stats
    rng=np.random.default_rng(4);D=rng.normal(size=(6,3,8));plans={c:dict(groups=[c]) for c in range(6)}
    summaries=[];pcas={}
    for c in range(6):
        z=rng.normal(size=(100,3));pcas[c,c]=pca_from_stats(len(z),z.sum(0),z.T@z,D[c])
        for split in ['train','eval']:
            for i in range(3):
                norm=np.ones((20,6));norm[:,c]=3
                pooled=rng.normal(size=(20,6,3))
                summaries.append((dict(category=c,category_name=str(c),split=split,image_id=c*100+i+(50 if split=='eval' else 0),annotation_id=c*100+i),dict(norm=norm,norm_rms=norm,gids=np.arange(6),pooled=pooled)))
    return summaries,pcas,D,plans

def test_numeric_reads_fixed_cross_category_groups(tmp_path):
    from semantic_study import numeric
    summaries,pcas,D,plans=fixture();original=json.dumps(plans)
    q=numeric(dict(kind='agf_g',seed=21),summaries,D,plans,tmp_path)
    assert len(q['summary'])==6 and all(r['worst_condition_AP']==1 for r in q['summary'])
    assert json.dumps(plans)==original
    # Changing training codes cannot change a diagnostic that uses evaluation
    # trajectories and frozen group assignments (only sensitivity scaling).
    for r,s in summaries:
        if r['split']=='train':s['pooled']*=100
    q2=numeric(dict(kind='agf_g',seed=21),summaries,D,plans,tmp_path)
    assert q['summary']==q2['summary']

def test_blind_cards_complete_inventory_and_review(tmp_path,monkeypatch):
    import torch
    import cv_study
    from semantic_study import numeric,make_cards,finish
    from semantic_review import score,read_ratings,SCORES
    summaries,pcas,D,plans=fixture()
    rgb=np.full((224,224,3),200,np.uint8);mask=np.zeros((224,224),np.uint8);mask[60:160,60:160]=1
    monkeypatch.setattr(cv_study,'source_crop',lambda *a:(rgb,mask))
    class Features:
        def get(self,k):return dict(x=np.ones((20,196,8),np.float32))
    class Model:
        def encode(self,x):return torch.ones((196,6,3))
    ctx=dict(out=tmp_path,device='cpu');results=[]
    # Exercise real card production once, then duplicate the fixture's card
    # content with different anonymous identities to test full paired scoring.
    row=dict(kind='agf_g',seed=21)
    make_cards(ctx,row,Model(),summaries,pcas,D,plans,Features(),None)
    root=tmp_path/'semantic_coherence';private=root/'private/review_key.json';key=json.loads(private.read_text());public=root/'blinded_review'
    assert len(key)==6
    for seed in [21,22,23]:
        for method in ['agf_g','supervised_vanilla','released_vanilla']:
            results.append(numeric(dict(kind=method,seed=seed),summaries,D,plans,root))
            if (method,seed)==('agf_g',21):continue
            for c in range(6):
                card=f'{seed}{method[0]}{len(key)}';key[f'{method}_{seed}_{c}']=dict(card_id=card,model=method,seed=seed,category=c,group=c)
                original=next(public.glob('*.png')).read_bytes();(public/(card+'.png')).write_bytes(original);(public/(card+'_variation.png')).write_bytes(original)
    private.write_text(json.dumps(key));finish(ctx,results)
    with zipfile.ZipFile(root/'blinded_review_public.zip') as z:
        assert len([n for n in z.namelist() if n.endswith('.png')])==108
        assert not any('private' in n or 'key' in n for n in z.namelist())
        assert all(not any(m in z.read(n).decode() for m in ['agf_g','supervised_vanilla','released_vanilla']) for n in z.namelist() if n.endswith(('.html','.csv')))
    assert not json.loads((root/'completion.json').read_text())['human_ratings_complete']
    paths=[]
    for rater in ['A','B']:
        path=public/('rater_'+rater+'_blank.csv');rows=list(csv.DictReader(path.open()))
        with pytest.raises(ValueError):read_ratings(path,[v['card_id'] for v in key.values()])
        mapping={v['card_id']:v for v in key.values()}
        for r in rows:
            for f in SCORES:r[f]=5 if mapping[r['card_id']]['model']=='agf_g' else 2
            r.update(common_concept_explanation='fixture '+rater,variation_explanation='fixture '+rater,confidence_1_to_3=2)
        filled=public/('filled_'+rater+'.csv')
        with filled.open('w',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
        paths.append(filled)
    scored=score(root,*paths);assert scored['complete'] and len(scored['comparisons'])==4
    assert all(q['difference']==3 for q in scored['comparisons'])
