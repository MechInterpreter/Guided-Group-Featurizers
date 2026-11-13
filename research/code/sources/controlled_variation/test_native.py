import copy,csv,json,zipfile
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from native_units import partial_nnls,ConceptUnits,summarize
from native_coherence import fit_pca,decoded_consistency,numeric,bsfunits,ALL_METHODS

def test_partial_nnls_matches_independent_scipy_with_dependent_atoms():
    from scipy.optimize import nnls
    gen=torch.Generator().manual_seed(5);B=torch.randn(3,4,9,generator=gen);B[1,3]=B[1,0];B[2]=0
    x=torch.randn(11,9,generator=gen);x[0]=0
    z,stats=partial_nnls(x,B,iterations=50)
    assert z.dtype==torch.float64 and stats['max_relative_kkt']<=1e-5
    for i in range(len(x)):
        for j in range(len(B)):
            expected,_=nnls(B[j].double().numpy().T,x[i].double().numpy(),maxiter=20000)
            assert np.allclose(z[i,j].numpy()@B[j].double().numpy(),expected@B[j].double().numpy(),atol=1e-4)

def test_concept_output_keeps_real_bias_gate_and_patch_aggregation():
    class Tokenizer:
        def __call__(self,x):return x.transpose(1,2),torch.tensor([1.,-2.,.5])[None,:,None].expand(len(x),-1,-1)
    # Tiny native token model; biased nonlinear projectors are deliberately retained.
    torch.manual_seed(3)
    native=SimpleNamespace(tokenizer=Tokenizer(),output_projectors=[torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.Tanh()) for i in range(3)],token_aggregator=torch.nn.Linear(3,4))
    model=SimpleNamespace(concept_model=native);x=np.arange(120,dtype=np.float32).reshape(10,4,3)/20
    score,parts=ConceptUnits(model,[2,0],'cpu').maps(x)
    mask,pres=native.tokenizer(torch.from_numpy(x));g=mask*pres
    assert np.allclose(score,g[:,[2,0]].transpose(1,2))
    for j,c in enumerate([2,0]):
        expected=native.token_aggregator.weight[:,c][None,:,None]*native.output_projectors[c](g[:,c])[:,None,:]
        assert np.allclose(parts[:,:,j],expected.detach().numpy())
    # Token contributions sum to the aggregator output minus its separate bias.
    _,allparts=ConceptUnits(model,[0,1,2],'cpu').maps(x)
    projected=torch.stack([native.output_projectors[c](g[:,c]) for c in range(3)],1)
    expected=native.token_aggregator(projected.transpose(1,2)).transpose(1,2)-native.token_aggregator.bias[None,:,None]
    assert np.allclose(allparts.sum(2),expected.detach().numpy(),atol=1e-6)

def test_shared_pooling_and_bsf_conversion_use_native_output_space():
    from test_semantic import fixture
    summaries,_,D,plans=fixture();row,obs=bsfunits(dict(kind='agf_g',seed=21),summaries,D,plans,list(range(6)))
    assert row['units']==list(range(6))
    for (_,s),(_,o) in zip(summaries,obs):assert np.allclose(o['vectors'],np.einsum('vgr,grd->vgd',s['pooled'],D))
    x=np.ones((2,4,3));f=np.tile([1,1,0,0],(2,1));score=np.ones((2,4,2));part=np.ones((2,4,2,3));part[:,:2]*=3
    s=summarize(x,f,score,part);assert np.allclose(s['vectors'],3) and np.allclose(s['contrasts'],2)

def test_full_output_trajectory_is_rotation_invariant_and_keeps_inactivity():
    rng=np.random.default_rng(4);z=rng.normal(size=(6,4,8));z[0]=0;Q=np.linalg.qr(rng.normal(size=(8,8)))[0]
    q=decoded_consistency(z,np.arange(6));q2=decoded_consistency(z@Q,np.arange(6))
    assert q['consistency']==pytest.approx(q2['consistency']) and q['zero_change_fraction']==pytest.approx(1/6)
    assert decoded_consistency(np.zeros_like(z),np.arange(6))['consistency']==0

def test_numeric_pca_is_train_only_and_three_coordinates_for_every_method():
    from test_semantic import fixture
    s,_,D,p=fixture();row,obs=bsfunits(dict(kind='agf_g',seed=21),s,D,p,list(range(6)))
    result,axes=numeric(row,obs,list(range(6)));assert all(q['worst_condition_AP']==1 for q in result['summary'])
    changed=copy.deepcopy(obs)
    for rec,o in changed:
        if rec['split']=='eval':o['vectors']*=100
    _,after=numeric(row,changed,list(range(6)))
    for c in axes:assert np.array_equal(axes[c]['V'],after[c]['V'])
    assert all(q['common_PC_dimensions']==3 for q in result['readouts'])

def test_bad_or_incomplete_competitor_states_rejected():
    from native_sources import validate_concept,check_context
    from coco_baselines import training
    from coco_baselines.storage import identity
    weights={'w':torch.ones(2)};state=dict(seed=21,protocol=dict(revision='concept-sae-development-repair-v2',epochs=[30,50,30]),stage_receipts=[dict(epochs_completed=n,best_diagnostics=dict(quality_pass=True)) for n in [30,50,30]],weights=weights,weights_sha=training.weight_sha(weights))
    validate_concept(state,21)
    for field,value in [('seed',22),('weights_sha','changed'),('stage_receipts',[dict(epochs_completed=10)]*3)]:
        bad=dict(state,**{field:value})
        with pytest.raises(ValueError):validate_concept(bad,21)
    ctx=dict(norm={'scale':1},packs={'train':{'identity':'train'},'select':{'identity':'dev'}})
    cfg=dict(train_identity='train',development_identity='dev',normalization_sha=identity(ctx['norm']));check_context(cfg,ctx)
    with pytest.raises(ValueError):check_context(dict(cfg,normalization_sha='other'),ctx)

def test_legacy_readthrough_never_writes_old():
    from legacy_cache import ReadThroughCache
    class C:
        def __init__(self,val):self.val=val;self.writes=[]
        def get(self,k):return self.val
        def put(self,*a,**kw):self.writes.append((a,kw));return kw
    new,old=C(None),C({'cached':True});c=ReadThroughCache(new,old)
    assert c.get('k')=={'cached':True};c.put('k',value=1);assert len(new.writes)==1 and not old.writes

def test_five_method_review_has_90_cards_and_eight_comparisons(tmp_path,monkeypatch):
    from test_semantic import fixture
    from native_coherence import make_cards,finish
    from semantic_review import score,SCORES
    import cv_study
    summaries,_,D,plans=fixture();_,obs=bsfunits(dict(kind='agf_g',seed=21),summaries,D,plans,list(range(6)))
    rgb=np.full((224,224,3),180,np.uint8);mask=np.zeros((224,224),np.uint8);mask[70:150,70:150]=1
    monkeypatch.setattr(cv_study,'source_crop',lambda *a:(rgb,mask))
    class Features:
        def get(self,k):return {'x':np.ones((20,196,8),np.float32)}
    class Adapter:
        def maps(self,x):return np.ones((len(x),196,6)),None
    row=dict(kind='agf_g',seed=21,units=list(range(6)),interface='fixture');ctx={'out':tmp_path}
    result,pcs=numeric(row,obs,list(range(6)));make_cards(ctx,row,Adapter(),obs,pcs,list(range(6)),Features(),None)
    root=tmp_path/'semantic_coherence';private=root/'private/review_key.json';key=json.loads(private.read_text());public=root/'blinded_review';results=[]
    for seed in [21,22,23]:
        for method in ALL_METHODS:
            q,_=numeric(dict(row,kind=method,seed=seed),obs,list(range(6)));results.append(q)
            if (method,seed)==('agf_g',21):continue
            for c in range(6):
                card=f'card{len(key)}';key[f'{method}_{seed}_{c}']=dict(card_id=card,model=method,seed=seed,category=c,group=c)
                original=next(public.glob('*.png')).read_bytes()
                for suffix in ['.png','_variation.png']:(public/(card+suffix)).write_bytes(original)
    private.write_text(json.dumps(key));finish(ctx,results)
    done=json.loads((root/'completion.json').read_text());assert set(done['expected_methods'])==set(ALL_METHODS)
    assert len(key)==90
    import pandas as pd
    assert len(pd.read_csv(root/'paired_primary_comparisons.csv'))==8
    with zipfile.ZipFile(root/'blinded_review_public.zip') as z:
        assert len([n for n in z.namelist() if n.endswith('.png')])==180
        assert not any('private' in n or 'key' in n for n in z.namelist())
    paths=[];bycard={v['card_id']:v for v in key.values()}
    for rater in ['A','B']:
        blank=public/('rater_'+rater+'_blank.csv');ratings=list(csv.DictReader(blank.open()))
        for r in ratings:
            for f in SCORES:r[f]=5 if bycard[r['card_id']]['model']=='agf_g' else 2
            r.update(common_concept_explanation='Synthetic test fixture '+rater,variation_explanation='Synthetic test fixture '+rater,confidence_1_to_3=2)
        dest=root/(rater+'.csv')
        with dest.open('w',newline='') as f:
            w=csv.DictWriter(f,list(ratings[0]));w.writeheader();w.writerows(ratings)
        paths.append(dest)
    assert len(score(root,*paths)['comparisons'])==8
