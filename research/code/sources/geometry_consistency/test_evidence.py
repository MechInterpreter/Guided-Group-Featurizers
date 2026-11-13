import copy,os
from pathlib import Path
import numpy as np
import pytest
import torch
from coco_baselines import training
from coco_adaptation.inputs import JournalStore
from coco_ggf.drive_recovery import install_drive_recovery
import analysis,reporting,supervised_control as sup,concept_sae as cs
from voc_training import partition,streaming_norm

@pytest.fixture
def native():
    path=os.environ.get('GGF_CONCEPT_SOURCE')
    if not path:raise RuntimeError('Provide the verified Concept-SAE source before running tests')
    return cs.source(path)

def test_weight_selection_development_guard_and_tie():
    def row(j,c,a,mse):return dict(candidate=j,reference_warm_mse=1.,development=dict(macro_capture=c,macro_ap=a,reconstruction=dict(mse=mse)))
    trials=[row(0,.2,.7,1.),row(1,.5,.8,1.1),row(2,.3,.75,1.01)]
    assert sup.choose(trials)['candidate']==2
    choice=sup.choose([row(0,.5,.9,1.5),row(1,.4,.8,1.2),row(2,.3,.8,1.3)])
    assert choice['candidate']==1 and not choice['reconstruction_guard_satisfied']
    assert sup.choose([row(j,.3,.8,1.) for j in range(3)])['candidate']==0

def test_VOC_split_is_disjoint_deterministic_and_annotation_blind():
    rr=[dict(image=str(j),official_split='train') for j in range(700)]+[dict(image='v'+str(j),official_split='val') for j in range(100)]
    a=partition(rr);b=partition(list(reversed(rr)));assert a==b
    assert list(a.values()).count('select')==256 and list(a.values()).count('train')==444
    assert all(a['v'+str(j)]=='eval' for j in range(100))

def test_VOC_normalization_uses_only_fit_and_matches_direct():
    x=np.random.default_rng(4).normal(size=(9,4,8)).astype(np.float16);idx=[0,2,3,5]
    a=streaming_norm(x,idx);train=x[idx].astype(np.float32);mu=train.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.sqrt(8/np.square(train.astype(float)-mu).sum(-1).mean())
    assert np.array_equal(a['mean'],mu) and a['scale']==pytest.approx(scale,rel=1e-12)
    x[[1,4,6,7,8]]=5000;b=streaming_norm(x,idx);assert np.array_equal(a['mean'],b['mean']) and a['scale']==b['scale']

@pytest.mark.parametrize('spectrum,entropy,pr,sr',[(np.ones(4),4,4,4),(np.array([4,0,0]),1,1,1),(np.array([3,1]),np.exp(-(.75*np.log(.75)+.25*np.log(.25))),1.6,4/3)])
def test_rank_conventions(spectrum,entropy,pr,sr):
    r=analysis.rank_stats(np.diag(spectrum));assert r['entropy_rank']==pytest.approx(entropy);assert r['participation_ratio']==pytest.approx(pr);assert r['coefficient_matrix_stable_rank']==pytest.approx(sr)
    assert r['covariance_stable_rank']==pytest.approx(np.square(spectrum).sum()/spectrum.max()**2)

def test_zero_variance_is_not_zero_rank():
    r=analysis.rank_stats(np.zeros((3,3)));assert r['zero_variance'] and r['entropy_rank'] is None

def test_native_curve_keeps_cancellation_and_negative_capture():
    target=np.array([[1.,0.],[0.,1.]])
    den,err=analysis.curve(target,[3*target,-2*target]);assert np.array_equal(den,[1,1]);assert np.array_equal(err,[[4,4],[0,0]])

def test_paired_bootstrap_identical_models_exact_zero():
    a=dict(den=np.ones((2,8)),errors=np.ones((2,4,8)),image_ids=list(range(8)),category_indices=[0,1]);r=reporting.paired_curves(a,a,21)
    assert r['difference_1_2_4']==[0,0,0] and np.array_equal(r['image_bootstrap95'],np.zeros((2,3)))
    b=copy.deepcopy(a);b['image_ids'][0]=999
    with pytest.raises(ValueError,match='cohorts'):reporting.paired_curves(a,b,21)

def test_probe_projection_equals_explicit_residual_removal():
    rng=np.random.default_rng(8);z=rng.normal(size=(7,5,3));D=rng.normal(size=(5,3,9));W=rng.normal(size=(9,4));x=rng.normal(size=(7,9))
    projected=np.einsum('grd,dc->grc',D,W);effect=np.einsum('pgr,grc->pgc',z,projected)
    for g in range(5):assert np.allclose((x-z[:,g]@D[g])@W,x@W-effect[:,g])

def test_seed_uncertainty_is_t_not_image_interval():
    r=reporting.stat([1,2,3]);assert r['sd']==1 and r['ci95'][1]-2==pytest.approx(4.302652729911275/np.sqrt(3))

def toy_pack(n=5,p=4,d=8):
    x=np.random.default_rng(8).normal(size=(n,p,d)).astype(np.float32);mm=[]
    for i in range(n):
        f=np.zeros((p,2),np.uint8);f[:2,0]=255;f[2:,1]=255;crowd=np.zeros(p,np.float32)
        if i%2:crowd[-1]=1
        mm.append(dict(fraction=f,crowd=crowd))
    return dict(x=x,ids=list(range(n)),masks=mm,identity='toy-independent-training')

@pytest.mark.parametrize('stage',[1,2,3])
def test_native_gradient_accumulation_matches_effective_batch(native,stage):
    torch.set_num_threads(1);torch.manual_seed(3);a=cs.build(native,2,p=4,d=8,seg=2,free=2);b=copy.deepcopy(a);p=toy_pack()
    aa=cs.configure(a,stage);bb=cs.configure(b,stage);oa=torch.optim.Adam([v for v in aa.parameters() if v.requires_grad],lr=.001);ob=torch.optim.Adam([v for v in bb.parameters() if v.requires_grad],lr=.001)
    va=cs.optimizer_step(a,aa,oa,p,list(range(5)),[0,1],'cpu',stage,microbatch=5);vb=cs.optimizer_step(b,bb,ob,p,list(range(5)),[0,1],'cpu',stage,microbatch=2)
    assert va==pytest.approx(vb,rel=2e-6)
    for ka,v in a.state_dict().items():assert torch.allclose(v,b.state_dict()[ka],atol=2e-6,rtol=2e-6),ka
    assert all(v.grad is None for v in a.parameters() if not v.requires_grad)

def test_released_native_additive_parts_match_unchanged_forward(native):
    torch.manual_seed(4);m=cs.build(native,2,p=4,d=8,seg=2,free=2);x=torch.randn(3,4,8);pred,cp,A,mask=cs.parts(m,x)
    actual=m(x)[0];assert torch.allclose(pred,actual,atol=1e-6)
    assert mask.shape==(3,2,4)
    nativepart=torch.einsum('bd,p->bpd',cp[:,0],A[:,0]);assert nativepart.shape==actual.shape

def test_stable_KL_matches_released_formula(native):
    m=cs.build(native,2,p=4,d=8,seg=2,free=2);merge=torch.cat([l[0].weight.detach() for l in m.concept_model.tokenizer.token_merge]);A=m.concept_model.token_aggregator.weight.T
    p=torch.softmax(merge.abs()/.02,1);q=torch.softmax(A.abs()/.02,1);old=(p*torch.log(p/q)).sum(1).mean()
    lp=torch.log_softmax(merge.abs()/.02,1);lq=torch.log_softmax(A.abs()/.02,1);new=(lp.exp()*(lp-lq)).sum(1).mean();assert float(old.detach())==pytest.approx(float(new.detach()),rel=2e-6)

def test_native_three_stage_resume_matches_uninterrupted(tmp_path,native,monkeypatch):
    torch.set_num_threads(1);original_build=cs.build
    monkeypatch.setattr(cs,'build',lambda mod,C:original_build(mod,C,p=4,d=8,seg=2,free=2))
    def ctx(path):
        s=JournalStore(path/'drive',dict(native_test=True),path/'local',False);install_drive_recovery(s)
        return dict(store=s,options=dict(concept_source=os.environ['GGF_CONCEPT_SOURCE']),cfg=dict(category_indices=[0,1]),packs=dict(train=toy_pack(),select=toy_pack()),device='cpu')
    a=ctx(tmp_path/'full');ma,ra=cs.fit(a,21);del ma
    b=ctx(tmp_path/'resume');step=cs.optimizer_step;calls=[]
    def stop(*args,**kw):
        calls.append(1)
        if len(calls)==4:raise KeyboardInterrupt()
        return step(*args,**kw)
    monkeypatch.setattr(cs,'optimizer_step',stop)
    with pytest.raises(KeyboardInterrupt):cs.fit(b,21)
    monkeypatch.setattr(cs,'optimizer_step',step);mb,rb=cs.fit(b,21)
    assert ra['weights_sha']==rb['weights_sha']
    for stage in [1,2,3]:
        x=a['store'].load(f'concept_sae_s21_stage{stage}');y=b['store'].load(f'concept_sae_s21_stage{stage}');assert x['history']==y['history']

def test_invalid_upstream_source_is_rejected(tmp_path):
    p=tmp_path/'replace_model.py';p.write_text('raise RuntimeError("untrusted")')
    with pytest.raises(ValueError,match='changed'):cs.source(p)

def test_main_protocol_documents_claim_limits():
    import experiment
    r=experiment.recipe();assert r['groups']==1024 and r['total_updates']==10640 and r['no_time_limit']
    assert 'fixed' in r['supervised_vanilla'] and 'no controller' in r['supervised_vanilla']
    assert 'own FIT-only' in r['voc'] and 'Concept-SAE' in r['additional_supervised']

def test_primary_freezes_all_three_seeds_before_scoring(tmp_path,monkeypatch):
    import checklist_run as run,experiment
    from test_experiment import FakeStore
    events=[];store=FakeStore();store.results=tmp_path;ctx=dict(store=store)
    monkeypatch.setattr(experiment,'endpoint',lambda c,s,k:events.append(('fit',s,k)) or dict(seed=s,kind=k))
    monkeypatch.setattr(experiment,'check_pairing',lambda *a:None)
    monkeypatch.setattr(experiment,'archive',lambda *a:None)
    monkeypatch.setattr(experiment,'freeze',lambda c,r,n:events.append(('freeze',len(r))))
    monkeypatch.setattr(run.evaluation,'coco_test',lambda c:{})
    monkeypatch.setattr(run.evaluation,'score',lambda *a:events.append(('score',)))
    monkeypatch.setattr(run.evaluation,'intervals',lambda *a:events.append(('intervals',)))
    monkeypatch.setattr(run.analysis,'run',lambda *a:events.append(('interpretation',)))
    monkeypatch.setattr(run.reporting,'checklist',lambda *a:None)
    import contextlib
    monkeypatch.setattr(run.reporting,'cost',lambda *a:contextlib.nullcontext())
    rows,_=run.primary(ctx,'coco',None)
    assert len(rows)==9 and events.index(('freeze',9))<events.index(('score',))<events.index(('interpretation',))

def test_real_consolidation_rank_probe_and_bootstrap_pipeline(tmp_path):
    # Tiny actual BSF, 196 real patch positions, native codes/decoder, two
    # annotation-defined concepts. No mock of analysis mathematics or storage.
    from test_scale_study import context
    ctx=context(tmp_path);ctx['stamps']=[]
    m=__import__('scale_models').model('released_vanilla',6,2,2,4)
    row=dict(key='actual_toy',kind='released_vanilla',seed=21,weights_sha=training.weight_sha(training.weights(m)))
    plan=analysis.train_plan(ctx,row,m)
    assert len(plan['plans'])==2 and all(1<=len(p['groups'])<=4 for p in plan['plans'])
    rank=analysis.ranks(ctx,row,m,plan);assert len(rank['blocks'])==6
    probe=analysis.fit_probe(ctx);assert probe['W'].shape==(5,2)
    result=analysis.heldout(ctx,row,m,ctx['packs']['select'],plan,probe,'toy')
    assert len(result['categories'])==2 and np.isfinite(result['macro_capture_1_2_4']).all()
    assert all(len(r['random_removed_AP'])==4 for r in result['categories'])
    # Completed audits are reused unchanged without encoding test images again.
    original=m.encode
    def forbidden(*a):raise AssertionError('Recomputed completed inference')
    m.encode=forbidden
    assert analysis.heldout(ctx,row,m,ctx['packs']['select'],plan,probe,'toy')==result
    m.encode=original

def test_fixed_weight_fit_really_has_heads_no_controller_and_exact_resume(tmp_path):
    from test_scale_study import context,targets
    from scale_fit import fit
    from scale_models import model
    torch.manual_seed(5);initial=training.weights(model('released_vanilla',6,2,2,4))
    a=context(tmp_path/'full');a['rule']['fixed_weights']=[1.,.2,.2]
    b=context(tmp_path/'resume');b['rule']['fixed_weights']=[1.,.2,.2]
    full=fit(a,'fixed','released_vanilla',6,2,21,initial,6,6,targets=targets(),fixed=True,stream=1)
    fit(b,'fixed','released_vanilla',6,2,21,initial,6,3,targets=targets(),fixed=True,stream=1)
    resumed=fit(b,'fixed','released_vanilla',6,2,21,initial,6,6,targets=targets(),fixed=True,stream=1)
    assert full['balance'] is None and tuple(full['head'].shape)==(2,2)
    assert training.weight_sha(full['weights'])==training.weight_sha(resumed['weights']) and torch.equal(full['head'],resumed['head'])

def test_pairing_enforces_heads_and_fixed_vs_adaptive_controller():
    import experiment as exp
    from coco_baselines.storage import identity
    from test_experiment import FakeStore
    targets=[dict(category_index=i,contrast_group=i,detector_group=i) for i in range(2)]
    w={'toy':torch.tensor([1.])};sha=training.weight_sha(w);values={};rows=[]
    for kind in exp.PRIMARY:
        key='toy_'+kind;supervised=kind!='released_vanilla'
        values[key]=dict(updates=3000,weights=w,sampled_batches_sha='same',sampled_noise_sha='same',
            head=torch.zeros(2,2) if supervised else None,balance={} if kind=='agf_g' else None)
        rows.append(dict(key=key,kind=kind,method='released_vanilla',seed=21,groups=1024,rank=12,active_groups=16,
            additional_updates=3000,total_training_updates=10640,cap=.4 if kind=='agf_g' else None,
            targets=targets,targets_sha=identity(targets),weights_sha=sha,warm_key='g1024_k16_s21_released_vanilla_warm',warm_weights_sha=sha))
    for base in ['mask','released_vanilla']:
        values['g1024_k16_s21_'+base+'_warm']=dict(updates=7640,weights=w,sampled_batches_sha='warm_same',sampled_noise_sha='warm_same',head=None,balance=None)
    ctx=dict(store=FakeStore(values),stamps=[],cfg=dict(category_indices=[0,1]));exp.check_pairing(ctx,rows)
    values['toy_supervised_vanilla']['balance']={}
    with pytest.raises(ValueError,match='classifier/controller'):exp.check_pairing(dict(ctx,_pairing_proofs={}),rows)

def test_vectorized_tied_bootstrap_matches_original_reference():
    from coco_ggf import study
    from bootstrap_fast import paired
    rng=np.random.default_rng(9);N=12;P=120;C=3
    positive=rng.random((P,C))<.35;valid=rng.random((P,C))<.9
    den=rng.random((N,C))+.1
    a=dict(image_ids=list(range(N)),category_indices=list(range(C)),objects=[7,8,9],
        positive=positive,valid=valid,patch_image_index=np.repeat(np.arange(N),10),
        denominator=den,error=rng.random((N,C)),detection_scores=rng.integers(0,5,size=(P,C)).astype(float))
    b=dict(a,error=rng.random((N,C)),detection_scores=rng.integers(0,5,size=(P,C)).astype(float))
    cfg=dict(bootstrap=37,fresh_test_seed=19)
    old=study.paired(a,b,cfg);new=paired(a,b,cfg,'cpu')
    assert old['valid_bootstrap_replicates']==new['valid_bootstrap_replicates']
    for key in ['capture_gain_ci95','ap_gain_ci95']:assert np.allclose(old[key],new[key],atol=1e-12)
    for x,y in zip(old['per_category'],new['per_category']):
        for key in ['capture_gain_ci95','ap_gain_ci95']:assert np.allclose(x[key],y[key],atol=1e-12)

def test_representative_plots_export(tmp_path):
    def s(v):return dict(mean=v,sd=1.,ci95=[v-2,v+2])
    means=[dict(kind=k,label=l,capture=s(40),patch_AP=s(70),VE=s(65)) for k,l in [('released_vanilla','Vanilla BSF'),('supervised_vanilla','Supervised Vanilla'),('agf_g','GGF'),('concept_sae','Concept-SAE (released; DINO/mask adaptation)')]]
    inter=[dict(kind=r['kind'],label=r['label'],capture_1_2_4=[s(35),s(43),s(48)],target_AP_drop=s(9),random_AP_drop=s(2)) for r in means[:3]]
    reporting.plot(tmp_path,means,inter)
    assert all((tmp_path/n).stat().st_size>1000 for n in ['baseline_comparison.png','baseline_comparison.pdf','consolidation_and_interventions.png','consolidation_and_interventions.pdf'])

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA parity/resume preflight executes in Colab before real fitting')
def test_CUDA_bootstrap_parity_and_fixed_fit_resume(tmp_path):
    from bootstrap_fast import paired
    rng=np.random.default_rng(13);N=8;P=80;C=2
    a=dict(image_ids=list(range(N)),category_indices=[0,1],objects=[1,2],positive=rng.random((P,C))<.4,
        valid=rng.random((P,C))<.9,patch_image_index=np.repeat(np.arange(N),10),
        denominator=rng.random((N,C))+.1,error=rng.random((N,C)),detection_scores=rng.integers(0,4,size=(P,C)))
    b=dict(a,detection_scores=rng.integers(0,4,size=(P,C)),error=rng.random((N,C)))
    cfg=dict(bootstrap=19,fresh_test_seed=8);cpu=paired(a,b,cfg,'cpu');gpu=paired(a,b,cfg,'cuda')
    assert np.allclose(cpu['ap_gain_ci95'],gpu['ap_gain_ci95'],atol=1e-10)
    from test_scale_study import context,targets
    from scale_models import model
    from scale_fit import fit
    initial=training.weights(model('released_vanilla',6,2,2,4))
    aa=context(tmp_path/'full');bb=context(tmp_path/'resume');aa['device']=bb['device']='cuda'
    full=fit(aa,'fixed','released_vanilla',6,2,21,initial,4,4,targets=targets(),fixed=True,stream=1)
    fit(bb,'fixed','released_vanilla',6,2,21,initial,4,2,targets=targets(),fixed=True,stream=1)
    resumed=fit(bb,'fixed','released_vanilla',6,2,21,initial,4,4,targets=targets(),fixed=True,stream=1)
    assert training.weight_sha(full['weights'])==training.weight_sha(resumed['weights'])

def test_full_run_uses_separate_VOC_training_and_prioritizes_evidence(tmp_path,monkeypatch):
    import contextlib,json,checklist_run as run,voc_training,backbone
    from test_experiment import FakeStore
    events=[]
    def context(name):
        s=FakeStore();s.results=tmp_path/name;s.results.mkdir();s.root=tmp_path/name/'checkpoints';s.run_id=name;s.output=tmp_path/name
        return dict(store=s,packs={},device='cuda',options=dict(source_output='old'),local=tmp_path/name)
    co=context('coco');vo=context('voc')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'Tesla T4')
    monkeypatch.setattr(run.exp,'prepare',lambda options:co);monkeypatch.setattr(run,'earlier',lambda ctx:None)
    def primary(ctx,cohort,coco):
        events.append(('primary',cohort));return [dict(seed=s,kind=k) for s in [21,22,23] for k in run.exp.PRIMARY],{}
    monkeypatch.setattr(run,'primary',primary)
    monkeypatch.setattr(backbone,'annotations',lambda *a:'annotation_metadata')
    monkeypatch.setattr(backbone,'backbone_preflight',lambda *a:None)
    def prepare(*a):events.append(('VOC_training_inputs',));return vo
    monkeypatch.setattr(voc_training,'prepare',prepare)
    monkeypatch.setattr(run.reporting,'cost',lambda *a:contextlib.nullcontext())
    monkeypatch.setattr(run.reporting,'checklist',lambda ctx,cohort,*a:events.append(('coverage',cohort)))
    monkeypatch.setattr(run.analysis,'run',lambda *a:None)
    monkeypatch.setattr(run.exp,'endpoint',lambda ctx,s,k:events.append(('supporting_fit',s,k)) or dict(seed=s,kind=k))
    monkeypatch.setattr(run.exp,'freeze',lambda *a:None)
    monkeypatch.setattr(run.exp,'archive',lambda ctx,*a:ctx['store'].results/'results.zip')
    monkeypatch.setattr(run.evaluation,'score',lambda *a:None);monkeypatch.setattr(run.evaluation,'intervals',lambda *a:None)
    monkeypatch.setattr(run.matched,'assert_sources',lambda *a:None)
    monkeypatch.setattr(run.matched,'versions',lambda:{})
    monkeypatch.setattr(cs,'run',lambda ctx,pack,cohort:events.append(('Concept_SAE',cohort)))
    receipt=tmp_path/'receipt.json'
    run.main(dict(source_output='old',receipt=str(receipt)))
    assert events.index(('primary','coco'))<events.index(('VOC_training_inputs',))<events.index(('primary','voc'))<events.index(('Concept_SAE','coco'))
    support=[e for e in events if e[0]=='supporting_fit'];assert len(support)==12
    assert events.index(('Concept_SAE','voc'))<events.index(support[0])
    result=json.loads(receipt.read_text());assert result['complete'] and result['VOC_results']!=result['results']

def test_result_tables_and_coverage_use_actual_completed_receipts(tmp_path,monkeypatch):
    from test_scale_study import context
    from experiment import PRIMARY
    ctx=context(tmp_path);s=ctx['store']
    for seed in [21,22,23]:
        for j,kind in enumerate(PRIMARY):
            key=f'm_{seed}_{kind}';row=dict(key=key,weights_sha=key)
            s.save(f'endpoint_{seed}_{kind}',row)
            report=dict(seed=seed,macro_capture=.3+.01*j,macro_ap=.7+.01*j,reconstruction=dict(ve=.65),total_learned_parameters=100)
            s.save(f'point_coco_{seed}_{kind}',dict(report=report))
            s.save('interpretability_coco_'+key,dict(macro_capture_1_2_4=[.3,.4,.5],macro_target_AP_drop=.05,macro_random_AP_drop=.01))
            s.save('consolidation_compact_coco_'+key,dict(den=np.ones((2,6)),errors=np.ones((2,4,6))*(1-.1*j),image_ids=list(range(6)),category_indices=[0,1]))
    out=reporting.summary(ctx,'coco')
    assert len(out['means'])==3 and len(out['paired_metric_differences'])==2
    assert (s.results/'paper_results_mean_sd.csv').is_file()
    latex=(s.results/'paper_results.tex').read_text()
    header=next(line for line in latex.splitlines() if line.startswith('Method'))
    assert header.endswith(chr(92)*2)
    monkeypatch.setattr(reporting,'paired_curves',lambda *a:(_ for _ in ()).throw(AssertionError('Recomputed cached CI')))
    out2=reporting.summary(ctx,'coco');assert out2['paired_consolidation']==out['paired_consolidation']
    reporting.checklist(ctx,'coco')
    coverage=__import__('json').loads((s.results/'checklist_coverage.json').read_text())
    assert any(r['status']=='manual paper revision remains necessary' for r in coverage['items'])

def test_fixed_control_propagates_interrupted_fit_key_for_outer_flush(monkeypatch):
    from test_experiment import FakeStore
    weights={'toy':torch.tensor([1.])};targets=[dict(category_index=0)]
    ctx=dict(store=FakeStore({'g1024_k16_s21_released_vanilla_targets':targets}),rule={},
        packs=dict(train=dict(x=np.zeros((1,196,4)))),device='cpu',active_key=None)
    monkeypatch.setattr(sup.scale_run,'initial',lambda *a,**k:weights)
    def stop(cc,key,*a,**k):
        if key.endswith('_warm'):return dict(weights=weights)
        cc['active_key']=key;raise KeyboardInterrupt()
    monkeypatch.setattr(sup,'fit',stop)
    with pytest.raises(KeyboardInterrupt):sup.trial(ctx,21,0)
    assert ctx['active_key']=='g1024_k16_s21_released_vanilla_fixed_supervision_candidate0'

def test_native_epoch_validation_interrupt_does_not_skip_selection(tmp_path,native,monkeypatch):
    torch.set_num_threads(1);original_build=cs.build
    monkeypatch.setattr(cs,'build',lambda mod,C:original_build(mod,C,p=4,d=8,seg=2,free=2))
    def ctx(path):
        s=JournalStore(path/'drive',dict(validation_boundary_test=True),path/'local',False);install_drive_recovery(s)
        return dict(store=s,options=dict(concept_source=os.environ['GGF_CONCEPT_SOURCE']),cfg=dict(category_indices=[0,1]),packs=dict(train=toy_pack(),select=toy_pack()),device='cpu')
    a=ctx(tmp_path/'full');ma,ra=cs.fit(a,21);del ma
    b=ctx(tmp_path/'resume');validation=cs.validation;calls=[]
    def stop(*args,**kwargs):
        calls.append(1)
        if len(calls)==3:raise KeyboardInterrupt()
        return validation(*args,**kwargs)
    monkeypatch.setattr(cs,'validation',stop)
    with pytest.raises(KeyboardInterrupt):cs.fit(b,21)
    assert b['store'].load('concept_sae_s21_stage1')['updates']==2
    monkeypatch.setattr(cs,'validation',validation);mb,rb=cs.fit(b,21)
    assert ra['weights_sha']==rb['weights_sha']
    for stage in [1,2,3]:
        assert a['store'].load(f'concept_sae_s21_stage{stage}')['history']==b['store'].load(f'concept_sae_s21_stage{stage}')['history']






