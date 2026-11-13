"""CPU synthetic checks for width-five exchange algebra, guards, and resume."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from coco_adaptation import core,inputs
from coco_baselines import metrics,training
from coco_baselines.storage import identity
from coco_ggf import five_exchanges as ex,exchange5_ops as ops


def fixture(rank=5):
    gen=torch.Generator().manual_seed(13);d=6;groups=4;atoms=groups*rank
    D=torch.randn(atoms,d,generator=gen);D=D/D.norm(dim=1,keepdim=True)
    w=dict(W_enc=torch.randn(d,atoms,generator=gen)*.2,W_dec=D,b_enc=torch.randn(atoms,generator=gen)*.1)
    targets=[dict(category_index=c,category=f'category{c}',contrast_group=c,detector_group=c,heat_mean=0.,heat_scale=1.) for c in range(2)]
    root=dict(weights=w,head=torch.zeros(2,2),spec=dict(kind='native',groups=groups,rank=rank,active=2,targets=targets,atom_ids=list(range(atoms))),
        source_weights_sha=training.weight_sha(w),source_head_sha='head',source_balance_sha='controller',source_optimizer_sha='adam',reference_mse=.31)
    cfg=dict(category_indices=[0,1],patch_sample_seed=61002,detection_patches=32,positive_fraction=.5,negative_fraction=.1,
        min_object_patch_mass=2.,min_background_patches=4,decision_min_images=1,batch=128,device='cpu',target_min_images=1,
        audit_image_batch=2,fresh_test_seed=99,bootstrap=10,score_ap_weight=.2,preview_images=0)
    masks=[];rng=np.random.default_rng(17);x=rng.normal(size=(4,196,d)).astype(np.float32)
    for i in range(4):
        frac=np.zeros((196,2),np.uint8);frac[:65,0]=255;frac[70:140,1]=255
        masks.append(dict(fraction=frac,crowd=np.zeros(196,np.float32),instances=[dict(category=c,annotation_id=2*i+c,fraction=frac[:,c].astype(np.float32)/255.) for c in range(2)]))
    pack=dict(x=x,ids=[101,102,103,104],masks=masks)
    ctx=dict(base=dict(groups=groups,block_rank=rank,active_groups=2),cfg=cfg,device='cpu',initial=root,
        protocol=dict(ex.settings(),block_rank=rank,groups=groups,active_groups=2,shard_images=2),packs=dict(train=pack,train_screen=pack,select=pack,confirm=pack))
    return ctx,root,pack


@pytest.mark.parametrize('rank',[3,4,5])
def test_swap_preserves_values_and_ungated_prediction(rank):
    ctx,root,pack=fixture(rank);sha=training.weight_sha(root['weights'])
    changed=ops.permute_snapshot(root,1,rank+2);ex.assert_permutation(root,changed)
    m,n=ops.model(ctx,root),ops.model(ctx,changed);x=torch.from_numpy(pack['x'][0])
    with torch.no_grad():
        a=(x@m.W_enc+m.b_enc)@m.W_dec;b=(x@n.W_enc+n.b_enc)@n.W_dec
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
        assert m.encode(x).shape==(196,4,rank)
        assert (m.encode(x).norm(dim=-1)>0).sum(1).max()<=2
    restored=ops.permute_snapshot(changed,1,rank+2)
    assert training.weight_sha(restored['weights'])==sha==training.weight_sha(root['weights'])
    assert torch.equal(changed['head'],root['head'])
    with pytest.raises(ValueError):ops.permute_snapshot(root,0,rank-1)
    with pytest.raises(ValueError):ops.permute_snapshot(root,-1,rank)
    with pytest.raises(ValueError):ops.permute_snapshot(root,0,4*rank)
    changed['weights']['W_dec'][0,0]+=.1
    with pytest.raises(ValueError):ex.assert_permutation(root,changed)


@pytest.mark.parametrize('rank',[3,5])
def test_profile_predictions_match_explicit_additive_contrasts(rank):
    ctx,root,pack=fixture(rank);m=ops.model(ctx,root)
    stats=ops.shard_stats(m,pack,0,4,ctx['cfg'])
    assert stats['remove'].shape==(2,rank) and stats['cross'].shape==(2,rank,4*rank)
    assert np.isfinite(stats['cross']).all()
    for key in ['gain','remove','cross']:stats[key]/=stats['energy'].reshape((-1,)+(1,)*(stats[key].ndim-1))
    stats['readout_groups']=np.array([0,1]);stats['category_indices']=np.array([0,1])
    proposals=ops.rank_proposals(stats,root['spec']['targets'],dict(ctx['protocol'],top_incoming=2))
    assert proposals
    for p in proposals[:12]:
        numerator=np.zeros(2);den=np.zeros(2);out,inc=p['outgoing'],p['incoming'];receiver=out//rank;donor=inc//rank
        with torch.no_grad():
            for i in range(4):
                x=torch.from_numpy(pack['x'][i]);raw=x@m.W_enc+m.b_enc;z=m.encode(x).reshape(196,-1)
                weights,_,cats=core.object_weights([pack['masks'][i]],[0,1],ctx['cfg']);wt=torch.from_numpy(weights).to(x)
                dx=wt@x;da=wt@raw;dz=wt@z
                for n,c in enumerate(cats):
                    ids=list(range(rank*int(c),rank*(int(c)+1)));pred=dz[n,ids]@m.W_dec[ids];after=pred.clone()
                    if c==receiver:after=pred-dz[n,out]*m.W_dec[out]+da[n,inc]*m.W_dec[inc]
                    elif c==donor:after=pred-dz[n,inc]*m.W_dec[inc]+da[n,out]*m.W_dec[out]
                    numerator[c]+=float((dx[n]-pred).square().sum()-(dx[n]-after).square().sum());den[c]+=float(dx[n].square().sum())
        gains=numerator/den
        assert p['target_predicted_gain']==pytest.approx(gains[p['category_index']],abs=2e-7)
        assert p['mean_predicted_gain']==pytest.approx(gains.mean(),abs=2e-7)


@pytest.mark.parametrize('rank',[3,5])
def test_removal_graph_matches_explicit_fixed_gate_errors(rank):
    ctx,root,pack=fixture(rank);atoms=[0,rank-1,rank,2*rank+1]
    data=ops.dependency_shard(ctx,pack,0,4,atoms);m=ops.model(ctx,root)
    numerator=np.zeros((2,len(atoms)));rec=np.zeros(len(atoms));error=np.zeros(2);den=np.zeros(2)
    with torch.no_grad():
        for i in range(4):
            x=torch.from_numpy(pack['x'][i]);pred,z3=m(x);z=z3.reshape(196,-1)
            weights,_,cats=core.object_weights([pack['masks'][i]],[0,1],ctx['cfg']);wt=torch.from_numpy(weights).to(x)
            dx=wt@x;dz=wt@z
            for j,atom in enumerate(atoms):
                after=pred-z[:,atom,None]*m.W_dec[atom]
                rec[j]+=float(((x-after).double().square()-(x-pred).double().square()).sum())
            for n,c in enumerate(cats):
                ids=list(range(rank*int(c),rank*(int(c)+1)));r=dx[n]-dz[n,ids]@m.W_dec[ids]
                error[c]+=float(r.double().square().sum());den[c]+=float(dx[n].double().square().sum())
                for j,atom in enumerate(atoms):
                    if atom//rank==c:
                        rr=r+dz[n,atom]*m.W_dec[atom]
                        numerator[c,j]+=float(rr.double().square().sum()-r.double().square().sum())
    np.testing.assert_allclose(data['capture_numerator'],numerator,atol=3e-6,rtol=2e-5)
    np.testing.assert_allclose(data['reconstruction_numerator'],rec,atol=3e-5,rtol=3e-5)
    np.testing.assert_allclose(data['denominator'],den,atol=1e-8)
    assert data['base_gate_packed'].shape==(4,196,1)


def report(c=.3,ap=.7,ve=.66,mse=.30):
    return dict(macro_capture=c,macro_ap=ap,reconstruction=dict(mse=mse,ve=ve,active_groups=2.,dead_groups=0,patches=784),
        categories=[dict(category_index=i,category=f'category{i}',capture=c,ap=ap) for i in range(2)])


def test_cumulative_guards_and_original_native_limit():
    cfg=ex.settings();root=report();current=report(c=.31)
    assert ex.native_gate(report(c=.32),current,root,cfg,.31,target=0)['passed']
    assert not ex.native_gate(root,root,root,cfg,.31)['passed']
    assert not ex.native_gate(report(c=.32,mse=.317),current,root,cfg,.31)['passed']
    assert not ex.native_gate(report(c=.32,ap=.698),current,root,cfg,.31)['passed']
    assert not ex.native_gate(report(c=.32,ve=.658),current,root,cfg,.31)['passed']
    candidate=report(c=.32);candidate['categories'][1]['capture']=.28
    assert not ex.native_gate(candidate,current,root,cfg,.31)['passed']
    with pytest.raises(ValueError):ex.native_gate(dict(candidate,categories=candidate['categories'][:1]),current,root,cfg)


def test_original_metadata_catalog_rejects_wrong_branches():
    rows=[dict(seed=s,arm=a,rank=5,core_parameters=7869440,maximum_active_coordinates=80,total_training_updates=13640,
        balanced=a!='vanilla5_continue',supervised=a!='vanilla5_continue',pcgrad=a=='pcgrad5_both',decoder_pcgrad=a=='pcgrad5_both') for a in ex.five.ARMS for s in ex.SEEDS]
    f=dict(complete=True,models=rows);r=dict(complete=True,stage='continuation5',models=[{}]*30,controls_retrained=False,new_test_gradient_updates=0,new_structural_exchanges=0)
    ex.validate_parent_catalog(f,r)
    for field,value in [('rank',3),('total_training_updates',10640),('pcgrad',True)]:
        changed=copy.deepcopy(f);changed['models'][0][field]=value
        with pytest.raises(ValueError):ex.validate_parent_catalog(changed,r)
    with pytest.raises(ValueError):ex.validate_parent_catalog(dict(f,models=rows[:-1]),r)
    with pytest.raises(ValueError):ex.validate_parent_catalog(f,dict(r,complete=False))


def test_resume_after_accepted_snapshot_before_commit(tmp_path,monkeypatch):
    ctx,root,pack=fixture();master=inputs.JournalStore(tmp_path/'out',dict(test='resume'),tmp_path/'local',False)
    s=ex.SeedStore(master,21);s.save('start',root)
    arm=dict(ctx,store=s,seed=21,source_row=dict(development=report()),initial=root)
    arm['protocol']=dict(ctx['protocol'],screening_images=4)
    proposal=dict(id='p0',category_index=0,category='category0',outgoing=0,incoming=5,receiver_group=0,donor_group=1)
    monkeypatch.setattr(ex,'preflight',lambda *a:None)
    monkeypatch.setattr(ex,'direction_profiles',lambda *a:{})
    monkeypatch.setattr(ex.coverage,'assignment_indices',lambda *a:(np.arange(4),dict(ids=pack['ids'])))
    monkeypatch.setattr(ops,'candidate_pool',lambda *a:[proposal])
    monkeypatch.setattr(ops,'graph_rank',lambda *a:[proposal])
    def graph(*a):s.save('dependency_graph',{});return {}
    monkeypatch.setattr(ops,'build_graph',graph);monkeypatch.setattr(ops,'gate_effects',lambda *a:{'tested':True})
    calls=[]
    def measured(arm,key,snap,split):
        calls.append(key);return report(c=.33) if snap['spec']['atom_ids'][0]==5 else report()
    monkeypatch.setattr(ex,'measured',measured)
    original=s.save;interrupted=[False]
    def save(key,value):
        if key=='exchange00' and not interrupted[0]:interrupted[0]=True;raise KeyboardInterrupt()
        return original(key,value)
    monkeypatch.setattr(s,'save',save)
    with pytest.raises(KeyboardInterrupt):ex.select_arm(arm)
    assert s.load('exchange00_accepted') is not None and s.load('selection_frozen') is None
    # Simulate fresh process/context after runtime loss.
    resumed=dict(ctx,store=ex.SeedStore(master,21),seed=21,source_row=dict(development=report()),initial=root)
    resumed['protocol']=arm['protocol']
    final=ex.select_arm(resumed);assert len(final['accepted_proposals'])==1 and final['new_gradient_updates']==0
    edited=resumed['store'].load('final_agfx');ex.assert_permutation(root,edited)
    before=len(calls);assert ex.select_arm(resumed)==final;assert len(calls)==before
    assert training.weight_sha(root['weights'])==root['source_weights_sha']


def test_all_seed_freeze_barrier_and_immutable_values(tmp_path):
    ctx,root,pack=fixture();s=inputs.JournalStore(tmp_path/'out',{'test':'freeze'},tmp_path/'local',False);ctx['store']=s
    with pytest.raises(FileNotFoundError):ex.verify_freeze(ctx)
    models=[]
    for seed in ex.SEEDS:
        ss=ex.SeedStore(s,seed);ss.save('start',root);edited=ops.permute_snapshot(root,0,5);ss.save('final_agfx',edited)
        models.append(dict(seed=seed,final_sha=training.weight_sha(edited['weights']),targets_sha=identity(root['spec']['targets'])))
    s.save('exchanges_frozen',dict(complete=True,models=models[:2]))
    with pytest.raises(ValueError):ex.verify_freeze(ctx)
    s.save('exchanges_frozen',dict(complete=True,models=models));assert len(ex.verify_freeze(ctx)['models'])==3
    edited['head'][0,0]=1;ex.SeedStore(s,23).save('final_agfx',edited)
    with pytest.raises(ValueError):ex.verify_freeze(ctx)


def test_quick_path_has_no_optimizer_or_extraction_calls():
    import ast,inspect
    tree=ast.parse(inspect.getsource(ex))
    banned={'backward','fit','fit_warm','fit_branch','extract','backbone','prepare_test','prepare_sweep','choose_targets','timing'}
    for n in ast.walk(tree):
        if isinstance(n,ast.Call):
            name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
            assert name not in banned,name


def test_nonfinite_or_wrong_control_cohort_rejected():
    p=dict(image_ids=[1,2],category_indices=[0,1],error=np.zeros((2,2)))
    ex.compact_matches(p,[1,2],[0,1])
    with pytest.raises(ValueError):ex.compact_matches(p,[2,1],[0,1])
    with pytest.raises(ValueError):ex.compact_matches(dict(p,error=np.zeros((2,1))),[1,2],[0,1])


def test_real_width5_dependency_pipeline_and_resume(tmp_path):
    ctx,root,pack=fixture(5)
    ctx['store']=inputs.JournalStore(tmp_path/'out',dict(test='graph'),tmp_path/'local',False)
    stats=ops.shard_stats(ops.model(ctx,root),pack,0,4,ctx['cfg'])
    for key in ['gain','remove','cross']:stats[key]/=stats['energy'].reshape((-1,)+(1,)*(stats[key].ndim-1))
    stats['readout_groups']=np.array([0,1]);stats['category_indices']=np.array([0,1])
    pool=ops.candidate_pool(stats,root['spec']['targets'],ctx['protocol']);assert pool
    graph=ops.build_graph(ctx,pool,pack)
    assert graph['capture_damage'].shape==(2,len(graph['atom_ids']))
    assert graph['ap_damage'].shape==graph['capture_damage'].shape
    assert graph['category_indices']==[0,1] and graph['image_ids']==pack['ids']
    ranked=ops.graph_rank(pool,graph,ctx['protocol']);assert len(ranked)==len(pool)
    assert all(np.isfinite(p['dependency_score']) for p in ranked)
    resumed=ops.build_graph(ctx,pool,pack)
    np.testing.assert_array_equal(resumed['ap_damage'],graph['ap_damage'])
    edited=ops.model(ctx,ops.permute_snapshot(root,0,5))
    effects=ops.gate_effects(ctx,edited,pack,graph,ranked[0])
    assert 0<=effects['patch_fraction_with_changed_gates']<=1 and len(effects['categories'])==2


def test_evaluation_reuses_controls_and_exports_all_seeds(tmp_path,monkeypatch):
    ctx,root,pack=fixture(5);s=inputs.JournalStore(tmp_path/'out',dict(test='evaluate'),tmp_path/'local',False)
    ctx.update(store=s,test_ids=pack['ids'],test_stamp={'normalized_sha':'toy'},checkpoint_stamps=[],stamps=[],roots=[],parent_catalog=[],sources={})
    native=core.validation(ops.model(ctx,root),pack,root['spec']['targets'],ctx['cfg'],'cpu',with_arrays=True)
    compact=ex.study.compact_audit(native.pop('_audit'),ex.matched.mapping(root['spec']['targets']),ctx['cfg'])
    records={};models=[]
    for seed in ex.SEEDS:
        ss=ex.SeedStore(s,seed);ss.save('start',root)
        edited=ops.permute_snapshot(root,0,5) if seed==23 else copy.deepcopy(root);ss.save('final_agfx',edited)
        models.append(dict(seed=seed,root_sha=training.weight_sha(root['weights']),final_sha=training.weight_sha(edited['weights']),
            targets_sha=identity(root['spec']['targets']),accepted_proposals=[{}] if seed==23 else [],original_native_eligible=True,
            development={'select':native}))
        for arm in ['guided5_continue','vanilla5_continue']:
            row=dict(seed=seed,arm=arm,rank=5,weights_sha=root['source_weights_sha'],targets_sha=identity(root['spec']['targets']),reference_mse=.31)
            report_value=dict(copy.deepcopy(native),**row,legacy_native_eligible=True)
            records[f'audit_s{seed}_{arm}']=dict(identity=ex.cache.report_signature(row,ctx['test_stamp'],ctx['cfg']),report=report_value)
            records[f'compact_s{seed}_{arm}']=compact
            ctx['parent_catalog'].append(row)
            if arm=='guided5_continue':ctx['roots'].append(row)
    class Source:
        run_id='toy';root=tmp_path/'read_only'
        def load(self,key):return copy.deepcopy(records[key])
    ctx['sources']['continued5']=Source();s.save('exchanges_frozen',dict(complete=True,models=models))
    monkeypatch.setattr(ex.matched,'source_key_stamp',lambda *a:dict(store='toy',key=a[1]))
    monkeypatch.setattr(ex.matched,'assert_sources',lambda *a:None)
    def load(context):ex.verify_freeze(context);context['packs']['test']=pack
    monkeypatch.setattr(ex,'load_test',load)
    actual=core.validation;calls=[]
    def validation(*a,**kw):calls.append(True);return actual(*a,**kw)
    monkeypatch.setattr(core,'validation',validation)
    # Synthetic CPU model shares the real native audit and paired-bootstrap path.
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    result=ex.evaluate(ctx)
    assert result['complete'] and not result['controls_retrained'] and result['new_gradient_updates']==0
    assert len(calls)==1  # Only the edited seed; two unchanged seeds reuse audits.
    assert len(result['models'])==9 and len(result['comparisons'])==3
    assert (s.results/'means.csv').exists() and (s.results/'paired.csv').exists()
    before=len(calls);assert ex.evaluate(ctx)==result;assert len(calls)==before
    assert all(r['capture_gain_pp']==r['AP_gain_pp']==r['VE_gain_pp']==0 for r in result['comparisons'][:2])
    assert (s.results/'figures/means.png').exists()
    source=tmp_path/'source.zip';source.write_bytes(b'example immutable source')
    archive=ex.export(ctx,source)
    import zipfile,hashlib
    with zipfile.ZipFile(archive) as z:
        hashes=json.loads(z.read('artifact_hashes.json'))
        assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())


def test_test_loader_requires_real_freeze_and_restores_legacy_hook(tmp_path,monkeypatch):
    ctx,root,pack=fixture();s=inputs.JournalStore(tmp_path/'out',dict(test='loader'),tmp_path/'local',False);ctx['store']=s;ctx['roots']=[]
    calls=[]
    def loader(sub):
        calls.append(True);ex.cache.verify_test_freeze(sub,ex.memory.required(sub['store'],'all_models_frozen'))
        sub['packs']['test']=pack;return pack
    monkeypatch.setattr(ex.cache,'load_test',loader)
    original_verify=ex.cache.verify_test_freeze
    with pytest.raises(FileNotFoundError):ex.load_test(ctx)
    assert not calls and ex.cache.verify_test_freeze is original_verify
    rows=[]
    for seed in ex.SEEDS:
        ss=ex.SeedStore(s,seed);ss.save('start',root);ss.save('final_agfx',root)
        rows.append(dict(seed=seed,final_sha=training.weight_sha(root['weights']),targets_sha=identity(root['spec']['targets'])))
        ctx['roots'].append({'seed':seed})
    s.save('exchanges_frozen',dict(complete=True,models=rows))
    assert ex.load_test(ctx) is pack and ctx['packs']['test'] is pack
    assert len(calls)==1 and ex.cache.verify_test_freeze is original_verify
