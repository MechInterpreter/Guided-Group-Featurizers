"""Synthetic pipeline oracles; none of these fixtures are research observations."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
import eval_common as common
import collect_evaluation as collector
import score_evaluation as score
import prepare_evaluation_reviews as reviews
import report_evaluation as reporting
from scale_models import model
from coco_baselines import training

@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)

def matrix(root,banks=2):
    root=Path(root);cats=list(range(75));ids={'calibration':list(range(20)),'audit':list(range(100,120))};records={};assets=[]
    defs=dict(categories=[dict(category_index=c,category_key=f'c{c}',category_name=f'c{c}',definition=f'Synthetic category{c}') for c in cats])
    common.save(root/'TARGET_DEFINITIONS.json',defs);common.save(root/'PROTOCOL.json',{'test':'synthetic only'})
    for role,ii in ids.items():
        fraction=np.zeros((20,196,75),np.uint8);fraction[:5]=255;crowd=np.zeros((20,196),np.float32)
        p=root/'labels'/f'{role}_masks.npz';common.npz(p,ids=np.array(ii),fraction=fraction,crowd=crowd)
        common.save(p.with_suffix('.json'),dict(path=str(p.relative_to(root)),sha256=common.sha(p),categories=cats))
        for i in ii:
            p=root/'images'/f'{i}.png';p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(224,224),(i%200,120,150)).save(p)
            records[str(i)]=dict(image_id=i,role=role)
            assets.append(dict(image_id=i,role=role,status='ready',crop_path=str(p.relative_to(root)),crop_sha256=common.sha(p)))
    common.save(root/'INPUTS.json',dict(ids=ids,categories=cats,records=records,folds=[dict(fold=0,guide=cats[38:],excluded=cats[:38]),dict(fold=1,guide=cats[:38],excluded=cats[38:])]))
    common.save(root/'IMAGE_ASSETS.json',dict(images=assets));bb=[]
    for k in range(banks):
        row=dict(bank_id=f'full_category_{"ggf" if k==0 else "vanilla"}_s21_f0',kind='ggf' if k==0 else 'vanilla',regime='full_category',seed=21,fold=0,groups=75,evaluation_folds=[0],capacity_matched=True,weights_sha=str(k)*64,signature=f'synthetic{k}',core_parameters=1383300,collection_wall_seconds=1.,observed_peak_GPU_bytes=1000,splits={})
        for role,ii in ids.items():
            maps=np.full((20,196,75),.01,np.float32);maps[:, :,3]=0;maps[:5,:,3]=1;maps[:,:,5]=2;maps[:,:,74]=0
            err=np.ones((75,75))*10;err[:,5]=0
            path=root/'responses'/row['bank_id']/f'{role}.npz'
            common.npz(path,ids=np.array(ii),scores=maps,pooled=np.partition(maps,-4,axis=1)[:,-4:,:].mean(1),capture_errors=err,capture_den=np.ones(75)*10,capture_count=np.ones(75,dtype=np.int64)*5,controls=np.tile([.1,10.,196*768,196*74],(20,1)))
            row['splits'][role]=[dict(path=str(path.relative_to(root)),sha256=common.sha(path),image_ids=ii)]
            if role=='calibration':
                lock=score.lock_numeric_readouts(root,row,cats);row['numeric_readout_sha256']=common.sha(lock)
        bb.append(row)
    common.save(root/'COLLECTION_COMPLETE.json',dict(complete=True,banks=bb,inputs_sha256=common.sha(root/'INPUTS.json'),audit_values=20*196*768,audit_total_variation=200.))
    return bb

def test_mask_AP_selector_is_not_capture_or_response_magnitude(tmp_path):
    banks=matrix(tmp_path,1);lock=common.read(tmp_path/'responses'/banks[0]['bank_id']/'NUMERIC_READOUT_LOCK.json')
    assert lock['AP_units']==[3]*75
    assert lock['Capture_units']==[5]*75
    score.numeric(tmp_path);result=common.read(tmp_path/'NUMERIC_COMPLETE.json')['results'][0]
    assert all(r['patch_AP']==1 and r['regional_Capture']==1 for r in result['categories'])
    assert all(r['within_target_images_AP'] is None for r in result['categories'])
    assert result['controls']['dead_groups']==1

def test_AP_unsupported_and_zero_tie_policy():
    s=np.zeros((2,196,75),np.float32);y=np.zeros((392,2),bool);v=np.ones_like(y);y[:196,0]=True
    units,table=score.choose_mask_AP(s,y,v)
    assert units==[0,None] and np.all(table[:,0]==.5) and np.isnan(table[:,1]).all()

def test_capture_oracle_uses_decoded_vector_not_norm():
    x=np.zeros((196,2));x[:2,0]=1;z=np.zeros((196,2,1));z[:2,0,0]=1
    D=np.array([[[1.,0.]],[[0.,1.]]]);frac=np.zeros((196,1),np.uint8);frac[:2]=255
    f=np.zeros(196);f[:2]=1;mask=dict(fraction=frac,crowd=np.zeros(196),instances=[dict(category=0,fraction=f)])
    error,den,n=common.capture_errors(x,z,D,mask,[0])
    assert np.array_equal(error,[[0.,1.]]) and den[0]==1 and n[0]==1

def test_portable_pool_and_view_hashes(tmp_path):
    banks=matrix(tmp_path,1);r=common.cache_views(tmp_path,banks[0]);a=common.read_pooled(tmp_path,banks[0],'audit')
    assert a['pooled'].shape==(20,75) and a['ids']==list(range(100,120))
    view=common.read(tmp_path/'portable'/banks[0]['bank_id']/'VIEWS.json')
    assert view['roles']['audit']['3'][0]['image_id']==100
    assert all(not v['patches'] for v in view['roles']['audit']['74'])
    p=tmp_path/'portable'/banks[0]['bank_id']/'POOLED.npz';p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError):common.read_pooled(tmp_path,banks[0],'audit')

def test_raw_chunk_corruption_stops(tmp_path):
    banks=matrix(tmp_path,1);p=tmp_path/banks[0]['splits']['audit'][0]['path'];p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError):common.read_maps(tmp_path,banks[0],'audit')

def test_training_matrix_source_and_architecture_guards():
    rows=[dict(seed=s,fold=0,kind=k,groups=75,width=12,top_k=32,total_updates=10640) for s in [21,22,23] for k in ['vanilla','supervised_vanilla']]
    reused=common.read(common.HERE/'REUSE_CATALOG.json')['models']
    r=dict(complete=True,regime='full_category',source_sha='s',protocol_sha256='p',new_endpoints=rows,reused_endpoints=reused)
    common.verify_training(r,'full_category','s','p')
    bad=copy.deepcopy(r);bad['new_endpoints'][0]['groups']=1024
    with pytest.raises(ValueError):common.verify_training(bad,'full_category','s','p')
    bad=copy.deepcopy(r);bad['new_endpoints'].pop()
    with pytest.raises(ValueError):common.verify_training(bad,'full_category','s','p')
    bad=copy.deepcopy(r);bad['reused_endpoints'][0]['weights_sha']='wrong'
    with pytest.raises(ValueError):common.verify_training(bad,'full_category','s','p')
    with pytest.raises(ValueError):common.verify_training(r,'full_category','different','p')

def test_native_BSF_and_SLincs_interfaces_no_invented_sum(tmp_path):
    torch.manual_seed(21);m=model('released_vanilla',75,12,32,768);weights=training.weights(m);p=tmp_path/'bsf.pt';torch.save(dict(weights=weights),p)
    row=dict(origin='reused',path=str(p),file_sha256=common.sha(p),weights_sha=training.weight_sha(weights),kind='ggf',groups=75)
    x=np.random.default_rng(1).normal(size=(196,768)).astype(np.float32);native=common.Native(row,tmp_path,'cpu');s,z,D,pred=native.factor(x)
    with torch.no_grad():expected=m.encode(torch.tensor(x)).norm(dim=-1).numpy()
    assert np.array_equal(s,expected) and pred.shape==(196,768) and (s>0).sum(1).max()<=32;native.close()
    dictionary=torch.eye(768)[:12].repeat(37,1);p=tmp_path/'slics.pt';torch.save(dict(D=dictionary),p)
    row=dict(origin='reused',path=str(p),file_sha256=common.sha(p),weights_sha=training.weight_sha({'D':dictionary}),kind='slics',groups=37)
    native=common.Native(row,tmp_path,'cpu');s,z,D,pred=native.factor(x)
    assert pred is None and z.dtype==np.float64 and D.dtype==np.float64
    expected=np.linalg.norm(np.maximum(x[:,:12],0),axis=1)
    assert np.allclose(s,expected[:,None],rtol=1e-6,atol=1e-6) and native.solver['max_relative_kkt']<=1e-5;native.close()

def test_collect_actual_native_chunks_resume_and_readout_lock(tmp_path):
    torch.manual_seed(21);m=model('released_vanilla',75,12,32,768);w=training.weights(m);p=tmp_path/'weights.pt';torch.save(dict(weights=w),p)
    row=dict(bank_id='synthetic_native',kind='ggf',origin='reused',groups=75,path=str(p),file_sha256=common.sha(p),weights_sha=training.weight_sha(w))
    packs={}
    for role,start in [('calibration',0),('audit',10)]:
        x=np.random.default_rng(start+2).normal(size=(2,196,768)).astype(np.float32);fraction=np.zeros((2,196,1),np.uint8);fraction[:,:4]=255
        masks=[dict(fraction=f,crowd=np.zeros(196),instances=[dict(category=0,fraction=(f[:,0]/255))]) for f in fraction]
        packs[role]=dict(x=x,ids=[start,start+1],masks=masks,identity=role)
        q=tmp_path/'labels'/f'{role}_masks.npz';common.npz(q,ids=np.array([start,start+1]),fraction=fraction,crowd=np.zeros((2,196)))
        common.save(q.with_suffix('.json'),dict(path=str(q.relative_to(tmp_path)),sha256=common.sha(q)))
    a=collector.collect_bank(tmp_path,row,packs,[0],tmp_path,'cpu',chunk=1)
    b=collector.collect_bank(tmp_path,row,packs,[0],tmp_path,'cpu',chunk=1)
    assert a==b and len(a['splits']['calibration'])==2 and a['numeric_readout_sha256']
    assert common.read_maps(tmp_path,a,'audit')['scores'].shape==(2,196,75)
    q=tmp_path/a['splits']['audit'][0]['path'];q.write_bytes(q.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):collector.collect_bank(tmp_path,row,packs,[0],tmp_path,'cpu',chunk=1)

def test_full_synthetic_annotation_gates_and_comparison(tmp_path):
    banks=matrix(tmp_path)
    for bank in banks:common.cache_views(tmp_path,bank)
    score.numeric(tmp_path);reviews.generic_reviews(tmp_path);reviews.presence_pack(tmp_path)
    folder=tmp_path/'reviews/coherence';public=common.read(folder/'PUBLIC.json')
    assert len(public['cases'])==150 and all(word not in json.dumps(public).lower() for word in ['vanilla','slics','ggf'])
    assert len(common.read(folder/'HUMAN_VALIDATION_SAMPLE.json')['case_ids'])<=30
    for context in ['context1','context2','context3']:
        d=common.read(folder/'templates'/f'{context}.json');d['reviewer']='synthetic-'+context
        for case,plan in zip(d['cases'],public['cases']):
            for pair in case['pairs']:
                code='unclear' if plan['invalid_pairs'][pair['pair']] else 'yes'
                pair.update(semantic=code,visual=code,shared_meaning='Synthetic fixture interpretation, not a real judgment')
        common.save(folder/'labels'/f'{context}.json',d)
    score.coherence(tmp_path)
    presence=tmp_path/'reviews/presence';private=common.read(presence/'PRIVATE.json');rows=[]
    for r in private['cases']:
        yes=list(range(75)) if r['image_id']%100<5 else []
        rows.append(dict(case_id=r['case_id'],yes=yes,no=[] if yes else list(range(75)),unclear=[],all_definitions_considered=True))
    common.save(presence/'labels'/'fixture.json',dict(public_sha256=common.sha(presence/'PUBLIC.json'),reviewer='synthetic fixture only',method_blind=True,cases=rows))
    reviews.lock_presence(tmp_path,'calibration');reviews.lock_presence(tmp_path,'audit')
    qa=tmp_path/'reviews/presence_qa';q=common.read(qa/'TEMPLATE.json');q['reviewer']='Independent synthetic QA only'
    byid={r['case_id']:r for r in rows}
    q['cases']=[copy.deepcopy(byid[r['case_id']]) for r in q['cases']];common.save(qa/'labels.json',q);reviews.presence_quality(tmp_path)
    score.select_presence(tmp_path);score.concentration(tmp_path)
    curves=common.read(tmp_path/'CONCENTRATION_COMPLETE.json')['results'][0]['rows'][0]['curves']
    assert curves[0]['metrics']['OR_coverage']==1 and curves[0]['metrics']['OR_FPR']==0
    with pytest.raises(FileNotFoundError):reviews.alignment_pack(tmp_path,'audit')
    for role in ['calibration','audit']:
        reviews.alignment_pack(tmp_path,role);f=tmp_path/'reviews'/('alignment_'+role);d=common.read(f/'TEMPLATE.json');pub=common.read(f/'PUBLIC.json');d['reviewer']='synthetic alignment only'
        for case,plan in zip(d['cases'],pub['cases']):case['codes']=['unclear' if bad else 'target' for bad in plan['invalid_views']]
        common.save(f/'labels.json',d);score.alignment(tmp_path,role)
    result=reporting.report(tmp_path)
    assert result['complete'] and len(result['matched_numeric_effects'])==5
    assert result['matched_numeric_effects'][0]['mean_difference']==0
    # A mutated private map cannot be silently relabeled into a different method.
    p=folder/'PRIVATE.json';d=common.read(p);d['cases'][0]['unit']=999;p.write_text(json.dumps(d))
    with pytest.raises(ValueError):score.coherence(tmp_path)

def test_category_fold_aggregation_is_category_weighted():
    rows=[]
    for f,values in [(0,{str(i):0. for i in range(38)}),(1,{str(i):1. for i in range(38,75)})]:
        rows.append(dict(regime='category_split',kind='ggf',stratum='excluded',capacity_matched=True,seed=21,fold=f,AP_by_category=values,planned_categories=len(values),AP_supported=len(values)))
    out=reporting.aggregate_numeric(rows)[0]
    assert out['mean_AP_supported']==37/75 and out['mean_AP_supported']!=.5

def test_missing_presence_and_invalid_pair_never_become_negative_or_clear(tmp_path):
    banks=matrix(tmp_path,1);common.cache_views(tmp_path,banks[0]);reviews.generic_reviews(tmp_path);reviews.presence_pack(tmp_path)
    with pytest.raises(ValueError):reviews.lock_presence(tmp_path,'audit')
    with pytest.raises(FileNotFoundError):score.concentration(tmp_path)
    result=reporting.report(tmp_path);assert not result['complete'] and result['stages']['coherence']=='pending'

def test_lineage_rejects_method_or_guidance_remapping(tmp_path,monkeypatch):
    norm=dict(mean=np.zeros(2),scale=1.);ctx=dict(options={'local':str(tmp_path)},norm=norm,packs={'train':{'identity':'train'},'select':{'identity':'dev'}},eligible=[0,1],folds=[dict(guide=[0],excluded=[1]),dict(guide=[1],excluded=[0])])
    cfg=dict(kind='ggf',seed=21,fold=1,normalization=common.identity(norm),train='train',development='dev',code_sha256='source',
        protocol=dict(groups=75,width=12,top_k=32,warm_updates=7640,guided_updates=3000),guide=[1],excluded=[0])
    class Source:
        def __init__(self,*a,**k):self.config=cfg
    monkeypatch.setattr(common.memory,'FrozenStore',Source)
    row=dict(origin='new',kind='ggf',seed=21,fold=1,regime='category_split',capacity_matched=True,output=str(tmp_path),run_id='r')
    common.verify_fit_lineage(row,ctx,{'training_source_sha256':'source'})
    cfg['guide']=[0,1]
    with pytest.raises(ValueError):common.verify_fit_lineage(row,ctx,{'training_source_sha256':'source'})
    cfg['guide']=[1];cfg['kind']='vanilla'
    with pytest.raises(ValueError):common.verify_fit_lineage(row,ctx,{'training_source_sha256':'source'})

def test_portable_view_coordinates_cannot_change_under_same_signature(tmp_path):
    banks=matrix(tmp_path,1);lock=common.cache_views(tmp_path,banks[0]);common.save(tmp_path/'PORTABLE_LOCK.json',{'banks':[lock]})
    p=tmp_path/'portable'/banks[0]['bank_id']/'VIEWS.json';d=common.read(p);d['roles']['audit']['3'][0]['patches']=[195];p.write_text(json.dumps(d))
    with pytest.raises(ValueError):reviews.generic_reviews(tmp_path)

def test_blinded_batch_assembly_retains_every_case_and_raw_source(tmp_path):
    banks=matrix(tmp_path,1);common.cache_views(tmp_path,banks[0]);reviews.generic_reviews(tmp_path)
    folder=tmp_path/'reviews/coherence';public={r['case_id']:r for r in common.read(folder/'PUBLIC.json')['cases']}
    for p in sorted((folder/'batches/context1').glob('*.json')):
        d=common.read(p);d['reviewer']='synthetic fresh batch '+p.stem
        for case in d['cases']:
            for pair in case['pairs']:
                code='unclear' if public[case['case_id']]['invalid_pairs'][pair['pair']] else 'yes'
                pair.update(semantic=code,visual=code,shared_meaning='Synthetic fixture only')
        common.save(folder/'batch_labels/context1'/p.name,d)
    reviews.assemble_context(tmp_path,'context1');result=common.read(folder/'labels/context1.json')
    assert len(result['cases'])==75 and len(result['sources'])==7
    first=next((folder/'batch_labels/context1').glob('*.json'));common.save(folder/'batch_labels/context1/duplicate.json',common.read(first))
    with pytest.raises(ValueError):reviews.assemble_context(tmp_path,'context1')

def test_seed21_adapter_changes_execution_only():
    import pilot_seed21_run as pilot
    import all75_run as original
    source=Path(original.__file__).read_bytes();function,counts=pilot.pilot_function()
    assert counts==dict(seed_loop=1,expected_count=1,receipt=1,receipt_name=2)
    assert Path(original.__file__).read_bytes()==source
    assert function.__globals__['configure'] is original.configure
    assert function.__globals__['cd_train'] is original.cd_train
    assert 'PILOT_SEED21_COMPLETE.json' in function.__code__.co_consts
    assert 'TRAINING_COMPLETE.json' not in function.__code__.co_consts

def test_pilot_receipt_cannot_masquerade_as_three_seed_complete():
    reused=[r for r in common.read(common.HERE/'REUSE_CATALOG.json')['models'] if r['seed']==21]
    rows=[dict(seed=21,fold=0,kind=k,groups=75,width=12,top_k=32,total_updates=10640) for k in ['vanilla','supervised_vanilla']]
    receipt=dict(complete=True,regime='full_category',source_sha='s',protocol_sha256='p',new_endpoints=rows,reused_endpoints=reused,
        scope='seed21_pilot',required_seeds=[21],full_three_seed_study_complete=False,
        execution_revision_sha256=common.sha(common.HERE/'pilot_seed21_run.py'),pilot_amendment_sha256=common.sha(common.HERE/'PILOT_AMENDMENT.json'))
    common.verify_training(receipt,'full_category','s','p',seeds=(21,))
    with pytest.raises(ValueError):common.verify_training(receipt,'full_category','s','p')
    receipt['execution_revision_sha256']='wrong'
    with pytest.raises(ValueError):common.verify_training(receipt,'full_category','s','p',seeds=(21,))

@pytest.mark.parametrize('regime,expected',[('full_category',2),('category_split',5)])
def test_seed21_driver_executes_both_folds_without_main_completion(tmp_path,monkeypatch,regime,expected):
    import pilot_seed21_run as pilot
    function,_=pilot.pilot_function();g=function.__globals__;calls=[];writes=[]
    class Store:
        def __init__(self,path):self.root=Path(path);self.root.mkdir(parents=True,exist_ok=True);self.require_drive=False
        def write(self,name,value):writes.append((name,value))
    norm={'mean':np.zeros(1),'scale':1.};nid=common.identity(norm)
    def prepare(options,require_drive):
        return dict(options=options,norm=norm,eligible=[0,1],folds=[dict(fold=0,guide=[0],excluded=[1]),dict(fold=1,guide=[1],excluded=[0])],
            packs={k:dict(x=np.zeros((1,196,768)),identity=k) for k in ['train','select','confirm']},test_stamp={'ids':[]},versions={},store=Store(tmp_path/'initial'))
    monkeypatch.setattr(g['cd_data'],'prepare',prepare)
    monkeypatch.setattr(g['cd_common'],'make_store',lambda output,*a,**k:Store(output))
    monkeypatch.setattr(g['cd_common'],'flush',lambda s:None)
    monkeypatch.setattr(g['matched'],'assert_sources',lambda ctx:None)
    monkeypatch.setattr(g['torch'].cuda,'is_available',lambda:True)
    monkeypatch.setattr(g['torch'].cuda,'get_device_name',lambda:'Synthetic wrapper test')
    monkeypatch.setattr(g['torch'].cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(g['torch'].cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(g['signal'],'signal',lambda *a:None)
    def fake_fit(ctx,fold,seed,kind):
        calls.append((seed,fold,kind));return dict(seed=seed,fold=fold,kind=kind,weights_sha=kind+str(fold),continuation_batches_sha='batch',continuation_noise_sha='noise')
    monkeypatch.setattr(g['cd_train'],'fit_bsf',fake_fit)
    monkeypatch.setattr(g['cd_train'],'training_context',g['cd_train'].training_context)
    g['checked_endpoint']=lambda ctx,row:row
    g['reuse_full_catalog']=lambda options:[dict(seed=s,kind=k,normalization_identity=nid) for s in [21,22,23] for k in ['ggf','slics']]
    g['reuse_native_category']=lambda options,folds:[]
    options=dict(regime=regime,output=str(tmp_path/'output'),local=str(tmp_path/'local'),experiment_sha='original training SHA')
    function(options,False)
    assert len(calls)==expected and {r[0] for r in calls}=={21}
    assert not any(name=='TRAINING_COMPLETE.json' for name,_ in writes)
    result=common.read(tmp_path/'output/PILOT_SEED21_COMPLETE.json')
    assert result['scope']=='seed21_pilot' and not result['full_three_seed_study_complete']
    if regime=='full_category':assert len(result['reused_endpoints'])==2
    else:assert {r[1] for r in calls}=={0,1}

@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU unavailable')
def test_GPU_native_maps_match_direct_encoder(tmp_path):
    torch.manual_seed(21);m=model('released_vanilla',75,12,32,768);w=training.weights(m);p=tmp_path/'gpu.pt';torch.save(dict(weights=w),p)
    row=dict(origin='reused',path=str(p),file_sha256=common.sha(p),weights_sha=training.weight_sha(w),kind='ggf',groups=75)
    x=np.random.default_rng(9).normal(size=(196,768)).astype(np.float32);native=common.Native(row,tmp_path,'cuda');s,z,D,pred=native.factor(x)
    with torch.no_grad():expected=native.model.encode(torch.tensor(x,device='cuda')).norm(dim=-1).cpu().numpy()
    assert np.array_equal(s,expected) and pred is not None;native.close()
