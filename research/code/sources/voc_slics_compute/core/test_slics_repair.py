import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.optimize import nnls
from coco_ggf import slics_repair as r
from coco_adaptation.inputs import JournalStore
torch.set_num_threads(2)

def config(**kw):return dict(r.settings(),**kw)

@pytest.mark.parametrize('fallback',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_nnls_matches_independent_scipy(dtype,fallback):
 torch.manual_seed(217);D=torch.randn(7,12,dtype=dtype);D/=D.norm(dim=1,keepdim=True)
 x=torch.randn(11,12,dtype=dtype);allow=torch.rand(11,7)>.35;allow[0]=False
 z,info=r.Solver(D,config(nnls_iterations=1000,nnls_tolerance=1e-5)).solve(x,allow,force_fallback=fallback)
 assert info['converged'] and info['max_relative_kkt']<=1e-5
 assert (z[~allow]==0).all() and (z>=0).all()
 for i in range(1,len(x)):
  take=allow[i].numpy();expected,_=nnls(D[take].double().numpy().T,x[i].double().numpy(),maxiter=10000)
  assert np.allclose(z[i,allow[i]].double().numpy(),expected,atol=1e-4)
 assert not z[0].any()

def test_zero_input_and_empty_allowed():
 D=torch.eye(4);x=torch.zeros(3,4);allow=torch.zeros(3,4,dtype=torch.bool)
 z,info=r.Solver(D,config()).solve(x,allow)
 assert not z.any() and info['unsupported_rows']==3 and info['max_relative_kkt']==0

def test_rowwise_relative_test_does_not_hide_small_rows():
 D=torch.eye(3);x=torch.tensor([[1e5,1e5,1e5],[1e-4,2e-4,3e-4]])
 z,info=r.Solver(D,config()).solve(x)
 assert torch.allclose(z,x) and info['max_relative_kkt']<1e-5

def test_solver_does_not_mutate_dictionary_or_labels():
 D=torch.eye(4);before=D.clone();allow=torch.tensor([[True,False,True,False]])
 old=allow.clone();r.Solver(D,config()).solve(torch.ones(1,4),allow)
 assert torch.equal(D,before) and torch.equal(allow,old)

def test_active_set_failure_is_not_silently_accepted(monkeypatch):
 def fail(*a,**k):raise RuntimeError('maximum iterations')
 monkeypatch.setattr(r,'scipy_nnls',fail)
 with pytest.raises(RuntimeError,match='fallback failed'):
  r.Solver(torch.eye(2),config()).solve(torch.ones(1,2),force_fallback=True)

def test_unconverged_fallback_is_rejected(monkeypatch):
 monkeypatch.setattr(r,'scipy_nnls',lambda A,b,maxiter:(np.zeros(A.shape[1]),1))
 with pytest.raises(RuntimeError,match='KKT'):
  r.Solver(torch.eye(2),config()).solve(torch.ones(1,2),force_fallback=True)

def test_native_and_topk_use_same_atoms_and_no_labels():
 D=torch.eye(6);cfg=config(groups=3,rank=2,active_groups=1)
 x=torch.tensor([[1.,2.,3.,4.,5.,6.]])
 dense=r.Model(D,cfg,'dense');top=r.Model(D,cfg,'top16')
 zd=dense.encode(x);zt=top.encode(x)
 assert (zd.norm(dim=-1)>0).sum()==3 and (zt.norm(dim=-1)>0).sum()==1
 assert dense.stats['calls']==1 and top.stats['calls']==2
 assert torch.equal(top.atoms(),dense.atoms()) and torch.equal(zt[:,2],zd[:,2])

def test_paper_atom_support_stays_nonzero():
 D=torch.eye(2,dtype=torch.float64);x=torch.tensor([[1.,0.],[0.,20.]],dtype=torch.float64)
 z=x.clone();allowed=torch.ones_like(z,dtype=torch.bool)
 new,info=r.old.atom_sweep(x,D.clone(),z,allowed,False)
 assert torch.equal(new,D) and info['residual_sse_after']==0

def test_paper_update_agrees_with_direct_svd_objective():
 rng=np.random.default_rng(32);E=rng.normal(size=(11,6));D=torch.tensor([[1.,0,0,0,0,0]],dtype=torch.float64)
 x=torch.tensor(E,dtype=torch.float64);z=torch.ones(11,1,dtype=torch.float64);allow=torch.ones_like(z,dtype=torch.bool)
 U,S,V=np.linalg.svd(E,full_matrices=False);atom=V[0];beta=E@atom
 if np.maximum(beta,0)@np.maximum(beta,0)<np.maximum(-beta,0)@np.maximum(-beta,0):atom=-atom;beta=-beta
 expected=np.square(E-np.maximum(beta,0)[:,None]*atom).sum()
 new,info=r.old.atom_sweep(x,D,z,allow,False)
 assert abs(info['residual_sse_after']-expected)<1e-8

def toy_context(tmp_path,name):
 cfg=config(groups=2,rank=2,active_groups=1,updates=4,evaluate_every=2,drive_every=1,nnls_iterations=400)
 rng=np.random.default_rng(41);x=rng.normal(size=(4,196,5)).astype(np.float32)
 masks=[dict(fraction=np.full((196,2),255,np.uint8),crowd=np.zeros(196),instances=[]) for _ in range(4)]
 s=JournalStore(tmp_path/name,dict(test=name),tmp_path/(name+'_local'),False)
 D=torch.tensor(rng.normal(size=(4,5)).astype(np.float32));D/=D.norm(dim=1,keepdim=True)
 s.save('ref_svd_s21',D)
 return dict(store=s,recipe=cfg,comparison=dict(r.old.settings(),groups=2,rank=2,images_per_update=1,warm_updates=2,guided_updates=2),
  cfg=dict(category_indices=[0,1]),device='cpu',packs=dict(train=dict(x=x,ids=list(range(4)),masks=masks),select=dict(x=x,ids=list(range(4)),masks=masks)),
  references=[dict(seed=21,arm='SVD mask',key='svd_s21')])

def fake_validation(ctx,D,key):
 mse=float(D.square().sum())
 return dict(reconstruction=dict(mse=mse,ve=1-mse),macro_capture=.2,macro_ap=.3,score=.26,categories=[])

def test_exact_resume_and_candidate_zero(tmp_path,monkeypatch):
 monkeypatch.setattr(r,'validation',fake_validation)
 a=toy_context(tmp_path,'full');b=toy_context(tmp_path,'resume')
 full=r.fit_seed(a,21);r.fit_seed(b,21,max_updates=1);resumed=r.fit_seed(b,21)
 assert torch.equal(full['D'],resumed['D']) and full['batch_sha']==resumed['batch_sha']
 assert [v['update'] for v in full['history']]==[0,2,4]
 assert full['best_update']==resumed['best_update']
 assert a['store'].load('candidate_s21_u0') is not None

def test_interrupted_sweep_never_commits_partial_dictionary(tmp_path,monkeypatch):
 monkeypatch.setattr(r,'validation',fake_validation);ctx=toy_context(tmp_path,'interrupt')
 initial=ctx['store'].load('ref_svd_s21').clone();original=r.old.atom_sweep
 def interrupted(x,D,z,allowed,guard):D.fill_(19);raise KeyboardInterrupt()
 monkeypatch.setattr(r.old,'atom_sweep',interrupted)
 with pytest.raises(KeyboardInterrupt):r.fit_seed(ctx,21)
 saved=ctx['store'].load('repaired_s21');assert saved['updates']==0 and torch.equal(saved['D'],initial)
 monkeypatch.setattr(r.old,'atom_sweep',original)
 assert r.fit_seed(ctx,21)['updates']==4

def test_checkpoint_zero_can_win(tmp_path,monkeypatch):
 ctx=toy_context(tmp_path,'zero');initial=ctx['store'].load('ref_svd_s21').clone()
 def val(ctx,D,key):
  v=fake_validation(ctx,D,key);v['reconstruction']['mse']=0. if torch.equal(D.cpu(),initial) else 1.;return v
 monkeypatch.setattr(r,'validation',val)
 st=r.fit_seed(ctx,21);assert st['best_update']==0 and torch.equal(st['best'],initial)

def test_pending_checkpoint_evaluation_is_completed_on_resume(tmp_path,monkeypatch):
 ctx=toy_context(tmp_path,'pending');original=fake_validation
 def fail(ctx,D,key):
  if key.endswith('u2'):raise KeyboardInterrupt()
  return original(ctx,D,key)
 monkeypatch.setattr(r,'validation',fail)
 with pytest.raises(KeyboardInterrupt):r.fit_seed(ctx,21)
 st=ctx['store'].load('repaired_s21');assert st['updates']==2 and st['last_evaluated']==0
 monkeypatch.setattr(r,'validation',original)
 st=r.fit_seed(ctx,21);assert [x['update'] for x in st['history']]==[0,2,4]

def test_pareto_retains_tradeoffs_and_ties():
 def v(i,c,a,e):return dict(update=i,validation=dict(macro_capture=c,macro_ap=a,reconstruction=dict(ve=e)))
 h=[v(0,.1,.1,.9),v(1,.2,.2,.8),v(2,.0,.0,.7),v(3,.2,.2,.8)]
 assert r.pareto(h)==[0,1,3]

def test_meter_reconstruction_matches_direct():
 cfg=config(groups=2,rank=2,active_groups=1);m=r.Model(torch.eye(4),cfg,'dense');meter=r.Meter(m)
 x=torch.tensor([[1.,-2,3,-4],[2.,3,-1,2.]])
 meter.encode(x[:1]);meter.encode(x[1:]);total=float((x-x.mean(0)).double().square().sum())
 got=meter.reconstruction(total);pred=x.clamp_min(0)
 assert abs(got['mse']-float((pred-x).square().mean()))<1e-8
 assert got['patches']==2 and got['active_groups']==2.0

def test_freeze_gate_prevents_any_test_load(tmp_path):
 ctx=toy_context(tmp_path,'gate')
 with pytest.raises(RuntimeError,match='Freeze all three'):r.load_audit_packs(ctx)

def test_read_only_frozen_store_methods():
 f=object.__new__(r.memory.FrozenStore)
 for method in ['save','write','event']:
  with pytest.raises(PermissionError):getattr(f,method)('x',{})

def test_export_crc_hash_inventory(tmp_path):
 ctx=toy_context(tmp_path,'export');source=tmp_path/'source.zip';source.write_bytes(b'trusted source')
 archive=r.export(ctx,source)
 import zipfile,hashlib
 with zipfile.ZipFile(archive) as z:
  hashes=json.loads(z.read('artifact_hashes.json'))
  assert hashes and all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in hashes.items())

def test_statistical_pairing_identity_is_enforced():
 with pytest.raises(ValueError,match='identities'):
  r.study.paired(dict(image_ids=[1],category_indices=[0],objects=[1]),dict(image_ids=[2],category_indices=[0],objects=[1]),{})

def test_all_category_audit_pipeline_with_real_mapping_and_compact_scores(tmp_path):
 fit=r.memory.config('cpu');fit.update(category_indices=list(range(75)),detection_patches=196,
  preview_images=0,min_category_images=1,decision_min_images=1,bootstrap=3,fresh_test_seed=71002,
  audit_image_batch=2,max_audit_groups=2,batch=128)
 rng=np.random.default_rng(31)
 def pack(ids):
  masks=[];x=rng.random((len(ids),196,75)).astype(np.float32)*.1
  for i,image in enumerate(ids):
   f=np.zeros((196,75),np.uint8);f[:10]=255;x[i,:10]+=1
   instances=[dict(category=c,annotation_id=image*100+c,fraction=np.r_[np.ones(10),np.zeros(186)]) for c in range(75)]
   masks.append(dict(fraction=f,crowd=np.zeros(196),instances=instances))
  return dict(x=x,masks=masks,ids=ids)
 mapping,test=pack([10,11,12]),pack([20,21,22])
 store=JournalStore(tmp_path/'real_audit',dict(test='pipeline'),tmp_path/'real_audit_local',False)
 cfg=config(rank=1,nnls_iterations=100)
 ctx=dict(store=store,cfg=fit,recipe=cfg,device='cpu',
  provenance=dict(categories=[dict(name='category_'+str(c),id=c+1) for c in range(75)]))
 D=torch.eye(75);m=r.Model(D,cfg)
 row=dict(seed=21,method='Repaired SLiCS',inference='dense',tag='integration',weights_sha=r.old.model_hash(D),
  selected_update=0,training_updates=4,training_seconds=.1,parameters=75*75)
 out=r.audit_model(ctx,m,row,mapping,test)
 assert len(out['categories'])==75 and out['summary']['reconstruction']['patches']==588
 assert out['summary']['solver']['max_relative_kkt']<=1e-5
 assert store.load('compact_integration')['category_indices']==list(range(75))
 # A resumed completed model audit must not solve coefficients again.
 m.encode=lambda x:(_ for _ in ()).throw(AssertionError('reran completed audit'))
 assert r.audit_model(ctx,m,row,mapping,test)==out
 alias=dict(row,tag='integration_alias',seed=22)
 reused=r.audit_model(ctx,m,alias,mapping,test)
 assert reused['summary']['seed']==22 and reused['summary']['reused_inference']
 assert reused['summary']['capture']==out['summary']['capture']

def test_real_synthetic_training_improves_supported_reconstruction():
 torch.manual_seed(90);D=torch.eye(6);z=torch.rand(12,6);allowed=z>.5;z=z*allowed;x=z@D
 recovered,info=r.Solver(D,config()).solve(x,allowed)
 revised,atom=r.old.atom_sweep(x,D.clone(),recovered,allowed,False)
 assert torch.allclose(recovered@D,x,atol=1e-6)
 assert atom['residual_sse_after']<1e-9

def test_invalid_solver_support_shape_fails():
 with pytest.raises(ValueError,match='support'):
  r.Solver(torch.eye(2),config()).solve(torch.ones(1,2),torch.ones(1,3,dtype=torch.bool))
