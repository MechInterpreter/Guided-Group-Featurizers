import ast,copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from coco_baselines import training,metrics
from coco_baselines.storage import identity
from coco_ggf import width_balance
from coco_ggf.capacity_fit import state_identity
from scale_fit import fit
from scale_models import model
import experiment as exp
from evaluation import stat,cfg_for
from voc import mask_pair,mapping,NAMES,safe_member,flickr_id

@pytest.mark.parametrize('key,allowed',[
 ('g1024_k16_s21_mask_warm',True),('g1024_k16_s22_released_vanilla_warm',True),
 ('g1024_k16_s23_released_vanilla_lr3000_p10000_released_vanilla',True),
 ('g1024_k16_s21_released_vanilla_lr3000_p10000_supervised_vanilla_c040',False),
 ('g4096_k32_s21_mask_warm',False),('g1024_k16_s24_mask_warm',False),
 ('g1024_k32_s21_mask_warm',False),('all_models_frozen',False),('test_complete',False),
 ('initializer_cat79',True),('mask_dictionary12',True)])
def test_reuse_scope(key,allowed):assert exp.ReuseStore.reusable(key)==allowed

class FakeStore:
 def __init__(self,values=None):self.values=values or {};self.calls=[]
 def load(self,k):return copy.deepcopy(self.values.get(k))
 def save(self,k,v,**kw):self.calls.append(k);self.values[k]=copy.deepcopy(v)
 def _width_drive_recovery_original_save(self,k,v,**kw):return self.save(k,v,**kw)
 def write(self,*args,**kw):pass
 def event(self,*args,**kw):pass

def test_pinned_endpoint_writes_blocked():
 s=exp.ReuseStore(FakeStore(),FakeStore(),[])
 for fn in [s.save,s.force_save]:
  with pytest.raises(ValueError):fn(next(iter(exp.PINS)),{})

def test_no_silent_advanced_state():
 key='g1024_k16_s23_mask_lr3000_p10000_agf_g_c040';s=exp.ReuseStore(FakeStore({key:dict(updates=3001)}),FakeStore(),[])
 with pytest.raises(ValueError):s.load(key)

def test_protocol_is_bounded_and_matched():
 r=exp.recipe();assert r['seeds']==[21,22,23];assert r['total_updates']==10640
 assert r['warm_updates']==7640 and r['guided_updates']==3000 and r['cap']==.4
 assert len(r['methods'])==7 and r['no_exchanges'] and r['no_capacity_sweep'] and r['no_time_limit']
 assert 'NOT tuned' in r['baseline_settings']
 assert all(len(v)==64 for v in exp.PINS.values())


def test_voc_palette_maps_semantics_and_keeps_void_unknown():
 c=np.zeros((224,224),np.uint8);o=c.copy();c[:,:112]=1;o[:,:112]=7;c[:16,:16]=255;o[:16,:16]=255
 g=dict(original=[224,224],resized=[224,224],box=[0,0,224,224],crop=224)
 m=mask_pair(c,o,g,{1:4},100)
 assert m['fraction'].shape==(196,80) and m['crowd'][0]==1
 assert m['fraction'][1,4]==255 and m['fraction'][7,4]==0
 assert m['instances'][0]['category']==4 and m['instances'][0]['annotation_id']==100*256+7
 assert m['instances'][0]['fraction'][0]==0 and m['instances'][0]['fraction'][1]==1

def test_voc_categories_full_and_aliases():
 cats=[dict(name=n) for n in NAMES];m=mapping(cats)
 assert len(m)==20 and cats[m[1]]['name']=='airplane' and cats[m[14]]['name']=='motorcycle' and cats[m[18]]['name']=='couch'
 with pytest.raises(ValueError):mapping(cats[:-1])

@pytest.mark.parametrize('name',['/tmp/evil','../evil','ok/../../evil','C:/evil','foo\\evil'])
def test_archive_traversal_rejected(name):assert not safe_member(name)

def test_flickr_metadata():
 assert flickr_id('http://farm2.staticflickr.com/1234/56789_abc.jpg')=='56789'
 assert flickr_id('56789')=='56789'
 assert flickr_id('https://example.com/abc.jpg') is None

def test_student_t_is_three_seed_not_normal_interval():
 r=stat([1,2,3]);assert r['mean']==2 and r['sd']==1
 assert np.isclose(r['high']-2,4.302652729911275/np.sqrt(3))
 with pytest.raises(ValueError):stat([1,2])

def test_ap_ties_and_capture_identity():
 assert metrics.ap([.1,.9,.9],[False,True,False])==.5
 assert metrics.capture(np.eye(3),np.eye(3))==1
 assert metrics.capture(np.eye(3),np.zeros((3,3)))==0

def test_voc_code_has_no_training_or_readout_selection():
 tree=ast.parse((Path(__file__).parent/'voc.py').read_text())
 calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
 assert not {'fit','train_targets','choose_targets','backward'}&set(calls)

def test_voc_config_changes_only_declared_category_denominator():
 c=dict(category_indices=list(range(75)),batch=512,positive_fraction=.5)
 ctx=dict(cfg=c,store=FakeStore({'voc_plan':dict(category_indices=[1,3,9])}))
 out=cfg_for(ctx,'voc');assert out==dict(c,category_indices=[1,3,9]) and c['category_indices']==list(range(75))

def test_paired_rng_adam_head_controller_resume_matches_uninterrupted():
 torch.set_num_threads(1);torch.manual_seed(7)
 initial=training.weights(model('released_vanilla',8,2,2,6))
 rng=np.random.default_rng(9);x=rng.normal(size=(4,196,6)).astype('float32')
 masks=[]
 for i in range(4):
  f=np.zeros((196,80),np.uint8);f[:98,0]=255
  masks.append(dict(fraction=f,crowd=np.zeros(196),instances=[dict(category=0,annotation_id=i,fraction=np.r_[np.ones(98),np.zeros(98)])]))
 pack=dict(x=x,ids=[1,2,3,4],masks=masks,identity='toy')
 cfg=dict(category_indices=[0],batch=64,audit_image_batch=1,bootstrap=2,preview_images=0,positive_fraction=.5,negative_fraction=.1,
  min_object_patch_mass=2,min_background_patches=4,decision_min_images=1,detection_patches=32,patch_sample_seed=8,score_ap_weight=.2)
 rule=dict(width=2,lr=.0003,images_per_update=2,snr=.1,clip=5,evaluate_every=5,local_save_every=2,fixed_weights=[1,.05,.01])
 targets=[dict(category_index=0,category='toy',contrast_group=0,detector_group=0,heat_mean=0,heat_scale=1)]
 def context():return dict(rule=rule,device='cpu',store=FakeStore(),packs=dict(train=pack,select=pack),balance_recipe=copy.deepcopy(width_balance.settings()['balance']),cfg=cfg)
 a=context();b=context()
 aa=fit(a,'toy','released_vanilla',8,2,21,initial,30,30,targets=targets,cap=.4,stream=1)
 fit(b,'toy','released_vanilla',8,2,21,initial,30,15,targets=targets,cap=.4,stream=1)
 bb=fit(b,'toy','released_vanilla',8,2,21,initial,30,30,targets=targets,cap=.4,stream=1)
 for key in ['weights','optimizer','head','balance','sample_rng','noise_rng','sampled_batches_sha','sampled_noise_sha']:
  assert identity(state_identity(aa[key]))==identity(state_identity(bb[key])),key


