import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from coco_adaptation.inputs import JournalStore
from coco_baselines.training import weight_sha
from zoo_benchmark import data
from focused_zoo import initialize,model,losses,fit,evaluate_arrays,recipe


def small():
    return dict(dimension=8,factors=4,groups=8,block_rank=3,active_factors=2,
        train_rows=32,select_rows=16,calibration_rows=48,test_rows=64,
        normalization_rows=100,data_seed=701,epochs=8,warm_epochs=1,
        batch=8,warmup=.1,log_epochs=1,lrs=[.0004,.003])


def test_frozen_scope_and_reused_baselines():
    ref=json.loads((Path(__file__).parent/'zoo_reference.json').read_text());cfg=ref['config'];r=recipe(cfg)
    assert cfg['train_rows']//cfg['batch']*cfg['epochs']==29200
    assert (cfg['dimension'],cfg['groups'],cfg['block_rank'],cfg['active_factors'])==(128,256,4,4)
    assert r['cap']==.6 and r['seeds']==[0,1,2] and len(ref['baseline_rows'])==9
    assert {s['seed'] for s in ref['baseline_rows']}=={0,1,2}
    for method,mean in ref['baseline_means'].items():
        rows=[v for v in ref['baseline_rows'] if v['method']==method]
        assert len(rows)==3 and np.mean([v['heldout_mean_r2'] for v in rows])==pytest.approx(mean['heldout_mean_r2'])


def test_initialization_does_not_read_true_component_coordinates():
    c=small();ds=data.generate(c,'goodfire','independent',701);pack=ds['packs']['train']
    torch.manual_seed(1);a=model(c,'cpu');initialize(a,pack,c)
    changed=copy.deepcopy(pack);changed.pop('coords')
    torch.manual_seed(1);b=model(c,'cpu');initialize(b,changed,c)
    assert weight_sha(a.state_dict())==weight_sha(b.state_dict())


def test_oracle_recovers_factor_and_reconstruction_exactly():
    c=small();ds=data.generate(c,'goodfire','independent',701)
    def codes(pack):
        z=np.zeros((len(pack['x']),c['groups'],3),np.float32)
        z[np.arange(len(z))[:,None],pack['ids']]=pack['coords'];return z
    D=np.zeros((c['groups'],3,c['dimension']));D[:c['factors']]=ds['truth']['frames']
    score=evaluate_arrays(ds,codes(ds['packs']['calibration']),codes(ds['packs']['test']),D)
    assert score['assigned_mean_r2']==pytest.approx(1.,abs=1e-12)
    assert score['ve']==pytest.approx(1.,abs=1e-12)
    assert score['factor_presence_ap']==pytest.approx(1.)
    assert score['assigned_above_90_count']==c['factors']


def test_wrong_group_assignment_is_not_reported_as_perfect_assigned_recovery():
    c=small();ds=data.generate(c,'goodfire','independent',701)
    def codes(pack):
        z=np.zeros((len(pack['x']),c['groups'],3),np.float32)
        z[np.arange(len(z))[:,None],pack['ids']+c['factors']]=pack['coords'];return z
    D=np.zeros((c['groups'],3,c['dimension']));D[c['factors']:]=ds['truth']['frames']
    score=evaluate_arrays(ds,codes(ds['packs']['calibration']),codes(ds['packs']['test']),D)
    assert score['heldout_mean_r2']==pytest.approx(1.,abs=1e-12)
    assert score['assigned_mean_r2']<0 and score['assigned_above_90_count']==0


def test_supervised_gradients_are_finite_and_reach_encoder_decoder_heads():
    c=small();ds=data.generate(c,'goodfire','independent',701);p=ds['packs']['train']
    torch.manual_seed(0);m=model(c,'cpu');head=torch.nn.Parameter(torch.zeros(c['factors'],2))
    ls=losses(m,torch.from_numpy(p['x']),torch.tensor(p['ids'].astype('int64')),torch.from_numpy(p['coords']),
        torch.tensor(ds['truth']['frames'],dtype=torch.float32),head,torch.stack([torch.zeros(c['factors']),torch.ones(c['factors'])]))
    assert all(torch.isfinite(v) and v>0 for v in ls)
    sum(ls).backward()
    assert m.W_enc.grad.norm()>0 and m.W_dec.grad.norm()>0 and head.grad.norm()>0


def test_resume_preserves_optimizer_controller_rng_and_exact_weights(tmp_path):
    torch.set_num_threads(1);c=small();ds=data.generate(c,'goodfire','independent',701)
    a=JournalStore(tmp_path/'a',{'test':1},tmp_path/'la',False)
    full=fit(a,'model',c,ds,0,.003,'cpu')
    b=JournalStore(tmp_path/'b',{'test':1},tmp_path/'lb',False)
    partial=fit(b,'model',c,ds,0,.003,'cpu',stop=11)
    assert partial['updates']==11 and partial['stats'] is not None
    resumed=fit(b,'model',c,ds,0,.003,'cpu')
    assert resumed['updates']==full['updates']==32
    assert resumed['controller']['reference_count']==full['controller']['reference_count']
    assert resumed['controller']['adaptations']==full['controller']['adaptations']>0
    assert weight_sha(full['weights'])==weight_sha(resumed['weights'])
    assert weight_sha({'head':full['head']})==weight_sha({'head':resumed['head']})
    assert resumed['history']==full['history']
    assert resumed['scheduler']==full['scheduler']


def test_resume_refuses_changed_data_identity(tmp_path):
    c=small();ds=data.generate(c,'goodfire','independent',701)
    s=JournalStore(tmp_path/'x',{'test':1},tmp_path/'local',False)
    fit(s,'model',c,ds,0,.003,'cpu',stop=3)
    changed=copy.deepcopy(ds);changed['fingerprints']['train']='changed'
    with pytest.raises(ValueError,match='identity changed'):fit(s,'model',c,changed,0,.003,'cpu')


def test_pinned_original_dataset_reused_without_regeneration(tmp_path,monkeypatch):
    import focused_zoo as z
    c=small();ds=data.generate(c,'goodfire','independent',701)
    p=tmp_path/'old'/'dataset.pt';p.parent.mkdir();torch.save(ds,p)
    ref=dict(original_dataset_relative_path='old/dataset.pt',original_dataset_sha256=z.sha_file(p),original_data_fingerprints=ds['fingerprints'])
    monkeypatch.setattr(data,'generate',lambda *a:(_ for _ in ()).throw(AssertionError('Must reuse original data')))
    found=z.dataset_for_reference(tmp_path,ref)
    assert z.fingerprints(found)==ds['fingerprints']


def test_corrupted_original_data_cannot_be_used_for_baseline_comparison(tmp_path):
    import focused_zoo as z
    p=tmp_path/'dataset.pt';p.write_bytes(b'invalid')
    ref=dict(original_dataset_relative_path='dataset.pt',original_dataset_sha256='bad')
    with pytest.raises(ValueError,match='checksum'):z.original_dataset(tmp_path,ref)


def test_released_model_sources_match_archived_baselines():
    import importlib,inspect,hashlib
    ref=json.loads((Path(__file__).parent/'zoo_reference.json').read_text())
    for info in ref['class_sources'].values():
        p=Path(inspect.getfile(importlib.import_module(info['module'])))
        assert hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==info['sha256']
