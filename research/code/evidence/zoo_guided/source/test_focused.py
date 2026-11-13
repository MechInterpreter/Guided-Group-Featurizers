import copy,inspect,json
from pathlib import Path
import numpy as np
import pytest
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation.inputs import JournalStore
from scale_models import model
from scale_plan import settings,METHODS
import focused_run as run
import focused_inputs as inp
import focused_evaluate as ev
import focused_protocol as rule

class Store:
    def __init__(self,p):self.results=p;self.results.mkdir(parents=True,exist_ok=True);self.saved={};self.output=p.parent;self.run_id='a'*64;self.calls=[]
    def load(self,k):self.calls.append(k);return copy.deepcopy(self.saved.get(k))
    def save(self,k,v,*a,**kw):self.saved[k]=copy.deepcopy(v);return v
    def write(self,n,v):
        p=self.results/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return p
    def event(self,*a,**k):pass

def row(seed,kind,lr=.0003,coefficient=.001):
    weights={'w':torch.tensor([seed],dtype=torch.float32)};targets=[]
    return dict(seed=seed,kind=kind,key=f'g4096_k32_s{seed}_{kind}_lr{int(lr*1e7)}',groups=4096,rank=12,active_groups=32,
        cap=.6 if kind=='agf_g' else None,lr=lr,coefficient=coefficient,additional_updates=3000,total_training_updates=10640,
        targets=targets,targets_sha=identity(targets),weights_sha=training.weight_sha(weights),weights=weights,
        reference_warm_mse=.3,development=dict(macro_capture=.4,macro_ap=.8,reconstruction=dict(mse=.2,ve=.7,active_groups=32)))

def test_fixed_recipe_49_endpoints_and_no_component_retraining():
    r=rule.recipe();assert r['total_updates']==10640 and r['additional_updates']==3000
    assert (r['groups'],r['width'],r['active_groups'],r['cap'])==(4096,12,32,.6)
    assert len(rule.expected_pairs())==49 and not r['configuration_selection'] and not r['additional_component_training']
    assert set(rule.ARMS)=={'agf_g',*METHODS}
    text=inspect.getsource(run.main)
    assert text.index('tuning=tune_baselines(ctx)')<text.index('freeze(ctx,tuning)')<text.index('load_holdout(ctx,coco)')
    assert text.index('verify_freeze(ctx)')<text.index('load_holdout(ctx,coco)')
    assert 'component_screen(' not in text and 'selection(' not in text and 'limit=12000' not in inspect.getsource(run)

def test_reuse_store_is_read_only_and_scope_limited(tmp_path):
    job=Store(tmp_path/'job');source=Store(tmp_path/'source');r=row(21,'agf_g');key=r['key']
    source.save(key,dict(weights=r.pop('weights'),updates=3000));source.save('test_complete','OLD')
    store=inp.ReuseStore(job,source,[r]);assert store.load(key)['updates']==3000
    assert store.load('test_complete') is None and 'test_complete' not in source.calls
    with pytest.raises(ValueError):store.save(key,{})
    with pytest.raises(ValueError):store._width_drive_recovery_original_save(key,{})
    store.save('new_fit',dict(updates=50));assert source.load('new_fit') is None

def test_advanced_prefix_is_rejected_not_retrained(tmp_path):
    job=Store(tmp_path/'job');source=Store(tmp_path/'source');key='g4096_k32_s21_released_vanilla_lr3000'
    source.save(key,dict(updates=12000));store=inp.ReuseStore(job,source,[])
    with pytest.raises(ValueError,match='beyond'):store.load(key)
    assert not job.saved

def test_pinned_pilot_skips_fit_and_mapping(tmp_path,monkeypatch):
    r=row(21,'agf_g');r.pop('weights');ctx=dict(pilot_rows=[r])
    monkeypatch.setattr(run,'run_branch',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Must reuse')))
    got=run.branch(ctx,21,'agf_g');assert got==r and got is not r
    assert run.audit_development(ctx,got)==r

def test_branch_never_uses_legacy12000_limit(monkeypatch):
    calls=[]
    def fit(ctx,g,k,seed,kind,**kw):calls.append((g,k,seed,kind,kw));r=row(seed,kind);r.pop('weights');return r
    monkeypatch.setattr(run,'run_branch',fit)
    run.branch(dict(pilot_rows=[]),24,'agf_g')
    assert calls[0][-1]['limit']==3000 and calls[0][-1]['cap']==.6

def test_tuning_uses_development_and_lasso_per_seed_activity(tmp_path,monkeypatch):
    s=Store(tmp_path/'job');calls=[]
    def branch(ctx,seed,kind,lr,coefficient):
        calls.append((kind,seed,lr,coefficient));r=row(seed,kind,lr,coefficient);r.pop('weights')
        r['development']['reconstruction']['mse']={.0001:.1,.0003:.2,.001:.3}[lr]
        if kind.endswith('group_lasso') and lr==.0001 and seed==23:r['development']['reconstruction']['active_groups']=90
        return r
    monkeypatch.setattr(run,'branch',branch)
    ctx=dict(store=s,rule=settings());out=run.tune_baselines(ctx)
    assert len(calls)==54 and calls[0][0]=='released_grassmannian'
    assert out['choices']['released_group_lasso']['lr']==.0003
    assert out['choices']['paper_group_lasso']['lr']==.0003
    assert out['choices']['released_vanilla']['lr']==.0001
    monkeypatch.setattr(run,'branch',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Cached tuning')))
    assert run.tune_baselines(ctx)==out

def test_no_eligible_lasso_is_retained_and_flagged(tmp_path,monkeypatch):
    s=Store(tmp_path/'job')
    def branch(ctx,seed,kind,lr,coefficient):
        r=row(seed,kind,lr,coefficient);r.pop('weights');r['development']['reconstruction']['active_groups']=80
        return r
    monkeypatch.setattr(run,'branch',branch);out=run.tune_baselines(dict(store=s,rule=settings()))
    assert not out['choices']['paper_group_lasso']['sparsity_matched'] and 'paper_group_lasso' in out['choices']

def test_freeze_exact49_and_mismatched_stream_rejected(tmp_path,monkeypatch):
    s=Store(tmp_path/'job');ctx=dict(store=s)
    def branch(ctx,seed,kind,**kw):
        r=row(seed,kind);s.save(r['key'],dict(updates=3000,weights=r.pop('weights'),sampled_batches_sha=f'b{seed}',sampled_noise_sha=f'n{seed}'));return r
    monkeypatch.setattr(run,'branch',branch)
    tuning=dict(choices={k:dict(lr=.0003,coefficient=.001) for k in METHODS})
    frozen=run.freeze(ctx,tuning);assert len(frozen['models'])==49 and not frozen['component_training_repeated']
    assert all(r['total_training_updates']==10640 for r in frozen['models'])
    assert run.freeze(ctx,{})==frozen
    s.saved.pop('all_models_frozen')
    def bad(ctx,seed,kind,**kw):
        r=branch(ctx,seed,kind,**kw)
        if kind=='paper_vanilla':s.saved[r['key']]['sampled_noise_sha']='DIFFERENT'
        return r
    monkeypatch.setattr(run,'branch',bad)
    with pytest.raises(ValueError,match='streams'):run.freeze(ctx,tuning)

def test_opened_source_holdout_is_rejected(tmp_path):
    s=Store(tmp_path/'source');s.root=tmp_path/'states';s.root.mkdir()
    inp.reject_opened_holdout(s);s.save('test_complete',dict(complete=True))
    with pytest.raises(ValueError):inp.reject_opened_holdout(s)
    s.saved.clear();(s.root/'features_test_000000').mkdir()
    with pytest.raises(ValueError):inp.reject_opened_holdout(s)

def test_endpoint_budget_and_readouts_cannot_change():
    r=row(21,'agf_g');r.pop('weights');rule.validate_row(r)
    for key,value in [('total_training_updates',19640),('cap',.4),('targets_sha','BAD')]:
        bad=dict(r,**{key:value})
        with pytest.raises(ValueError):rule.validate_row(bad)

def test_local_cache_trim_requires_verified_remote_copy(tmp_path):
    local=tmp_path/'local';job=JournalStore(tmp_path/'drive',dict(test='trim'),local/'cache',False)
    source=Store(tmp_path/'source');store=inp.ReuseStore(job,source,[])
    key='g4096_k32_s24_released_vanilla_lr3000';warm='g4096_k32_s24_released_vanilla_warm'
    job.save(key,dict(updates=3000,weights=torch.ones(1)));job.save(warm,dict(updates=7640,weights=torch.ones(1)))
    inp.trim_verified_local(dict(store=store,local=local),dict(key=key))
    assert not (job.rescue/key).exists() and not (job.rescue/warm).exists()
    assert job.load(key)['updates']==3000 and job.load(warm)['updates']==7640
    job.save(key,dict(updates=3001,weights=torch.ones(1)*2),drive=False)
    inp.trim_verified_local(dict(store=store,local=local),dict(key=key))
    assert (job.rescue/key).exists() and job.load(key)['updates']==3001
    with pytest.raises(ValueError):inp.trim_verified_local(dict(store=store,local=tmp_path/'wrong'),dict(key=key))

def test_result_archive_crc_and_hashes(tmp_path):
    s=Store(tmp_path/'output/results');s.write('small.json',dict(a=1));ctx=dict(store=s)
    a=tmp_path/'a.zip';a.write_bytes(b'core');b=tmp_path/'b.zip';b.write_bytes(b'wrapper')
    out=run.export(ctx,dict(source_zip=str(a),wrapper_zip=str(b)))
    import zipfile,hashlib
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in json.loads(z.read('artifact_hashes.json')).items())

def test_actual49_model_evaluation_and_resume_cache(tmp_path,monkeypatch):
    torch.set_num_threads(1);rng=np.random.default_rng(9);x=rng.normal(size=(6,196,4)).astype(np.float32);masks=[]
    for i in range(6):
        f=np.zeros((196,2),np.uint8);f[:80,0]=255;f[85:165,1]=255
        masks.append(dict(fraction=f,crowd=np.zeros(196,np.float32),instances=[dict(category=c,annotation_id=100*i+c,fraction=f[:,c]/255.) for c in [0,1]]))
    pack=dict(x=x,ids=list(range(1,7)),masks=masks,identity=identity(dict(ids=list(range(1,7)))))
    cfg=dict(category_indices=[0,1],negative_fraction=.1,positive_fraction=.5,min_object_patch_mass=2,min_background_patches=4,
        decision_min_images=1,target_min_images=1,patch_sample_seed=61002,detection_patches=32,preview_images=0,audit_image_batch=1,
        batch=128,score_ap_weight=.2,bootstrap=4,fresh_test_seed=12)
    s=JournalStore(tmp_path/'drive',dict(test=True),tmp_path/'local',False)
    ctx=dict(store=s,rule=settings(),cfg=cfg,device='cpu',norm=dict(scale=1.,mean=np.zeros(4)),stamps=[],checkpoint_stamps=[])
    targets=[dict(category_index=c,category=chr(97+c),contrast_group=c,detector_group=c,heat_mean=0.,heat_scale=1.) for c in [0,1]]
    rows=[]
    for seed in rule.SEEDS:
        torch.manual_seed(seed);weights=training.weights(model('released_vanilla',6,2,2,4))
        for kind in rule.ARMS:
            key=f'{kind}_{seed}';s.save(key,dict(weights=weights,updates=3000))
            rows.append(dict(key=key,seed=seed,kind=kind,method='released_vanilla',groups=6,rank=2,active_groups=2,coefficient=.001,
                targets=targets,targets_sha=identity(targets),weights_sha=training.weight_sha(weights),head_parameters=4 if kind=='agf_g' else 0,
                cap=.6 if kind=='agf_g' else None,total_training_updates=10640))
    s.save('all_models_frozen',dict(complete=True,models=rows))
    monkeypatch.setattr(ev,'validate_row',lambda r:r)
    report=ev.run(ctx,pack)
    assert report['complete'] and len(report['models'])==49 and len(report['paired_comparisons'])==42
    assert all(abs(r['capture_gain_pp'])<1e-9 and abs(r['AP_gain_pp'])<1e-9 for r in report['paired_comparisons'])
    monkeypatch.setattr(ev,'evaluate',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Cached audit must skip inference')))
    assert ev.run(ctx,pack)==report
    s.save('all_models_frozen',dict(complete=True,models=rows[:-1]))
    with pytest.raises(ValueError):ev.verify_freeze(ctx)

@pytest.mark.parametrize('method',METHODS)
def test_actual_baseline_fit_restarts_exactly(tmp_path,method):
    from test_scale_study import context
    from scale_fit import fit
    a=context(tmp_path/'resume');torch.manual_seed(5);weights=training.weights(model(method,6,2,2,4))
    first=fit(a,'baseline',method,6,2,21,weights,4,2,stream=1)
    second=fit(a,'baseline',method,6,2,21,weights,4,4,stream=1)
    b=context(tmp_path/'whole');whole=fit(b,'baseline',method,6,2,21,weights,4,4,stream=1)
    assert first['updates']==2 and second['updates']==4
    assert training.weight_sha(second['weights'])==training.weight_sha(whole['weights'])
    assert second['sampled_batches_sha']==whole['sampled_batches_sha'] and second['sampled_noise_sha']==whole['sampled_noise_sha']
