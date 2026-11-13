import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from coco_adaptation import inputs
from coco_baselines import training
from coco_ggf import twelve_rank as rank,twelve_capacity as exp
from rank_core import Moments,atomic_json,block_statistics,fingerprint,sha,summarize


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):yield
    torch.set_num_threads(old)


def fixture(tmp_path,monkeypatch):
    protocol=dict(analysis_source_sha=rank.RANK_SOURCE,capacity_run=exp.PARENT_RUN,normalization_sha='norm',train_ids=[1,2])
    run_id=fingerprint(protocol);monkeypatch.setattr(rank,'RANK_RUN',run_id)
    source=tmp_path/'old_rank';folder=source/'results'/run_id[:20]
    atomic_json(folder/'protocol.json',protocol)
    atomic_json(folder/'summary.json',dict(run_id=run_id,complete=True,models=[{}]*30))
    atomic_json(folder/'cohort.json',dict(identity='cohort',ids=[1,2]))
    store=inputs.JournalStore(tmp_path/'new',dict(normalization_sha='norm'),tmp_path/'local',False)
    ctx=dict(base=dict(groups=32,block_rank=8,active_groups=16),store=store,roots=[],stamps=[],checkpoint_stamps=[],device='cpu',
        packs=dict(train=dict(x=np.random.default_rng(3).normal(size=(2,196,768)).astype(np.float32),ids=[1,2],identity='cohort')))
    targets=[dict(category_index=i,category='c'+str(i)) for i in range(32)]
    frozen=[]
    for seed in exp.SEEDS:
        for arm in exp.SOURCE_ARMS:
            row=dict(arm=arm,seed=seed,weights_sha=arm+str(seed));ctx['roots'].append(row)
            m=Moments(32,8);m.add_dense(np.random.default_rng(seed).normal(size=(40,32,8)))
            blocks,_=block_statistics(m,np.broadcast_to(np.eye(8),(32,8,8)))
            meta=dict(arm=arm,seed=seed,rank=8,active_groups=16)
            records=[dict(r,**meta) for r in summarize(blocks)]
            dest=folder/'models'/(arm+'_s'+str(seed))
            atomic_json(dest/'summary.json',records);atomic_json(dest/'blocks.json',blocks)
            hashes={p.name:sha(p) for p in dest.iterdir()}
            signature=fingerprint(dict(protocol=run_id,train='cohort',model=row['weights_sha'],rank=8,k=16))
            atomic_json(dest/'complete.json',dict(signature=signature,complete=True,patches=392000,file_hashes=hashes))
        for arm in exp.ARMS:
            width,k,sup=exp.arm_spec(arm);cfg=dict(ctx['base'],block_rank=width)
            torch.manual_seed(seed);model=training.make_model('vanilla',cfg)
            weights=training.weights(model);key=arm+'_s'+str(seed)
            store.save(key,dict(weights=weights))
            frozen.append(dict(seed=seed,arm=arm,rank=width,active_groups=k,weights_sha=training.weight_sha(weights),key=key,supervised=sup,targets=targets))
    monkeypatch.setattr(exp,'verify_freeze',lambda ctx:dict(complete=True,models=frozen))
    monkeypatch.setattr(exp,'validate_state',lambda *args:None)
    return ctx,source


def test_rank_source_requires_exact_model_cohort_and_checksums(tmp_path,monkeypatch):
    ctx,source=fixture(tmp_path,monkeypatch)
    folder=rank.verify_source(ctx,source)
    ctx['roots'][0]['weights_sha']='different'
    with pytest.raises(ValueError,match='different model'):rank.verify_source(ctx,source)
    ctx['roots'][0]['weights_sha']=exp.SOURCE_ARMS[0]+'21'
    ctx['packs']['train']['identity']='wrong'
    with pytest.raises(ValueError,match='cohort differs'):rank.verify_source(ctx,source)
    ctx['packs']['train']['identity']='cohort'
    (folder/'models/agf_8d16_s21/blocks.json').write_text('changed')
    with pytest.raises(ValueError,match='file changed'):rank.verify_source(ctx,source)


def test_rank_scheduler_reuses_six_sources_and_resumes_all_twelve_new_models(tmp_path,monkeypatch):
    ctx,source=fixture(tmp_path,monkeypatch)
    originals={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
    calls=[];analyze=rank.analyze_model
    def counted(*args,**kw):calls.append(1);return analyze(*args,**kw)
    monkeypatch.setattr(rank,'analyze_model',counted)
    result=rank.run(ctx,source)
    assert result['complete'] and result['new_models_analyzed']==12 and result['saved_8d_models_reused']==6
    assert len(calls)==12 and len(result['three_seed_means'])==36
    folder=ctx['store'].results/'effective_rank'
    assert len(list((folder/'models').iterdir()))==18
    assert (folder/'figures/initialized_slots.png').exists()
    def forbidden(*args,**kw):raise AssertionError('Completed rank inference repeated')
    monkeypatch.setattr(rank,'analyze_model',forbidden)
    again=rank.run(ctx,source)
    assert result==again
    assert originals=={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
