"""Checks for reserve copy semantics, profiling, controls and phase ordering."""
import copy,json
import numpy as np
import pytest
import torch
from coco_adaptation import core,inputs
from coco_baselines import training
from coco_baselines.storage import identity
from coco_ggf import reserve_additions as add,warm_capacity as cap,capacity_reuse as reuse
from coco_ggf import capacity_fit as engine
from test_capacity_sweep import fixture,single_thread


def test_copy_keeps_every_original_slot_and_donor_value(tmp_path):
    ctx,st,_=fixture(tmp_path);before=training.weight_sha(st['weights']);donors=[13,82]
    out=add.copy_additions(st['weights'],32,donors,2)
    add.assert_preserved_start(st['weights'],out,32);add.assert_padding(out,32,2,True)
    for slot,atom in enumerate(donors):
        assert torch.equal(out['W_dec'][slot*6+5],st['weights']['W_dec'][atom])
        assert torch.equal(out['W_enc'][:,slot*6+5],st['weights']['W_enc'][:,atom])
        assert torch.equal(out['b_enc'][slot*6+5],st['weights']['b_enc'][atom])
        assert torch.equal(out['W_dec'][(atom//5)*6+atom%5],st['weights']['W_dec'][atom])
    assert training.weight_sha(st['weights'])==before
    with pytest.raises(ValueError,match='reserve'):add.copy_additions(st['weights'],32,[3,82],2)


def test_vectorized_profile_matches_explicit_raw_addition_error(tmp_path):
    ctx,st,targets=fixture(tmp_path);m=core.new_model('vanilla',ctx['base'],st['weights'],'cpu');p=ctx['packs']['train']
    stats=add.profile_piece(m,p,0,4,targets,ctx['cfg'],2);explicit=np.zeros((2,150));energy=np.zeros(2)
    with torch.no_grad():
        for i in range(4):
            x=torch.from_numpy(p['x'][i]);raw=x@m.W_enc+m.b_enc;z=m.encode(x).reshape(196,-1)
            ww,_,cats=core.object_weights([p['masks'][i]],[0,1],ctx['cfg']);wt=torch.from_numpy(ww)
            dx=wt@x;da=wt@raw;dz=wt@z
            for row,c in enumerate(cats):
                current=dz[row,c*5:c*5+5]@m.W_dec[c*5:c*5+5];error=(dx[row]-current).square().sum()
                for atom in range(10,160):
                    edited=current+da[row,atom]*m.W_dec[atom]
                    explicit[c,atom-10]+=float(error-(dx[row]-edited).square().sum())
                energy[c]+=float(dx[row].square().sum())
    np.testing.assert_allclose(stats['gain'],explicit,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(stats['energy'],energy,rtol=1e-6)


def test_targeted_ranking_and_random_pool_are_train_only_and_seeded():
    p=dict(gain=np.array([[1.,3.,2.],[4.,2.,1.]]),energy=np.ones(2),counts=np.ones(2),
        foreground=np.array([[0.,1.,1.],[1.,1.,0.]]),background=np.zeros((2,3)))
    a=add.choose_directions(p,21,2);b=add.choose_directions(p,21,2)
    assert a==b and a['targeted']==[11,10]
    assert a['random'][0] in [11,12] and a['random'][1] in [10,11]
    assert 'test' not in a


def test_train_profile_resumes_complete_chunks_and_reuses_saved_selection(tmp_path,monkeypatch):
    ctx,st,targets=fixture(tmp_path);row=dict(seed=21,arm='guided5_continue',weights_sha=training.weight_sha(st['weights']),targets=targets)
    ctx.update(roots=[row])
    monkeypatch.setattr(cap,'source_state',lambda ctx,row:copy.deepcopy(st))
    monkeypatch.setattr(add,'screen_pack',lambda ctx:ctx['packs']['train'])
    original_settings=add.settings
    monkeypatch.setattr(add,'settings',lambda:dict(original_settings(),profile_chunk=2))
    original_piece=add.profile_piece;calls=[0]
    def interrupted(*args,**kw):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('synthetic profile interruption')
        return original_piece(*args,**kw)
    with monkeypatch.context() as p:
        p.setattr(add,'profile_piece',interrupted)
        with pytest.raises(RuntimeError,match='profile interruption'):add.selection_for(ctx,21)
    assert ctx['store'].load('reserve_profile_s21')['next_offset']==2
    chosen=add.selection_for(ctx,21)
    model=core.new_model('vanilla',ctx['base'],st['weights'],'cpu')
    explicit=original_piece(model,ctx['packs']['train'],0,4,targets,ctx['cfg'],2)
    expected=add.choose_directions(explicit,21,2)
    assert chosen['targeted']==expected['targeted'] and chosen['random']==expected['random']
    def forbidden(*a,**kw):raise AssertionError('saved profile reran')
    monkeypatch.setattr(add,'profile_piece',forbidden)
    assert add.selection_for(ctx,21)==chosen


def test_disabled_padding_stays_zero_after_real_guided_updates(tmp_path):
    ctx,st,targets=fixture(tmp_path);w=add.copy_additions(st['weights'],32,[13,82],2)
    opt=cap.widen_optimizer(st['optimizer'],st['weights'],32,6,True)
    sub=dict(ctx,base=dict(ctx['base'],block_rank=6),recipe=add.settings())
    out=engine.fit(sub,'padding',w,targets,10,111021,True,True,10.,initial_head=st['head'],initial_balance=copy.deepcopy(st['balance']),initial_optimizer=opt)
    add.assert_padding(out['weights'],32,2,True)
    bad=copy.deepcopy(out['weights']);bad['W_dec'][2*6+5,0]=.01
    with pytest.raises(ValueError,match='padding'):add.assert_padding(bad,32,2,True)


def test_padded_no_addition_control_reproduces_source_prediction(tmp_path):
    ctx,st,_=fixture(tmp_path)
    w={k:cap.expand_tensor(v,k,32,5,6) for k,v in st['weights'].items()}
    a=core.new_model('vanilla',ctx['base'],st['weights'],'cpu');b=core.new_model('vanilla',dict(ctx['base'],block_rank=6),w,'cpu')
    with torch.no_grad():
        x=torch.from_numpy(ctx['packs']['train']['x'][0]);torch.testing.assert_close(a(x)[0],b(x)[0],rtol=1e-5,atol=1e-6)
    add.assert_padding(w,32,2,False)


def test_exact_partial_addition_resume_and_matching_noise(tmp_path):
    runs=[]
    for name,donors in [('targeted',[13,82]),('random',[14,83]),('ordinary',None)]:
        ctx,st,targets=fixture(tmp_path/name)
        w=add.copy_additions(st['weights'],32,donors,2) if donors else {k:cap.expand_tensor(v,k,32,5,6) for k,v in st['weights'].items()}
        opt=cap.widen_optimizer(st['optimizer'],st['weights'],32,6,True);sub=dict(ctx,base=dict(ctx['base'],block_rank=6),recipe=add.settings())
        def fit(key,limit=None):return engine.fit(sub,key,w,targets,4,111021,True,True,10.,initial_head=st['head'],initial_balance=copy.deepcopy(st['balance']),initial_optimizer=opt,max_updates=limit)
        full=fit('full');fit('resume',2);end=fit('resume')
        for key in ['weights','optimizer','head','balance']:
            assert identity(engine.state_identity(full[key]))==identity(engine.state_identity(end[key]))
        runs.append(full)
    assert len({r['sampled_batches_sha'] for r in runs})==1
    assert len({r['sampled_noise_sha'] for r in runs})==1


def test_all_seed_priority_run_writes_initial_final_diagnostics_and_skips_complete(tmp_path,monkeypatch):
    ctx,_,targets=fixture(tmp_path/'main',d=768);ctx.update(roots=[],stamps=[],checkpoint_stamps=[])
    saved={}
    for seed in add.SEEDS:
        for sup in [True,False]:
            _,st,_=fixture(tmp_path/f'root_{seed}_{sup}',sup,d=768);arm='guided5_continue' if sup else 'vanilla5_continue'
            root=dict(seed=seed,arm=arm,supervised=sup,targets=targets,targets_sha=identity(targets),reference_mse=10.,
                weights_sha=training.weight_sha(st['weights']),optimizer_sha=identity(engine.state_identity(st['optimizer'])),balance_sha=identity(engine.state_identity(st['balance'])))
            ctx['roots'].append(root);saved[(seed,arm)]=st
    monkeypatch.setattr(cap,'source_state',lambda ctx,row:copy.deepcopy(saved[(row['seed'],row['arm'])]))
    monkeypatch.setattr(add,'UPDATES',2)
    selections={s:dict(targeted=[13,82],random=[14,83]) for s in add.SEEDS}
    monkeypatch.setattr(add,'selection_for',lambda ctx,seed:selections[seed])
    result=add.run(ctx)
    assert result['complete'] and len(result['models'])==12 and len(result['initial_models'])==12
    assert result['development_only'] and not result['test_features_opened']
    assert len(result['comparisons'])==12
    for row in result['models']:
        assert row['added_directions']==(2 if row['arm'] in ['targeted','random'] else 0)
        assert set(row['reports'])=={'select','confirm'}
        assert 'fixed_category_slot_diagnostic' in row['reports']['select']
    assert (ctx['store'].results/'final_select_means.csv').exists()
    def forbidden(*a,**kw):raise AssertionError('completed priority stage retrained')
    monkeypatch.setattr(add,'fit_branch',forbidden)
    assert identity(add.run(ctx))==identity(result)


def test_missing_or_different_previous_capacity_config_does_not_import(tmp_path):
    ctx,_,_=fixture(tmp_path);ctx['local']=tmp_path;ctx.update(checkpoint_stamps=[],sources={})
    assert not reuse.restore(ctx)['reused']
    candidate=dict(ctx['store'].config,source_sha=reuse.PREVIOUS_SHA);rid=identity(candidate)
    manifest=ctx['store'].output/'results'/rid[:20]/'manifest.json';manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(run_id=rid,config=dict(candidate,extra='wrong'))))
    with pytest.raises(ValueError,match='manifest differs'):reuse.restore(ctx)


def test_exact_previous_progress_is_imported_without_overwriting_newer_state(tmp_path):
    ctx,st,_=fixture(tmp_path);ctx.update(local=tmp_path,checkpoint_stamps=[],stamps=[],sources={})
    cfg=dict(ctx['store'].config,source_sha=reuse.PREVIOUS_SHA)
    old=inputs.JournalStore(ctx['store'].output,cfg,tmp_path/'old_local',False)
    key='agf_6d16_s21';old.save(key,dict(updates=250,weights=st['weights']))
    result=reuse.restore(ctx);assert result['reused']
    assert ctx['store'].load(key)['updates']==250
    ctx['store'].save(key,dict(updates=500,weights=st['weights']))
    reuse.restore(ctx);assert ctx['store'].load(key)['updates']==500
    assert old.load(key)['updates']==250


def test_worker_priority_order_and_no_test_before_capacity_freeze():
    from pathlib import Path
    worker=(Path(__file__).resolve().parents[1]/'capacity_worker.py').read_text()
    assert worker.index('reserve.run(')<worker.index('capacity_reuse.restore(')<worker.index('exp.run(')<worker.index('exp.evaluate(')
    assert 'reserve_prescreen_complete' in worker
    assert 'load_test(' not in __import__('inspect').getsource(add.run)
    ctx=dict(packs={'test':{}})
    with pytest.raises(ValueError,match='development only'):add.run(ctx)


def test_short_protocol_budget_and_copy_scope():
    cfg=add.settings();assert cfg['planned_new_updates']==6000 and cfg['updates_per_arm']==500
    assert cfg['balance']['bounds']['capture'][1]==.4
    assert cfg['balance']['bounds']['detection'][1]==.4
    assert cfg['concept_groups']==75 and cfg['no_removal_graph']
