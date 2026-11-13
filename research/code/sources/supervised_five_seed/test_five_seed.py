import copy,inspect,json
from pathlib import Path
import numpy as np
import pytest
import torch
import supervised_study as s
import five_seed as f
from coco_baselines import training
from coco_baselines.storage import identity
from test_supervised import Store
from precision_retry import EXPECTED_RUN,BASE_SHA

def fixture(tmp_path):
    parent=Store(tmp_path/'parent');parent.run_id=EXPECTED_RUN
    parent.root=tmp_path/'parent_checkpoints';parent.root.mkdir()
    plan=json.loads((Path(s.__file__).parent/'predecessor_holdout_fixture.json').read_text())['plan']
    parent.save('fresh_holdout_plan',plan);rows=[]
    for seed in f.SEEDS:
        for arm in s.ARMS+(['current_ggf'] if seed in [21,22,23] else []):
            native=arm.startswith('slics');key=f'{arm}_s{seed}';D=torch.eye(3)
            state=dict(D=D,updates=s.TOTAL) if native else dict(weights={'w':torch.ones(2)*seed},updates=s.GUIDED)
            weights={'D':D} if native else state['weights'];parent.save(key,state)
            rows.append(dict(seed=seed,arm=arm,key=key,weights_sha=training.weight_sha(weights),
                method='native_slics' if native else 'released_vanilla',targets=[],targets_sha=identity([]),
                total_training_updates=s.TOTAL,selected_total_updates=s.TOTAL,groups=4096 if arm=='current_ggf' else 75,
                capacity_scope='anchor' if arm=='current_ggf' else 'matched'))
    choice=dict(complete=True,ggf75_lr=.001)
    ctx=dict(store=parent,local=tmp_path/'local',norm=dict(scale=1),stamps=[],checkpoint_stamps=[],packs={},cfg={})
    options=dict(local=str(tmp_path/'local'),output=str(tmp_path/'output'),wrapper_sha=BASE_SHA)
    return ctx,rows,choice,options

def patch_scope(monkeypatch,rows):
    protocol=f.evaluation_protocol(s.protocol())
    for obj,key in [(s,'SEEDS'),(s,'protocol'),(s,'final_rows'),(s,'summarize'),(s,'plot')]:
        monkeypatch.setattr(obj,key,getattr(obj,key))
    f.install_evaluation_scope(rows,protocol)
    return protocol

def open_fixture(tmp_path,monkeypatch):
    ctx,rows,choice,options=fixture(tmp_path);calls=[]
    monkeypatch.setattr(f.object_schedule,'reserve_predecessor_holdout',lambda ctx:calls.append('reserve'))
    result,protocol=f.open_evaluation(ctx,rows,choice,options,s.protocol(),False)
    return ctx,rows,choice,options,result,protocol,calls

def test_exact_scope_and_original_protocol_is_not_mutated():
    old=s.protocol();before=copy.deepcopy(old);new=f.evaluation_protocol(old)
    assert old==before and old['seeds']==list(range(21,28))
    assert new['seeds']==list(range(21,26)) and new['omitted_seeds']==[26,27]
    assert new['screen_seeds']==[21,22,23] and new['cap']==old['cap']
    assert new['total']==10640 and new['small_lrs']==old['small_lrs']

@pytest.mark.parametrize('defect',['missing','duplicate','seed26','budget','unfinished','targets'])
def test_scope_never_accepts_missing_or_changed_endpoints(tmp_path,defect):
    _,rows,_,_=fixture(tmp_path)
    if defect=='missing':rows.pop()
    elif defect=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    elif defect=='seed26':rows[-1]['seed']=26
    elif defect=='budget':rows[-1]['total_training_updates']=9999
    elif defect=='unfinished':rows[-1]['selected_total_updates']=9999
    else:rows[-1]['targets_sha']='changed'
    with pytest.raises(ValueError):f.validate_rows(rows)

def test_separate_identity_while_parent_checkpoints_and_manifest_scope_stay_intact(tmp_path,monkeypatch):
    parent,rows,choice,options=fixture(tmp_path);before=copy.deepcopy(parent['store'].values)
    monkeypatch.setattr(f.object_schedule,'reserve_predecessor_holdout',lambda ctx:None)
    child,protocol=f.open_evaluation(parent,rows,choice,options,s.protocol(),False)
    assert child['store'].run_id!=EXPECTED_RUN and s.SEEDS==list(range(21,28))
    assert parent['store'].load('all_models_frozen') is None
    assert child['store'].load('fresh_holdout_plan')==parent['store'].load('fresh_holdout_plan')
    assert set(parent['store'].values)==set(before)
    for key,st in before.items():
        if 'weights' in st:assert training.weight_sha(st['weights'])==training.weight_sha(parent['store'].load(key)['weights'])
    assert protocol['seeds']==[21,22,23,24,25]
    assert (child['store'].results/'execution_five_seed/five_seed.py').exists()

def test_parent_model_reads_are_pinned_and_writes_refused(tmp_path,monkeypatch):
    parent,rows,_,_,child,_,_=open_fixture(tmp_path,monkeypatch);row=rows[0];store=child['store']
    expected=parent['store'].load(row['key']);actual=store.load(row['key'])
    assert training.weight_sha(expected['weights'])==training.weight_sha(actual['weights'])
    with pytest.raises(PermissionError):store.save(row['key'],expected)
    with pytest.raises(PermissionError):store._width_drive_recovery_original_save(row['key'],expected)
    assert store.load('not_a_permitted_parent_key') is None
    parent['store'].values[row['key']]['weights']['w']+=1
    with pytest.raises(ValueError,match='changed'):store.load(row['key'])

def test_freeze_23_rows_and_test_gate_never_calls_training_or_seeds26_27(tmp_path,monkeypatch):
    parent,rows,choice,_,child,_,_=open_fixture(tmp_path,monkeypatch)
    patch_scope(monkeypatch,rows)
    monkeypatch.setattr(s,'trim_local',lambda *a:None)
    monkeypatch.setattr(s,'native_fit',lambda *a:(_ for _ in ()).throw(AssertionError('training called')))
    monkeypatch.setattr(s,'small_fit',lambda *a:(_ for _ in ()).throw(AssertionError('training called')))
    with pytest.raises(FileNotFoundError):s.verify_frozen(child)
    frozen=s.freeze(child,choice)
    assert len(frozen['models'])==23 and frozen['protocol']['seeds']==[21,22,23,24,25]
    assert parent['store'].load('all_models_frozen') is None
    monkeypatch.setattr(s,'load_holdout',lambda ctx,coco:'five-seed-test')
    assert s.extract_after_freeze(child,None)=='five-seed-test'
    with pytest.raises(ValueError):s.final_rows(child,26,choice)
    frozen['models'].pop();child['store'].save('all_models_frozen',frozen)
    with pytest.raises(ValueError,match='Incomplete'):s.verify_frozen(child)

def test_resume_opened_evaluation_reuses_exact_lock_and_cached_progress(tmp_path,monkeypatch):
    parent,rows,choice,options,child,_,calls=open_fixture(tmp_path,monkeypatch)
    child['store'].save('audit_signature_index',{'a':'complete'})
    child['store'].save('final_joint_progress',dict(completed_images=17))
    def forbidden(*args):raise AssertionError('Own opened holdout incorrectly required unopened again')
    monkeypatch.setattr(f.object_schedule,'assert_unopened',forbidden)
    again,_=f.open_evaluation(parent,rows,choice,options,s.protocol(),False)
    assert again['store'].run_id==child['store'].run_id and calls==['reserve']
    assert again['store'].load('audit_signature_index')=={'a':'complete'}
    assert again['store'].load('final_joint_progress')['completed_images']==17
    changed=copy.deepcopy(rows);changed[0]['weights_sha']='changed'
    with pytest.raises(ValueError,match='lock changed'):f.open_evaluation(parent,changed,choice,options,s.protocol(),False)

def test_first_evaluation_refuses_already_opened_parent(tmp_path,monkeypatch):
    ctx,rows,choice,options=fixture(tmp_path)
    ctx['store'].save('audit_signature_index',{})
    with pytest.raises(ValueError,match='unopened'):f.open_evaluation(ctx,rows,choice,options,s.protocol(),False)

def test_five_seed_summary_uses_five_and_two_additional_orders(tmp_path,monkeypatch):
    parent,rows,choice,options,child,_,_=open_fixture(tmp_path,monkeypatch)
    patch_scope(monkeypatch,rows);real_plot=s.plot;monkeypatch.setattr(s,'plot',lambda *a:None)
    reports=[]
    for row in rows:
        v=dict(macro_capture=.3+row['seed']*.001,macro_ap=.7,reconstruction=dict(ve=.8,active_groups=32),
            parameters=123,seconds=1.,categories=[dict(category='fixture',category_index=0,capture=.3,ap=.7)])
        reports.append(dict(row=row,report=v))
    result=s.summarize(child,reports,[])
    assert result['evaluation_protocol']['seeds']==[21,22,23,24,25]
    for r in result['means']:
        assert r['seed_group']!='all_seven'
        assert r['seeds']=={'all_five':5,'development_tuning_orders':3,'additional_orders':2,'capacity_anchor_three':3}[r['seed_group']]
    assert all(r['n']==5 for r in result['seed_intervals'])
    assert (child['store'].results/'supervised_models.csv').read_text().count('\n')==24
    real_plot(child,result['means'])
    assert (child['store'].results/'figures/supervised_comparison.png').stat().st_size>1000

def test_plot_labels_and_worker_order_are_five_seed_only(tmp_path,monkeypatch):
    _,rows,_,_=fixture(tmp_path);captured=[]
    real=f.revise
    def capture(fn,replacements):
        source=inspect.getsource(fn)
        for a,b in replacements:source=source.replace(a,b)
        captured.append(source);return real(fn,replacements)
    monkeypatch.setattr(f,'revise',capture);patch_scope(monkeypatch,rows)
    assert '(5 seeds)' in captured[-1] and '(7 seeds)' not in captured[-1]
    assert 'all_seven' not in captured[0] and 'seven seeds' not in captured[-1]
    assert "['matched_ggf_final','slics_final_partial','current_ggf']" in captured[-1]
    worker=(Path(f.__file__).parent/'five_seed_worker.py').read_text()
    assert 'for seed in five_seed.SEEDS:' in worker
    assert 'study.selection(' not in worker and 'for seed in study.SEEDS' not in worker
    assert worker.index('study.final_rows(ctx,seed,choice)')<worker.index('five_seed.install_evaluation_scope')<worker.index('study.freeze(')<worker.index('study.extract_after_freeze(')
    assert "PRECISION[0].store=PRECISION[1].store=ctx['store']" in worker
