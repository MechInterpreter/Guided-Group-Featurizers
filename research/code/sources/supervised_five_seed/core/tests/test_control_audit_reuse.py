"""Regression for missing root audits despite a completed continuation source."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from coco_ggf import control_audit_reuse as repair,next_experiments as prior
from test_width_balance import fixture


class Source:
    run_id=prior.PREVIOUS_RUN
    def __init__(self,data):self.data=data
    def load(self,key):return copy.deepcopy(self.data.get(key))
    def save(self,*args):raise AssertionError('Immutable source must never be written')


def setup(tmp_path,monkeypatch):
    ctx,_=fixture(tmp_path,'recovery',4)
    ctx.update(roots=[],test_stamp={'cohort':'same'},test_ids=[1,2],checkpoint_stamps=[])
    data={'experiment_complete':dict(complete=True,models=[])}
    for seed in prior.SEEDS:
        for arm in ['cap040','vanilla4']:
            row=dict(seed=seed,arm=arm,weights_sha=f'{seed}_{arm}',targets_sha='targets',rank=4)
            ctx['roots'].append(row)
            report=dict(row,macro_capture=.3,macro_ap=.7,reconstruction=dict(ve=.8,mse=.2),
                categories=[dict(category_index=0,capture=.3,ap=.7)])
            tag=f's{seed}_{arm}'
            data['audit_'+tag]=dict(identity=prior.base.report_signature(row,ctx['test_stamp'],ctx['cfg']),report=report)
            data['compact_'+tag]=dict(image_ids=[1,2],category_indices=[0],values=np.array([.3,.7]))
            data['experiment_complete']['models'].append(report)
    source=Source(data);ctx['sources']={'previous':source}
    monkeypatch.setattr(prior.matched,'assert_sources',lambda c:None)
    monkeypatch.setattr(prior.matched,'source_key_stamp',lambda s,k:dict(run_id=s.run_id,key=k))
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**kw:pytest.fail('No training during recovery'))
    return ctx,source


def test_missing_root_audits_are_reused_without_training_and_resume_is_idempotent(tmp_path,monkeypatch):
    ctx,source=setup(tmp_path,monkeypatch)
    before=prior.engine.state_identity(source.data)
    ctx['store'].save('frozen_width5',{'complete':True,'model_hash':'unchanged'})
    result=repair.restore(ctx)
    assert result['new_gradient_updates']==0 and not result['new_inference']
    assert len(result['restored_keys'])==12
    for key in result['restored_keys']:
        assert prior.engine.state_identity(ctx['store'].load(key))==prior.engine.state_identity(source.load(key))
    assert repair.restore(ctx)==result
    assert len(ctx['checkpoint_stamps'])==12
    assert prior.engine.state_identity(source.data)==before
    assert ctx['store'].load('frozen_width5')=={'complete':True,'model_hash':'unchanged'}


@pytest.mark.parametrize('fault',['source_run','incomplete','missing_pair','signature','summary','cohort','categories','destination'])
def test_incompatible_audits_stop_before_any_recovery_writes(tmp_path,monkeypatch,fault):
    ctx,source=setup(tmp_path,monkeypatch);tag='s23_cap040'
    if fault=='source_run':source.run_id='wrong'
    elif fault=='incomplete':source.data['experiment_complete']['complete']=False
    elif fault=='missing_pair':del source.data['compact_'+tag]
    elif fault=='signature':source.data['audit_'+tag]['identity']='wrong'
    elif fault=='summary':source.data['experiment_complete']['models'][-2]=dict(source.data['experiment_complete']['models'][-2],macro_capture=.4)
    elif fault=='cohort':source.data['compact_'+tag]['image_ids']=[2,1]
    elif fault=='categories':source.data['compact_'+tag]['category_indices']=[1]
    elif fault=='destination':ctx['store'].save('compact_'+tag,{'incompatible':True})
    with pytest.raises((ValueError,FileNotFoundError)):repair.restore(ctx)
    assert ctx['store'].load('audit_s21_cap040') is None
    assert ctx['store'].load('control_audit_reuse') is None
