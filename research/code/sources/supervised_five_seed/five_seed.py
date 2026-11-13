"""Five-seed evaluation fork; retain original training identities and checkpoints."""
import copy,json,hashlib
from pathlib import Path
import supervised_study as study
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation.inputs import JournalStore
from coco_ggf.drive_recovery import install_drive_recovery
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched
import object_schedule
from precision_retry import EXPECTED_RUN
from training_precision import revise

SEEDS=[21,22,23,24,25]
HERE=Path(__file__).resolve().parent

def evaluation_protocol(original):
    p=copy.deepcopy(original)
    if p['seeds']!=list(range(21,28)) or p['screen_seeds']!=SEEDS[:3]:
        raise ValueError('Unexpected original seven-seed protocol')
    p.update(version=4,seeds=SEEDS.copy(),evaluation_revision='five-seed-scope-amendment-v1',
        original_seed_plan=list(range(21,28)),omitted_seeds=[26,27],
        seed_scope='Five training orders (21–25), common deterministic SVD initializer and fixed data; 21–23 tuned the LR, 24–25 are additional orders.',
        scope_amendment='User reduced seven to five for compute/time before test evaluation. Not a completed seven-seed study.',
        headline_reporting='Completed-training fixed-budget comparison; checkpoint-selected results and their original statistical contrast remain separately available.',
        training_protocol_and_fingerprints='Retained from original run; no loss, solver threshold, budget, optimizer, data or checkpoint-selector change.',
        training_arithmetic_changed_on_precision_recovery_batches=True)
    return p


def validate_rows(rows):
    expected={(seed,arm) for seed in SEEDS for arm in study.ARMS}|{(seed,'current_ggf') for seed in SEEDS[:3]}
    if len(rows)!=23 or {(r['seed'],r['arm']) for r in rows}!=expected:
        raise ValueError('All five completed seed pairs and three saved capacity anchors are required')
    for r in rows:
        if r['total_training_updates']!=study.TOTAL or identity(r['targets'])!=r['targets_sha']:
            raise ValueError('Changed training budget/readouts')
        if r['arm'] in ['matched_ggf_final','slics_final_partial'] and r['selected_total_updates']!=study.TOTAL:
            raise ValueError('Do not evaluate an unfinished fixed-budget checkpoint')


class EvaluationStore:
    """New evaluation writes; narrowly permitted, hash-checked model reads from parent."""
    def __init__(self,job,parent,rows):
        self.job=job;self.parent=parent
        self.pins={r['key']:(r['weights_sha'],r['method']) for r in rows}
    def __getattr__(self,name):return getattr(self.job,name)
    def load(self,key):
        if key in self.pins:
            st=memory.required(self.parent,key);sha,method=self.pins[key]
            weights={'D':st['D']} if method=='native_slics' else st['weights']
            if training.weight_sha(weights)!=sha:raise ValueError('Parent model changed after five-seed freeze: '+key)
            return st
        return self.job.load(key)
    def save(self,key,value,*args,**kwargs):
        if key in self.pins:raise PermissionError('Evaluation cannot rewrite a trained model')
        return self.job.save(key,value,*args,**kwargs)
    def force_save(self,key,value,*args,**kwargs):
        if key in self.pins:raise PermissionError('Evaluation cannot rewrite a trained model')
        return self.job._width_drive_recovery_original_save(key,value,*args,**kwargs)
    @property
    def _width_drive_recovery_original_save(self):return self.force_save


def open_evaluation(ctx,rows,choice,options,original_protocol,require_drive=True):
    validate_rows(rows)
    parent=ctx['store']
    if parent.run_id!=EXPECTED_RUN:raise ValueError('Wrong saved training run')
    protocol=evaluation_protocol(original_protocol)
    plan=memory.required(parent,'fresh_holdout_plan')
    if identity(plan)!=object_schedule.RESERVED_PLAN_SHA:raise ValueError('Reserved holdout changed')
    if not choice.get('complete'):raise ValueError('Development settings must be frozen')
    pins=[{k:r[k] for k in ['seed','arm','key','weights_sha','targets_sha','selected_total_updates','total_training_updates']} for r in rows]
    hashes=json.loads((HERE/'REVISION_HASHES.json').read_text())
    payload=dict(study='ggf_slics_five_seed_evaluation',version=1,parent_run=EXPECTED_RUN,
        execution_revision_sha=identity(hashes),
        parent_source_sha=options['wrapper_sha'],evaluation_protocol=protocol,
        settings_sha=identity(choice),models=pins,holdout_plan_sha=identity(plan),
        numerical_versions=matched.versions(),normalization_sha=identity(ctx['norm']))
    new_id=identity(payload);lockpath=parent.results/'five_seed_scope_lock.json'
    lock=dict(parent_run=EXPECTED_RUN,evaluation_run=new_id,config_sha=identity(payload),
        seeds=SEEDS,omitted_seeds=[26,27],holdout_plan_sha=identity(plan),
        amendment_before_test=True,reason='User reduced scope for compute/time; no test metrics used to choose five seeds.')
    if lockpath.exists():
        if json.loads(lockpath.read_text())!=lock:raise ValueError('Five-seed scope/model/settings lock changed')
        # This exact evaluation may already have opened its own holdout. Resume it;
        # do not falsely demand that its already-reserved test be unopened again.
    else:
        object_schedule.assert_unopened(parent)
        object_schedule.reserve_predecessor_holdout(ctx)
        # Persist the scope before any test extraction; the parent manifest and
        # seven-seed freeze markers are never rewritten.
        parent.write('five_seed_scope_lock.json',lock)
    local=Path(options['local'])/'five_seed_evaluation'
    job=JournalStore(options['output'],payload,local/'rescue',require_drive)
    install_drive_recovery(job)
    store=EvaluationStore(job,parent,rows)
    out=dict(ctx,store=store,local=local,active_key=None)
    local.mkdir(parents=True,exist_ok=True)
    store._width_drive_recovery_original_save('fresh_holdout_plan',copy.deepcopy(plan),drive=True)
    store.save('supervised_settings_frozen',copy.deepcopy(choice))
    store.write('fresh_holdout_plan.json',plan)
    store.write('five_seed_scope_amendment.json',lock)
    store.write('protocol.json',payload)
    store.write('README.json',dict(seeds=SEEDS,total_models=23,
        matched_pairs=5,capacity_anchors=3,training='Saved states reused; only unfinished training through seed 25 is resumed.',
        limits=protocol['seed_scope'],scope_amendment=protocol['scope_amendment'],
        precision='Strict float64 recovery retained. Earlier precision sources copied with this execution revision.'))
    folder=store.results/'execution_five_seed';folder.mkdir(exist_ok=True)
    for name in list(hashes)+['REVISION_HASHES.json']:(folder/name).write_bytes((HERE/name).read_bytes())
    return out,protocol


def install_evaluation_scope(rows,protocol):
    """Only after original-identity training has finished, switch evaluation scope."""
    validate_rows(rows)
    summary=revise(study.summarize,[
        ("('all_seven',SEEDS)","('all_five',SEEDS)")])
    plot=revise(study.plot,[
        ("['all_seven','capacity_anchor_three']","['all_five','capacity_anchor_three']"),
        ("'GGF75x12 (7 seeds)'","'GGF75x12 (5 seeds)'"),
        ("'SLiCS patch adaptation (7 seeds)'","'SLiCS patch adaptation (5 seeds)'"),
        ("'SLiCS75x12 joint (7 seeds)'","'SLiCS75x12 joint (5 seeds)'"),
        ("['matched_ggf','slics_object_partial','current_ggf']","['matched_ggf_final','slics_final_partial','current_ggf']"),
        ("names=dict(matched_ggf=","names=dict(matched_ggf_final="),
        ("slics_object_partial='SLiCS patch adaptation (5 seeds)'","slics_final_partial='SLiCS patch adaptation (5 seeds)'"),
        ('Prospective supervised comparison | seven seeds | new holdout',
         'Completed-training comparison | five seeds | scope amended before test')])
    catalog={seed:[copy.deepcopy(r) for r in rows if r['seed']==seed] for seed in SEEDS}
    def frozen_rows(ctx,seed,choice):
        if seed not in catalog:raise ValueError('Seed outside the fixed five-seed scope')
        return copy.deepcopy(catalog[seed])
    study.SEEDS=SEEDS.copy();study.protocol=lambda:copy.deepcopy(protocol)
    study.final_rows=frozen_rows;study.summarize=summary;study.plot=plot
    if len(study.expected_pairs())!=23:raise ValueError('Five-seed freeze inventory mismatch')
