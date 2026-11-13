"""Read-only prefix migration and resumable final joint reconstruction."""
import copy,hashlib,json,re,time
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity,sha_file
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched

LEGACY_RUN='d8a0a7c7bd2e6ffb880cee729c8fdfba2adea8ca3709f887e5c58e5ae77fb0fd'
RESERVED_PLAN_SHA='af34b44ced90edaae136b8108e287c8a4aabec3cadf45cb1f147f122212c07aa'
RESERVED_EXCLUSIONS_SHA='82d66e78cf2a506bcff0555a7239282afc853261e756e2f413e9f4255e9844d0'
HERE=Path(__file__).resolve().parent

def check_legacy(options):
    path=Path(options['output'])/'results'/LEGACY_RUN[:20]/'manifest.json'
    expected=json.loads((HERE/'legacy_manifest.json').read_text())
    if not path.exists():raise FileNotFoundError('Saved predecessor manifest is required; no silent restart: '+str(path))
    if json.loads(path.read_text())!=expected or identity(expected['config'])!=LEGACY_RUN:
        raise ValueError('Saved predecessor identity differs from the pinned training run')
    return memory.FrozenStore(Path(options['output'])/'checkpoints'/LEGACY_RUN[:20],Path(options['local'])/'read_previous')

def object_report(report):
    """Validation facts reused from the predecessor never influence an MSE selector."""
    report=copy.deepcopy(report)
    report['reconstruction']=dict(status='deferred',mse=None,ve=None,active_groups=None,dead_groups=None,
        note='Not used in object selection; final joint reconstruction remains required')
    return report

def assert_unopened(source):
    """Fail closed on any evidence that this journal could have opened test data."""
    for key in ['all_models_frozen','supervised_test_complete','audit_signature_index']:
        if source.load(key) is not None:raise ValueError('Reserved holdout is not provably unopened: '+key)
    for name in ['all_models_frozen.json','supervised_summary.json','supervised_models.csv']:
        if (source.results/name).exists():raise ValueError('Reserved holdout has a freeze/test marker: '+name)
    if any((source.results/'reports').glob('audit_*')):raise ValueError('Reserved holdout has saved test reports')
    if any(p.name.startswith(('features_test_','audit_','final_joint_','final_objects_','image_ci_')) for p in source.root.iterdir()):
        raise ValueError('Reserved holdout has test feature or inference artifacts')
    status=source.results/'completion_status.json'
    if status.exists():
        record=json.loads(status.read_text())
        if any(record.get(k,False) for k in ['complete','all_models_frozen','test_complete']):
            raise ValueError('Reserved holdout has completed freeze/test status')
    events=source.results/'events.jsonl'
    if events.exists():
        for line in events.read_text().splitlines():
            event=json.loads(line)
            if event.get('stage')=='supervised_test_model_complete' or (event.get('stage')=='extraction' and event.get('split')=='test'):
                raise ValueError('Reserved holdout was opened according to its event journal')

def reserve_predecessor_holdout(ctx,coco=None):
    """Transfer the exact annotation-only reservation; do not redraw or score it."""
    store=ctx['store'];previous=store.previous
    if previous.run_id!=LEGACY_RUN:raise ValueError('Unexpected holdout reservation source')
    assert_unopened(previous)
    plan=previous.load('fresh_holdout_plan')
    if plan is None or identity(plan)!=RESERVED_PLAN_SHA:raise ValueError('Pinned reserved holdout plan changed or is missing')
    exclusions_path=previous.results/'other_study_holdout_excluded.json'
    exclusions=json.loads(exclusions_path.read_text())
    if identity(exclusions)!=RESERVED_EXCLUSIONS_SHA:raise ValueError('Original other-study exclusions changed')
    ids=plan['splits']['test'];categories=ctx['cfg']['category_indices']
    if len(ids)!=ctx['rule']['final_holdout_images'] or len(set(ids))!=len(ids):raise ValueError('Reserved holdout size changed')
    if plan['category_indices']!=categories or len(categories)!=75:raise ValueError('Retain the exact 75-category task')
    if any(plan['coverage']['split_counts']['test'][c]<10 for c in categories):raise ValueError('Reserved holdout does not meet every category quota')
    used=set(i for values in ctx['provenance']['splits'].values() for i in values)
    used.update(exclusions['ids']);used.update(exclusions.get('other_study_ids',[]))
    used.update(plan['excluded_prior_ids'])
    for source in ctx['source_stores']:
        old=source.load('fresh_holdout_plan')
        if old is not None:used.update(old['splits']['test'])
    # Strong-mask study's historical fresh cohorts also remain excluded.
    strong=ctx.get('sources',{}).get('strong')
    if strong is not None:
        old=memory.required(strong,'fresh_plan')
        used.update(i for values in old['splits'].values() for i in values)
    if set(ids)&used:raise ValueError('Reserved holdout overlaps a training/development/other-study cohort')
    # No sibling source revision may have already tested these reserved IDs.
    from coco_adaptation.inputs import ReadStore
    for file in store.results.parent.glob('*/fresh_holdout_plan.json'):
        if file.parent in [store.results,previous.results]:continue
        sibling_plan=json.loads(file.read_text())
        if not set(ids)&set(sibling_plan['splits']['test']):continue
        sibling=ReadStore(store.output/'checkpoints'/file.parent.name,ctx['local']/'read_sibling_reservations')
        assert_unopened(sibling)
    current=store.load('fresh_holdout_plan')
    if current is not None and identity(current)!=RESERVED_PLAN_SHA:raise ValueError('Current reserved holdout changed')
    if current is None:store._width_drive_recovery_original_save('fresh_holdout_plan',copy.deepcopy(plan),drive=True)
    store.write('fresh_holdout_plan.json',plan)
    store.write('other_study_holdout_excluded.json',exclusions)
    store.write('holdout_reservation_reuse.json',dict(source_run=LEGACY_RUN,plan_sha=RESERVED_PLAN_SHA,
        annotation_reservation_only=True,source_frozen=False,source_test_features_or_reports_found=False,
        all_75_category_quotas_retained=True,holdout_features_opened=False,
        reason='Changing an evaluation schedule does not consume an unopened annotation-only reservation.'))
    ctx['checkpoint_stamps'].append(matched.source_key_stamp(previous,'fresh_holdout_plan'))
    print('REUSED VERIFIED UNOPENED HOLDOUT: 1500 images; all 75 categories; annotation scan skipped; test features remain closed.',flush=True)
    return copy.deepcopy(plan)

class PrefixReuse:
    def __init__(self,current,previous,ctx):
        self.current=current;self.previous=previous;self.ctx=ctx
        cfg=current.config;old=previous.config
        for name in ['core_sha','numerical_versions','source_runs','normalization_sha','train_identity','development_identity','pilot_pins']:
            if cfg[name]!=old[name]:raise ValueError('Incompatible saved training inputs: '+name)
        if current.run_id==previous.run_id:raise ValueError('Evaluation revision must have a separate run identity')
    def __getattr__(self,name):return getattr(self.current,name)
    @staticmethod
    def reusable(key):
        return bool(key=='native_svd12' or
            re.fullmatch(r'native_slics12_s(?:21|22|23|24|25|26|27)(?:_object)?',key) or
            re.fullmatch(r'small75_s(?:21|22|23|24|25|26|27)_(?:warm|targets|lr(?:1000|2999|10000)_ggf(?:_object)?)',key))
    def load(self,key):
        value=self.current.load(key)
        if value is not None or not self.reusable(key):return value
        value=self.previous.load(key)
        if value is None:return None
        import supervised_study as study
        stamp=matched.source_key_stamp(self.previous,key)
        value=copy.deepcopy(value)
        if re.fullmatch(r'native_slics12_s\d+',key):
            seed=int(key.rsplit('s',1)[1]);D0=study.native_initializer(self.ctx)
            args=dict(source=training.weight_sha({'D':D0}),train=self.ctx['packs']['train']['identity'],
                select=self.ctx['packs']['select']['identity'],seed=seed,solver=study.native_solver_config(),total=study.TOTAL)
            expected=identity(dict(args,protocol=self.previous.config['protocol']))
            if value['fingerprint']!=expected:raise ValueError('Incompatible native SLiCS training fingerprint')
            value['fingerprint']=identity(dict(args,protocol=study.protocol()))
            if not 0<=value['updates']<=study.TOTAL:raise ValueError('Invalid imported native update count')
        if key.endswith('_object'):
            # Only the earliest highest-object-score checkpoint was retained in either protocol.
            parent_key=key[:-len('_object')];parent=self.load(parent_key)
            if parent is None:raise ValueError('Selected prefix has no training state')
            selected=parent.get('selected',{}).get('object')
            if selected is None or selected['updates']!=value['updates'] or selected['score']!=value['score']:
                raise ValueError('Imported object candidate differs from its training record')
            val=value['validation'];score=val['macro_capture']+.2*val['macro_ap']
            if not np.isfinite(score) or score!=value['score']:raise ValueError('Imported object score is invalid')
            value['validation']=object_report(val)
        if isinstance(value,dict) and 'selected' in value:
            value['selected']={k:v for k,v in value['selected'].items() if k=='object'}
            for item in value.get('history',[]):
                if 'validation' in item and 'macro_capture' in item['validation']:
                    item['validation']=object_report(item['validation'])
        if key=='native_svd12':
            if value['D'].shape!=(900,self.ctx['packs']['train']['x'].shape[-1]) or value['sha']!=training.weight_sha({'D':value['D']}):
                raise ValueError('Imported initializer shape/hash mismatch')
        self.current._width_drive_recovery_original_save(key,value,drive=True)
        self.ctx['checkpoint_stamps'].append(stamp)
        self.current.write('imported_prefixes/'+key+'.json',dict(previous_run=self.previous.run_id,key=key,
            previous_stamp=stamp,updates=value.get('updates') if isinstance(value,dict) else None,
            scientific_protocol_changed=True,training_arithmetic_changed=False))
        print('IMPORTED SAVED TRAINING PREFIX:',key,'updates',value.get('updates') if isinstance(value,dict) else None,flush=True)
        return value
    def save(self,*args,**kwargs):return self.current.save(*args,**kwargs)
    @property
    def _width_drive_recovery_original_save(self):return self.current._width_drive_recovery_original_save

@torch.no_grad()
def final_reconstruction(ctx,model,pack,weights_sha):
    """Exact unchanged joint solver; cache by frozen weights and full cohort."""
    from supervised_study import pack_variance
    key='final_joint_'+identity(dict(weights=weights_sha,ids=pack['ids'],cohort=pack.get('identity'),
        norm=identity(ctx.get('norm',{})),solver=model.cfg,arithmetic='image-streamed-float64-SSE-v1'))[:32]
    s=ctx['store'];n=len(pack['ids']);d=model.d
    st=s.load(key)
    if st is None:st=dict(completed_images=0,total_images=n,sse=0.,patches=0,active=0.,alive=np.zeros(model.n_groups,bool),seconds=0.)
    if st['total_images']!=n or not 0<=st['completed_images']<=n:raise ValueError('Final reconstruction cache mismatch')
    if st.get('complete'):return st['report']
    try:
        for i in range(st['completed_images'],n):
            x=torch.as_tensor(np.array(pack['x'][i],copy=True),device=ctx['device']);start=time.monotonic()
            # No partial-concept projections, pruning, new approximation or relaxed KKT.
            joint=model.solve(x).reshape(-1,model.n_groups,model.group_size)
            prediction=joint.flatten(1)@model.D
            sse=float((prediction-x).double().square().sum());active=joint.norm(dim=-1)>1e-6
            # Commit only after the entire image has passed the solver's verification.
            st.update(completed_images=i+1,sse=st['sse']+sse,patches=st['patches']+len(x),
                active=st['active']+float(active.sum()),alive=st['alive']|active.any(0).cpu().numpy(),
                seconds=st['seconds']+time.monotonic()-start)
            s.save(key,st,drive=(i+1)%4==0 or i+1==n)
            if (i+1)%4==0 or i==0 or i+1==n:
                s.event('final_joint_reconstruction',images_completed=i+1,images_total=n,seconds=st['seconds'],weights_sha=weights_sha)
            del x,joint,prediction,active
    except BaseException:
        s._width_drive_recovery_original_save(key,st,drive=True)
        raise
    if st['patches']!=196*n:raise ValueError('Final joint reconstruction missed patches')
    report=dict(status='computed',mse=st['sse']/(st['patches']*d),ve=1-st['sse']/pack_variance(pack),
        active_groups=st['active']/st['patches'],dead_groups=int((~st['alive']).sum()),patches=st['patches'],
        seconds=st['seconds'],inference='joint NNLS; never a sum of partial projections')
    if not np.isfinite([report['mse'],report['ve']]).all():raise ValueError('Nonfinite final reconstruction')
    st.update(complete=True,report=report);s._width_drive_recovery_original_save(key,st,drive=True)
    return report
