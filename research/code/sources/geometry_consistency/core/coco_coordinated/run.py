"""A bounded coordinated-edit diagnostic that never retrains parent baselines.

Public notebook API: config(device), prepare(parent_output, output, local,
source_sha, cfg, require_drive=True), run_study(context), evaluate(context),
export(context, source). Parent artifacts are read-only and checksum verified.
"""
import copy
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch

from coco_baselines import metrics, training
from coco_baselines.storage import identity, serial, sha_file
from coco_adaptation import core, inputs, run as old_run

EXPECTED_PARENT = '35196154aba794047bf8d1bbed231c35de7f76e2bdffb5a450ccae9abe3dfebd'
_NATIVE_AUDIT_ARRAYS = metrics.audit_arrays


def config(device='cuda'):
    return dict(version=1, study='coco_coordinated_edit_diagnostic', device=device,
        methods=['vanilla', 'grassmannian', 'group_lasso'], seed=1, rounds=1,
        updates=300, evaluate_every=75, local_save_every=20,
        proposal_pairs=2, proposal_strengths=[.5, 1.], proposal_steps=40,
        drive_save_every=75, proposal_group_pool=4, candidates_per_round=4,
        proposal_seed=62102, proposal_object_limit=512,
        proposal_collateral_object_limit=64,proposal_collateral_weight=1.,
        proposal_opt_lr=.04,proposal_min_pair_gain=0.,
        target_min_images=10, min_category_images=10, decision_min_images=3,
        bootstrap=100, preview_images=3, audit_image_batch=4,
        detection_patches=32,patch_sample_seed=61002,
        positive_fraction=.5,negative_fraction=.1,min_object_patch_mass=2.,
        min_background_patches=4,max_audit_groups=4,capture_threshold=.8,
        reuse_inspected_test=True, train_baselines=False, extract_activations=False,
        min_target_capture_gain=.005, min_target_ap_gain=-.005,
        transaction_seed=62101,global_reconstruction_tolerance=.02,
        accept_reconstruction_tolerance=.01,macro_tradeoff_tolerance=.005,
        category_preservation_tolerance=.03,score_ap_weight=.2,
        lr=3e-4,snr=.1,images_per_update=8,batch=2048,grad_clip=5.,
        lambda_capture=.05,lambda_detection=.01,
        lambda_preservation=.02,lambda_pair_context=.02,
        anchor_energy_floor=1e-6,protect_capture_threshold=.5,protect_ap_threshold=.8,
        preserve_groups_per_update=4,preserve_patch_limit=128,diagnostic_grad_every=50)


class FrozenStore(inputs.ReadStore):
    """Parent write methods fail even if a caller accidentally invokes one."""
    def save(self, *args, **kwargs):
        raise PermissionError('Frozen parent checkpoints may only be read')
    def write(self, *args, **kwargs):
        raise PermissionError('Frozen parent reports may only be read')
    def event(self, *args, **kwargs):
        raise PermissionError('Frozen parent journals may only be read')
    def load(self, key):
        self._key(key)
        found = self._records(self.root, key)
        for _, _, path in found:
            try:
                return torch.load(path, map_location='cpu', weights_only=False)
            except (RuntimeError, EOFError, OSError):
                continue
        if list((self.root/key).glob('slot*.json')):
            raise OSError('No valid parent generation remains: '+key)
        return None


def required(store, key):
    value = store.load(key)
    if value is None:
        raise FileNotFoundError('Missing frozen source '+key+'. No retraining or extraction fallback is permitted.')
    return value


def streaming_array_sha(array, rows=32):
    """Same byte identity as historical array_sha without a whole-array bytes copy."""
    a = np.asarray(array)
    h = hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode())
    for lo in range(0, len(a), rows):
        h.update(np.ascontiguousarray(a[lo:lo+rows]).tobytes())
    return h.hexdigest()


@torch.no_grad()
def streaming_measure(model, x, device, batch=2048):
    """Same global held-out mean VE; float64 accumulation in bounded batches."""
    model.eval()
    mean = np.zeros(x.shape[1], np.float64)
    for lo in range(0, len(x), batch):
        mean += np.asarray(x[lo:lo+batch], np.float64).sum(0)
    mean /= len(x)
    total = sse = active = 0.
    count = 0
    alive = np.zeros(model.n_groups, bool)
    for lo in range(0, len(x), batch):
        chunk = np.asarray(x[lo:lo+batch], np.float64)
        total += float(np.square(chunk-mean).sum())
        xb = torch.as_tensor(np.array(x[lo:lo+batch], copy=True), dtype=torch.float32, device=device)
        pred, z = model(xb)
        sse += float((pred-xb).double().square().sum())
        on = z.norm(dim=-1)>1e-6
        active += float(on.sum()); alive |= on.any(0).cpu().numpy(); count += len(xb)
    return dict(mse=sse/(count*x.shape[1]), ve=1-sse/max(total,1e-12),
        active_groups=active/count, dead_groups=int((~alive).sum()), patches=count)


def memory_safe_metrics():
    metrics.array_sha = streaming_array_sha
    training.measure = streaming_measure
    metrics.audit_arrays = safe_audit_arrays


class _ReadOnlyAuditView:
    """Keep the backing mmap intact while copying only each requested batch."""
    def __init__(self,array):
        self.array=array
        self.shape=array.shape
        self.dtype=array.dtype
    def __len__(self):return len(self.array)
    def __getitem__(self,key):return np.array(self.array[key],copy=True)
    def __array__(self,dtype=None,copy=None):
        value=np.asarray(self.array,dtype=dtype)
        return value.copy() if copy else value


def safe_audit_arrays(model,x,masks,ids,cfg,device):
    # PyTorch warns even for read-only use of an immutable mmap. A tiny batch
    # copy makes the tensor writable without ever copying the whole split.
    array=x if np.asarray(x).flags.writeable else _ReadOnlyAuditView(x)
    return _NATIVE_AUDIT_ARRAYS(model,array,masks,ids,cfg,device)


def normalized_pack(source, split, provenance, norm, local, chunk=20):
    """Stream saved half-precision shards into a verified, read-only float32 mmap."""
    expected = list(provenance['splits'][split])
    signature = identity(dict(run=source.run_id, split=split, ids=expected,
        mean=streaming_array_sha(norm['mean']), scale=float(norm['scale']),
        normalization_train_sha=norm['train_sha']))
    folder = Path(local)/'normalized'; folder.mkdir(parents=True, exist_ok=True)
    path = folder/(signature+'.npy'); meta = path.with_suffix('.json')
    valid = False
    if path.exists() and meta.exists():
        record = json.loads(meta.read_text())
        valid = record.get('identity') == signature and record.get('sha256') == sha_file(path)
    mmap = None
    masks = []; ids = []; source_records = []
    raw_h = hashlib.sha256(str((len(expected),196,768)).encode()+str(np.dtype('float32')).encode())
    if not valid:
        staging = path.with_suffix('.pending.npy')
        mmap = np.lib.format.open_memmap(staging, mode='w+', dtype=np.float32,
                                       shape=(len(expected),196,768))
    for lo in range(0, len(expected), chunk):
        key = f'features_{split}_{lo:06d}'
        shard = required(source, key)
        want = expected[lo:lo+chunk]
        if list(shard['ids']) != want or len(shard['masks']) != len(want):
            raise ValueError('Saved IDs/masks/features disagree at '+key)
        raw = shard['x'].detach().cpu().float().numpy() if torch.is_tensor(shard['x']) else np.asarray(shard['x'],np.float32)
        if raw.shape != (len(want),196,768) or not np.isfinite(raw).all():
            raise ValueError('Unexpected activation shard shape or values: '+key)
        raw_h.update(np.ascontiguousarray(raw).tobytes())
        if mmap is not None:
            mmap[lo:lo+len(want)] = (raw-norm['mean'])*norm['scale']
        ids.extend(want); masks.extend(shard['masks'])
        generation = source._records(source.root,key)[0]
        source_records.append(dict(key=key,meta_sha=sha_file(generation[1]),payload_sha=sha_file(generation[2])))
        del shard, raw
    if split == 'train' and raw_h.hexdigest() != norm['train_sha']:
        raise ValueError('Saved training activation bytes disagree with saved normalization')
    if ids != expected:
        raise ValueError('Saved split order changed')
    if mmap is not None:
        mmap.flush(); del mmap
        os.replace(staging,path)
        meta.write_text(json.dumps(dict(identity=signature,sha256=sha_file(path),
            raw_float32_sha=raw_h.hexdigest(),source_records=source_records),indent=2))
    else:
        if json.loads(meta.read_text()).get('source_records') != source_records:
            raise ValueError('Cached normalized data source generations changed')
    return dict(x=np.load(path,mmap_mode='r'),ids=ids,masks=masks,
                normalized_path=str(path),identity=signature,raw_float32_sha=raw_h.hexdigest())


def source_snapshot(stores):
    records=[]
    for s in stores:
        records.append(dict(path=str(s.results/'manifest.json'),sha256=sha_file(s.results/'manifest.json')))
        for name in ('all_models_frozen.json','models_frozen.json','summary.json'):
            p=s.results/name
            if p.exists(): records.append(dict(path=str(p),sha256=sha_file(p)))
    return records


def assert_immutable(records):
    for row in records:
        if sha_file(row['path']) != row['sha256']:
            raise RuntimeError('Frozen parent file changed: '+row['path'])


def _discover(output, predicate, local):
    matched=[]
    for manifest in (Path(output)/'results').glob('*/manifest.json'):
        record=json.loads(manifest.read_text())
        if predicate(record['config']):
            s=FrozenStore(Path(output)/'checkpoints'/manifest.parent.name,local)
            matched.append(s)
    if len(matched)!=1:
        raise ValueError(f'Expected exactly one matching frozen source in {output}, found {len(matched)}')
    return matched[0]


def prepare(parent_output, output, local, source_sha, cfg, require_drive=True):
    if cfg.get('train_baselines') or cfg.get('extract_activations'):
        raise ValueError('This diagnostic does not train baselines or extract activations')
    if cfg.get('rounds')!=1:
        raise ValueError('One bounded round is predeclared; use a new protocol before expanding search')
    parent_output=Path(parent_output);local=Path(local)
    if Path(output).resolve()==parent_output.resolve() or parent_output.resolve() in Path(output).resolve().parents:
        raise ValueError('New diagnostic output must be separate from the frozen parent output')
    parent=FrozenStore(parent_output/'checkpoints'/EXPECTED_PARENT[:20],local/'read_parent')
    if parent.run_id != EXPECTED_PARENT:
        raise ValueError('Wrong stratified parent run')
    provenance=json.loads((parent.results/'provenance.json').read_text())
    norm=required(parent,'normalization')
    if np.asarray(norm['mean']).shape!=(196,768) or not np.isfinite(norm['mean']).all() or not np.isfinite(norm['scale']) or norm['scale']<=0:
        raise ValueError('Saved training normalization is invalid')
    plan=json.loads((parent.results/'coverage_plan.json').read_text())
    if identity({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']:
        raise ValueError('Saved coverage plan checksum identity is invalid')
    if provenance['splits']!=plan['splits'] or parent.config['data_plan']!=plan['fingerprint']:
        raise ValueError('Frozen stratification plan/provenance identity disagrees')
    cohort=plan['coverage']['category_indices']
    cfg=copy.deepcopy(cfg);cfg['category_indices']=cohort
    versions={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','scikit-learn','einops')}
    methods={};sources=[parent]
    for method in cfg['methods']:
        native=_discover(parent_output/f'native_{method}_s1',
            lambda c:c['data_plan']==parent.config['data_plan'] and c['config']['methods']==[method] and c['config']['seeds']==[1]
                and c['source_sha']==parent.config['source_sha'] and c['numerical_versions']==parent.config['numerical_versions'],local/'read_native')
        frozen=json.loads((native.results/'models_frozen.json').read_text())
        if not frozen['complete'] or len(frozen['choices'])!=1:
            raise ValueError('Native models did not freeze')
        choice=frozen['choices'][0]
        if not choice['sparsity_matched']:raise ValueError('Unmatched native checkpoint cannot be substituted')
        state=required(native,choice['key'])
        if training.weight_sha(state['best']) != choice['weights_sha']:
            raise ValueError('Frozen native weight SHA mismatch')
        del state
        adaptation=_discover(parent_output/f'adaptation_{method}_s1',
            lambda c:c['parent_run_id']==native.run_id and c['config']['methods']==[method] and c['config']['category_indices']==cohort
                and c['source_sha']==parent.config['source_sha'] and c['numerical_versions']==parent.config['numerical_versions'],local/'read_adaptation')
        catalog=json.loads((adaptation.results/'all_models_frozen.json').read_text())
        arms=catalog['models'];start=next(a for a in arms if a['arm']=='supervised_fixed_equal_search_compute')
        state=required(adaptation,start['key'])
        if training.weight_sha(state['best']) != start['weights_sha']:
            raise ValueError('Strong fixed start hash differs from frozen catalog')
        del state
        native_arm=next(a for a in arms if a['arm']=='native')
        if native_arm['weights_sha'] != choice['weights_sha']:
            raise ValueError('Native choice differs from adaptation frozen catalog')
        methods[method]=dict(native=native,adaptation=adaptation,choice=choice,
            start=start,base=native.config['config'])
        sources.extend([native,adaptation])
    stamps=source_snapshot(sources)
    for m in methods.values():
        for source,key in [(m['native'],m['choice']['key']),(m['adaptation'],m['start']['key'])]:
            record=source._records(source.root,key)[0]
            stamps.extend(dict(path=str(path),sha256=sha_file(path)) for path in record[1:])
    identity_config=dict(config=cfg,parent_run_id=parent.run_id,source_sha=source_sha,
        source_models={k:dict(native=v['choice'],start=v['start'],native_run=v['native'].run_id,
                             adaptation_run=v['adaptation'].run_id) for k,v in methods.items()},
        normalization=dict(train_sha=norm['train_sha'],mean_sha=streaming_array_sha(norm['mean']),scale=float(norm['scale'])),
        numerical_versions=versions)
    store=inputs.JournalStore(output,identity_config,local/'checkpoints',require_drive)
    memory_safe_metrics()
    packs={}
    for split in ('train','select','confirm','mapping'):
        packs[split]=normalized_pack(parent,split,provenance,norm,local, parent.config['config']['extract_chunk'])
        store.event('verified_saved_activations',split=split,images=len(packs[split]['ids']),mmap=packs[split]['normalized_path'])
    sets=[set(p['ids']) for p in packs.values()]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError('Frozen source split overlap')
    ctx=dict(store=store,parent=parent,methods=methods,packs=packs,provenance=provenance,
        normalization=norm,local=local,cfg=cfg,source_stamps=stamps,source_sha=source_sha)
    store.write('source_preflight.json',dict(passed=True,parent_run_id=parent.run_id,
        cohort=cohort,categories=len(cohort),sources=identity_config['source_models'],
        normalized_packs={k:{f:v[f] for f in ('identity','raw_float32_sha','normalized_path')} for k,v in packs.items()},
        parent_files=stamps,baselines_retrained=False,backbone_loaded=False,
        test_status='Previously inspected parent 800-image holdout: diagnostic only; no new independent confirmation.',
        memory='Saved shards normalized one at a time into read-only local memory maps. Global-mean VE and byte identities streamed.',
        environment=dict(current=versions,parent=parent.config['numerical_versions'],same=versions=={k:parent.config['numerical_versions'].get(k) for k in versions},
                         parent_missing_versions=[k for k in versions if k not in parent.config['numerical_versions']])))
    for method,m in methods.items():
        for name in ('summary.json','all_models_frozen.json'):
            store.write(f'frozen_parent/{method}/{name}',json.loads((m['adaptation'].results/name).read_text()))
    return ctx


def _weights(context, method):
    m=context['methods'][method]
    state=required(m['adaptation'],m['start']['key'])
    w=state['best']
    if training.weight_sha(w)!=m['start']['weights_sha']:
        raise ValueError('Frozen strong start changed during study')
    return w


def _audit(context, method, weights, key):
    store=context['store'];old=store.load(key)
    sha=training.weight_sha(weights)
    if old is not None:
        if old['weights_sha']!=sha:raise ValueError('Stale training audit')
        return old['audit']
    cfg=context['cfg'];base=context['methods'][method]['base']
    model=core.new_model(method,base,weights,cfg['device']);p=context['packs']['train']
    audit=metrics.audit_arrays(model,p['x'],p['masks'],p['ids'],cfg,cfg['device'])
    del model
    store.save(key,dict(weights_sha=sha,audit=audit))
    return audit


def run_study(context):
    from . import engine
    s=context['store'];cfg=context['cfg'];packs=context['packs'];completed=s.load('all_models_frozen')
    if completed is not None:
        context['study']=completed
        return completed
    results={}
    try:
        for mi,method in enumerate(cfg['methods']):
            start=_weights(context,method);base=context['methods'][method]['base']
            audit=_audit(context,method,start,method+'_strong_training_audit')
            targets=core.choose_targets(audit,cfg,context['provenance']['categories'])
            if {t['category_index'] for t in targets}!=set(cfg['category_indices']):
                raise ValueError('Strong start training target coverage differs from frozen 75-category cohort')
            plan=s.load(method+'_proposal_plan')
            if plan is None:
                began=time.monotonic();proposals=engine.propose(audit,targets,cfg)
                plan=dict(proposals=proposals,seconds=time.monotonic()-began,targets=targets,
                    training_weights_sha=training.weight_sha(start),selection_or_test_used=False)
                s.save(method+'_proposal_plan',plan)
            if plan['training_weights_sha']!=training.weight_sha(start) or plan['targets']!=targets:
                raise ValueError('Proposal source changed')
            s.write('proposals/'+method+'.json',plan)
            candidates=[];controls={};fit_updates=0;fit_seconds=0.
            original_select=engine.score(start,method,base,packs['select'],targets,cfg)
            original_confirm=engine.score(start,method,base,packs['confirm'],targets,cfg)
            for j,proposal in enumerate(plan['proposals']):
                transaction=engine.targets_for_proposal(targets,proposal,audit)
                control_context={k:proposal[k] for k in ('category_index','pair','receiver','donor')}
                transaction_cfg=dict(cfg,transaction_category=proposal['category_index'],
                    control_context_proposal=control_context)
                target_id=identity(dict(targets=transaction,context=control_context));control_key=method+'_control_'+target_id[:12]
                seed=cfg['transaction_seed']+mi*100
                if target_id not in controls:
                    st=engine.fit(s,control_key,method,base,start,packs['train'],packs['select'],transaction,transaction_cfg,seed,proposal=None)
                    controls[target_id]=dict(key=control_key,state=st,targets=transaction)
                    fit_updates+=st['updates'];fit_seconds+=st.get('seconds',0.)
                control=controls[target_id]
                key=f'{method}_candidate{j}'
                s.event('coordinated_transaction',method=method,key=key,category=proposal['category'],receiver=proposal['receiver'],pair=proposal['pair'],target_mapping_sha=target_id)
                st=engine.fit(s,key,method,base,start,packs['train'],packs['select'],transaction,transaction_cfg,seed,proposal=proposal)
                fit_updates+=st['updates'];fit_seconds+=st.get('seconds',0.)
                a=engine.score(st['best'],method,base,packs['select'],transaction,cfg)
                b=engine.score(control['state']['best'],method,base,packs['select'],transaction,cfg)
                initial=engine.score(start,method,base,packs['select'],transaction,cfg)
                decision=engine.compare(a,b,initial,cfg,base,method,proposal['category_index'])
                if not st.get('best_valid',True):
                    decision['accepted']=False
                    decision.setdefault('reasons',[]).append('no_eligible_candidate_checkpoint')
                if not control['state'].get('best_valid',True):
                    decision['accepted']=False
                    decision.setdefault('reasons',[]).append('no_eligible_matched_control_checkpoint')
                row=dict(key=key,control_key=control_key,proposal=proposal,targets=transaction,
                    selection=decision,candidate=a,control=b,initial=initial,
                    candidate_best_update=st['best_update'],candidate_updates=st['updates'],weights_sha=training.weight_sha(st['best']))
                s.write('decisions/'+key+'.json',row);candidates.append(row)
                del st,a,b,initial
            eligible=[r for r in candidates if r['selection']['accepted']]
            # Candidate ranking never reads confirmation or test. One nomination only.
            nominee=max(eligible,key=lambda r:(r['selection'].get('target_capture_gain',-np.inf),r['selection'].get('target_ap_gain',-np.inf),r['key'])) if eligible else None
            confirmation=None;chosen_key=None;accepted=False;fallback_preservation=None
            if nominee is not None:
                t=nominee['targets'];st=required(s,nominee['key']);ctrl=required(s,nominee['control_key'])
                a=engine.score(st['best'],method,base,packs['confirm'],t,cfg)
                b=engine.score(ctrl['best'],method,base,packs['confirm'],t,cfg)
                ini=engine.score(start,method,base,packs['confirm'],t,cfg)
                confirmation=engine.compare(a,b,ini,cfg,base,method,nominee['proposal']['category_index'])
                accepted=bool(confirmation['accepted']);chosen_key=nominee['key'] if accepted else nominee['control_key']
                chosen_targets=t
            elif candidates:
                chosen_key=candidates[0]['control_key'];chosen_targets=candidates[0]['targets']
            else:
                chosen_targets=targets
            # Fallback continued model also must preserve the original strong start.
            if chosen_key is not None and not accepted:
                fallback=required(s,chosen_key)
                scores=engine.score(fallback['best'],method,base,packs['confirm'],targets,cfg)
                fc=engine.preservation_only(scores,original_confirm,cfg,base,method)
                fallback_preservation=dict(fc,control_key=chosen_key,best_valid=fallback.get('best_valid',True))
                if not fc['accepted'] or not fallback.get('best_valid',True):
                    chosen_key=None;chosen_targets=targets
            selected=start if chosen_key is None else required(s,chosen_key)['best']
            selected_key=method+'_selected';s.save(selected_key,dict(best=selected,targets=chosen_targets,updates=0 if chosen_key is None else cfg['updates']))
            # Equal model-update control includes every candidate and distinct transaction control.
            full_cfg=dict(cfg,updates=fit_updates)
            if chosen_key is None and not accepted:
                # A fully rejected continued path returns the original targets
                # and objective. Do not retain a rejected transaction's target
                # exemption/context anchors in the all-update fixed control.
                full_cfg.pop('transaction_category',None)
                full_cfg.pop('control_context_proposal',None)
            elif nominee is not None:
                full_cfg.update(transaction_category=nominee['proposal']['category_index'],
                    control_context_proposal={k:nominee['proposal'][k] for k in ('category_index','pair','receiver','donor')})
            elif candidates:
                full_cfg.update(transaction_category=candidates[0]['proposal']['category_index'],
                    control_context_proposal={k:candidates[0]['proposal'][k] for k in ('category_index','pair','receiver','donor')})
            full_key=method+'_fixed_total_updates'
            if fit_updates:
                full=engine.fit(s,full_key,method,base,start,packs['train'],packs['select'],chosen_targets,full_cfg,cfg['transaction_seed']+mi*100,proposal=None)
                full_seconds=full.get('seconds',0.)
            else:
                full=dict(best=start,updates=0);s.save(full_key,full);full_seconds=0.
            comparison_category=nominee['proposal']['category_index'] if nominee is not None else (candidates[0]['proposal']['category_index'] if candidates else targets[0]['category_index'])
            whole_budget_decisions={}
            for split in ('select','confirm'):
                adapted=engine.score(selected,method,base,packs[split],chosen_targets,cfg)
                whole=engine.score(full['best'],method,base,packs[split],chosen_targets,cfg)
                common_initial=engine.score(start,method,base,packs[split],chosen_targets,cfg)
                dec=engine.compare(adapted,whole,common_initial,cfg,base,method,comparison_category)
                whole_budget_decisions[split]=dict(dec,accepted_structural_edit=accepted,
                    whole_control_preservation=engine.preservation_only(whole,common_initial,cfg,base,method),
                    structural_advantage_over_equal_total_updates=bool(accepted and dec['accepted']))
            s.save(method+'_frozen_start',dict(best=start,updates=0,targets=targets))
            result=dict(method=method,seed=1,accepted_edits=int(accepted),selected_key=selected_key,
                selected_source_key=chosen_key,selected_weights_sha=training.weight_sha(selected),
                targets=chosen_targets,initial_targets=targets,nominee=None if nominee is None else nominee['key'],confirmation=confirmation,
                fallback_preservation=fallback_preservation,
                equal_total_update_comparisons=whole_budget_decisions,
                candidates=[{k:r[k] for k in ('key','control_key','proposal','selection','candidate_best_update','weights_sha')} for r in candidates],
                arms=[dict(arm='frozen_strong_start',key=method+'_frozen_start'),dict(arm='coordinated_adaptive',key=selected_key),dict(arm='fixed_equal_all_search_updates',key=full_key)],
                accounting=dict(candidate_and_transaction_control_updates=fit_updates,
                    full_fixed_control_updates=full['updates'],proposal_optimization_steps=sum({
                        (p['category_index'],tuple(p['pair'])):p.get('training_surrogate',{}).get('total_rotation_gradient_updates',0)
                        for p in plan['proposals']}.values()),
                    proposal_seconds=plan['seconds'],search_fit_seconds=fit_seconds,full_control_fit_seconds=full_seconds,
                    caveat='Equal model-update budget; rotation optimization uses different smaller parameter/work units and is reported separately. Wall-clock costs are not claimed equal.'),
                original_selection=original_select,original_confirmation=original_confirm)
            s.save(method+'_study',result);s.write('study/'+method+'.json',result);results[method]=result
            assert_immutable(context['source_stamps'])
            s.event('variant_frozen',method=method,accepted_edits=int(accepted),candidates=len(candidates),search_updates=fit_updates)
        assert_immutable(context['source_stamps'])
        out=dict(methods=results,complete=True,test_used_for_training_or_selection=False,
                 scope='One diagnostic training seed, fixed width/rank, owner-aligned supervised edits. No native baseline retraining.')
        s.save('all_models_frozen',out);s.write('all_models_frozen.json',out);context['study']=out
        return out
    except BaseException as exc:
        s.write('failure.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),baseline_retraining_permitted=False))
        raise


def evaluate(context):
    from . import engine
    s=context['store'];cfg=context['cfg'];study=required(s,'all_models_frozen')
    if not study['complete']:raise RuntimeError('Freeze all edits before test evaluation')
    old=s.load('evaluation_complete')
    if old is not None:return old
    if not cfg['reuse_inspected_test']:
        return dict(status='weights_frozen',test='No new extraction requested; independent test must be declared separately')
    test=normalized_pack(context['parent'],'test',context['provenance'],context['normalization'],context['local'],context['parent'].config['config']['extract_chunk'])
    if any(set(test['ids']) & set(p['ids']) for p in context['packs'].values()):raise ValueError('Diagnostic test overlaps a fitting/mapping split')
    reports=[];comparisons=[]
    for method,branch in study['methods'].items():
        base=context['methods'][method]['base']
        arm_catalog=list(branch['arms'])
        control_keys=list(dict.fromkeys(r['control_key'] for r in branch['candidates']))
        arm_catalog.extend(dict(arm='transaction_fixed_control_'+str(i),key=k) for i,k in enumerate(control_keys))
        frozen_audits={}
        for arm in arm_catalog:
            state=required(s,arm['key']);weights=state['best']
            cached=s.load('evaluation_'+arm['key'])
            needs_bootstrap=arm['arm'] in ('coordinated_adaptive','fixed_equal_all_search_updates')
            if cached is not None:
                if cached['report']['summary']['weights_sha']!=training.weight_sha(weights):
                    raise ValueError('Stale final evaluation checkpoint')
                reports.append(cached['report'])
                if needs_bootstrap:frozen_audits[arm['arm']]=(cached['audit'],cached['calibration'])
                s.event('reused_diagnostic_model_evaluation',method=method,arm=arm['arm'])
                continue
            model=core.new_model(method,base,weights,cfg['device']);mp=context['packs']['mapping']
            mapping=metrics.audit_arrays(model,mp['x'],mp['masks'],mp['ids'],cfg,cfg['device'])
            calibration=metrics.calibrate(mapping,cfg,context['provenance']['categories'])
            calibration['label_use']='Final mapping annotations choose evaluation groups only; supervised training/proposals separately use training masks.'
            audit=metrics.audit_arrays(model,test['x'],test['masks'],test['ids'],cfg,cfg['device'])
            rows=metrics.evaluate(audit,calibration,cfg,62190)
            cap=[r['contrast']['single_group_capture'] for r in rows if r['category_index'] in cfg['category_indices'] and r['contrast'] is not None]
            ap=[r['detection']['ap'] for r in rows if r['category_index'] in cfg['category_indices'] and r['detection'] is not None]
            if len(cap)!=len(cfg['category_indices']) or len(ap)!=len(cfg['category_indices']):
                raise ValueError('Final supported-category denominator changed; retain missing cases rather than reporting a favorable subset')
            summary=dict(method=method,seed=cfg['seed'],arm=arm['arm'],key=arm['key'],weights_sha=training.weight_sha(weights),
                mean_single_group_contrast_capture=float(np.mean(cap)),mean_single_group_ap=float(np.mean(ap)),
                reconstruction=training.measure(model,test['x'].reshape(-1,model.d),cfg['device'],cfg['batch']),
                categories=len(cap),images=len(test['ids']),accepted_edits=branch['accepted_edits'] if arm['arm']=='coordinated_adaptive' else 0,
                test_status='Reused previously inspected 800-image holdout; development diagnostic, not independent confirmation')
            report=dict(summary=summary,categories=rows)
            report['frozen_training_receiver_scores']=core.category_scores(audit,branch['targets'],cfg)
            report['original_training_mapping_scores']=core.category_scores(audit,branch['initial_targets'],cfg)
            s.write('reports/'+arm['key']+'.json',report)
            s.write('mappings/'+arm['key']+'.json',dict(mappings=calibration['mappings'],weights_sha=summary['weights_sha'],label_use=calibration['label_use']))
            # Only the two paired comparison audits retain arrays; other model
            # resume records are small reports. Raw activations stay in parent.
            s.save('evaluation_'+arm['key'],dict(report=report,calibration=calibration,
                audit=audit if needs_bootstrap else None))
            reports.append(report)
            if needs_bootstrap:frozen_audits[arm['arm']]=(audit,calibration)
            del model,mapping
            s.event('diagnostic_model_evaluated',method=method,arm=arm['arm'],capture=summary['mean_single_group_contrast_capture'],ap=summary['mean_single_group_ap'])
        a,fa=frozen_audits['coordinated_adaptive'];b,fb=frozen_audits['fixed_equal_all_search_updates']
        paired=old_run.paired_intervals(a,b,fa,fb,dict(cfg,fresh_test_seed=62190))
        paired.update(method=method,accepted_edits=branch['accepted_edits'],
            note='Reused diagnostic holdout; image bootstrap excludes selection/multiple-comparison/seed uncertainty.')
        comparisons.append(paired)
        del frozen_audits
        assert_immutable(context['source_stamps'])
    old_run.export_categories(s,reports)
    assert_immutable(context['source_stamps'])
    out=dict(models=[r['summary'] for r in reports],comparisons=comparisons,
        accepted_edits=sum(v['accepted_edits'] for v in study['methods'].values()),
        baseline_training_updates=0,backbone_extractions=0,
        independent_confirmation=False,test_status='Previously inspected parent holdout',
        scope='Annotation-grounded object contrast and patch detection proxies; neither full concept-manifold recovery nor an exact feature-splitting rate.')
    native_rows=[]
    for method,m in context['methods'].items():
        prior=json.loads((m['adaptation'].results/'summary.json').read_text())
        row=next(r for r in prior['reports'] if r['arm']=='native')
        native_rows.append(dict(row,evidence_status='Frozen parent report; not retrained or reevaluated in this diagnostic'))
    out['frozen_native_baseline_reports']=native_rows
    plot(context,out)
    s.write('summary.json',out);s.save('evaluation_complete',out)
    return out


def plot(context,report):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    cfg=context['cfg'];methods=cfg['methods'];fig,axes=plt.subplots(len(methods),3,figsize=(15,4*len(methods)),squeeze=False)
    labels=['Frozen native BSF','Strong fixed start','Matched fixed path','Coordinated AGF','Fixed: all search updates']
    colors=['#aab8c5','#80a99f','#b8bbbf','#008d84','#bb9ccc']
    for i,method in enumerate(methods):
        branch=context['store'].load('all_models_frozen')['methods'][method]
        rows={r['arm']:r for r in report['models'] if r['method']==method}
        native=next(r for r in report['frozen_native_baseline_reports'] if r['method']==method)
        nominee=next((r for r in branch['candidates'] if r['key']==branch['nominee']),None)
        path_key=nominee['control_key'] if nominee is not None else (branch['candidates'][0]['control_key'] if branch['candidates'] else None)
        path=next((r for r in report['models'] if r['method']==method and r['key']==path_key),rows['frozen_strong_start'])
        selected=[native,rows['frozen_strong_start'],path,rows['coordinated_adaptive'],rows['fixed_equal_all_search_updates']]
        for ax,field,title in zip(axes[i],['mean_single_group_contrast_capture','mean_single_group_ap','ve'],['Single-group object contrast capture','Object-patch detection AP','Reconstruction VE']):
            values=[100*(r['reconstruction']['ve'] if field=='ve' else r[field]) for r in selected]
            ax.bar(np.arange(5),values,color=colors)
            ax.set_xticks(np.arange(5),labels,rotation=25,ha='right',fontsize=8)
            ax.set_title(method+': '+title);ax.set_ylabel('%');ax.grid(axis='y',alpha=.2)
            for j,v in enumerate(values):ax.text(j,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('Reused COCO holdout: exploratory coordinated-edit diagnostic',fontsize=13)
    fig.tight_layout();folder=context['store'].results/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'coordinated_comparison.png',dpi=170);fig.savefig(folder/'coordinated_comparison.pdf');plt.close(fig)


def export(context, source):
    s=context['store'];assert_immutable(context['source_stamps'])
    path=s.output/f'agf_coco_coordinated_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for prefix,root in [('results',s.results),('source',Path(source))]:
            for p in root.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pt','.pyc'):
                    archive.write(p,prefix+'/'+p.relative_to(root).as_posix())
    s.event('export_complete',path=str(path),bytes=path.stat().st_size)
    return path
