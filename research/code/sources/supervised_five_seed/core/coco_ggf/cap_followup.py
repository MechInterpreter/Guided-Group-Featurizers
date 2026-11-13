"""Saved-cap evaluation and paired higher-cap development experiments.

Test evaluation performs inference only. The separate sweep never loads test
features and restarts each cap from the same saved reconstruction warm-up.
"""
import copy
import gc
import json
import traceback
from pathlib import Path
import numpy as np
import torch
from coco_baselines import data, training
from coco_baselines.storage import identity, sha_file
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import weight_bound as bound, width_balance as width, matched_baselines as matched, study
from .width_balance_fit import fit, state_identity
from .drive_recovery import install_drive_recovery

BOUND_RUN = '842bd916b90c62d783787b26000dc57a844c1346b4f21ead8480101cd646040f'
SEEDS = [21, 22, 23]
CAPS = [.6, .8, 1.]
CONTROLS = ['mask4_balanced', 'vanilla4', 'mask3_fixed']


def safe_destination(output, paths):
    dest = Path(output).resolve()
    for path in paths:
        root = Path(path).resolve()
        if dest == root or dest in root.parents or root in dest.parents:
            raise ValueError('New output overlaps a read-only source: ' + str(root))


def cap_recipe(cap):
    if type(cap) not in (int, float) or float(cap) not in CAPS:
        raise ValueError('Supported new caps: 0.60, 0.80, 1.00')
    recipe = bound.settings()
    recipe.update(study='ggf_cap_sweep_development', upper_bound=float(cap))
    recipe['balance']['bounds']['capture'][1] = float(cap)
    recipe['balance']['bounds']['detection'][1] = float(cap)
    return recipe


def caps_checked(caps):
    caps=list(caps)
    if not caps or len(caps)!=len(set(caps)):
        raise ValueError('Choose distinct caps from 0.60, 0.80, 1.00')
    for cap in caps:cap_recipe(cap)
    return sorted(float(c) for c in caps)


def phase_identity(ctx, initial, targets, updates, rng_seed, ref, balance_recipe,
                   head=None, balance=None):
    return identity(dict(initial=training.weight_sha(initial), targets=targets,
        updates=updates, seed=rng_seed, balanced=True, supervised=True,
        reference_mse=ref, initial_head_sha=None if head is None else training.weight_sha({'head': head}),
        initial_balance=state_identity(balance), base=ctx['base'], cfg=ctx['cfg'],
        balance=balance_recipe, train_ids=ctx['provenance']['splits']['train'],
        select_ids=ctx['provenance']['splits']['select']))


def verify_cap040(ctx, seed):
    row, warm, old1, old2 = bound.verify_seed(ctx, seed)
    source = ctx['sources']['cap040']
    p1 = memory.required(source, f'phase1_cap040_s{seed}')
    p2 = memory.required(source, f'phase2_cap040_s{seed}')
    result = memory.required(source, f'result_s{seed}')
    recipe = bound.settings()['balance']
    for phase, initial, old, n, rng_seed, head, balance in [
        (p1, warm['best'], old1, 2500, 71000+seed, None, None),
        (p2, p1['weights'], old2, 500, 82000+seed, p1['head'], p1['balance'])]:
        matched.complete(phase, n)
        expected = phase_identity(ctx, initial, row['targets'], n, rng_seed,
                                  row['reference_mse'], recipe, head, balance)
        if phase['fingerprint'] != expected:
            raise ValueError('Saved 0.40 phase lineage changed')
        for key in ['sampled_batches_sha', 'sampled_noise_sha']:
            if phase[key] != old[key]:
                raise ValueError('Saved 0.40 paired schedule/noise changed: '+key)
        matched.valid_weights(phase['weights'], 'vanilla', ctx['base'])
        if phase['history'][-1]['update'] != n:
            raise ValueError('Missing final development endpoint')
        bound.validate_metrics(phase['history'][-1]['validation'])
    if (result['seed'] != seed or result['new_updates'] != 3000 or result['test_access'] or
        result['baseline_weights_sha'] != row['weights_sha'] or
        result['treatment_weights_sha'] != training.weight_sha(p2['weights']) or
        result['targets_sha'] != row['targets_sha'] or
        result['warm_weights_sha'] != training.weight_sha(warm['best']) or
        identity(result['baseline']) != identity(old2['history'][-1]['validation']) or
        identity(result['treatment']) != identity(p2['history'][-1]['validation'])):
        raise ValueError('Saved 0.40 result does not identify its final checkpoint')
    declared = next(r for r in ctx['completed_cap040']['results'] if r['seed']==seed)
    if identity(result) != identity(declared):
        raise ValueError('Saved 0.40 result and completed summary differ')
    if tuple(p2['head'].shape) != (75, 2) or not torch.isfinite(p2['head']).all():
        raise ValueError('Invalid saved 0.40 detection head')
    for key in [f'phase1_cap040_s{seed}', f'phase2_cap040_s{seed}', f'result_s{seed}']:
        ctx['checkpoint_stamps'].append(matched.source_key_stamp(source, key))
    frozen = dict(row, arm='cap040', label='Mask GGF 4: cap 0.40', origin='saved_cap040',
        key=f'phase2_cap040_s{seed}', weights_sha=training.weight_sha(p2['weights']),
        head_sha=training.weight_sha({'head': p2['head']}), development=result['treatment'])
    return frozen


def open_sources(paths, output, local, source_sha, mode, require_drive=True):
    safe_destination(output, paths.values())
    local = Path(local)
    parent = matched.discover(paths['parent'], local/'read_parent', run=bound.PARENT_RUN)
    cap = matched.discover(paths['cap040'], local/'read_cap040', run=BOUND_RUN)
    original = matched.discover(paths['original'], local/'read_original', run=parent.config['source_runs']['original'])
    pc, bc = parent.config, cap.config
    if (pc['recipe'] != width.settings() or bc['recipe'] != bound.settings() or
        bc['parent_run'] != parent.run_id or bc['activation_run'] != original.run_id or
        original.run_id != memory.EXPECTED_PARENT or bc['fit_config'] != pc['fit_config'] or
        bc['base'] != dict(pc['base'], block_rank=4, local_save_updates=50) or
        bc['versions'] != matched.versions() or pc['versions'] != matched.versions()):
        raise ValueError('Saved source protocol, capacity, lineage or numerical versions differ')
    completed = memory.required(cap, 'requested_screen_complete')
    summary = json.loads((cap.results/'summary.json').read_text())
    if identity(completed) != identity(summary):
        raise ValueError('Saved completion checkpoint and summary differ')
    if not completed['complete'] or not completed['study_complete'] or completed['requested_seeds'] != SEEDS or completed['test_access']:
        raise ValueError('All three 0.40 seeds must finish before this experiment')
    if sorted(r['seed'] for r in completed['results']) != SEEDS:
        raise ValueError('Missing or duplicate saved seed results')
    frozen = memory.required(parent, 'all_models_frozen')
    if not frozen['complete'] or len(frozen['models']) != 15:
        raise ValueError('The five-arm source is incomplete')
    catalog = {(r['seed'],r['arm']):r for r in frozen['models']}
    if set(catalog) != {(s,a) for s in SEEDS for a in width.ARMS}:
        raise ValueError('Invalid source model catalog')
    provenance = json.loads((original.results/'provenance.json').read_text())
    norm = memory.required(original, 'normalization')
    if (bc['normalization_sha'] != identity(norm) or
        bc['train_ids'] != provenance['splits']['train'] or bc['development_ids'] != provenance['splits']['select']):
        raise ValueError('Saved normalization or development identities differ')
    plan=json.loads((original.results/'coverage_plan.json').read_text())
    if (provenance['splits'] != plan['splits'] or
        np.asarray(norm['mean']).shape!=(196,768) or not np.isfinite(norm['mean']).all() or
        not np.isfinite(norm['scale']) or norm['scale']<=0):
        raise ValueError('Invalid normalization or inconsistent split plan')
    if (len(bc['train_ids']), len(bc['development_ids'])) != (2000,400) or set(bc['train_ids']) & set(bc['development_ids']):
        raise ValueError('Invalid source split allocation')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a CUDA GPU runtime')
    if mode == 'sweep' and torch.cuda.get_device_name() != bc['gpu']:
        raise RuntimeError('Paired training requires the saved GPU model: '+bc['gpu'])
    payload = dict(study='ggf_cap040_test' if mode=='test' else 'ggf_higher_cap_development',
        version=1, seeds=SEEDS, source_sha=source_sha, parent_run=parent.run_id,
        cap040_run=cap.run_id, versions=matched.versions(), base=bc['base'], fit_config=bc['fit_config'],
        supported_caps=[] if mode=='test' else CAPS, maximum_planned_gradient_updates=0 if mode=='test' else 27000,
        test_access=mode=='test', controls_retrained=False, reconstruction_limit_changed=False,
        endpoint='Final 2500 + 500 guided updates; no development-best checkpoint selection')
    store = inputs.JournalStore(output, payload, local/'rescue', require_drive)
    install_drive_recovery(store)
    ctx = dict(store=store, parent=parent, catalog=catalog,
        sources=dict(width_balance=parent, cap040=cap, original=original),
        base=copy.deepcopy(bc['base']), cfg=copy.deepcopy(bc['fit_config']), recipe=bound.settings(),
        device='cuda', local=local, stamps=memory.source_snapshot([parent,cap,original]),
        checkpoint_stamps=[], packs={}, provenance=provenance, norm=norm, seeds=SEEDS)
    ctx['completed_cap040']=completed
    for path in [original.results/'provenance.json', original.results/'coverage_plan.json']:
        ctx['stamps'].append(dict(path=str(path), sha256=sha_file(path)))
    for source, keys in [(parent,['all_models_frozen']), (cap,['requested_screen_complete']), (original,['normalization'])]:
        ctx['checkpoint_stamps'].extend(matched.source_key_stamp(source,k) for k in keys)
    ctx['cap040_rows'] = [verify_cap040(ctx,seed) for seed in SEEDS]
    store.write('protocol.json', payload)
    memory.memory_safe_metrics()
    return ctx


def cached_split(ctx, split, cache_local):
    source = ctx['sources']['original']
    chunk = source.config['config']['extract_chunk']
    for lo in range(0,len(ctx['provenance']['splits'][split]),chunk):
        ctx['checkpoint_stamps'].append(matched.source_key_stamp(source,f'features_{split}_{lo:06d}'))
    ctx['store'].event('loading_cached_split',split=split,images=len(ctx['provenance']['splits'][split]))
    return memory.normalized_pack(source,split,ctx['provenance'],ctx['norm'],Path(cache_local),chunk)


def report_signature(row, test_stamp, cfg):
    return identity(dict(weights=row['weights_sha'], targets=row['targets_sha'],
                         rank=row['rank'], test=test_stamp, cfg=cfg))


def check_report(report, row, cfg):
    if report['weights_sha'] != row['weights_sha'] or report['targets_sha'] != row['targets_sha'] or report['rank'] != row['rank']:
        raise ValueError('Saved control report identifies different weights/readouts')
    if sorted(r['category_index'] for r in report['categories']) != cfg['category_indices']:
        raise ValueError('Scoring categories cannot shrink or repeat')
    if any(not np.isfinite([r['capture'], r['ap']]).all() for r in report['categories']):
        raise FloatingPointError('Undefined category score')
    bound.validate_metrics(report)


def prepare_test(paths, output, local, cache_local, source_sha, require_drive=True):
    ctx = open_sources(paths, output, local, source_sha, 'test', require_drive)
    source = matched.discover(paths['strong'], ctx['local']/'read_strong', run=ctx['parent'].config['source_runs']['strong'])
    ctx['sources']['strong'] = source
    ctx['stamps'] += memory.source_snapshot([source])
    plan = memory.required(source,'fresh_plan')
    stamp = memory.required(source,'activation_identity_test')
    ids = plan['splits']['test']
    if (len(ids)!=1500 or len(set(ids))!=1500 or stamp['ids']!=ids or
        stamp['normalization_sha']!=identity(ctx['norm']) or stamp!=ctx['parent'].config['test_identity'] or
        set(ids)&(set(ctx['provenance']['splits']['train'])|set(ctx['provenance']['splits']['select']))):
        raise ValueError('Wrong cached test cohort, normalization, or overlapping splits')
    ctx.update(test_ids=ids,test_stamp=stamp,test_chunk=source.config['config']['extract_chunk'])
    for key in ['fresh_plan','activation_identity_test']:
        ctx['checkpoint_stamps'].append(matched.source_key_stamp(source,key))
    for lo in range(0,1500,ctx['test_chunk']):
        ctx['checkpoint_stamps'].append(matched.source_key_stamp(source,f'features_test_{lo:06d}'))
    reused=[]
    for seed in SEEDS:
        for arm in CONTROLS:
            row=ctx['catalog'][(seed,arm)]
            matched.check_targets(row['targets'],ctx['cfg'])
            tag=f's{seed}_{arm}'
            saved=memory.required(ctx['parent'],'audit_'+tag)
            if saved['identity']!=report_signature(row,stamp,ctx['cfg']):
                raise ValueError('Saved control audit uses different test data or scoring settings')
            report=saved['report'];check_report(report,row,ctx['cfg'])
            compact=memory.required(ctx['parent'],'compact_'+tag)
            if compact['image_ids']!=ids or compact['category_indices']!=ctx['cfg']['category_indices']:
                raise ValueError('Saved compact control belongs to a different cohort')
            ctx['store'].save('audit_'+tag,saved)
            ctx['store'].save('compact_'+tag,compact)
            ctx['store'].write('reports/'+tag+'.json',width.json_safe(report))
            ctx['checkpoint_stamps'] += [matched.source_key_stamp(ctx['parent'],k) for k in ['audit_'+tag,'compact_'+tag]]
            reused.append(dict(seed=seed,arm=arm,weights_sha=row['weights_sha'],test_signature=saved['identity']))
            del compact,saved,report
    # Freeze every treatment before opening any experimental test feature.
    frozen=dict(complete=True,models=ctx['cap040_rows'],reused_controls=reused,new_gradient_updates=0)
    ctx['store'].save('all_models_frozen',frozen)
    ctx['store'].write('all_models_frozen.json',width.json_safe(frozen))
    ctx['store'].write('test_identity.json',stamp)
    ctx['packs']['select']=cached_split(ctx,'select',cache_local)
    ctx['store'].write('source_checkpoint_hashes.json',ctx['checkpoint_stamps'])
    matched.assert_sources(ctx)
    return ctx


def load_test(ctx):
    frozen=memory.required(ctx['store'],'all_models_frozen')
    if not frozen['complete'] or [r['seed'] for r in frozen['models']]!=SEEDS:
        raise ValueError('Freeze all three saved endpoints before test access')
    verify_test_freeze(ctx,frozen)
    source=ctx['sources']['strong'];ids=ctx['test_ids'];stamp=ctx['test_stamp'];norm=ctx['norm']
    folder=ctx['local']/'normalized';folder.mkdir(exist_ok=True)
    path=folder/('final_test_'+stamp['normalized_sha']+'.npy');meta=path.with_suffix('.json')
    valid=path.exists() and meta.exists() and json.loads(meta.read_text()).get('file_sha')==sha_file(path)
    target=None
    if not valid:target=np.lib.format.open_memmap(path.with_suffix('.pending.npy'),mode='w+',dtype=np.float32,shape=(1500,196,768))
    masks=[]
    for lo in range(0,len(ids),ctx['test_chunk']):
        shard=memory.required(source,f'features_test_{lo:06d}');want=ids[lo:lo+ctx['test_chunk']]
        if shard['ids']!=want or shard['normalization_sha']!=identity(norm) or len(shard['masks'])!=len(want):
            raise ValueError('Cached test shard identity changed')
        raw=shard['x'].float().numpy() if torch.is_tensor(shard['x']) else np.asarray(shard['x'],np.float32)
        if raw.shape!=(len(want),196,768) or not np.isfinite(raw).all():raise ValueError('Invalid cached test activations')
        if target is not None:target[lo:lo+len(want)]=data.normalize(raw,norm)
        masks.extend(shard['masks']);del shard,raw
        if lo%200==0:ctx['store'].event('loading_cached_test',completed=lo+len(want),total=1500)
    if target is not None:
        target.flush();del target
        import os
        os.replace(path.with_suffix('.pending.npy'),path)
    x=np.load(path,mmap_mode='r')
    if x.shape!=(1500,196,768) or x.dtype!=np.float32 or memory.streaming_array_sha(x)!=stamp['normalized_sha']:
        raise ValueError('Cached test differs from the previously scored activation identity')
    meta.write_text(json.dumps(dict(file_sha=sha_file(path),normalized_sha=stamp['normalized_sha'])))
    ctx['packs']['test']=dict(x=x,ids=ids,masks=masks)
    matched.assert_sources(ctx)
    return ctx['packs']['test']


def verify_test_freeze(ctx, frozen):
    if identity(frozen['models']) != identity(ctx['cap040_rows']):
        raise ValueError('Frozen treatment catalog changed')
    for row in frozen['models']:
        weights=get_test_weights(ctx,row)
        matched.valid_weights(weights,'vanilla',ctx['base'])


def get_test_weights(ctx,row):
    state=memory.required(ctx['sources']['cap040'],row['key'])
    if training.weight_sha(state['weights'])!=row['weights_sha'] or training.weight_sha({'head':state['head']})!=row['head_sha']:
        raise ValueError('Frozen 0.40 checkpoint changed')
    return state['weights']


def audit_treatment(ctx,row):
    store=ctx['store'];tag=f"s{row['seed']}_cap040"
    signature=report_signature(row,ctx['test_stamp'],ctx['cfg'])
    old=store.load('audit_'+tag)
    if old is not None:
        if old['identity']!=signature:raise ValueError('Cached treatment audit identity changed')
        check_report(old['report'],row,ctx['cfg'])
        memory.required(store,'compact_'+tag)
        return old['report']
    model=core.new_model('vanilla',ctx['base'],get_test_weights(ctx,row),ctx['device'])
    store.event('test_scoring_start',model=tag,images=1500,new_gradient_updates=0)
    with torch.inference_mode(), matched.progress(ctx,model,tag):
        report=core.validation(model,ctx['packs']['test'],row['targets'],ctx['cfg'],ctx['device'],with_arrays=True)
        compact=study.compact_audit(report.pop('_audit'),matched.mapping(row['targets']),ctx['cfg'])
        activity=matched.activity(model,ctx['packs']['test'],ctx['device'],ctx['cfg']['batch'])
        dev=training.measure(model,ctx['packs']['select']['x'].reshape(-1,model.d),ctx['device'],ctx['cfg']['batch'])
    for key in ['mse','ve']:
        if abs(dev[key]-row['development']['reconstruction'][key])>1e-5:
            raise ValueError('Saved 0.40 development reconstruction does not reproduce')
    report.update({k:row[k] for k in ['seed','arm','label','rank','origin','core_parameters','head_parameters','maximum_active_coordinates','weights_sha','targets_sha']})
    report.update(activity=activity,development_reconstruction=dev,
        common_native_MSE_limit=row['reference_mse']*1.02,
        legacy_native_eligible=bool(dev['mse']<=row['reference_mse']*1.02))
    check_report(report,row,ctx['cfg'])
    store.save('compact_'+tag,compact)
    store.save('audit_'+tag,dict(identity=signature,report=report))
    store.write('reports/'+tag+'.json',width.json_safe(report))
    folder=store.results/'image_contrasts';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/(tag+'.npz'),image_ids=np.asarray(compact['image_ids']),
        category_indices=np.asarray(compact['category_indices']),denominator=compact['denominator'],error=compact['error'])
    store.event('test_scoring_complete',model=tag,capture=report['macro_capture'],AP=report['macro_ap'],VE=report['reconstruction']['ve'])
    del model,compact;gc.collect();torch.cuda.empty_cache()
    return report


def evaluate(ctx):
    verify_test_freeze(ctx,memory.required(ctx['store'],'all_models_frozen'))
    if 'test' not in ctx['packs']:raise RuntimeError('Load the verified cached test first')
    s=ctx['store'];reports=[];comparisons=[]
    for seed in SEEDS:
        row=next(r for r in ctx['cap040_rows'] if r['seed']==seed)
        treatment=audit_treatment(ctx,row);reports.append(treatment)
        for arm in CONTROLS:
            tag=f's{seed}_{arm}';control=memory.required(s,'audit_'+tag)['report'];reports.append(control)
            sig=identity(dict(treatment=report_signature(row,ctx['test_stamp'],ctx['cfg']),
                control=memory.required(s,'audit_'+tag)['identity'],bootstrap=ctx['cfg']['bootstrap']))
            key=f'paired_s{seed}_cap040_vs_{arm}'
            cached=s.load(key)
            if cached is None:
                s.event('paired_intervals_start',seed=seed,control=arm,resamples=ctx['cfg']['bootstrap'])
                a=memory.required(s,f'compact_s{seed}_cap040');b=memory.required(s,'compact_'+tag)
                ci=study.paired(a,b,ctx['cfg']);del a,b;gc.collect()
                cached=dict(identity=sig,intervals=ci);s.save(key,cached)
            if cached['identity']!=sig:raise ValueError('Cached paired interval identity changed')
            change=dict(seed=seed,treatment='cap040',control=arm,
                capture_gain_pp=100*(treatment['macro_capture']-control['macro_capture']),
                AP_gain_pp=100*(treatment['macro_ap']-control['macro_ap']),
                VE_gain_pp=100*(treatment['reconstruction']['ve']-control['reconstruction']['ve']),
                paired_intervals=cached['intervals'],reconstruction_guard_passed=treatment['legacy_native_eligible'])
            comparisons.append(change);s.write('paired/'+key+'.json',width.json_safe(change))
    out=dict(complete=True,models=reports,comparisons=comparisons,new_gradient_updates=0,
        controls_retrained=False,new_structural_exchanges=0,test_previously_inspected=True,
        interpretation='Same previously inspected 1500-image test. Fixed endpoint inference; conditional paired image intervals, no multiple-comparison correction. This is GGF, not structural AGF.')
    test_tables(ctx,out)
    matched.assert_sources(ctx)
    s.save('experiment_complete',out);s.write('summary.json',width.json_safe(out))
    return out


def test_tables(ctx,out):
    s=ctx['store'];rows=[];categories=[]
    for r in out['models']:
        rows.append(dict(seed=r['seed'],arm=r['arm'],capture_pct=100*r['macro_capture'],
            patch_AP_pct=100*r['macro_ap'],VE_pct=100*r['reconstruction']['ve'],MSE=r['reconstruction']['mse'],
            native_eligible=r['legacy_native_eligible']))
        categories.extend(dict(seed=r['seed'],arm=r['arm'],**c) for c in r['categories'])
    matched.csv_write(s.results/'test_comparison.csv',rows)
    matched.csv_write(s.results/'all_categories.csv',categories)
    means=[]
    for arm in ['cap040']+CONTROLS:
        rr=[r for r in rows if r['arm']==arm]
        means.append(dict(arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in ['capture_pct','patch_AP_pct','VE_pct']},eligible_seeds=sum(r['native_eligible'] for r in rr)))
    matched.csv_write(s.results/'three_seed_means.csv',means)
    gains=[]
    for r in out['comparisons']:
        ci=r['paired_intervals']
        gains.append({k:r[k] for k in ['seed','treatment','control','capture_gain_pp','AP_gain_pp','VE_gain_pp','reconstruction_guard_passed']}|
            dict(capture_ci_low=100*ci['capture_gain_ci95'][0],capture_ci_high=100*ci['capture_gain_ci95'][1],
                 AP_ci_low=100*ci['ap_gain_ci95'][0],AP_ci_high=100*ci['ap_gain_ci95'][1]))
    matched.csv_write(s.results/'paired_comparison.csv',gains)
    plot_means(s,means,'cap040_test','Previously inspected test; three seeds',
        {'cap040':'GGF 4\nCap 0.40','mask4_balanced':'GGF 4\nCap 0.20','vanilla4':'Vanilla\nBSF 4','mask3_fixed':'Original\nGGF 3'})


def prepare_sweep(paths,output,local,cache_local,source_sha,require_drive=True,caps=(.6,)):
    # No strong-test store or experimental test shard is opened in this mode.
    ctx=open_sources(paths,output,local,source_sha,'sweep',require_drive)
    ctx['caps']=caps_checked(caps)
    ctx['store'].write('requested_caps.json',dict(caps=ctx['caps'],seeds=SEEDS,
        policy='Execution scope only; expanding caps does not change any per-trial identity'))
    for split in ['train','select']:ctx['packs'][split]=cached_split(ctx,split,cache_local)
    ctx['store'].write('source_checkpoint_hashes.json',ctx['checkpoint_stamps'])
    matched.assert_sources(ctx)
    return ctx


def run_sweep(ctx):
    results=[];s=ctx['store']
    if 'test' in ctx['packs']:raise ValueError('The development sweep cannot access test features')
    for seed in SEEDS:
        row,warm,old1,old2=bound.verify_seed(ctx,seed)
        saved40=next(r for r in ctx['cap040_rows'] if r['seed']==seed)
        baselines={.2:old2['history'][-1]['validation'],.4:saved40['development']}
        for cap in ctx.get('caps',CAPS):
            code=f'cap{int(round(cap*100)):03d}'
            sub=dict(ctx,recipe=cap_recipe(cap))
            s.event('higher_cap_seed_start',seed=seed,upper_bound=cap)
            first=fit(sub,f'phase1_{code}_s{seed}',warm['best'],row['targets'],2500,71000+seed,True,True,reference_mse=row['reference_mse'])
            second=fit(sub,f'phase2_{code}_s{seed}',first['weights'],row['targets'],500,82000+seed,True,True,
                reference_mse=row['reference_mse'],initial_head=first['head'],initial_balance=first['balance'])
            for new,old,n in [(first,old1,2500),(second,old2,500)]:
                matched.complete(new,n)
                for key in ['sampled_batches_sha','sampled_noise_sha']:
                    if new[key]!=old[key]:raise ValueError('Paired cap schedule/noise diverged: '+key)
            metrics=second['history'][-1]['validation'];bound.validate_metrics(metrics)
            result=dict(seed=seed,upper_bound=cap,development=metrics,baselines=baselines,
                comparisons={str(b):bound.compare(seed,v,metrics,row['reference_mse']) for b,v in baselines.items()},
                weights_sha=training.weight_sha(second['weights']),head_sha=training.weight_sha({'head':second['head']}),
                targets_sha=row['targets_sha'],warm_weights_sha=training.weight_sha(warm['best']),
                final_weights=second['history'][-1]['weights_next'],new_updates=3000,test_access=False,
                reconstruction_limit=row['reference_mse']*1.02)
            s.save(f'result_{code}_s{seed}',result);s.write(f'reports/{code}_s{seed}.json',width.json_safe(result))
            results.append(result)
            del first,second;gc.collect();torch.cuda.empty_cache()
        del warm,old1,old2;gc.collect()
    requested=ctx.get('caps',CAPS)
    out=dict(complete=True,study_complete=requested==CAPS,seeds=SEEDS,caps=requested,results=results,new_gradient_updates=9000*len(requested),
        test_access=False,controls_retrained=False,new_warmups=0,new_structural_exchanges=0,
        selection_rule='Report all final endpoints and original reconstruction guard. Prefer Pareto tradeoffs on development; do not select caps using test scores.')
    sweep_tables(ctx,out);matched.assert_sources(ctx)
    s.save('experiment_complete',out);s.write('summary.json',width.json_safe(out))
    return out


def sweep_preflight(ctx):
    # Disposable full-size loss check from the existing validated implementation.
    # Its saved RNG restoration keeps all actual paired trials unchanged.
    checked=bound.preflight(ctx)
    pending={}
    for seed in SEEDS:
        for cap in ctx.get('caps',CAPS):
            label=f'cap{int(round(cap*100)):03d}_s{seed}'
            pending[label]=0
            for phase,n in [(1,2500),(2,500)]:
                saved=ctx['store'].load(f'phase{phase}_{label}')
                pending[label]+=n-(0 if saved is None else saved['updates'])
    out=dict(checked,pending_updates=pending,
        estimated_gradient_hours=sum(pending.values())*checked['seconds_per_update']/3600,
        note='Estimate uses the existing 0.40 full-size loss/controller preflight; higher-cap numerics are checked on every trial update. Development scoring, source verification and Drive writes add time.')
    ctx['store'].write('sweep_preflight.json',out)
    return out


def sweep_tables(ctx,out):
    s=ctx['store'];rows=[];gains=[]
    for result in out['results']:
        seed=result['seed'];v=result['development']
        rows.append(dict(seed=seed,upper_bound=result['upper_bound'],capture_pct=100*v['macro_capture'],
            patch_AP_pct=100*v['macro_ap'],VE_pct=100*v['reconstruction']['ve'],MSE=v['reconstruction']['mse'],
            native_eligible=v['reconstruction']['mse']<=result['reconstruction_limit'],origin='new_trial'))
        gains.extend(dict(upper_bound=result['upper_bound'],baseline_bound=float(b),**change) for b,change in result['comparisons'].items())
    for seed in SEEDS:
        first=next(r for r in out['results'] if r['seed']==seed)
        for cap,v in first['baselines'].items():
            rows.append(dict(seed=seed,upper_bound=float(cap),capture_pct=100*v['macro_capture'],
                patch_AP_pct=100*v['macro_ap'],VE_pct=100*v['reconstruction']['ve'],MSE=v['reconstruction']['mse'],
                native_eligible=v['reconstruction']['mse']<=first['reconstruction_limit'],origin='reused_control'))
    matched.csv_write(s.results/'development_comparison.csv',rows)
    matched.csv_write(s.results/'development_gains.csv',gains)
    matched.csv_write(s.results/'final_loss_weights.csv',[
        dict(seed=r['seed'],upper_bound=r['upper_bound'],reconstruction_weight=r['final_weights'][0],
             contrast_weight=r['final_weights'][1],detection_weight=r['final_weights'][2]) for r in out['results']])
    means=[]
    for cap in [.2,.4]+out['caps']:
        rr=[r for r in rows if r['upper_bound']==cap]
        means.append(dict(arm=str(cap),**{k:float(np.mean([r[k] for r in rr])) for k in ['capture_pct','patch_AP_pct','VE_pct']},eligible_seeds=sum(r['native_eligible'] for r in rr)))
    matched.csv_write(s.results/'three_seed_development_means.csv',means)
    s.write('selection_policy.json',dict(test_access=False,all_results_retained=True,
        reconstruction_limit_changed=False,rule=out['selection_rule']))
    plot_means(s,means,'higher_caps_development','Development only; three seeds',{str(c):f'Cap {c:.2f}' for c in [.2,.4]+CAPS})


def plot_means(store,means,name,title,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),layout='constrained')
    for ax,key,label in zip(axes,['capture_pct','patch_AP_pct','VE_pct'],['Capture','Patch AP','Reconstruction VE']):
        for j,r in enumerate(means):
            ax.bar(j,r[key],color=plt.get_cmap('viridis')(0.2+0.6*j/max(1,len(means)-1)))
            ax.text(j,r[key]+1,f'{r[key]:.2f}%',ha='center',fontsize=9)
        ax.set_xticks(range(len(means)),[labels[r['arm']] for r in means],fontsize=9)
        ax.set_ylim(0,100);ax.set_title(label);ax.set_ylabel('%');ax.grid(axis='y',alpha=.15)
    fig.suptitle(title);folder=store.results/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/(name+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)


def export(ctx,source_path):
    result=matched.export(ctx,source_path)
    name='ggf_cap040_test' if ctx['store'].config.get('test_access',False) else 'ggf_higher_caps'
    dest=result.with_name(name+'_'+ctx['store'].run_id[:12]+'_results.zip')
    result.replace(dest);return dest


def failure(ctx,stage,exc,source_path):
    try:
        ctx['store'].write('failure.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        print('Partial evidence:',export(ctx,source_path),flush=True)
    except BaseException as err:print('Partial export failed:',type(err).__name__,str(err),flush=True)
