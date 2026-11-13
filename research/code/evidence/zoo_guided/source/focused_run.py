"""Baseline-first seven-seed GGF confirmation at a fixed10640-update budget."""
import argparse,copy,gc,json,os,subprocess,sys,traceback,zipfile
from pathlib import Path
import numpy as np
from coco_baselines import training
from coco_baselines.storage import identity,sha_file
from coco_adaptation import inputs
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched
from scale_run import run_branch,development,preflight,flush
from focused_protocol import *
from focused_inputs import ReuseStore,completed_components,reject_opened_holdout,annotations,backbone_preflight,trim_verified_local

def branch(ctx,seed,kind,lr=.0003,coefficient=.001):
    # Completed pilot endpoints are never passed into fit or rewritten.
    for row in ctx['pilot_rows']:
        if row['seed']==seed and row['kind']==kind and lr==row['lr'] and coefficient==row['coefficient']:
            print('Reusing frozen pilot:',kind,'seed',seed,'/',TOTAL,'total updates',flush=True)
            return validate_row(copy.deepcopy(row))
    row=run_branch(ctx,GROUPS,TOPK,seed,kind,cap=CAP,limit=ADDITIONAL,lr=lr,coefficient=coefficient)
    return validate_row(row)

def audit_development(ctx,row):
    if 'development' not in row:row['development']=development(ctx,row)['report']
    trim_verified_local(ctx,row)
    return row

def tune_baselines(ctx):
    s=ctx['store'];old=s.load('baseline_settings_selected')
    if old is not None:return old
    choices={};details=[]
    for method in BASELINE_ORDER:
        trials=[]
        for j,lr in enumerate(ctx['rule']['baseline_learning_rates']):
            coefficient=ctx['rule']['paper_lasso_coefficients'][j] if method=='paper_group_lasso' else .001
            rows=[]
            for seed in SCREEN:
                row=audit_development(ctx,branch(ctx,seed,method,lr,coefficient));rows.append(row)
                s.write('baseline_progress.json',dict(stage='three_seed_baseline_tuning',method=method,lr=lr,coefficient=coefficient,models=rows,complete=False))
            activity=[r['development']['reconstruction']['active_groups'] for r in rows]
            if not np.isfinite(activity+[r['development']['reconstruction']['mse'] for r in rows]).all():raise ValueError('Nonfinite baseline development score/activity')
            eligible=all(abs(a-TOPK)<=.2*TOPK for a in activity) if method.endswith('group_lasso') else True
            trial=dict(lr=lr,coefficient=coefficient,mse=float(np.mean([r['development']['reconstruction']['mse'] for r in rows])),
                activity=activity,sparsity_eligible=eligible,models=rows)
            trials.append(trial)
            s.write('baseline_tuning_progress.json',dict(completed=details,current_method=method,trials=trials))
        feasible=[r for r in trials if r['sparsity_eligible']]
        best=min(feasible or trials,key=lambda r:(r['mse'],r['lr']))
        choices[method]=dict(lr=best['lr'],coefficient=best['coefficient'],sparsity_matched=bool(feasible),
            selection='minimum three-seed development MSE; per-seed20% activity tolerance for Lasso; retain any mismatch explicitly')
        details.append(dict(method=method,trials=trials,chosen=choices[method],selected_models=best['models']))
        s.event('baseline_settings',method=method,lr=best['lr'],coefficient=best['coefficient'],sparsity_matched=bool(feasible))
    out=dict(complete=True,choices=choices,details=details,test_opened=False,total_updates=TOTAL)
    s.save('baseline_settings_selected',out);s.write('baseline_settings_selected.json',out)
    return out

def three_seed_summary(ctx,tuning):
    s=ctx['store'];old=s.load('three_seed_development_complete')
    if old is not None:return old
    rows=[]
    for kind in ARMS:
        setting=tuning['choices'].get(kind,dict(lr=.0003,coefficient=.001))
        selected=next((x['selected_models'] for x in tuning['details'] if x['method']==kind),None)
        for seed in SCREEN:
            row=copy.deepcopy(next(r for r in selected if r['seed']==seed)) if selected else audit_development(ctx,branch(ctx,seed,kind,**{k:setting[k] for k in ['lr','coefficient']}))
            rows.append(row)
    means=[]
    for kind in ARMS:
        reports=[r['development'] for r in rows if r['kind']==kind]
        entry=dict(kind=kind,seeds=3,total_updates=TOTAL,split='previously inspected development selection set')
        for name,get in [('capture_pct',lambda r:r['macro_capture']),('patch_AP_pct',lambda r:r['macro_ap']),('VE_pct',lambda r:r['reconstruction']['ve'])]:
            values=np.array([100*get(r) for r in reports]);entry[name]=float(values.mean());entry[name+'_seed_sd']=float(values.std(ddof=1))
        entry['mean_active_groups']=float(np.mean([r['reconstruction']['active_groups'] for r in reports]));means.append(entry)
    out=dict(complete=True,models=rows,means=means,fresh_holdout_opened=False)
    s.save('three_seed_development_complete',out);s.write('three_seed_development_summary.json',out)
    matched.csv_write(s.results/'three_seed_development_means.csv',means)
    print('THREE-SEED DEVELOPMENT COMPARISON READY; fresh holdout still closed',json.dumps(means),flush=True)
    return out

def freeze(ctx,tuning):
    s=ctx['store'];old=s.load('all_models_frozen')
    if old is not None:
        if old.get('recipe')!=recipe():raise ValueError('Frozen recipe changed')
        return old
    rows=[]
    # Complete the four additional GGF starts, then the selected baseline starts.
    # Seeds21-23 return the already completed compatible pilot/tuning endpoints.
    for kind in ARMS:
        setting=tuning['choices'].get(kind,dict(lr=.0003,coefficient=.001))
        for seed in SEEDS:
            row=audit_development(ctx,branch(ctx,seed,kind,lr=setting['lr'],coefficient=setting['coefficient']))
            rows.append(row);s.write('final_suite_progress.json',dict(complete=False,models=rows,total_expected=len(expected_pairs()),total_updates=TOTAL))
    if len(rows)!=len(expected_pairs()) or {(r['seed'],r['kind']) for r in rows}!=expected_pairs():raise ValueError('Incomplete49-model catalog')
    streams={}
    for row in rows:
        validate_row(row);state=memory.required(s,row['key'])
        if state['updates']!=ADDITIONAL or training.weight_sha(state['weights'])!=row['weights_sha']:raise ValueError('Frozen endpoint changed')
        pair=(state['sampled_batches_sha'],state['sampled_noise_sha'])
        if row['seed'] in streams and streams[row['seed']]!=pair:raise ValueError('Sampling/noise streams differ within a seed')
        streams[row['seed']]=pair
        row['additional_optimizer_step_seconds']=state.get('seconds')
        row['common_native_warm_mse']=next(r['reference_warm_mse'] for r in rows if r['seed']==row['seed'] and r['kind']=='released_vanilla')
        del state;gc.collect()
    out=dict(complete=True,models=rows,recipe=recipe(),test_opened=False,all_models_frozen_before_test=True,
        matched='Capacity/group dimensions, additional-update counts and image/noise streams; not equal parameters, supervision or FLOPs.',
        source_run=STUDY_RUN,component_training_repeated=False)
    s.save('all_models_frozen',out);s.write('all_models_frozen.json',out)
    print('ALL49 ENDPOINTS FROZEN AT10640 UPDATES; fresh holdout can now open',flush=True)
    return out

def export(ctx,options,complete=False):
    s=ctx['store'];s.write('completion_status.json',dict(complete=complete,recipe=recipe(),
        components_imported=s.load('components_imported') is not None,baseline_tuning_complete=s.load('baseline_settings_selected') is not None,
        three_seed_development_complete=s.load('three_seed_development_complete') is not None,models_frozen=s.load('all_models_frozen') is not None,
        test_complete=s.load('test_complete') is not None,costs_complete=s.load('costs_complete') is not None))
    out=s.output/f'ggf_confirmation_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        hashes={}
        for p in sorted(s.results.rglob('*')):
            if p.is_file():
                name='results/'+p.relative_to(s.results).as_posix();z.write(p,name);hashes[name]=sha_file(p)
        for name,key in [('core_source.zip','source_zip'),('focused_source.zip','wrapper_zip')]:
            p=Path(options[key]);z.write(p,name);hashes[name]=sha_file(p)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    import hashlib
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in hashes.items()):raise OSError('Result ZIP integrity failed')
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);o=json.loads(Path(p.parse_args().config).read_text())
    if o['source_sha']!=CORE_SHA or o['baselines']!='both' or not o['fresh']:raise ValueError('Wrong pinned source or baseline/holdout scope')
    local=Path(o['local']);wrapper=Path(__file__).parent;core=Path(o['core_source'])
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(wrapper),str(core)]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.check_call([sys.executable,'-m','pytest','-q',str(wrapper/'test_focused.py'),str(core/'tests/test_scale_study.py'),
        '--disable-warnings','--basetemp',str(local/('tests_'+os.urandom(4).hex()))],cwd=wrapper,env=env)
    import torch
    from threadpoolctl import threadpool_limits
    from scale_data import prepare,holdout_plan,load_holdout
    from coco_ggf.drive_recovery import install_drive_recovery
    from coco_ggf.cap_followup import safe_destination
    from focused_evaluate import run as evaluate,verify_freeze
    from focused_costs import run as costs
    if matched.versions()!=o['expected_versions']:raise ValueError('Numerical package versions differ from the saved runs')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Select a T4 GPU')
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    ctx=None;stage='prepare'
    try:
        safe_destination(o['job_output'],[o['source_output'],o['components_output'],*o['paths'].values()])
        # Feature/context verification writes only into the new output tree.
        setup=dict(o,output=str(Path(o['job_output'])/'input_verification'))
        ctx=prepare(setup)
        if ctx['store'].run_id!=STUDY_RUN:raise ValueError('Pinned numerical/training identity differs; restore the source environment')
        source=matched.discover(o['source_output'],local/'read_scale',run=STUDY_RUN)
        component=matched.discover(o['components_output'],local/'read_components',prefix=COMPONENT_PREFIX)
        reject_opened_holdout(source)
        ctx['stamps']+=memory.source_snapshot([source,component])
        job=inputs.JournalStore(o['job_output'],dict(wrapper_sha=o['wrapper_sha'],core_sha=CORE_SHA,
            source_run=STUDY_RUN,component_run=component.run_id,normalization_sha=identity(ctx['norm']),
            train=ctx['packs']['train']['identity'],selection=ctx['packs']['select']['identity'],recipe=recipe()),local/'job_rescue',True)
        install_drive_recovery(job)
        rows,imported=completed_components(ctx,source,component,job.results)
        ctx.update(store=ReuseStore(job,source,rows),pilot_rows=rows)
        ctx['store'].save('components_imported',imported);ctx['store'].write('components_imported.json',imported)
        zoo=json.loads(Path(o['zoo_receipt']).read_text())
        if not zoo['complete'] or zoo['wrapper_sha']!=o['wrapper_sha']:raise ValueError('Complete synthetic stage first with the same notebook source')
        for name in ['zoo_complete.json','zoo_means.csv','zoo_models.csv','zoo_geometry.csv','zoo_factors.csv','PROTOCOL.json','data_audit.json','archived_baselines.json','zoo_selected.json']:
            src=Path(zoo['results'])/name;dest=ctx['store'].results/'synthetic_zoo'/name
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
        ctx['store'].write('recipe_frozen.json',recipe());ctx['store'].write('baseline_implementation_audit.json',__import__('scale_models').BASELINE_AUDIT)
        ctx['store'].results.joinpath('PROTOCOL.md').write_bytes((wrapper/'PROTOCOL.md').read_bytes())
        # Setup authentication/replay uses only a training image. No final features.
        stage='cached_backbone_and_annotation_plan';coco=annotations(ctx,o['source_output'])
        saved_plan=source.load('fresh_holdout_plan')
        if saved_plan is not None and ctx['store'].load('fresh_holdout_plan') is None:ctx['store'].save('fresh_holdout_plan',saved_plan)
        plan=holdout_plan(ctx,coco);ctx['store'].write('fresh_holdout_plan.json',plan)
        backbone_preflight(ctx,coco,o['source_output']);preflight(ctx)
        stage='missing_baselines_then_development_tuning';tuning=tune_baselines(ctx);three_seed_summary(ctx,tuning);export(ctx,o)
        stage='seven_seed_fixed_budget';freeze(ctx,tuning);export(ctx,o)
        stage='untouched_holdout_evaluation';verify_freeze(ctx);pack=load_holdout(ctx,coco);evaluate(ctx,pack);del pack;gc.collect();export(ctx,o)
        stage='measured_forward_costs';costs(ctx)
        matched.assert_sources(ctx);archive=export(ctx,o,True)
        receipt=dict(complete=True,results=str(ctx['store'].results),result_zip=str(archive),wrapper_sha=o['wrapper_sha'],
            models=49,total_updates=TOTAL,component_training_repeated=False)
        Path(o['receipt']).write_text(json.dumps(receipt,indent=2));print('GGF CONFIRMATION COMPLETE:',archive,flush=True)
    except BaseException as exc:
        if ctx is not None:
            try:flush(ctx)
            except BaseException as extra:print('Checkpoint flush issue:',repr(extra),flush=True)
            if hasattr(ctx.get('store'), 'job'):
                try:
                    ctx['store'].write('last_interruption.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
                    print('Partial evidence ZIP:',export(ctx,o),flush=True)
                except BaseException as extra:print('Partial export issue:',repr(extra),flush=True)
        raise
    finally:
        if ctx is not None:ctx['packs'].clear();gc.collect()

if __name__=='__main__':main()
