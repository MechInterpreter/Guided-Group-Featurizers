"""Three-arm, three-seed continuation of saved 4D checkpoints."""
import copy,gc,json,time
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core,inputs
from coco_coordinated import run as memory
from . import cap_followup as base,caps_pipeline as caps
from . import pcgrad_fit as engine
matched=base.matched
ARMS=['ggf_continue','ggf_pcgrad','vanilla_continue']
LABELS=dict(ggf_continue='GGF: ordinary continuation',ggf_pcgrad='GGF: encoder PCGrad',
    vanilla_continue='Vanilla BSF: continuation',cap040='Starting GGF: cap 0.40',vanilla4='Starting Vanilla BSF')

def settings():
    return dict(seeds=base.SEEDS,arms=ARMS,updates_per_arm=3000,balance=base.bound.settings()['balance'],
        pcgrad='Weighted PCGrad on encoder parameters shared by all three losses; sum projected gradients. Decoder and classifier use ordinary weighted gradients.',
        optimizer='Carry source Adam moments, head and controller; no learning-rate change or phase reset.',
        pairing='Fresh identical image/noise streams seeded 93000 + seed. Separate saved RNG for projection order.',
        endpoint='Final 3000-update endpoint; all native-limit failures retained.',
        data='Existing 2000 training, 400 development, and previously inspected 1500 test images; no test selection.',
        success='Compare all three metrics with the starting checkpoint and ordinary continuation; retain per-seed outcomes and unchanged native reconstruction guard.',
        source='https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf')

def prepare(paths,output,local,cache_local,source_sha,previous_test_output=None,require_drive=True):
    base.safe_destination(output,paths.values());local=Path(local)
    # Reuse the verified source and test-metadata loader. This scratch store is
    # local only; no experimental test features are opened by prepare_test.
    ctx=base.prepare_test(paths,local/'source_checks',local,cache_local,source_sha,False)
    if torch.cuda.get_device_name()!=ctx['sources']['cap040'].config['gpu']:
        raise RuntimeError('Use the same Tesla T4 as the saved paired sources')
    caps.reuse_previous_cap040_audits(ctx,previous_test_output)
    old=ctx['store'];payload=dict(old.config,study='ggf_pcgrad_continuation',continuation_recipe=settings(),
        maximum_planned_gradient_updates=27000,test_access='Only after all nine continuation endpoints freeze')
    new=inputs.JournalStore(output,payload,local/'rescue',require_drive);base.install_drive_recovery(new)
    for seed in base.SEEDS:
        for arm in base.CONTROLS+['cap040']:
            tag=f's{seed}_{arm}'
            for prefix in ['audit_','compact_']:
                value=old.load(prefix+tag)
                if value is not None:new.save(prefix+tag,value)
    ctx['store']=new;ctx['recipe']=settings();ctx['roots']=[]
    ctx['packs']['train']=base.cached_split(ctx,'train',cache_local)
    for row in ctx['cap040_rows']:
        ctx['roots'].append(dict(row,source_name='cap040'))
    for seed in base.SEEDS:
        row=ctx['catalog'][(seed,'vanilla4')]
        state=memory.required(ctx['parent'],row['key']);matched.complete(state,500)
        if (row['rank']!=4 or row['supervised'] or row['balanced'] or
            training.weight_sha(state['weights'])!=row['weights_sha'] or identity(row['targets'])!=row['targets_sha']):
            raise ValueError('The saved Vanilla endpoint/readouts changed')
        matched.valid_weights(state['weights'],'vanilla',ctx['base'])
        ctx['checkpoint_stamps'].append(matched.source_key_stamp(ctx['parent'],row['key']))
        ctx['roots'].append(dict(row,source_name='width_balance',development=state['history'][-1]['validation']))
        del state
    ctx['store'].write('protocol.json',payload)
    ctx['store'].write('source_checkpoint_hashes.json',ctx['checkpoint_stamps'])
    matched.assert_sources(ctx);return ctx

def source_state(ctx,row):
    state=memory.required(ctx['sources'][row['source_name']],row['key']);matched.complete(state,500)
    if training.weight_sha(state['weights'])!=row['weights_sha']:
        raise ValueError('Starting checkpoint weights changed')
    if row.get('head_sha') is not None and training.weight_sha({'head':state['head']})!=row['head_sha']:
        raise ValueError('Starting checkpoint classifier changed')
    return state

def preflight(ctx):
    """Disposable full-size optimizer/controller checks; actual fits reseed."""
    old=ctx['store'].load('continuation_preflight')
    if old is not None:return old
    timings={};rng_state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state_all()
    try:
        for arm in ARMS:
            root=next(r for r in ctx['roots'] if r['seed']==21 and r['arm']==('vanilla4' if arm=='vanilla_continue' else 'cap040'))
            st=source_state(ctx,root);supervised=arm!='vanilla_continue'
            sub=dict(ctx,recipe=settings())
            out=engine.fit(sub,'preflight_'+arm,st['weights'],root['targets'],2,93021,
                supervised,supervised,root['reference_mse'],st['head'] if supervised else None,
                st['balance'] if supervised else None,pcgrad=arm=='ggf_pcgrad',initial_optimizer=st['optimizer'])
            timings[arm]=out['seconds']/2;del st,out;gc.collect()
    finally:
        torch.set_rng_state(rng_state);torch.cuda.set_rng_state_all(cuda_state)
    pending={}
    for seed in base.SEEDS:
        for arm in ARMS:
            saved=ctx['store'].load(f'continuation_{arm}_s{seed}')
            pending[f'{arm}_s{seed}']=3000-(0 if saved is None else saved['updates'])
    result=dict(gpu=torch.cuda.get_device_name(),seconds_per_update=timings,pending_updates=pending,
        estimated_gradient_hours=sum(n*timings[k.rsplit('_s',1)[0]] for k,n in pending.items())/3600,
        disposable_updates=6,actual_trial_rng_unchanged=True)
    ctx['store'].save('continuation_preflight',result);ctx['store'].write('preflight.json',result);return result

def run(ctx):
    if 'test' in ctx['packs']:raise ValueError('Test features must stay closed during training')
    rows=[]
    for seed in base.SEEDS:
        seed_rows=[]
        for arm in ARMS:
            root=next(r for r in ctx['roots'] if r['seed']==seed and r['arm']==('vanilla4' if arm=='vanilla_continue' else 'cap040'))
            initial=source_state(ctx,root);supervised=arm!='vanilla_continue';key=f'continuation_{arm}_s{seed}'
            state=engine.fit(ctx,key,initial['weights'],root['targets'],3000,93000+seed,supervised,supervised,
                root['reference_mse'],initial['head'] if supervised else None,initial['balance'] if supervised else None,
                pcgrad=arm=='ggf_pcgrad',initial_optimizer=initial['optimizer'])
            matched.complete(state,3000);matched.valid_weights(state['weights'],'vanilla',ctx['base'])
            v=state['history'][-1]['validation']
            if not supervised:
                model=core.new_model('vanilla',ctx['base'],state['weights'],ctx['device'])
                with torch.inference_mode():v=core.validation(model,ctx['packs']['select'],root['targets'],ctx['cfg'],ctx['device'])
                del model
            base.bound.validate_metrics(v)
            row=dict(root,arm=arm,label=LABELS[arm],origin='continued_saved_endpoint',key=key,
                source_name='continuation',supervised=supervised,balanced=supervised,pcgrad=arm=='ggf_pcgrad',
                weights_sha=training.weight_sha(state['weights']),
                head_sha=training.weight_sha({'head':state['head']}) if supervised else None,
                head_parameters=150 if supervised else 0,new_gradient_updates=3000,
                total_training_updates=13640,development=v,starting_arm=root['arm'],starting_weights_sha=root['weights_sha'],
                optimizer_sha=identity(engine.state_identity(state['optimizer'])),
                phase_batch_sha=state['sampled_batches_sha'],phase_noise_sha=state['sampled_noise_sha'])
            seed_rows.append(row)
            ctx['store'].write(f'development/{arm}_s{seed}.json',base.width.json_safe(row))
            del state,initial;gc.collect();torch.cuda.empty_cache()
        for key in ['phase_batch_sha','phase_noise_sha']:
            if len({r[key] for r in seed_rows})!=1:raise ValueError('Continuation schedule or noise diverged: '+key)
        rows.extend(seed_rows)
    frozen=dict(complete=True,models=rows,seeds=base.SEEDS,arms=ARMS,new_gradient_updates=27000,test_features_opened=False)
    old=ctx['store'].load('continuation_models_frozen')
    if old is not None and identity(old)!=identity(frozen):raise ValueError('Frozen continuation endpoints changed')
    ctx['store'].save('continuation_models_frozen',frozen);ctx['store'].write('continuation_models_frozen.json',base.width.json_safe(frozen))
    ctx['store'].save('all_models_frozen',dict(complete=True,models=ctx['cap040_rows']))
    ctx['rows']=rows;tables(ctx,[dict(r,**r['development']) for r in rows],[],development=True)
    matched.assert_sources(ctx);return frozen

def verify_freeze(ctx):
    frozen=memory.required(ctx['store'],'continuation_models_frozen')
    if (not frozen['complete'] or len(frozen['models'])!=9 or
        {(r['seed'],r['arm']) for r in frozen['models']}!={(s,a) for s in base.SEEDS for a in ARMS}):
        raise ValueError('Freeze all nine continuation endpoints before test')
    if identity(frozen['models'])!=identity(ctx['rows']):raise ValueError('Frozen endpoint catalog changed')
    for row in ctx['rows']:
        state=memory.required(ctx['store'],row['key']);matched.complete(state,3000)
        if (training.weight_sha(state['weights'])!=row['weights_sha'] or
            identity(engine.state_identity(state['optimizer']))!=row['optimizer_sha'] or
            (row['supervised'] and training.weight_sha({'head':state['head']})!=row['head_sha'])):
            raise ValueError('Frozen checkpoint changed')

def load_test(ctx):
    verify_freeze(ctx)
    return base.load_test(ctx)

def audit(ctx,row):
    s=ctx['store'];tag=f"s{row['seed']}_{row['arm']}";signature=base.report_signature(row,ctx['test_stamp'],ctx['cfg'])
    old=s.load('audit_'+tag)
    if old is not None:
        if old['identity']!=signature:raise ValueError('Cached test identity changed')
        base.check_report(old['report'],row,ctx['cfg']);memory.required(s,'compact_'+tag);return old['report']
    state=memory.required(s,row['key']);matched.complete(state,3000)
    model=core.new_model('vanilla',ctx['base'],state['weights'],ctx['device']);del state
    s.event('test_scoring_start',model=tag,images=len(ctx['test_ids']),new_gradient_updates=0)
    with torch.inference_mode(),matched.progress(ctx,model,tag):
        result=core.validation(model,ctx['packs']['test'],row['targets'],ctx['cfg'],ctx['device'],with_arrays=True)
        compact=base.study.compact_audit(result.pop('_audit'),matched.mapping(row['targets']),ctx['cfg'])
        dev=training.measure(model,ctx['packs']['select']['x'].reshape(-1,model.d),ctx['device'],ctx['cfg']['batch'])
    for k in ['mse','ve']:
        if abs(dev[k]-row['development']['reconstruction'][k])>1e-5:raise ValueError('Frozen development reconstruction does not reproduce')
    result.update({k:row[k] for k in ['seed','arm','label','rank','origin','core_parameters','head_parameters','maximum_active_coordinates','weights_sha','targets_sha']})
    result.update(legacy_native_eligible=dev['mse']<=row['reference_mse']*1.02,development_reconstruction=dev,
        common_native_MSE_limit=row['reference_mse']*1.02,total_training_updates=13640)
    base.check_report(result,row,ctx['cfg']);s.save('compact_'+tag,compact);s.save('audit_'+tag,dict(identity=signature,report=result))
    s.write('reports/'+tag+'.json',base.width.json_safe(result))
    del model,compact;gc.collect();torch.cuda.empty_cache();return result

def evaluate(ctx):
    verify_freeze(ctx)
    if 'test' not in ctx['packs']:raise RuntimeError('Load the verified test after freezing all endpoints')
    s=ctx['store'];reports=[];comparisons=[]
    pairs=[('ggf_continue','cap040'),('ggf_pcgrad','cap040'),('vanilla_continue','vanilla4'),
        ('ggf_pcgrad','ggf_continue'),('ggf_continue','vanilla_continue'),('ggf_pcgrad','vanilla_continue')]
    for seed in base.SEEDS:
        lookup={}
        for row in [r for r in ctx['rows'] if r['seed']==seed]:
            lookup[row['arm']]=audit(ctx,row)
        root=next(r for r in ctx['cap040_rows'] if r['seed']==seed)
        lookup['cap040']=base.audit_treatment(ctx,root)
        lookup['vanilla4']=memory.required(s,f'audit_s{seed}_vanilla4')['report']
        reports.extend(lookup.values())
        for a,b in pairs:
            ta=f's{seed}_{a}';tb=f's{seed}_{b}'
            sig=identity(dict(a=memory.required(s,'audit_'+ta)['identity'],b=memory.required(s,'audit_'+tb)['identity'],bootstrap=ctx['cfg']['bootstrap']))
            key=f'paired_s{seed}_{a}_vs_{b}';old=s.load(key)
            if old is None:
                s.event('paired_intervals_start',seed=seed,treatment=a,control=b)
                old=dict(identity=sig,intervals=base.study.paired(memory.required(s,'compact_'+ta),memory.required(s,'compact_'+tb),ctx['cfg']));s.save(key,old)
            if old['identity']!=sig:raise ValueError('Paired comparison changed')
            av,bv=lookup[a],lookup[b]
            comparisons.append(dict(seed=seed,treatment=a,control=b,paired_intervals=old['intervals'],
                capture_gain_pp=100*(av['macro_capture']-bv['macro_capture']),AP_gain_pp=100*(av['macro_ap']-bv['macro_ap']),
                VE_gain_pp=100*(av['reconstruction']['ve']-bv['reconstruction']['ve']),
                all_three_improve=all(av[k]>bv[k] for k in ['macro_capture','macro_ap']) and av['reconstruction']['ve']>bv['reconstruction']['ve']))
    out=dict(complete=True,models=reports,comparisons=comparisons,new_training_updates=27000,
        new_test_gradient_updates=0,new_structural_exchanges=0,test_previously_inspected=True,
        uncertainty='Conditional paired whole-image intervals; 200 resamples; no seed-level or multiplicity correction.',
        interpretation='GGF continuation experiment, not structural AGF; report all outcomes including reconstruction-limit failures.')
    tables(ctx,reports,comparisons);matched.assert_sources(ctx)
    s.save('experiment_complete',out);s.write('summary.json',base.width.json_safe(out));return out

def tables(ctx,reports,comparisons,development=False):
    s=ctx['store'];prefix='development' if development else 'test';rows=[];categories=[]
    for r in reports:
        rows.append(dict(seed=r['seed'],arm=r['arm'],capture_pct=100*r['macro_capture'],patch_AP_pct=100*r['macro_ap'],
            VE_pct=100*r['reconstruction']['ve'],MSE=r['reconstruction']['mse']))
        categories.extend(dict(seed=r['seed'],arm=r['arm'],**c) for c in r.get('categories',[]))
    matched.csv_write(s.results/(prefix+'_comparison.csv'),rows);matched.csv_write(s.results/(prefix+'_categories.csv'),categories)
    arms=ARMS if development else ARMS+['cap040','vanilla4'];means=[]
    for arm in arms:
        rr=[r for r in rows if r['arm']==arm]
        if len(rr)!=3:raise ValueError('Retain all three seeds for every arm')
        means.append(dict(arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in ['capture_pct','patch_AP_pct','VE_pct']}))
    matched.csv_write(s.results/(prefix+'_means.csv'),means)
    if comparisons:
        flat=[]
        for r in comparisons:
            ci=r['paired_intervals'];flat.append({k:v for k,v in r.items() if k!='paired_intervals'}|
                dict(capture_ci_low=100*ci['capture_gain_ci95'][0],capture_ci_high=100*ci['capture_gain_ci95'][1],AP_ci_low=100*ci['ap_gain_ci95'][0],AP_ci_high=100*ci['ap_gain_ci95'][1]))
        matched.csv_write(s.results/'paired_comparison.csv',flat)
    base.plot_means(s,means,'pcgrad_'+prefix,prefix.title()+'; three seeds',LABELS)

def export(ctx,source_path):
    original=matched.export(ctx,source_path);target=original.with_name('ggf_pcgrad_'+ctx['store'].run_id[:12]+'_results.zip')
    original.replace(target);return target

failure=base.failure
