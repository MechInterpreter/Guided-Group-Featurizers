"""Three-seed GGF component ablation. Reuse pilot warm-ups and tested controls."""
import argparse,copy,gc,json,os,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from types import MethodType
import numpy as np
import torch
import pilot_inputs as source
from coco_baselines import training
from coco_baselines.storage import identity,sha_file
from coco_adaptation import inputs
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched,width_balance,study
from coco_ggf.drive_recovery import install_drive_recovery
from scale_fit import fit
from scale_run import flush
from scale_models import model
from scale_metrics import evaluate

SEEDS=[21,22,23]
KINDS=['released_vanilla','mask_only','fixed_guided','agf_g']
LABELS=dict(released_vanilla='Vanilla BSF',mask_only='Mask + reconstruction',fixed_guided='Fixed-weight guidance',agf_g='GGF')
CONTROL_PREFIX='445e2b6040fc'
# The initial delivered protocol identifies the training math. Later figure-only
# typography changes must not create another set of six training jobs.
TRAINING_PROTOCOL_SOURCE='33a2dc0ae36d8b757764486aff416e96c3437851dda0b930f151835327a1a149'


def fixed_policy(history):
    # Use only training-generated weights at these already logged steps.
    # Validation scores in the same records never enter this calculation.
    steps=[500,1000,1500,2000,2500,3000]
    selected=[r for r in history if r['update'] in steps]
    if [r['update'] for r in selected]!=steps:raise ValueError('Require all six logged TRAIN-weight records')
    weights=np.asarray([r['weights'] for r in selected],float)
    if weights.shape!=(6,3) or not np.isfinite(weights).all() or not np.allclose(weights[:,0],1.) or (weights[:,1:]<=0).any() or (weights[:,1:]>.600001).any():
        raise ValueError('Invalid logged adaptive weights')
    return dict(rule='Per-seed arithmetic mean of weights USED at logged updates500,1000,1500,2000,2500,3000; not a full-step time average',
        logged_updates=steps,logged_weights=weights.tolist(),fixed_weights=weights.mean(0).tolist(),validation_or_test_scores_used=False)


def install_transport(store):
    install_drive_recovery(store);prior=store.save
    def save(self,key,value,drive=True):
        if drive and isinstance(value,dict) and 'updates' in value and 'target' in value:
            drive=value['updates']%500==0 or value['updates']>=value['target']
        return prior(key,value,drive=drive)
    store.save=MethodType(save,store)
    store.write('drive_recovery.json',dict(local_checkpoint_updates=100,drive_checkpoint_updates=500,force_save_at_stop_and_boundaries=True,
        model_updates_unchanged=True,maximum_abrupt_loss_updates=499,source_stores_read_only=True))


def prepare(o,require_drive=True):
    # This read-only helper verifies all six completed pilot endpoints. Its own
    # writable input-verification journal is isolated beneath our new output.
    base_o=dict(o,wrapper_sha=TRAINING_PROTOCOL_SOURCE,output=str(Path(o['output'])/'verified_inputs'),local=str(Path(o['local'])/'inputs'))
    base=source.prepare(base_o,require_drive)
    parent=base['parent'];local=Path(o['local']);policies={};warms={};rows=base['rows']
    original_rule=copy.deepcopy(parent.config['recipe'])
    for seed in SEEDS:
        guided=next(r for r in rows if r['seed']==seed and r['kind']=='agf_g')
        path=parent.results/'histories'/(guided['key']+'.json')
        policy=fixed_policy(json.loads(path.read_text()));policy.update(seed=seed,history_sha256=sha_file(path))
        policies[str(seed)]=policy;base['stamps'].append(dict(path=str(path),sha256=sha_file(path)))
        key=f'g4096_k32_s{seed}_mask_warm';state=memory.required(parent,key)
        if (state['updates'],state['target'],state['groups'],state['active'],state['method'])!=(7640,7640,4096,32,'released_vanilla'):
            raise ValueError('Wrong saved masked warm-up')
        warm_targets=memory.required(parent,f'g4096_k32_s{seed}_mask_targets')
        if identity(warm_targets)!=guided['targets_sha']:raise ValueError('Pilot TRAIN readouts differ from saved warm-up mappings')
        p=local/f'warm_s{seed}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(state['weights'],p)
        warms[str(seed)]=dict(key=key,weights_sha=training.weight_sha(state['weights']),file_sha=sha_file(p),updates=7640)
        base['checkpoint_stamps'].append(matched.source_key_stamp(parent,key));del state;gc.collect()
    controls=matched.discover(o['control_output'],local/'read_test_controls',prefix=CONTROL_PREFIX)
    completed=memory.required(controls,'test_complete')
    if not completed['complete'] or len(completed['models'])!=6 or completed['fresh_holdout_opened']:
        raise ValueError('Finish the exact six-model pilot test evaluation first')
    if identity(controls.config['cfg'])!=identity(base['cfg']) or controls.config['test_identity']!=base['test_stamp']:
        raise ValueError('Saved controls use another test cohort or metric configuration')
    for r in completed['models']:
        old=next(x for x in rows if (x['seed'],x['kind'])==(r['seed'],r['kind']))
        if any(r[k]!=old[k] for k in ['weights_sha','targets_sha','total_training_updates']):raise ValueError('Saved control endpoint changed')
    base['stamps']+=memory.source_snapshot([controls]);base['checkpoint_stamps'].append(matched.source_key_stamp(controls,'test_complete'))
    source.source_protocol.cache.safe_destination(o['output'],[o['pilot_output'],o['control_output'],*o['paths'].values()])
    payload=dict(study='ggf_three_seed_components',wrapper_sha=TRAINING_PROTOCOL_SOURCE,core_sha=source.CORE_SHA,pilot_run=parent.run_id,
        control_run=controls.run_id,versions=matched.versions(),seeds=SEEDS,groups=4096,width=12,k=32,cap=.6,
        original_training_rule=original_rule,policies=policies,warms=warms,cohort=base['test_stamp'],cfg=base['cfg'],
        new_warmups=0,reused_controls=6,new_fits=6,new_updates=18000,new_exchanges=0,fresh_holdout_opened=False)
    s=inputs.JournalStore(o['output'],payload,local/'rescue',require_drive);install_transport(s)
    provenance=json.loads((base['sources']['original'].results/'provenance.json').read_text())
    ctx=dict(base,store=s,local=local,base_context=base,controls=controls,policies=policies,warms=warms,provenance=provenance,
        active_key=None,rule=dict(original_rule,local_save_every=100,drive_save_every=500),balance_recipe=copy.deepcopy(width_balance.settings()['balance']))
    ctx['packs']={}
    for split in ['train','select']:
        ctx['packs'][split]=source.source_protocol.cache.cached_split(ctx,split,o['cache_local'])
        expected=parent.config['training_identity' if split=='train' else 'selection_identity']
        if ctx['packs'][split]['identity']!=expected:raise ValueError('Cached training/development identity differs from pilot')
    s.write('protocol.json',payload);s.write('fixed_weight_policy.json',dict(policies=policies,chosen_before_new_fits=True))
    s.write('active_source.json',dict(actual_archive_sha=o['wrapper_sha'],training_protocol_archive_sha=TRAINING_PROTOCOL_SOURCE,
        scope='Training math and saved-fit identities unchanged; figure renderer may receive typography updates.'))
    matched.assert_sources(ctx);return ctx


def train_components(ctx):
    s=ctx['store'];old=s.load('components_frozen')
    if old is not None:return verify_freeze(ctx)
    rows=copy.deepcopy(ctx['base_context']['rows'])
    for seed in SEEDS:
        parent=next(r for r in rows if r['seed']==seed and r['kind']=='agf_g')
        expected_streams=json.loads((ctx['base_context']['local']/'verified_weights'/f's{seed}_agf_g.json').read_text())['streams']
        for kind in ['mask_only','fixed_guided']:
            key=f'{kind}_s{seed}';p=ctx['local']/f'warm_s{seed}.pt';weights=torch.load(p,map_location='cpu',weights_only=True)
            if training.weight_sha(weights)!=ctx['warms'][str(seed)]['weights_sha']:raise ValueError('Warm-up weight cache changed')
            sub=dict(ctx,rule=dict(ctx['rule'],fixed_weights=ctx['policies'][str(seed)]['fixed_weights']))
            start=time.monotonic();ctx['active_key']=key
            state=fit(sub,key,'released_vanilla',4096,32,seed,weights,3000,3000,
                targets=parent['targets'] if kind=='fixed_guided' else None,cap=.6,learning_rate=.0003,fixed=kind=='fixed_guided',coefficient=.001,stream=1)
            ctx['active_key']=None
            if state['updates']!=3000 or [state['sampled_batches_sha'],state['sampled_noise_sha']]!=expected_streams:
                raise ValueError('Ablation update budget or paired training stream differs')
            row=dict(parent,key=key,kind=kind,weights_sha=training.weight_sha(state['weights']),cap=None,
                head_parameters=150 if kind=='fixed_guided' else 0,
                fixed_weights=ctx['policies'][str(seed)]['fixed_weights'] if kind=='fixed_guided' else [1.,0.,0.],
                additional_updates=3000,total_training_updates=10640,origin='new_component',warm_weights_sha=ctx['warms'][str(seed)]['weights_sha'])
            m=model('released_vanilla',4096,12,32,ctx['packs']['train']['x'].shape[-1],.001).to(ctx['device']);m.load_state_dict(state['weights'])
            row['development']=evaluate(m,ctx['packs']['select'],row['targets'],ctx['cfg'],ctx['device'])
            rows.append(row);s.write('training_progress.json',dict(complete=False,models=rows));s.event('component_complete',seed=seed,kind=kind,
                wall_minutes=(time.monotonic()-start)/60,capture=100*row['development']['macro_capture'],AP=100*row['development']['macro_ap'],VE=100*row['development']['reconstruction']['ve'])
            del state,weights,m;gc.collect();torch.cuda.empty_cache()
    expected={(seed,k) for seed in SEEDS for k in KINDS}
    if len(rows)!=12 or {(r['seed'],r['kind']) for r in rows}!=expected:raise ValueError('Require all twelve component/control endpoints')
    frozen=dict(complete=True,models=rows,new_training_updates=18000,new_warmups=0,reused_controls=6,fixed_policies=ctx['policies'],test_features_opened=False)
    s.save('components_frozen',frozen);s.write('components_frozen.json',frozen);return verify_freeze(ctx)


def verify_freeze(ctx):
    frozen=memory.required(ctx['store'],'components_frozen');rows=frozen['models']
    if not frozen['complete'] or len(rows)!=12 or {(r['seed'],r['kind']) for r in rows}!={(s,k) for s in SEEDS for k in KINDS}:
        raise ValueError('Freeze all twelve endpoints before new test inference')
    for row in rows:
        if row['kind'] not in ['mask_only','fixed_guided']:continue
        state=memory.required(ctx['store'],row['key'])
        if state['updates']!=3000 or training.weight_sha(state['weights'])!=row['weights_sha'] or identity(row['targets'])!=row['targets_sha']:
            raise ValueError('Frozen component identity changed')
        del state
    matched.assert_sources(ctx);return frozen


def evaluate_all(ctx):
    frozen=verify_freeze(ctx);s=ctx['store'];rows=frozen['models']
    # Six controls already evaluated; import exact compact audits without inference.
    cohort=identity(dict(test=ctx['test_stamp'],cfg=ctx['cfg'],normalization=identity(ctx['norm'])))
    for row in rows:
        if row['kind'] not in source.KINDS:continue
        tag=f's{row["seed"]}_{row["kind"]}';key='audit_'+tag;saved=memory.required(ctx['controls'],key)
        if saved['signature']!=identity(dict(weights=row['weights_sha'],targets=row['targets_sha'],cohort=cohort)):raise ValueError('Control audit signature changed')
        if s.load(key) is None:s.save(key,saved)
    pack=source.load_cached_test(ctx['base_context']);reports=[]
    for row in rows:
        tag=f's{row["seed"]}_{row["kind"]}';key='audit_'+tag;signature=identity(dict(weights=row['weights_sha'],targets=row['targets_sha'],cohort=cohort));saved=s.load(key)
        if saved is None:
            state=memory.required(s,row['key']);m=model('released_vanilla',row['groups'],row['rank'],row['active_groups'],pack['x'].shape[-1],.001).to(ctx['device']);m.load_state_dict(state['weights']);del state
            with matched.progress(ctx,m,tag):report=evaluate(m,pack,row['targets'],ctx['cfg'],ctx['device'],compact=True)
            compact=report.pop('_compact');native=next(x['reference_warm_mse'] for x in rows if x['seed']==row['seed'] and x['kind']=='released_vanilla')
            report.update(seed=row['seed'],kind=row['kind'],weights_sha=row['weights_sha'],targets_sha=row['targets_sha'],total_training_updates=10640,
                native_warm_eligible=row['development']['reconstruction']['mse']<=native*1.02)
            saved=dict(signature=signature,report=report,compact=compact);s.save(key,saved);s.write('reports/'+tag+'.json',report)
            del m,compact;gc.collect();torch.cuda.empty_cache()
        if saved['signature']!=signature:raise ValueError('Saved component test audit changed')
        reports.append(saved['report']);del saved
    ctx['base_context']['packs'].clear();del pack;gc.collect();means=[]
    table=[dict(seed=r['seed'],kind=r['kind'],label=LABELS[r['kind']],capture_pct=100*r['macro_capture'],patch_AP_pct=100*r['macro_ap'],
        VE_pct=100*r['reconstruction']['ve'],native_warm_eligible=r['native_warm_eligible']) for r in reports]
    for kind in KINDS:
        rr=[r for r in table if r['kind']==kind];entry=dict(kind=kind,label=LABELS[kind],seeds=3)
        for key in ['capture_pct','patch_AP_pct','VE_pct']:
            stat=source.seed_stats([r[key] for r in rr]);entry.update({key:stat['mean'],key+'_seed_sd':stat['seed_sd']})
        means.append(entry)
    matched.csv_write(s.results/'test_models.csv',table);matched.csv_write(s.results/'test_means.csv',means)
    s.write('point_metrics.json',dict(complete=True,models=reports,means=means));print('COMPONENT TEST MEANS',json.dumps(means),flush=True)
    comparisons=[]
    for a,b,component in [('mask_only','released_vanilla','Initialization'),('fixed_guided','mask_only','Fixed object guidance'),('agf_g','fixed_guided','Adaptive schedule versus matched-scale fixed weights')]:
        for seed in SEEDS:
            aa=memory.required(s,f'audit_s{seed}_{a}');bb=memory.required(s,f'audit_s{seed}_{b}');key=f'pair_s{seed}_{a}_vs_{b}'
            signature=identity(dict(a=aa['signature'],b=bb['signature'],cfg=ctx['cfg']));saved=s.load(key)
            if saved is None:
                s.event('paired_intervals_start',seed=seed,treatment=a,control=b);saved=dict(signature=signature,intervals=study.paired(aa['compact'],bb['compact'],ctx['cfg']));s.save(key,saved)
            if saved['signature']!=signature:raise ValueError('Paired component signature changed')
            ar,br=aa['report'],bb['report'];ci=saved['intervals']
            comparisons.append(dict(component=component,seed=seed,treatment=a,control=b,capture_gain_pp=100*(ar['macro_capture']-br['macro_capture']),
                AP_gain_pp=100*(ar['macro_ap']-br['macro_ap']),VE_gain_pp=100*(ar['reconstruction']['ve']-br['reconstruction']['ve']),
                capture_ci_low=100*ci['capture_gain_ci95'][0],capture_ci_high=100*ci['capture_gain_ci95'][1],AP_ci_low=100*ci['ap_gain_ci95'][0],AP_ci_high=100*ci['ap_gain_ci95'][1],
                valid_bootstraps=ci['valid_bootstrap_replicates'],invalid_bootstraps=ci['invalid_full_cohort_replicates']))
            matched.csv_write(s.results/'component_paired_images.csv',comparisons);del aa,bb,saved;gc.collect()
    seed_means=[]
    for component in sorted({r['component'] for r in comparisons}):
        rr=[r for r in comparisons if r['component']==component];entry=dict(component=component)
        for key in ['capture_gain_pp','AP_gain_pp','VE_gain_pp']:
            stat=source.seed_stats([r[key] for r in rr]);entry.update({key:stat['mean'],key+'_seed_sd':stat['seed_sd'],key+'_seed_ci_low':stat['seed_ci95'][0],key+'_seed_ci_high':stat['seed_ci95'][1]})
        seed_means.append(entry)
    matched.csv_write(s.results/'component_paired_seeds.csv',seed_means)
    matched.csv_write(s.results/'test_categories.csv',[dict(seed=r['seed'],kind=r['kind'],**v) for r in reports for v in r['categories']])
    result=dict(complete=True,models=reports,means=means,comparisons=comparisons,seed_means=seed_means,new_updates=18000,new_warmups=0,reused_controls=6,
        test_previously_inspected=True,fresh_holdout_opened=False,seven_seed_study=False,exchanges=False,
        scope='Fixed weights use TRAIN logs only. Matched capacity, warm-ups, TRAIN readouts, image/noise streams and update budgets; supervision and compute differ. No test selection.')
    s.save('components_complete',result);s.write('component_summary.json',result);plot(s.results/'figures',means);matched.assert_sources(ctx);return result


def plot(folder,means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(exist_ok=True);fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axs,['capture_pct','patch_AP_pct','VE_pct'],['Capture','Patch AP','Reconstruction VE']):
        bars=ax.barh(range(4),[r[key] for r in means],color=['#879cad','#9ab8b4','#9b82b2','#087f83'],xerr=[r[key+'_seed_sd'] for r in means],capsize=3)
        ax.bar_label(bars,labels=[f'{r[key]:.2f}%' for r in means],padding=4);ax.set(yticks=range(4),yticklabels=[r['label'] for r in means],xlabel='Percent; error bars = seed SD',title=title,xlim=(min(0,min(r[key] for r in means)-5),max(105,max(r[key] for r in means)+15)))
        ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    fig.suptitle('GGF components / three seeds / 4,096 groups /12D /Top-32 /10,640 updates')
    for ext in ['pdf','png']:fig.savefig(folder/('component_comparison.'+ext),dpi=200)
    plt.close(fig)


def export(ctx,o,complete):
    s=ctx['store'];s.write('completion_status.json',dict(complete=complete,training_complete=s.load('components_frozen') is not None,
        evaluation_complete=s.load('components_complete') is not None,new_warmups=0,reused_controls=6,planned_new_updates=18000,exchanges=False,seven_seed_study=False))
    path=s.output/f'ggf_components_{s.run_id[:12]}_results.zip';hashes={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(s.results.rglob('*')):
            if p.is_file():
                n='results/'+p.relative_to(s.results).as_posix();z.write(p,n);hashes[n]=sha_file(p)
        for n,k in [('core_source.zip','source_zip'),('component_source.zip','wrapper_zip')]:z.write(o[k],n);hashes[n]=sha_file(o[k])
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    import hashlib
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in hashes.items()):raise ValueError('Export checksum failed')
    return path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);o=json.loads(Path(ap.parse_args().config).read_text())
    if o['source_sha']!=source.CORE_SHA or matched.versions()!=o['expected_versions'] or o['fresh']:raise ValueError('Pinned sources/versions and cached-test mode required')
    w=Path(__file__).resolve().parent
    subprocess.check_call([sys.executable,'-m','pytest','-q',str(w/'test_components.py'),'--disable-warnings','--basetemp',str(Path(o['local'])/('selftest_'+os.urandom(4).hex()))],cwd=w)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Select T4')
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    ctx=None;stage='prepare_saved_inputs'
    try:
        ctx=prepare(o);ctx['store'].results.joinpath('PROTOCOL.md').write_bytes((w/'PROTOCOL.md').read_bytes())
        stage='figure1_inference'
        from figure_current import make as make_figure
        make_figure(ctx);export(ctx,o,False)
        if o.get('figure_only',False):
            print('FIGURE ONLY COMPLETE. No ablation training started.',flush=True);return
        stage='six_component_fits';train_components(ctx);export(ctx,o,False)
        ctx['packs'].clear();gc.collect();stage='component_evaluation_and_intervals';result=evaluate_all(ctx);archive=export(ctx,o,True)
        Path(o['receipt']).write_text(json.dumps(dict(complete=True,result_zip=str(archive),results=str(ctx['store'].results),wrapper_sha=o['wrapper_sha'],new_warmups=0,reused_controls=6,new_fits=6,seven_seed_study=False),indent=2))
        print('GGF COMPONENT ABLATION COMPLETE; STOPPED.',json.dumps(result['means']),archive,flush=True)
    except BaseException as exc:
        if ctx is not None:
            try:flush(ctx);ctx['store'].write('interruption.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));print('Partial evidence:',export(ctx,o,False),flush=True)
            except BaseException as secondary:print('Save/export issue:',repr(secondary),flush=True)
        raise


if __name__=='__main__':main()
