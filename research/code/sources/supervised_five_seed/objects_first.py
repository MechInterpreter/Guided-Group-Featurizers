"""Resume the pinned, already-opened five-seed test; object metrics before CPU VE."""
import copy,gc,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
import supervised_study as study
import five_seed
from precision_retry import EXPECTED_RUN,BASE_SHA
from training_precision import revise
from coco_baselines import training
from coco_baselines.storage import identity,sha_file
from coco_adaptation.inputs import JournalStore
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched
from coco_ggf.drive_recovery import install_drive_recovery

HERE=Path(__file__).resolve().parent
FIVE_RUN='a9c638e1ec0bd97434d7cda7217d46ee46334c10820b393e038775d89e716822'
SEEDS=[21,22,23,24,25]

def pinned_manifest(output):
    expected=json.loads((HERE/'FIVE_MANIFEST.json').read_text())
    if expected['run_id']!=FIVE_RUN or identity(expected['config'])!=FIVE_RUN:
        raise ValueError('Pinned five-seed identity is invalid')
    path=Path(output)/'results'/FIVE_RUN[:20]/'manifest.json'
    if not path.exists() or json.loads(path.read_text())!=expected:
        raise ValueError('Restore the exact active five-seed evaluation; no replacement run will start')
    return expected

def check_frozen(store,manifest):
    frozen=memory.required(store,'all_models_frozen');cfg=manifest['config']
    five_seed.validate_rows(frozen['models'])
    fields=['seed','arm','key','weights_sha','targets_sha','selected_total_updates','total_training_updates']
    observed=[{k:r[k] for k in fields} for r in frozen['models']]
    if (not frozen.get('complete') or frozen['protocol']!=cfg['evaluation_protocol'] or observed!=cfg['models']
            or frozen['settings_sha']!=cfg['settings_sha'] or frozen['holdout_plan_sha']!=cfg['holdout_plan_sha']):
        raise ValueError('Frozen five-seed weights, readouts, settings or cohort changed')
    return frozen

def evidence_files(ctx,mode):
    hashes=json.loads((HERE/'REVISION_HASHES.json').read_text())
    folder=ctx['store'].results/'execution_objects_first'/identity(hashes)[:16]
    folder.mkdir(parents=True,exist_ok=True)
    for name in list(hashes)+['REVISION_HASHES.json']:(folder/name).write_bytes((HERE/name).read_bytes())
    ctx['store'].write('objects_first_execution_'+mode+'.json',dict(
        run_id=FIVE_RUN,mode=mode,revision=hashes,settings_and_models_changed=False,
        test_cohort_changed=False,training_performed=False,
        CPU_reconstruction='Same joint NNLS objective, strict 1e-5 KKT, active-set maxiter 20000. CPU-only mode bypasses GPU FISTA and uses float64 active-set directly; previous verified image-prefix results are retained.',
        precision_note='Solver backend/intermediate precision changes are execution provenance, not new training or a relaxed tolerance. Nonunique NNLS codes can have different activity; the reconstructed objective remains the verified quantity.'))

def resume_gpu(ctx,options):
    manifest=pinned_manifest(options['output'])
    if ctx['store'].run_id!=EXPECTED_RUN:raise ValueError('Wrong parent training run')
    lock=json.loads((ctx['store'].results/'five_seed_scope_lock.json').read_text())
    if lock['evaluation_run']!=FIVE_RUN:raise ValueError('The five-seed scope lock differs')
    local=Path(options['local'])/'five_seed_evaluation'
    job=JournalStore(options['output'],manifest['config'],local/'rescue',True);install_drive_recovery(job)
    frozen=check_frozen(job,manifest)
    store=five_seed.EvaluationStore(job,ctx['store'],frozen['models'])
    out=dict(ctx,store=store,local=local,active_key=None)
    if identity(ctx['norm'])!=manifest['config']['normalization_sha']:raise ValueError('Normalization changed')
    five_seed.install_evaluation_scope(frozen['models'],frozen['protocol'])
    study.verify_frozen(out)
    # Small CPU handoff; no training activations or GPU backbone are needed later.
    saved=dict(run_id=FIVE_RUN,cfg=ctx['cfg'],norm=ctx['norm'],parent_run=EXPECTED_RUN,
        protocol=frozen['protocol'],category_indices=ctx['cfg']['category_indices'])
    previous=store.load('objects_first_cpu_context')
    if previous is not None and identity(previous)!=identity(saved):raise ValueError('CPU handoff context changed')
    store.save('objects_first_cpu_context',saved)
    evidence_files(out,'GPU_objects')
    return out

def resume_cpu(options):
    manifest=pinned_manifest(options['output'])
    local=Path(options['local'])/'five_seed_evaluation'
    job=JournalStore(options['output'],manifest['config'],local/'rescue',True);install_drive_recovery(job)
    frozen=check_frozen(job,manifest)
    saved=memory.required(job,'objects_first_cpu_context')
    if saved['run_id']!=FIVE_RUN or identity(saved['norm'])!=manifest['config']['normalization_sha']:
        raise ValueError('CPU handoff identity/normalization changed')
    parent=memory.FrozenStore(Path(options['output'])/'checkpoints'/EXPECTED_RUN[:20],local/'read_training')
    ctx=dict(store=five_seed.EvaluationStore(job,parent,frozen['models']),local=local,
        cfg=saved['cfg'],norm=saved['norm'],device='cpu',active_key=None,stamps=[],checkpoint_stamps=[],packs={})
    five_seed.install_evaluation_scope(frozen['models'],frozen['protocol'])
    # CPU never loads the large anchor weights; their finished object reports
    # are required from the GPU pass instead. Native weights are verified on read.
    evidence_files(ctx,'CPU_intervals_and_reconstruction')
    return ctx

def cached_test_pack(ctx):
    s=ctx['store'];plan=memory.required(s,'fresh_holdout_plan')
    if identity(plan)!=json.loads((HERE/'FIVE_MANIFEST.json').read_text())['config']['holdout_plan_sha']:
        raise ValueError('Cached test cohort differs from the frozen cohort')
    # Read existing feature shards only. Missing shards fail; no extraction,
    # backbone, HF authentication or new holdout selection is possible here.
    pack=memory.normalized_pack(s,'test',dict(splits=plan['splits']),ctx['norm'],ctx['local'],ctx['cfg']['extract_chunk'])
    pack['identity']=identity(dict(ids=pack['ids'],normalization=identity(ctx['norm'])))
    from coco_adaptation import coverage
    observed=coverage.evidence(pack,ctx['cfg']).all(2).sum(0)
    if not np.array_equal(observed,np.asarray(plan['coverage']['split_counts']['test'])):
        raise ValueError('Cached mask coverage changed')
    print('REUSED EXACT CACHED TEST:',len(pack['ids']),'images; no feature extraction.',flush=True)
    return pack

def audit_signature(ctx,row,pack):
    stamp=identity(dict(ids=pack['ids'],normalization=identity(ctx['norm']),cfg=ctx['cfg']))
    return identity(dict(weights=row['weights_sha'],targets=row['targets_sha'],cohort=stamp,readout=row.get('readout','native_bsf')))

def object_key(row):return f'objects_s{row["seed"]}_{row["arm"]}'

def object_evaluator():
    # Keep original object arithmetic, raw-cache identity and cheap GGF VE.
    # Remove only the blocking final SLiCS reconstruction call.
    return revise(study.evaluate_state,[
        ("    if compact and isinstance(m,NativeSLiCS):\n        report['reconstruction']=final_reconstruction(ctx,m,pack,row['weights_sha'])",
         "    # Joint SLiCS reconstruction is handled by the separate CPU notebook.")])

def objects_pass(ctx,pack,evaluator=None):
    s=ctx['store'];frozen=study.verify_frozen(ctx);evaluator=evaluator or object_evaluator()
    index=s.load('objects_signature_index') or {};oldindex=s.load('audit_signature_index') or {}
    models=[]
    for row in frozen['models']:
        key=object_key(row);sig=audit_signature(ctx,row,pack);saved=s.load(key)
        if saved is None:
            old=s.load(f'audit_s{row["seed"]}_{row["arm"]}')
            if old is None and sig in oldindex:old=memory.required(s,oldindex[sig])
            if old is None and sig in index:old=memory.required(s,index[sig])
            if old is not None:
                if old['signature']!=sig:raise ValueError('Completed audit signature changed')
                saved=dict(old,row=row,reused_inference=True)
                print('REUSED COMPLETED OBJECT METRICS:',row['seed'],row['arm'],flush=True)
            else:
                report=evaluator(ctx,row,pack,True);compact=report.pop('_compact')
                saved=dict(row=row,signature=sig,report=report,compact=compact)
            s.save(key,saved);index[sig]=key;s.save('objects_signature_index',index)
        if saved['signature']!=sig:raise ValueError('Object audit changed')
        models.append(dict(row=row,report=saved['report']))
        s.write('object_reports/'+key+'.json',dict(row=row,report=saved['report']))
        s.event('five_seed_object_model_complete',seed=row['seed'],arm=row['arm'],completed=len(models),total=23,
            capture=100*saved['report']['macro_capture'],AP=100*saved['report']['macro_ap'])
        del saved;gc.collect()
    return object_summary(ctx,models)

def object_summary(ctx,models,pairs=None):
    from coco_ggf.matched_baselines import csv_write
    five_seed.validate_rows([r['row'] for r in models])
    s=ctx['store'];table=[]
    for item in models:
        r=item['row'];v=item['report'];rec=v['reconstruction']
        table.append(dict(seed=r['seed'],arm=r['arm'],capture_pct=100*v['macro_capture'],patch_AP_pct=100*v['macro_ap'],
            VE_pct=None if rec.get('ve') is None else 100*rec['ve'],reconstruction_status=rec.get('status','computed'),
            groups=r['groups'],selected_updates=r['selected_total_updates'],total_fitted_updates=study.TOTAL))
    means=[]
    for arm in study.ARMS+['current_ggf']:
        groups=[('capacity_anchor_three',study.SCREEN)] if arm=='current_ggf' else [('all_five',SEEDS),('development_tuning_orders',SEEDS[:3]),('additional_orders',SEEDS[3:])]
        for label,seeds in groups:
            rr=[r for r in table if r['arm']==arm and r['seed'] in seeds]
            if sorted(r['seed'] for r in rr)!=seeds:raise ValueError('Incomplete five-seed summary')
            out=dict(arm=arm,seed_group=label,seeds=len(seeds))
            for metric in ['capture_pct','patch_AP_pct']:
                st=study.seed_stat([r[metric] for r in rr]);out[metric]=st['mean'];out[metric+'_sd']=st['sd']
                out[metric+'_ci_low']=st['ci_low'];out[metric+'_ci_high']=st['ci_high']
            out['VE_pct']=float(np.mean([r['VE_pct'] for r in rr])) if all(r['VE_pct'] is not None for r in rr) else None
            means.append(out)
    stats=[]
    for a,b in [('matched_ggf','slics_object_partial'),('matched_ggf_final','slics_final_partial')]:
        for metric in ['capture_pct','patch_AP_pct']:
            differences=[next(r[metric] for r in table if r['seed']==seed and r['arm']==a)-next(r[metric] for r in table if r['seed']==seed and r['arm']==b) for seed in SEEDS]
            stats.append(dict(treatment=a,control=b,metric=metric,**study.seed_stat(differences),
                primary=a=='matched_ggf',scope='Training-order variation on fixed cohort/common initializer; not population uncertainty'))
    primary=[r for r in stats if r['primary']]
    for row,p in zip(primary,study.holm([r['p_two_sided'] for r in primary])):row['holm_p_two_sided']=p
    for row in stats:row.setdefault('holm_p_two_sided',None)
    csv_write(s.results/'object_models.csv',table);csv_write(s.results/'object_means.csv',means)
    csv_write(s.results/'object_paired_seeds.csv',stats)
    if pairs:csv_write(s.results/'object_paired_images.csv',pairs)
    out=dict(object_metrics_complete=True,reconstruction_complete=False,seed_intervals_complete=True,
        whole_image_intervals_complete=pairs is not None,seeds=SEEDS,means=means,seed_intervals=stats,image_intervals=pairs or [],
        original_evaluation_protocol=study.protocol(),note='All frozen models retained; pending SLiCS VE is null, never zero. Main figure uses completed training. No retuning or readout fitting.')
    s.save('objects_phase_complete',out);s.write('objects_summary.json',out);object_plot(ctx,means)
    return out

def object_plot(ctx,means):
    import matplotlib;matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    rows=[r for r in means if r['seed_group']=='all_five' and r['arm'] in ['matched_ggf_final','slics_final_partial']]
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for ax,metric,title in zip(axes,['capture_pct','patch_AP_pct'],['Capture','Patch AP']):
        bars=ax.barh(range(2),[r[metric] for r in rows],xerr=[r[metric+'_sd'] for r in rows],color=['#00877c','#9576ae'],capsize=3)
        ax.set_yticks(range(2),['GGF (5 seeds)','SLiCS patch adaptation (5 seeds)']);ax.invert_yaxis()
        ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set_xlim(0,105);ax.set_title(title);ax.set_xlabel('Percent; error bars = seed SD')
    fig.suptitle('Completed-training comparison | unchanged frozen test cohort')
    folder=ctx['store'].results/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/('objects_first.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

def load_object_models(ctx):
    # GPU already verified all model hashes. CPU rechecks the frozen catalog and
    # cached report pins without loading 4,096-group GPU anchors just for VE.
    rows=check_frozen(ctx['store'],json.loads((HERE/'FIVE_MANIFEST.json').read_text()))['models'];models=[]
    ids=memory.required(ctx['store'],'fresh_holdout_plan')['splits']['test']
    pack=dict(ids=ids,identity=identity(dict(ids=ids,normalization=identity(ctx['norm']))))
    for row in rows:
        saved=memory.required(ctx['store'],object_key(row))
        if saved['row']['weights_sha']!=row['weights_sha'] or saved['row']['targets_sha']!=row['targets_sha']:
            raise ValueError('Cached object weights/readouts changed')
        if saved['signature']!=audit_signature(ctx,row,pack):raise ValueError('Cached object data/config signature changed')
        models.append(dict(row=row,report=saved['report']))
    return models

def cpu_intervals(ctx,models):
    s=ctx['store'];pairs=[]
    for seed in SEEDS:
        for a,b in [('matched_ggf','slics_object_partial'),('matched_ggf_final','slics_final_partial')]:
            key=f'image_ci_s{seed}_{a}_{b}';saved=s.load(key)
            if saved is None:
                x=memory.required(s,object_key(dict(seed=seed,arm=a)));y=memory.required(s,object_key(dict(seed=seed,arm=b)))
                s.event('CPU_bootstrap_start',seed=seed,treatment=a,control=b)
                ci=study.study.paired(x['compact'],y['compact'],ctx['cfg'])
                saved=dict(seed=seed,treatment=a,control=b,capture_ci_low_pp=100*ci['capture_gain_ci95'][0],capture_ci_high_pp=100*ci['capture_gain_ci95'][1],
                    AP_ci_low_pp=100*ci['ap_gain_ci95'][0],AP_ci_high_pp=100*ci['ap_gain_ci95'][1],valid_bootstraps=ci['valid_bootstrap_replicates'],
                    invalid_bootstraps=ci['invalid_full_cohort_replicates'],scope='Paired whole-image bootstrap conditional on fitted models')
                s.save(key,saved);s.event('CPU_bootstrap_complete',seed=seed,treatment=a)
                del x,y;gc.collect()
            pairs.append(saved)
    return object_summary(ctx,models,pairs)

class CPUJointModel:
    def __init__(self,D,cfg):
        from coco_ggf.slics_repair import Solver
        self.D=D.detach().cpu().double().clone();self.cfg=cfg;self.d=D.shape[1];self.group_size=12;self.n_groups=len(D)//12
        self.solver=Solver(self.D,cfg)
    @torch.no_grad()
    def solve(self,x):
        # Exact same original NNLS objective/tolerance/maxiter. No GPU FISTA;
        # float64 coefficients avoid discarding an already verified solution.
        z,info=self.solver.solve(x.cpu().double(),force_fallback=True)
        if info['max_relative_kkt']>self.cfg['nnls_tolerance']:raise RuntimeError('CPU NNLS certificate failed')
        return z

def cpu_reconstruction(ctx,pack,models):
    # Direct final_reconstruction already accepts float64 D/coefficients and
    # uses exactly the existing key, allowing the saved image prefix to resume.
    import object_schedule
    s=ctx['store'];complete=[]
    for item in models:
        row=item['row'];report=copy.deepcopy(item['report'])
        if row['method']=='native_slics' and report['reconstruction'].get('status')!='computed':
            st=memory.required(s,row['key'])
            if training.weight_sha({'D':st['D']})!=row['weights_sha']:raise ValueError('Frozen SLiCS changed')
            model=CPUJointModel(st['D'],row['solver']);del st
            s.event('CPU_reconstruction_model_start',seed=row['seed'],arm=row['arm'])
            report['reconstruction']=object_schedule.final_reconstruction(ctx,model,pack,row['weights_sha'])
            del model;gc.collect()
        saved=dict(row=row,report=report)
        complete.append(saved);s.save(f'CPU_final_s{row["seed"]}_{row["arm"]}',saved)
        s.write('CPU_reports/'+f's{row["seed"]}_{row["arm"]}.json',saved)
    pairs=[]
    for seed in SEEDS:
        for a,b in [('matched_ggf','slics_object_partial'),('matched_ggf_final','slics_final_partial')]:
            pairs.append(memory.required(s,f'image_ci_s{seed}_{a}_{b}'))
    result=study.summarize(ctx,complete,pairs)
    s.write('CPU_reconstruction_complete.json',dict(complete=True,seeds=SEEDS,backend='float64 CPU active-set; previous verified image prefixes retained'))
    return result

def export_phase(ctx,source_zip,phase):
    s=ctx['store'];path=s.output/f'ggf_supervised_{s.run_id[:12]}_{phase}_results.zip';hashes={}
    if phase=='complete':
        if s.load('supervised_test_complete') is None:raise ValueError('Full metrics are not complete')
        s.write('completion_status.json',dict(complete=True,settings_frozen=True,all_models_frozen=True,
            test_complete=True,object_metrics_complete=True,reconstruction_complete=True,bsfs_trained=0,modified_bsf_control=False))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(s.results.rglob('*')):
            if file.is_file():
                name='results/'+file.relative_to(s.results).as_posix();z.write(file,name);hashes[name]=sha_file(file)
        z.write(source_zip,'source.zip');hashes['source.zip']=sha_file(source_zip)
        z.writestr('artifact_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise OSError('Export CRC failed')
        for name,digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise OSError('Export checksum mismatch')
    return path
