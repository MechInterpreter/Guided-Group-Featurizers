"""Run all authorized empirical checklist work in priority order; no time cutoff."""
import argparse,copy,gc,json,platform,time,traceback
from pathlib import Path
import torch
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched
import experiment as exp
import evaluation,analysis,reporting,scale_run

def earlier(ctx):
    """Read through exact compatible former-job warm-ups/endpoints, never scores.

    Old adaptive supervised-Vanilla states are intentionally not imported into
    the new fixed-weight control. New primary receipts/evaluations are rebuilt.
    """
    root=Path(ctx['options']['source_output']).parent/'GGF_COCO_1024_supervised_VOC_v1'
    if not (root/'results').exists():return
    hits=[]
    for path in (root/'results').glob('*/manifest.json'):
        rec=json.loads(path.read_text());cfg=rec['config']
        if cfg.get('wrapper_sha')=='ba9fcb0574875022d5a440a6a036b5bda69ac3fc78277eef224883c771aa1911' and cfg.get('source_run')==exp.SOURCE_RUN and cfg.get('training_identity')==ctx['packs']['train']['identity'] and cfg.get('selection_identity')==ctx['packs']['select']['identity'] and cfg.get('normalization_sha')==identity(ctx['norm']) and cfg.get('versions')==matched.versions():hits.append(rec['run_id'])
    if len(hits)>1:raise ValueError('Ambiguous compatible prior job')
    if not hits:return
    prior=matched.discover(root,ctx['local']/'read_prior_job',run=hits[0]);ctx['sources']['previous_job']=prior;ctx['stamps']+=memory.source_snapshot([prior]);original=ctx['store'].load;seen=set()
    def load(key):
        value=original(key)
        if value is None and exp.ReuseStore.reusable(key) and 'fixed_supervision' not in key:
            value=prior.load(key)
            if value is not None and key not in seen:ctx['checkpoint_stamps'].append(matched.source_key_stamp(prior,key));seen.add(key);print('COMPATIBLE PREVIOUS-JOB REUSE',key,flush=True)
        return value
    ctx['store'].load=load

def primary(ctx,cohort,coco):
    rows=[]
    for seed in exp.SEEDS:
        for kind in exp.PRIMARY:
            with reporting.cost(ctx,f'{cohort}_{seed}_{kind}_endpoint'):rows.append(exp.endpoint(ctx,seed,kind))
        exp.check_pairing(ctx,[r for r in rows if r['seed']==seed]);exp.archive(ctx)
    exp.freeze(ctx,rows,'primary_frozen')
    if cohort=='coco':pack=evaluation.coco_test(ctx)
    else:
        ctx['store'].save('voc_plan',dict(category_indices=ctx['cfg']['category_indices']));pack=ctx['packs']['eval']
    with reporting.cost(ctx,cohort+'_primary_evaluation'):evaluation.score(ctx,rows,pack,cohort);evaluation.intervals(ctx,rows,cohort)
    exp.archive(ctx)
    with reporting.cost(ctx,cohort+'_interpretability'):analysis.run(ctx,rows,pack,cohort,coco)
    ctx['store'].save('primary_complete',dict(complete=True,models=rows));reporting.checklist(ctx,cohort);exp.archive(ctx)
    print('PRIORITY EVIDENCE SAVED',cohort,str(ctx['store'].results),flush=True);return rows,pack

def main(options):
    ctx=voc=None
    try:
        if not torch.cuda.is_available():raise RuntimeError('Select a T4 GPU runtime')
        if 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Use T4 to match the saved T4 stochastic streams; cross-device continuation has not been validated')
        torch.set_num_threads(1)
        print('STAGE1 verify immutable sources, resume state and locked experimental recipe',flush=True);ctx=exp.prepare(options);earlier(ctx)
        ctx['store'].write('hardware.json',dict(python=platform.python_version(),torch=torch.__version__,GPU=torch.cuda.get_device_name(),versions=matched.versions(),
            core_linear_forward_FLOPs_per_patch=4*768*1024*12,FLOPs_scope='Encoder/decoder dense linear multiply-adds only; excludes Top-k, losses, backward/controller, mapping/evaluation; not total method FLOPs'))
        print('STAGE2 primary COCO controlled comparison, then numerical/qualitative interpretation',flush=True)
        rows,cp=primary(ctx,'coco',None)
        # Image/backbone assets are loaded once, after primary scores are durable.
        from backbone import annotations,backbone_preflight
        coco=annotations(ctx,options['source_output']);backbone_preflight(ctx,coco,options['source_output'])
        # Atlas needs image metadata; quantitative analysis above does not.
        with reporting.cost(ctx,'coco_direction_atlas'):analysis.run(ctx,rows,cp,'coco',coco)
        reporting.checklist(ctx,'coco');exp.archive(ctx)
        print('STAGE3 actual VOC training with fixed COCO architecture/budget; own FIT/dev/eval split',flush=True)
        from voc_training import prepare as voc_prepare
        with reporting.cost(ctx,'VOC_assets_features'):voc=voc_prepare(ctx,coco)
        vrows,vp=primary(voc,'voc',coco)
        print('STAGE4 additional published supervised method: pinned released Concept-SAE task adaptation',flush=True)
        import concept_sae
        for cc,pp,cohort in [(ctx,cp,'coco'),(voc,vp,'voc')]:
            with reporting.cost(cc,cohort+'_Concept_SAE'):concept_sae.run(cc,pp,cohort)
            reporting.checklist(cc,cohort);exp.archive(cc)
        print('STAGE5 supporting released/paper-equation Grassmannian and Group Lasso on COCO',flush=True)
        for kind in exp.EXTRA:
            with reporting.cost(ctx,kind+'_endpoints'):more=[exp.endpoint(ctx,seed,kind) for seed in exp.SEEDS]
            rows.extend(more);exp.freeze(ctx,rows,'frozen_through_'+kind)
            with reporting.cost(ctx,kind+'_evaluation'):evaluation.score(ctx,more,cp,'coco');evaluation.intervals(ctx,more,'coco')
            with reporting.cost(ctx,kind+'_interpretability'):analysis.run(ctx,more,cp,'coco',coco)
            reporting.checklist(ctx,'coco');exp.archive(ctx)
        exp.freeze(ctx,rows,'all_models_frozen');matched.assert_sources(ctx)
        reporting.checklist(voc,'voc',True);vzip=exp.archive(voc,True)
        ctx['store'].write('VOC_result_location.json',dict(results=str(voc['store'].results),archive=str(vzip),checkpoints=str(voc['store'].root)))
        ctx['store'].save('VOC_completed_evidence',dict(complete=True,run_id=voc['store'].run_id,results=str(voc['store'].results),archive=str(vzip)))
        reporting.checklist(ctx,'coco',True);path=exp.archive(ctx,True)
        Path(options['receipt']).write_text(json.dumps(dict(complete=True,archive=str(path),results=str(ctx['store'].results),VOC_archive=str(vzip),VOC_results=str(voc['store'].results))))
        print('ALL PLANNED EMPIRICAL WORK COMPLETE. Paper editing and mentor review remain.',flush=True)
    except BaseException as exc:
        for cc in [voc,ctx]:
            if cc is None:continue
            try:
                scale_run.flush(cc);cc['store'].write('failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));exp.archive(cc)
            except BaseException as cleanup:print('CHECKPOINT CLEANUP NEEDS ATTENTION',type(cleanup).__name__,str(cleanup),flush=True)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(json.loads(Path(a.config).read_text()))
