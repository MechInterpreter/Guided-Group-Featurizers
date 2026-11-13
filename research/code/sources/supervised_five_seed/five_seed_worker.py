"""Run All through seed 25 only, then the complete five-seed test evaluation."""
import argparse,copy,gc,json,traceback
from pathlib import Path
from training_precision import install
PRECISION=install()

def main():
    import torch
    from threadpoolctl import threadpool_limits
    from coco_ggf import matched_baselines as matched
    import supervised_study as study
    from supervised_live import install_observers
    import five_seed
    args=argparse.ArgumentParser();args.add_argument('--config',required=True)
    options=json.loads(Path(args.parse_args().config).read_text())
    if options['stages']!=['all']:raise ValueError('Five-seed notebook requires its complete Run All stage')
    if matched.versions()!=options['expected_versions']:raise ValueError('Pinned numerical versions differ')
    if not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime; T4 supported')
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    install_observers();ctx=None;stage='prepare'
    try:
        ctx=study.prepare(options);original_protocol=copy.deepcopy(study.protocol())
        stage='annotations';coco=study.annotations(ctx,options['source_output'])
        stage='backbone_train_replay'
        source=options['baseline_output'] if (Path(options['baseline_output'])/'input_assets/dinov3/verified.json').exists() else options['source_output']
        study.backbone_preflight(ctx,coco,source)
        # Never retune LR; it was already selected on the original three seeds.
        choice=study.memory.required(ctx['store'],'supervised_settings_frozen')
        stage='finish_saved_training_through_seed25';rows=[]
        for seed in five_seed.SEEDS:
            print('FIVE-SEED STUDY: finish/reuse seed',seed,'of 21–25; seeds 26–27 will not run.',flush=True)
            rows.extend(study.final_rows(ctx,seed,choice))
            ctx['store'].write('five_seed_training_progress.json',dict(seeds=five_seed.SEEDS,completed_through=seed,models=len(rows),test_opened=False))
        stage='five_seed_scope_lock_and_freeze'
        ctx,protocol=five_seed.open_evaluation(ctx,rows,choice,options,original_protocol)
        PRECISION[0].store=PRECISION[1].store=ctx['store']
        five_seed.install_evaluation_scope(rows,protocol)
        study.freeze(ctx,choice);study.export(ctx,options['wrapper_zip'])
        Path(options['receipt']).write_text(json.dumps(dict(complete=False,primary_seeds=5,results=str(ctx['store'].results),run_id=ctx['store'].run_id)))
        stage='new_holdout_extraction';pack=study.extract_after_freeze(ctx,coco)
        stage='five_seed_test_evaluation_and_intervals';study.audit(ctx,pack);del pack;gc.collect()
        matched.assert_sources(ctx);archive=study.export(ctx,options['wrapper_zip'],True)
        Path(options['receipt']).write_text(json.dumps(dict(complete=True,primary_seeds=5,results=str(ctx['store'].results),
            result_zip=str(archive),run_id=ctx['store'].run_id,omitted_seeds=[26,27],fresh_holdout_images=1500)))
        print('FIVE-SEED SUPERVISED COMPARISON COMPLETE:',archive,flush=True)
    except BaseException as exc:
        if ctx is not None:
            try:study.flush(ctx)
            except BaseException as extra:print('Checkpoint flush failed:',repr(extra),flush=True)
            try:
                ctx['store'].write('last_interruption.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
                partial=study.export(ctx,options['wrapper_zip'])
                Path(options['receipt']).write_text(json.dumps(dict(complete=False,primary_seeds=5,results=str(ctx['store'].results),partial_zip=str(partial),stage=stage)))
                print('Partial evidence ZIP:',partial,flush=True)
            except BaseException as extra:print('Partial export failed:',repr(extra),flush=True)
        raise
    finally:
        if ctx is not None:ctx['packs'].clear();gc.collect()

if __name__=='__main__':main()
