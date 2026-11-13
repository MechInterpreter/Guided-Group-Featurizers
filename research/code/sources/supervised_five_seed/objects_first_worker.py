import argparse,gc,json,traceback
from pathlib import Path
from training_precision import install
PRECISION=install()

def main():
    import torch
    from threadpoolctl import threadpool_limits
    import supervised_study as study
    import objects_first as task
    from supervised_live import install_observers
    from coco_ggf import matched_baselines as matched
    args=argparse.ArgumentParser();args.add_argument('--config',required=True)
    options=json.loads(Path(args.parse_args().config).read_text())
    if not torch.cuda.is_available():raise RuntimeError('Use T4 for the object-metric notebook')
    if matched.versions()!=options['expected_versions']:raise ValueError('Pinned numerical versions differ')
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    install_observers();ctx=None;stage='restore_exact_existing_evaluation'
    try:
        base=study.prepare(options)
        ctx=task.resume_gpu(base,options);PRECISION[0].store=PRECISION[1].store=ctx['store']
        # Already-extracted test shards only. There are no fits, LR selection,
        # annotations download, backbone extraction or joint reconstruction.
        stage='read_existing_test_features';pack=task.cached_test_pack(ctx)
        stage='all_five_capture_and_AP_first';result=task.objects_pass(ctx,pack)
        del pack;gc.collect();archive=task.export_phase(ctx,options['wrapper_zip'],'objects')
        Path(options['receipt']).write_text(json.dumps(dict(complete=True,object_metrics_complete=True,
            reconstruction_complete=False,whole_image_intervals_complete=False,primary_seeds=5,
            results=str(ctx['store'].results),result_zip=str(archive),run_id=ctx['store'].run_id)))
        print('ALL FIVE CAPTURE/AP RESULTS SAVED:',archive,flush=True)
        for row in result['means']:
            if row['seed_group']=='all_five' and row['arm'] in ['matched_ggf_final','slics_final_partial']:
                print('COMPLETED TRAINING:',row,flush=True)
        print('GPU WORK FINISHED. Run the CPU companion for image-bootstrap intervals and joint reconstruction.',flush=True)
    except BaseException as exc:
        if ctx is not None:
            ctx['store'].write('objects_first_interruption.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
            try:
                partial=task.export_phase(ctx,options['wrapper_zip'],'objects_partial')
                Path(options['receipt']).write_text(json.dumps(dict(complete=False,results=str(ctx['store'].results),partial_zip=str(partial),stage=stage)))
                print('Partial evidence ZIP:',partial,flush=True)
            except BaseException as extra:print('Partial export failed:',repr(extra),flush=True)
        raise
    finally:
        if ctx is not None:ctx['packs'].clear();gc.collect()

if __name__=='__main__':main()
