import argparse,gc,importlib.metadata,json,traceback
from pathlib import Path

def main():
    import torch
    from threadpoolctl import threadpool_limits
    import objects_first as task
    import supervised_study as study
    from supervised_live import install_observers
    args=argparse.ArgumentParser();args.add_argument('--config',required=True)
    options=json.loads(Path(args.parse_args().config).read_text())
    expected=options['expected_versions']
    for name,version in expected.items():
        observed=importlib.metadata.version(name)
        if (observed.split('+')[0]!=version.split('+')[0] if name=='torch' else observed!=version):
            raise ValueError('Pinned CPU numerical package differs: '+name)
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    install_observers();ctx=None;stage='restore_CPU_handoff'
    try:
        ctx=task.resume_cpu(options)
        ctx['store'].write('CPU_runtime.json',dict(device='cpu',versions={p:importlib.metadata.version(p) for p in expected},
            solver='float64 SciPy active-set NNLS directly; unchanged KKT and maxiter; no CUDA tensor operations'))
        task.memory.required(ctx['store'],'objects_phase_complete')
        models=task.load_object_models(ctx)
        stage='CPU_image_bootstrap_intervals_first';object_result=task.cpu_intervals(ctx,models)
        interim=task.export_phase(ctx,options['wrapper_zip'],'objects_with_intervals')
        Path(options['receipt']).write_text(json.dumps(dict(complete=False,object_metrics_complete=True,whole_image_intervals_complete=True,
            reconstruction_complete=False,results=str(ctx['store'].results),partial_zip=str(interim))))
        print('ALL FIVE OBJECT METRICS AND INTERVALS SAVED:',interim,flush=True)
        for row in object_result['seed_intervals']:
            if row['treatment']=='matched_ggf_final':print('COMPLETED-TRAINING PAIRED RESULT:',row,flush=True)
        stage='CPU_reconstruction';pack=task.cached_test_pack(ctx)
        task.cpu_reconstruction(ctx,pack,models);del pack;gc.collect()
        archive=task.export_phase(ctx,options['wrapper_zip'],'complete')
        Path(options['receipt']).write_text(json.dumps(dict(complete=True,object_metrics_complete=True,whole_image_intervals_complete=True,
            reconstruction_complete=True,primary_seeds=5,results=str(ctx['store'].results),result_zip=str(archive))))
        print('FIVE-SEED COMPARISON COMPLETE:',archive,flush=True)
    except BaseException as exc:
        if ctx is not None:
            ctx['store'].write('CPU_interruption.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
            try:
                partial=task.export_phase(ctx,options['wrapper_zip'],'CPU_partial')
                Path(options['receipt']).write_text(json.dumps(dict(complete=False,results=str(ctx['store'].results),partial_zip=str(partial),stage=stage)))
                print('Partial evidence ZIP:',partial,flush=True)
            except BaseException as extra:print('Partial export failed:',repr(extra),flush=True)
        raise
    finally:
        if ctx is not None:ctx['packs'].clear();gc.collect()

if __name__=='__main__':main()
