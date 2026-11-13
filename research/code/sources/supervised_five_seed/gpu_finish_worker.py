import argparse,gc,hashlib,importlib.metadata,json,time,traceback
from pathlib import Path


def main():
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import objects_first as task
    import supervised_study as study
    import object_schedule
    from gpu_joint_nnls import GPUJointModel,benchmark
    from supervised_live import install_observers
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    options=json.loads(Path(parser.parse_args().config).read_text())
    for name,version in options['expected_versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Pinned numerical version differs: '+name)
    if not torch.cuda.is_available():raise RuntimeError('Use a GPU runtime, such as T4')
    torch.set_num_threads(2);threadpool_limits(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    observers=install_observers();ctx=None;stage='restore_gpu_reconstruction'
    try:
        # The saved lightweight handoff restores frozen catalogs without training
        # features/backbone, then only the execution device changes to CUDA.
        ctx=task.resume_cpu(options);ctx['device']='cuda';store=ctx['store']
        hashes=json.loads((Path(__file__).parent/'REVISION_HASHES.json').read_text())
        revision=task.identity(hashes)
        store.write('GPU_joint_execution.json',dict(revision=revision,run_id=task.FIVE_RUN,
            gpu=torch.cuda.get_device_name(),versions={p:importlib.metadata.version(p) for p in options['expected_versions']},
            full_dictionary=True,KKT_tolerance=1e-5,fallback_maxiter=20000,training=False,cohort_changed=False,
            existing_image_prefixes_retained=True,solver='Float32 batched FISTA, certified-row compaction, cached float64 active-set refinement, full SciPy fallback',
            note='Execution solver differs. All rows require original-space float64 KKT; nonunique coefficients/activity may differ. Prior verified CPU/GPU prefixes remain in final totals.'))
        task.memory.required(store,'objects_phase_complete')
        models=task.load_object_models(ctx)
        stage='reuse_saved_intervals'
        task.cpu_intervals(ctx,models)
        stage='load_cached_test_features';pack=task.cached_test_pack(ctx)
        completed=[]
        for item in models:
            row=item['row'];report=dict(item['report'])
            if row['method']=='native_slics' and report['reconstruction'].get('status')!='computed':
                state=task.memory.required(store,row['key'])
                if task.training.weight_sha({'D':state['D']})!=row['weights_sha']:raise ValueError('Frozen model changed')
                cache_key='final_joint_'+task.identity(dict(weights=row['weights_sha'],ids=pack['ids'],cohort=pack.get('identity'),
                    norm=task.identity(ctx.get('norm',{})),solver=row['solver'],arithmetic='image-streamed-float64-SSE-v1'))[:32]
                prior=store.load(cache_key)
                if prior is not None and prior.get('complete'):
                    report['reconstruction']=prior['report']
                    print('REUSED COMPLETE RECONSTRUCTION:',row['seed'],row['arm'],flush=True)
                else:
                    stage='GPU_reconstruction_s'+str(row['seed'])+'_'+row['arm']
                    model=GPUJointModel(state['D'],row['solver']);del state
                    image=0 if prior is None else prior['completed_images']
                    if not 0<=image<len(pack['ids']):raise ValueError('Invalid saved image prefix')
                    key='GPU_benchmark_'+task.identity(dict(weights=row['weights_sha'],revision=revision,gpu=torch.cuda.get_device_name()))[:24]
                    if store.load(key) is None:
                        print('BENCHMARK: eight fixed patches; comparing full NNLS objectives and speed.',flush=True)
                        probe=benchmark(model,torch.as_tensor(np.array(pack['x'][image],copy=True),device='cuda'))
                        probe.update(seed=row['seed'],weights_sha=row['weights_sha'],image_index=image,revision=revision)
                        store.save(key,probe);store.write('GPU_benchmarks/'+key+'.json',probe)
                        print('GPU BENCHMARK:',json.dumps(probe),flush=True)
                        if probe['observed_probe_speedup']<1:
                            print('Probe is not faster. Continuing the unchanged requested full evaluation; full-image timing is authoritative.',flush=True)
                    store.event('GPU_reconstruction_model_start',seed=row['seed'],arm=row['arm'],images_saved=image,total_images=len(pack['ids']))
                    monitor=observers.monitor
                    with monitor.operation('GPU joint reconstruction',announce=True,seed=row['seed'],images_saved=image,total_images=len(pack['ids'])):
                        report['reconstruction']=object_schedule.final_reconstruction(ctx,model,pack,row['weights_sha'])
                    store.write('GPU_solver_last_s'+str(row['seed'])+'.json',model.last_stats)
                    del model;gc.collect();torch.cuda.empty_cache()
            saved=dict(row=row,report=report);completed.append(saved)
            store.save(f'GPU_final_s{row["seed"]}_{row["arm"]}',saved)
            store.write(f'GPU_reports/s{row["seed"]}_{row["arm"]}.json',saved)
        pairs=[]
        for seed in task.SEEDS:
            for a,b in [('matched_ggf','slics_object_partial'),('matched_ggf_final','slics_final_partial')]:
                pairs.append(task.memory.required(store,f'image_ci_s{seed}_{a}_{b}'))
        study.summarize(ctx,completed,pairs)
        store.write('GPU_reconstruction_complete.json',dict(complete=True,seeds=task.SEEDS,revision=revision))
        archive=task.export_phase(ctx,options['wrapper_zip'],'complete')
        Path(options['receipt']).write_text(json.dumps(dict(complete=True,object_metrics_complete=True,whole_image_intervals_complete=True,
            reconstruction_complete=True,results=str(store.results),result_zip=str(archive))))
        print('FIVE-SEED COMPARISON COMPLETE:',archive,flush=True)
    except BaseException as exc:
        if ctx is not None:
            ctx['store'].write('GPU_joint_interruption.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
            try:
                archive=task.export_phase(ctx,options['wrapper_zip'],'GPU_joint_partial')
                Path(options['receipt']).write_text(json.dumps(dict(complete=False,stage=stage,results=str(ctx['store'].results),partial_zip=str(archive))))
                print('VERIFIED IMAGE PREFIX / PARTIAL ZIP SAVED:',archive,flush=True)
            except BaseException as extra:print('Partial export failed:',repr(extra),flush=True)
        raise
    finally:
        if ctx is not None:ctx['packs'].clear();gc.collect()


if __name__=='__main__':main()
