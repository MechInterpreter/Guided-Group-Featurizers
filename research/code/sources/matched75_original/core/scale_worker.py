"""One resumable worker, no late token or configuration prompts."""
import argparse,gc,json,os,subprocess,sys,traceback
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    options=json.loads(Path(args.config).read_text());source=Path(__file__).resolve().parent;local=Path(options['local']);local.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    marker=local/'scale_selftests.json'
    if not marker.exists() or json.loads(marker.read_text()).get('source_sha')!=options['source_sha']:
        subprocess.check_call([sys.executable,'-m','pytest','-q',str(source/'tests/test_scale_study.py'),str(source/'test_rank.py'),str(source/'tests/test_five_exchanges.py'),
            '--disable-warnings','--basetemp',str(local/('selftest_'+os.urandom(4).hex()))],cwd=source,env=env)
        marker.write_text(json.dumps(dict(source_sha=options['source_sha'],passed=True)))
    import torch
    from threadpoolctl import threadpool_limits
    from coco_ggf import matched_baselines as matched
    from scale_data import prepare,annotations,holdout_plan,backbone_preflight,load_holdout
    from scale_run import factorial,caps,tune_baselines,final_suite,preflight,export,flush
    from scale_evaluate import run as evaluate
    from scale_analysis import run as analyze
    if matched.versions()!=options['expected_versions']:raise ValueError('Numerical packages differ from the saved source')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Select a Tesla T4 runtime')
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    ctx=None;stage='prepare'
    try:
        ctx=prepare(options);ctx['store'].results.joinpath('PROTOCOL.md').write_bytes((source/'PROTOCOL_SCALE.md').read_bytes())
        coco=annotations(ctx)
        if ctx['rule']['fresh']:
            stage='fresh_holdout_plan';holdout_plan(ctx,coco)
            stage='backbone_access_and_replay';backbone_preflight(ctx,coco)
        stage='T4_preflight';preflight(ctx)
        stage='group_count_and_topk_screen';configuration=factorial(ctx);export(ctx,options['source_zip'])
        stage='adaptive_cap_screen';capchoice=caps(ctx,configuration);export(ctx,options['source_zip'])
        stage='baseline_development_tuning';tuning=tune_baselines(ctx,configuration['groups'],configuration['active_groups']);export(ctx,options['source_zip'])
        stage='seven_seed_models_and_exchanges';final_suite(ctx,configuration,capchoice,tuning);export(ctx,options['source_zip'])
        stage='final_holdout_evaluation';pack=load_holdout(ctx,coco);evaluate(ctx,pack);del pack;gc.collect();export(ctx,options['source_zip'])
        stage='effective_rank_coding_proxy_and_costs';analyze(ctx)
        stage='export';archive=export(ctx,options['source_zip'],True)
        Path(options['receipt']).write_text(json.dumps(dict(complete=True,result_zip=str(archive),results=str(ctx['store'].results),
            source_sha=options['source_sha'],run_id=ctx['store'].run_id,fresh_holdout=ctx['rule']['fresh']),indent=2))
        print('ALL SCALE STUDY PHASES COMPLETE:',archive,flush=True)
    except BaseException as exc:
        if ctx is not None:
            try:
                flush(ctx);ctx['store'].write('last_interruption.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
                print('Partial evidence:',export(ctx,options['source_zip']),flush=True)
            except BaseException as secondary:print('Recovery/export also failed:',type(secondary).__name__,str(secondary),flush=True)
        raise
    finally:
        if ctx is not None:ctx['packs'].clear();gc.collect()


if __name__=='__main__':main()
