"""Fresh pinned-package worker for either inference-only or continuation."""
import argparse,json,os,subprocess,sys,gc
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    o=json.loads(Path(args.config).read_text());root=Path(__file__).resolve().parent
    env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    tests=['test_pcgrad_continuation.py','test_caps_pipeline.py','test_cap_followup.py','test_width_balance.py','test_weight_bound.py','test_weight_bound_transport.py']
    subprocess.check_call([sys.executable,'-m','pytest','-q',*[str(root/'tests'/n) for n in tests],
        '--disable-warnings','--basetemp',str(Path(o['local'])/('selftest_'+os.urandom(6).hex()))],cwd=root,env=env)
    import torch
    from threadpoolctl import threadpool_limits
    from coco_ggf import completed_caps_test,pcgrad_continuation
    exp=completed_caps_test if o['mode']=='completed_test' else pcgrad_continuation
    if exp.matched.versions()!=o['expected_versions']:raise RuntimeError('Fresh worker versions do not match source')
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    ctx=None;stage='prepare'
    try:
        if o['mode']=='completed_test':
            ctx=exp.prepare(o['paths'],o['training_output'],o['output'],o['local'],o['cache_local'],o['source_sha'],o['previous_test_output'])
        else:
            ctx=exp.prepare(o['paths'],o['output'],o['local'],o['cache_local'],o['source_sha'],o['previous_test_output'])
        ctx['store'].results.joinpath('PROTOCOL.md').write_bytes((root/'PROTOCOL_WORKFLOWS.md').read_bytes())
        print('Progress:',ctx['store'].results/'progress.json',flush=True)
        if o['mode']!='completed_test':
            stage='preflight';timing=exp.preflight(ctx)
            print('Estimated remaining gradient hours:',round(timing['estimated_gradient_hours'],2),'plus scoring and Drive overhead',flush=True)
            stage='continuation_and_freeze';exp.run(ctx)
            ctx['packs'].pop('train',None);gc.collect()
        stage='cached_test_loading';exp.load_test(ctx)
        stage='test_evaluation';summary=exp.evaluate(ctx)
        stage='export';result=exp.export(ctx,o['source_zip'])
        receipt=dict(complete=summary['complete'],result_zip=str(result),results=str(ctx['store'].results),mode=o['mode'])
        Path(o['receipt']).write_text(json.dumps(receipt,indent=2))
        print('ALL STAGES COMPLETE:',result,flush=True)
    except BaseException as exc:
        if ctx is not None:exp.failure(ctx,stage,exc,o['source_zip'])
        raise
if __name__=='__main__':main()
