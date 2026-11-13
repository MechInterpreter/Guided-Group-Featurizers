"""Pinned fresh Colab process. One priority stage per notebook cell."""
import argparse,gc,json,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--stage',required=True,choices=['width5','pcgrad_both','longer']);args=p.parse_args()
    o=json.loads(Path(args.config).read_text());root=Path(__file__).resolve().parent
    env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    marker=Path(o['local'])/'selftest_passed.json'
    if not marker.exists() or json.loads(marker.read_text()).get('source_sha')!=o['source_sha']:
        tests=['test_next_experiments.py','test_pcgrad_continuation.py','test_width_balance.py','test_weight_bound.py','test_weight_bound_transport.py']
        subprocess.check_call([sys.executable,'-m','pytest','-q',*[str(root/'tests'/n) for n in tests],
            '--disable-warnings','--basetemp',str(Path(o['local'])/('selftest_'+os.urandom(6).hex()))],cwd=root,env=env)
        marker.write_text(json.dumps(dict(source_sha=o['source_sha'],passed=True)))
    import torch
    from threadpoolctl import threadpool_limits
    from coco_ggf import next_experiments as exp
    if exp.matched.versions()!=o['expected_versions']:raise RuntimeError('Fresh worker versions differ from the saved source')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Select Tesla T4')
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True,warn_only=True)
    ctx=None;stage='prepare'
    try:
        ctx=exp.prepare(o['paths'],o['output'],o['local'],o['cache_local'],o['source_sha'],o['previous_output'],o['previous_test_output'])
        ctx['store'].results.joinpath('PROTOCOL.md').write_bytes((root/'PROTOCOL_NEXT.md').read_bytes())
        print('Priority:',exp.PRIORITY,'Current:',args.stage,flush=True)
        print('Drive checkpoints:',ctx['store'].root,flush=True)
        stage=args.stage;result=exp.run_stage(ctx,args.stage)
        ctx['packs'].pop('test',None);ctx['packs'].pop('train',None);gc.collect()
        stage='export';archive=exp.export(ctx,o['source_zip'])
        receipt=dict(complete=result['complete'],stage=args.stage,result_zip=str(archive),results=str(ctx['store'].results))
        Path(o['local'],args.stage+'_completion.json').write_text(json.dumps(receipt,indent=2))
        print('STAGE COMPLETE:',args.stage,archive,flush=True)
    except BaseException as exc:
        if ctx is not None:exp.failure(ctx,stage,exc,o['source_zip'])
        raise

if __name__=='__main__':main()
