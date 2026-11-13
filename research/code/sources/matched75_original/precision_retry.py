"""Preserve passing partial NNLS calls; retry only a strict-KKT failure in float64."""
import functools,hashlib,json,time
from pathlib import Path
import torch

BASE_SHA='dfed82ac09e358dd8fc5e8d6de1bb524519cc0965cfea45f1318367b7eab90cf'
EXPECTED_RUN='e8aff97082f2f0d1942306ac1cae391aca2deef1374244b2e694903be6b310ec'
HERE=Path(__file__).resolve().parent
FAILURE='Partial NNLS failed strict KKT verification'

class PrecisionRetry:
    def __init__(self,original,emit=None):
        self.original=original;self.emit=emit or (lambda event:print('[precision] '+json.dumps(event),flush=True))
        self.retries=0;self.store=None
    @torch.no_grad()
    def __call__(self,x,D,cfg,width):
        try:
            return self.original(x,D,cfg,width)
        except RuntimeError as exc:
            # Do not mask OOM, SciPy iteration exhaustion, invalid data, or other errors.
            if str(exc)!=FAILURE or (x.dtype==torch.float64 and D.dtype==torch.float64):raise
        self.retries+=1
        event=dict(operation='partial_NNLS_float64_retry',retry=self.retries,rows=len(x),groups=len(D)//width,
            tolerance=cfg['nnls_tolerance'],nnls_iterations=cfg['nnls_iterations'],training_weights_changed=False)
        self.emit(dict(event,status='started'))
        start=time.monotonic()
        # Exact same dictionary values, observations, objective, iteration budget,
        # nonnegativity, active-set fallback and terminal KKT threshold. Retain
        # verified float64 coefficients: recasting them caused the original failure.
        z,info=self.original(x.double(),D.double(),cfg,width)
        if z.dtype!=torch.float64 or not torch.isfinite(z).all() or bool((z<0).any()):
            raise RuntimeError('Float64 NNLS retry returned invalid coefficients')
        if not 0<=info['max_relative_kkt']<=cfg['nnls_tolerance']:
            raise RuntimeError('Float64 NNLS retry did not meet the unchanged strict KKT threshold')
        event.update(status='verified',seconds=time.monotonic()-start,max_relative_kkt=info['max_relative_kkt'],
            fallback_rows=info['cpu_fallback_rows'])
        self.emit(event)
        if self.store is not None:self.store.write('last_partial_precision_retry.json',event)
        # Existing diagnostics remain intact; no unexpected keys or metric changes.
        return z,info

def verify_source(folder):
    folder=Path(folder)
    for name,digest in json.loads((folder/'FILE_HASHES.json').read_text()).items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe numerical source member')
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('Original source changed: '+name)

def check_run(options):
    if options['wrapper_sha']!=BASE_SHA:raise ValueError('Precision repair requires the exact current numerical source')
    if hashlib.sha256(Path(options['wrapper_zip']).read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Numerical archive changed')
    manifest=Path(options['output'])/'results'/EXPECTED_RUN[:20]/'manifest.json'
    expected=json.loads((HERE/'ACTIVE_MANIFEST.json').read_text())
    if not manifest.exists() or json.loads(manifest.read_text())!=expected:raise ValueError('Existing saved run differs; refusing a new training run')
    from coco_baselines.storage import identity
    if identity(expected['config'])!=EXPECTED_RUN:raise ValueError('Pinned manifest identity is invalid')

def install():
    import supervised_study as study
    verify_source(Path(study.__file__).parent)
    expected=json.loads((HERE/'ACTIVE_MANIFEST.json').read_text())
    if study.protocol()!=expected['config']['protocol']:raise ValueError('Scientific protocol differs from the running study')
    original_prepare=study.prepare
    original_partial=study.partial_nnls
    retry=PrecisionRetry(original_partial)
    @functools.wraps(original_prepare)
    def prepare(options,require_drive=True):
        check_run(options)
        ctx=original_prepare(options,require_drive=require_drive)
        if ctx['store'].run_id!=EXPECTED_RUN:raise ValueError('Refusing to change the saved run identity')
        retry.store=ctx['store']
        folder=ctx['store'].results/'execution_precision'
        folder.mkdir(exist_ok=True)
        hashes=json.loads((HERE/'REVISION_HASHES.json').read_text())
        for name in list(hashes)+['REVISION_HASHES.json']:(folder/name).write_bytes((HERE/name).read_bytes())
        ctx['store'].write('execution_precision_revision.json',dict(revision='strict-partial-float64-retry-v1',
            original_run_id=EXPECTED_RUN,original_source_sha=BASE_SHA,revision_files=hashes,
            only_failed_partial_inference_retried=True,scientific_objective_changed=False,
            tolerance_changed=False,training_arithmetic_changed=False,all_previous_checkpoints_reused=True,
            note='Float32 successful calls unchanged; strict-KKT failures retry the same solve in float64.'))
        print('VERIFIED SAME SAVED RUN:',EXPECTED_RUN,'; strict partial NNLS precision retry enabled.',flush=True)
        return ctx
    study.partial_nnls=retry
    study.prepare=prepare
    return retry,original_partial,original_prepare
