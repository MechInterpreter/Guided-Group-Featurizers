"""Observability only: retain every original numerical call, result, and exception."""
import functools
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class Monitor:
    def __init__(self, path=None, interval=30., emit=None):
        self.path = Path(path) if path else None
        self.interval = interval
        self.emit = emit or (lambda text: print(text, flush=True))
        self.stack = []
        self.fallback_calls = 0
        self.fallback_seconds = 0.
        self.last_write = -float('inf')
        self.last_message = -float('inf')
        self.last_completed = None

    def snapshot(self):
        now = time.monotonic()
        data = dict(worker_pid=os.getpid(), time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    operation=' > '.join(item['name'] for item in self.stack) or 'between operations',
                    operation_started_monotonic=self.stack[-1]['started'] if self.stack else now,
                    fallback_calls_completed=self.fallback_calls,
                    fallback_cpu_seconds=self.fallback_seconds,
                    last_completed_work=self.last_completed)
        for item in self.stack:
            data.update(item['details'])
        return data

    def write(self, force=False):
        now = time.monotonic()
        if self.path is None or (not force and now - self.last_write < 1.):
            return
        self.last_write = now
        # Logging failure must not alter, reject or restart a numerical operation.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.writing')
            tmp.write_text(json.dumps(self.snapshot(), allow_nan=False))
            os.replace(tmp, self.path)
        except OSError:
            pass

    def message(self, text, force=False):
        now = time.monotonic()
        if force or now - self.last_message >= self.interval:
            self.emit('[live] ' + text)
            self.last_message = now

    @contextmanager
    def operation(self, name, announce=False, **details):
        item = dict(name=name, started=time.monotonic(), details=details)
        self.stack.append(item)
        self.write(force=announce)
        if announce:
            self.message('START ' + name + ' ' + json.dumps(details), force=True)
        try:
            yield item
        except BaseException as exc:
            if announce:
                self.message(f'INTERRUPTED {name}: {type(exc).__name__}', force=True)
            raise
        else:
            if announce:
                self.message(f'FINISHED {name} in {time.monotonic() - item["started"]:.1f}s', force=True)
        finally:
            self.stack.pop()
            self.write(force=announce)


class Observers:
    def __init__(self, monitor):
        self.monitor = monitor
        self.patches = []
        self.audit_installed = False

    def replace(self, obj, name, fn):
        previous = getattr(obj, name)
        self.patches.append((obj, name, previous))
        setattr(obj, name, fn)

    def close(self):
        for obj, name, previous in reversed(self.patches):
            setattr(obj, name, previous)
        self.patches.clear()

    def install_audit(self):
        if self.audit_installed:
            return
        from coco_baselines import metrics
        original = metrics.audit_arrays
        monitor = self.monitor

        @functools.wraps(original)
        def audit(model, x, masks, ids, cfg, device, *args, **kwargs):
            previous = model.encode
            had_own = 'encode' in vars(model)
            own = vars(model).get('encode')
            total = len(ids) * 196
            with monitor.operation('patch evaluation', announce=True, images_total=len(ids),
                                   patches_total=total, patches_completed=0, images_completed=0) as item:
                @functools.wraps(previous)
                def encode(batch):
                    result = previous(batch)
                    item['details']['patches_completed'] += len(batch)
                    done = item['details']['patches_completed']
                    item['details']['images_completed'] = done // 196
                    monitor.last_completed = dict(kind='evaluation batch', patches_completed=done,
                                                  patches_total=total, time_monotonic=time.monotonic())
                    monitor.write(force=True)
                    monitor.message(f'Evaluation completed {done // 196}/{len(ids)} images '
                                    f'({done}/{total} patches)', force=done == total or done == len(batch))
                    return result
                model.encode = encode
                try:
                    return original(model, x, masks, ids, cfg, device, *args, **kwargs)
                finally:
                    if had_own:
                        model.encode = own
                    else:
                        delattr(model, 'encode')
        self.replace(metrics, 'audit_arrays', audit)
        self.audit_installed = True

    def install(self):
        import supervised_study as study
        from coco_ggf import slics_repair
        import scipy.optimize
        monitor = self.monitor
        original_prepare = study.prepare

        @functools.wraps(original_prepare)
        def prepare(*args, **kwargs):
            with monitor.operation('verify inputs and restore saved run', announce=True):
                ctx = original_prepare(*args, **kwargs)
            # prepare installs the original memory-safe audit. Observe it afterwards.
            self.install_audit()
            original_event = ctx['store'].event

            @functools.wraps(original_event)
            def event(stage, **values):
                result = original_event(stage, **values)
                if stage in ['slics_progress', 'study_checkpoint', 'study_progress']:
                    shown = {k:values[k] for k in ['seed', 'key', 'update', 'total', 'capture', 'AP', 'VE',
                              'seconds_per_update', 'step_seconds'] if k in values}
                    monitor.last_completed = dict(kind=stage, **shown, time_monotonic=time.monotonic())
                    monitor.message(stage + ' ' + json.dumps(shown), force=True)
                    monitor.write(force=True)
                return result
            self.replace(ctx['store'], 'event', event)
            return ctx
        self.replace(study, 'prepare', prepare)

        original_native = study.native_fit

        @functools.wraps(original_native)
        def native(ctx, seed):
            saved = ctx['store'].load(f'native_slics12_s{seed}')
            details = dict(seed=seed, updates_saved=0 if saved is None else saved['updates'],
                           last_completed_validation=None if saved is None else saved.get('last_evaluated'),
                           updates_total=study.TOTAL)
            del saved
            with monitor.operation('SLiCS fit/resume', announce=True, **details):
                return original_native(ctx, seed)
        self.replace(study, 'native_fit', native)

        original_fit = study.fit

        @functools.wraps(original_fit)
        def fit(ctx, key, *args, **kwargs):
            with monitor.operation('GGF fit/resume', announce=True, model_key=key):
                return original_fit(ctx, key, *args, **kwargs)
        self.replace(study, 'fit', fit)

        original_evaluate = study.evaluate_model

        @functools.wraps(original_evaluate)
        def evaluate(model, pack, *args, **kwargs):
            with monitor.operation('model evaluation', announce=True, model_type=type(model).__name__,
                                   readout=getattr(model, 'readout', 'encoder'), images_total=len(pack['ids'])):
                result = original_evaluate(model, pack, *args, **kwargs)
                ve=result['reconstruction']['ve']
                shown='deferred (joint reconstruction runs after model freeze)' if ve is None else f'{100*ve:.2f}%'
                monitor.message(f"COMPLETED METRICS: Capture {100*result['macro_capture']:.2f}%, patch AP {100*result['macro_ap']:.2f}%, VE {shown}",force=True)
                return result
        self.replace(study, 'evaluate_model', evaluate)

        original_solve = slics_repair.Solver.solve

        @functools.wraps(original_solve)
        def solve(solver, x, *args, **kwargs):
            with monitor.operation('joint/supported NNLS', nnls_rows=len(x), nnls_atoms=len(solver.D)):
                result = original_solve(solver, x, *args, **kwargs)
                if result[1].get('cpu_fallback_rows', 0):
                    monitor.message('NNLS completed: ' + json.dumps(result[1]))
                return result
        self.replace(slics_repair.Solver, 'solve', solve)

        original_nnls = scipy.optimize.nnls

        @functools.wraps(original_nnls)
        def fallback(A, b, *args, **kwargs):
            with monitor.operation('CPU NNLS fallback', fallback_dimensions=list(A.shape)):
                # Record entry before a potentially long C call holding the GIL.
                monitor.write(force=True)
                started = time.monotonic()
                result = original_nnls(A, b, *args, **kwargs)
                monitor.fallback_calls += 1
                monitor.fallback_seconds += time.monotonic() - started
                monitor.message(f'CPU NNLS fallbacks completed: {monitor.fallback_calls}; '
                                f'cumulative fallback time {monitor.fallback_seconds:.1f}s')
                return result
        # Cover both the captured import used for joint solves and the dynamic
        # scipy.optimize import in partial_nnls. Each calls the original once.
        self.replace(scipy.optimize, 'nnls', fallback)
        self.replace(slics_repair, 'scipy_nnls', fallback)
        return self


def install_observers():
    monitor = Monitor(os.environ.get('GGF_LIVE_STATUS'))
    monitor.message('Live progress enabled; heartbeat is provided by the notebook parent process.', force=True)
    return Observers(monitor).install()
