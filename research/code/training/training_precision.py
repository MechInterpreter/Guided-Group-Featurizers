"""Strict NNLS precision recovery for partial readouts, training and joint readouts.

Passing float32 calls retain their arithmetic. Only the exact terminal KKT
failure retries in double; an affected atom sweep uses double intermediates and
returns its dictionary in the original storage dtype. No tolerance is relaxed.
"""

import functools, hashlib, inspect, json, textwrap, time
from pathlib import Path
import torch
from precision_retry import (
    PrecisionRetry,
    verify_source,
    check_run,
    BASE_SHA,
    EXPECTED_RUN,
)

HERE = Path(__file__).resolve().parent
JOINT_FAILURE = "NNLS failed the per-row float64 KKT check; source model unchanged"


class JointPrecisionRetry:
    def __init__(self, original, emit=None):
        self.original = original
        self.emit = emit or (
            lambda e: print("[precision] " + json.dumps(e), flush=True)
        )
        self.retries = 0
        self.store = None

    @torch.no_grad()
    def __call__(self, solver, x, allowed=None, force_fallback=False):
        try:
            return self.original(solver, x, allowed, force_fallback=force_fallback)
        except RuntimeError as exc:
            if str(exc) != JOINT_FAILURE or solver.D.dtype == torch.float64:
                raise
        self.retries += 1
        start = time.monotonic()
        event = dict(
            operation="supported_or_joint_NNLS_float64_retry",
            retry=self.retries,
            rows=len(x),
            directions=len(solver.D),
            supported=allowed is not None,
            tolerance=solver.cfg["nnls_tolerance"],
            nnls_iterations=solver.cfg["nnls_iterations"],
            force_fallback=bool(force_fallback),
        )
        self.emit(dict(event, status="started"))
        # Call the unwrapped implementation, preventing recursive retries.
        promoted = type(solver)(solver.D.double(), solver.cfg)
        z, info = self.original(
            promoted, x.double(), allowed, force_fallback=force_fallback
        )
        if (
            z.dtype != torch.float64
            or not torch.isfinite(z).all()
            or bool((z < 0).any())
        ):
            raise RuntimeError("Float64 joint NNLS retry returned invalid coefficients")
        support = (
            torch.ones_like(z, dtype=torch.bool)
            if allowed is None
            else allowed.to(z.device)
        )
        if bool((z[~support] != 0).any()):
            raise RuntimeError("Float64 joint NNLS retry violated support")
        # Independently recompute the terminal certificate from the actual result.
        errors = promoted._errors(z, x.double().to(z.device), support)
        if (
            not torch.isfinite(errors).all()
            or float(errors.max()) > solver.cfg["nnls_tolerance"]
            or not 0 <= info["max_relative_kkt"] <= solver.cfg["nnls_tolerance"]
        ):
            raise RuntimeError(
                "Float64 joint NNLS retry did not meet the unchanged strict KKT threshold"
            )
        event.update(
            status="verified",
            seconds=time.monotonic() - start,
            max_relative_kkt=float(errors.max()),
            fallback_rows=info["cpu_fallback_rows"],
        )
        self.emit(event)
        if self.store is not None:
            self.store.write("last_joint_precision_retry.json", event)
        return z, dict(
            info,
            precision_recovery="float64_after_terminal_KKT_failure",
            first_attempt_dtype=str(solver.D.dtype),
        )


def precision_atom_sweep(original, x, D, z, allowed, guard):
    if z.dtype == D.dtype:
        return original(x, D, z, allowed, guard)
    if z.dtype != torch.float64 or D.dtype != torch.float32:
        raise TypeError("Unexpected NNLS/atom precision combination")
    changed, info = original(x.double(), D.double(), z, allowed, guard)
    # Keep the stored model format/fingerprint compatible; only intermediates
    # on a precision-recovery batch are double. The update remains atomic.
    return changed.to(dtype=D.dtype), dict(
        info,
        precision_recovery="float64_intermediates",
        stored_dictionary_dtype=str(D.dtype),
    )


def revise(function, replacements):
    """Guard each exact, reviewable source edit; retain original module globals."""
    source = textwrap.dedent(inspect.getsource(function))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(
                "Unexpected source for " + function.__qualname__ + ": " + old
            )
        source = source.replace(old, new)
    namespace = {}
    exec(
        compile(
            source,
            str(HERE / "training_precision.py") + "::" + function.__qualname__,
            "exec",
        ),
        inspect.unwrap(function).__globals__,
        namespace,
    )
    return namespace[function.__name__]


def patch_paths(study, schedule):
    original_native = study.native_fit
    original_sweep = study.atom_sweep
    # Look up atom_sweep in the original globals so observers/test substitutes
    # remain effective. No decorator, sampling or checkpoint logic changes.
    study.__dict__["_ggf_precision_atom_sweep"] = precision_atom_sweep
    native = revise(
        original_native,
        [
            (
                "if bool(cols.any()):z[:,cols],info=Solver(D[cols],cfg).solve(xx,allow[:,cols])",
                "if bool(cols.any()):\n                    coefficients,info=Solver(D[cols],cfg).solve(xx,allow[:,cols])\n                    z=z.to(dtype=coefficients.dtype)\n                    z[:,cols]=coefficients",
            ),
            (
                "changed,atom=atom_sweep(xx,D,z,allow,False)",
                "changed,atom=_ggf_precision_atom_sweep(atom_sweep,xx,D,z,allow,False)",
            ),
        ],
    )
    forward = revise(
        study.NativeSLiCS.forward,
        [("z.flatten(1)@self.D,z", "z.flatten(1)@self.D.to(dtype=z.dtype),z")],
    )
    evaluate = revise(
        study.evaluate_model,
        [
            (
                "joint.flatten(1)@self.parent.atoms().reshape(-1,self.d)",
                "joint.flatten(1)@self.parent.atoms().reshape(-1,self.d).to(dtype=joint.dtype)",
            )
        ],
    )
    reconstruction = revise(
        schedule.final_reconstruction,
        [
            (
                "joint.flatten(1)@model.D",
                "joint.flatten(1)@model.D.to(dtype=joint.dtype)",
            )
        ],
    )
    study.native_fit = native
    study.NativeSLiCS.forward = forward
    study.evaluate_model = evaluate
    schedule.final_reconstruction = reconstruction
    study.final_reconstruction = reconstruction
    return dict(native_fit=original_native, atom_sweep=original_sweep)


def install():
    import supervised_study as study
    import object_schedule as schedule
    from coco_ggf.slics_repair import Solver

    if getattr(Solver.solve, "_ggf_precision_installed", False):
        raise RuntimeError("Precision overlay already installed")
    verify_source(Path(study.__file__).parent)
    expected = json.loads((HERE / "ACTIVE_MANIFEST.json").read_text())
    if study.protocol() != expected["config"]["protocol"]:
        raise ValueError("Scientific protocol differs from the running study")
    original_prepare = study.prepare
    partial = PrecisionRetry(study.partial_nnls)
    joint = JointPrecisionRetry(Solver.solve)
    original_paths = patch_paths(study, schedule)

    @functools.wraps(Solver.solve)
    def solve(self, x, allowed=None, force_fallback=False):
        return joint(self, x, allowed, force_fallback)

    solve._ggf_precision_installed = True
    Solver.solve = solve
    study.partial_nnls = partial

    @functools.wraps(original_prepare)
    def prepare(options, require_drive=True):
        check_run(options)
        ctx = original_prepare(options, require_drive=require_drive)
        if ctx["store"].run_id != EXPECTED_RUN:
            raise ValueError("Refusing to change the saved run identity")
        partial.store = joint.store = ctx["store"]
        hashes = json.loads((HERE / "REVISION_HASHES.json").read_text())
        revision = hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest()
        folder = ctx["store"].results / "execution_training_precision" / revision[:16]
        folder.mkdir(parents=True, exist_ok=True)
        for name in list(hashes) + ["REVISION_HASHES.json"]:
            (folder / name).write_bytes((HERE / name).read_bytes())
        ctx["store"].write(
            "execution_training_precision_revision.json",
            dict(
                revision="strict-training-and-joint-float64-retry-v2",
                revision_sha=revision,
                original_run_id=EXPECTED_RUN,
                original_source_sha=BASE_SHA,
                revision_files=hashes,
                tolerance_changed=False,
                objective_changed=False,
                budgets_changed=False,
                training_arithmetic_changed_on_precision_recovery_batches=True,
                stored_dictionary_dtype="unchanged float32",
                all_previous_checkpoints_reused=True,
                note="Passing calls unchanged. Exact terminal KKT failures retry in float64. Affected atom sweeps use double intermediates then restore dictionary storage dtype. Joint reconstruction retains verified coefficient precision. Earlier execution overlays are preserved.",
            ),
        )
        print(
            "VERIFIED SAME SAVED RUN:",
            EXPECTED_RUN,
            "; training and joint NNLS precision recovery enabled.",
            flush=True,
        )
        return ctx

    study.prepare = prepare
    return partial, joint, original_paths
