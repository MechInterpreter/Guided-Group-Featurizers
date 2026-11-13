"""Locked category-transfer protocol and resumable, isolated storage."""

import copy, hashlib, json, os, time, socket
from pathlib import Path
import numpy as np
import torch
from coco_baselines.storage import identity, sha_file, serial
from coco_adaptation.inputs import JournalStore
from coco_ggf.drive_recovery import install_drive_recovery

METHODS = ["ggf", "supervised_vanilla", "vanilla", "slics", "concept_sae"]
LABELS = dict(
    ggf="GGF",
    supervised_vanilla="Supervised Vanilla",
    vanilla="Vanilla BSF",
    slics="SLiCS",
    concept_sae="Concept-SAE",
)
SPEC = dict(
    revision="ggf-category-disjoint-v1",
    seeds=[21, 22, 23],
    groups=1024,
    width=12,
    top_k=16,
    warm_updates=7640,
    guided_updates=3000,
    lr=0.0003,
    cap=0.4,
    fixed_weights=[1.0, 0.2, 0.2],
    fixed_weight_candidates=[[0.05, 0.01], [0.2, 0.2], [0.4, 0.4]],
    images_per_update=8,
    snr=0.1,
    clip=5.0,
    evaluate_every=500,
    local_save_every=50,
    methods=METHODS,
    concept_epochs=[30, 50, 30],
    concept_tokenizer_LRs=[0.0001, 0.0003, 0.001],
    concept_free_tokens=36,
    nnls_tolerance=1e-5,
    nnls_iterations=2000,
    patch_sample=32,
    random_controls=3,
    probe_ridge_trace_fraction=0.001,
    intervention_relative_norm=0.1,
    image_strengths=[0.05, 0.10, 0.20],
    image_categories_per_fold=6,
    train_images=2000,
    development_images=400,
    calibration_images=400,
    evaluation_images=1500,
    readout="One existing unit selected by maximum regional Capture on confirm/calibration. Use the SAME unit for AP, fingerprints and interventions.",
    ordering="For each seed complete both category folds. Within each fold fit all five methods before calibration/evaluation. Publish interim results after each five-method block. Complete every primary block before supporting image edits.",
    scope="Category supervision transfer on previously inspected COCO allocations. Excluded objects can appear without annotations in reconstruction training. Not zero-shot readout selection, discovery of native concepts, or an independent new dataset.",
    inference="Existing native units only. BSFs: 1024 groups. SLiCS: one width-12 dictionary per guided category, independent NNLS projection. Concept-SAE: guided-category tokens plus 36 free tokens. No excluded-category slot is trained.",
    support="Do not shrink the fixed category denominator for failed/unsupported metrics. Save diagnostics and stop.",
    primary="Excluded-category focus, reproducibility, Capture, same-unit AP, shared evaluation-only ridge probe suppression and matched controls.",
    endpoint="Fixed BSF and SLiCS endpoints. Concept-SAE uses its repaired guided-only development selection and mask-quality gate. No settings change after evaluation is opened.",
)


def category_split(categories, eligible):
    order = sorted(
        eligible,
        key=lambda c: (
            hashlib.sha256(
                ("ggf-category-disjoint-v1|" + str(categories[c]["id"])).encode()
            ).hexdigest(),
            categories[c]["id"],
        ),
    )
    n = (len(order) + 1) // 2
    return [
        dict(fold=0, guide=sorted(order[n:]), excluded=sorted(order[:n])),
        dict(fold=1, guide=sorted(order[:n]), excluded=sorted(order[n:])),
    ]


def safe_output(output, sources):
    out = Path(output).resolve()
    for path in sources:
        src = Path(path).resolve()
        if out == src or src in out.parents or out in src.parents:
            raise ValueError("Output overlaps an original source: " + str(src))


def make_store(output, local, payload, drive=True):
    s = JournalStore(output, payload, local, drive)
    install_drive_recovery(s)
    return s


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(
        json.dumps(clean(serial(value)), indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def store_path(ctx, fold, seed, kind):
    return Path(ctx["options"]["output"]) / "fits" / f"fold{fold}_seed{seed}_{kind}"


def fit_payload(ctx, fold, seed, kind):
    return dict(
        protocol=SPEC,
        code_sha256=ctx["options"]["experiment_sha"],
        core_sha256=ctx["options"]["core_sha"],
        fold=fold,
        seed=seed,
        kind=kind,
        guide=ctx["folds"][fold]["guide"],
        excluded=ctx["folds"][fold]["excluded"],
        train=ctx["packs"]["train"]["identity"],
        development=ctx["packs"]["select"]["identity"],
        normalization=identity(ctx["norm"]),
        versions=ctx["versions"],
    )


def finish_model(s, state, kind, fold, seed):
    from coco_baselines.training import weight_sha

    weights = state["D"] if kind == "slics" else state["weights"]
    digest = weight_sha({"D": weights} if kind == "slics" else weights)
    s.save(
        "frozen",
        dict(
            kind=kind,
            fold=fold,
            seed=seed,
            weights=weights,
            weights_sha=digest,
            training_complete=True,
            guide=s.config["guide"],
            excluded=s.config["excluded"],
        ),
    )
    row = dict(
        kind=kind,
        fold=fold,
        seed=seed,
        weights_sha=digest,
        output=str(s.output),
        run_id=s.run_id,
        guide=s.config["guide"],
        excluded=s.config["excluded"],
    )
    s.write("FROZEN.json", row)
    return row


def flush(store):
    """Publish the latest COMPLETE local generations after a handled stop."""
    for folder in store.rescue.iterdir():
        if folder.is_dir() and list(folder.glob("slot*.json")):
            local = store._records(store.rescue, folder.name)
            if not local:
                continue
            _, meta, payload = local[0]
            record = json.loads(meta.read_text())
            # A completed Drive save was already read-back verified. Do not
            # reupload every old optimizer and evaluation shard on each stop.
            committed = False
            for remote in (store.root / folder.name).glob("slot*.json"):
                try:
                    r = json.loads(remote.read_text())
                    if (
                        r["run_id"] == store.run_id
                        and r["sha256"] == record["sha256"]
                        and remote.with_suffix(".pt").exists()
                    ):
                        committed = True
                        break
                except (OSError, ValueError, KeyError):
                    continue
            if committed:
                continue
            state = torch.load(payload, map_location="cpu", weights_only=False)
            store._width_drive_recovery_original_save(folder.name, state, drive=True)


class Lease:
    def __init__(self, root):
        self.path = Path(root) / "active_worker.json"
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if time.time() - old["time"] < 180:
                raise RuntimeError(
                    "Another worker has a recent Drive heartbeat. Stop it before starting a second writer. After an abrupt disconnect allow three minutes for the lease to expire."
                )
            self.path.unlink()
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump(dict(owner=self.owner, time=time.time()), f)

    def beat(self):
        if json.loads(self.path.read_text())["owner"] != self.owner:
            raise RuntimeError("Drive worker lease ownership changed")
        write_json(self.path, dict(owner=self.owner, time=time.time()))

    def release(self):
        if (
            self.path.exists()
            and json.loads(self.path.read_text()).get("owner") == self.owner
        ):
            self.path.unlink()
