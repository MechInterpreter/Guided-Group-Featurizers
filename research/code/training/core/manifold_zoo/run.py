"""Native baseline-only reconstruction of the reported Manifold Zoo experiment.
Released classes, original training/selection/audit functions; no old repair method.
"""

import copy, csv, hashlib, importlib.metadata, inspect, json, os, time, traceback
from pathlib import Path
import numpy as np
import torch
from bsf import VanillaBSF, GrassmannianBSF, GroupLassoBSF
from bsf.train import recon_r2, l0_dead
from . import data, metrics
from .storage import ArtifactStore

CLASSES = dict(
    vanilla=VanillaBSF, grassmannian=GrassmannianBSF, group_lasso=GroupLassoBSF
)


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def sha_weights(w):
    h = hashlib.sha256()
    for k, v in sorted(w.items()):
        a = v.detach().cpu().numpy()
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def weights(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


class Store:
    def __init__(self, root, cfg, source_sha, local=None, require_drive=True):
        self.output = Path(root)
        self.cfg = cfg
        self.require_drive = require_drive
        self.cache = {}
        self.output.mkdir(parents=True, exist_ok=True)
        versions = {
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "scipy", "scikit-learn", "einops")
        }
        identity = dict(config=cfg, source_sha=source_sha, versions=versions)
        self.local = ArtifactStore(
            Path(local or "/content/agf_native_goodfire_rescue"), identity
        )
        self.drive = ArtifactStore(self.output / "checkpoints", identity)
        self.root = self.drive.root
        self.results = self.output / "results"
        self.results.mkdir(exist_ok=True)
        self.write("manifest.json", dict(identity=identity, run=self.drive.identity))

    def check_mount(self):
        if self.require_drive and not Path("/content/drive/MyDrive").is_dir():
            raise OSError(
                "Drive disconnected. Local rescue is saved; remount and rerun. No empty replacement Drive folder was created."
            )

    def load(self, key):
        if key in self.cache:
            return self.cache[key]
        candidates = []
        for s in (self.local, self.drive):
            r = s.load(key)
            if r is not None:
                candidates.append(r)
        if not candidates:
            return None
        out = (
            max(candidates, key=lambda r: r.get("_saved_ns", 0))
            if all(isinstance(r, dict) for r in candidates)
            else candidates[0]
        )
        if key == "manifold_zoo" or key.startswith("score_"):
            self.cache[key] = out
        return out

    def save(self, key, value, drive=True):
        value = (
            dict(value, _saved_ns=time.time_ns()) if isinstance(value, dict) else value
        )
        self.local.save(key, value)
        if key == "manifold_zoo" or key.startswith("score_"):
            self.cache[key] = value
        if drive:
            self.check_mount()
            self.drive.root.mkdir(parents=True, exist_ok=True)
            self.drive.save(key, value)
        return value

    def result(self, key, fn):
        old = self.load(key)
        if old is not None:
            print("Reused", key, flush=True)
            return old
        return self.save(key, fn())

    def write(self, name, value):
        raw = json.dumps(serial(value), indent=2, allow_nan=False)
        local = self.local.root / "json" / name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(raw)
        self.check_mount()
        path = self.results / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(raw)
        os.replace(tmp, path)
        return value

    def event(self, stage, **kw):
        r = dict(
            stage=stage,
            time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **serial(kw),
        )
        lp = self.local.root / "events.jsonl"
        with lp.open("a") as f:
            f.write(json.dumps(r, allow_nan=False) + "\n")
        self.write("progress.json", r)
        self.check_mount()
        (self.results / "events.jsonl").write_bytes(lp.read_bytes())
        print(stage, kw, flush=True)


def make_model(method, cfg):
    d, g, b, k = (
        cfg["dimension"],
        cfg["groups"],
        cfg["block_rank"],
        cfg["active_factors"],
    )
    if method == "sae":
        return VanillaBSF(d, g * b, 1, k * b)
    if method == "group_lasso":
        return GroupLassoBSF(d, g, b, target_l0=k)
    return CLASSES[method](d, g, b, k)


@torch.no_grad()
def encode(m, x, device, batch=2048):
    m.eval()
    z = []
    for lo in range(0, len(x), batch):
        z.append(
            m.encode(torch.as_tensor(x[lo : lo + batch], device=device)).cpu().numpy()
        )
    return np.concatenate(z), m.decoder_atoms().detach().cpu().numpy().reshape(
        m.n_groups, m.group_size, m.d
    )


def measure(m, x, device):
    z, D = encode(m, x, device)
    pred = np.einsum("ngr,grd->nd", z, D, optimize=True)
    on = np.linalg.norm(z, axis=2) > 1e-6
    return dict(
        mse=float(np.mean((pred.astype(float) - x) ** 2)),
        active_groups=float(on.sum(1).mean()),
        dead_groups=int((~on.any(0)).sum()),
        ve=float(1 - np.sum((pred - x) ** 2) / np.sum((x - x.mean(0)) ** 2)),
    )


def train_native(store, key, method, x, val, seed, lr, device, max_updates=None):
    """Resumable equivalent of released bsf.train(snr=0), not a replacement model.

    Preserve the trainer's floor batches, torch.randperm, Adam, OneCycleLR,
    loss/normalize calls, and logging-subsample RNG consumption. Additional
    validation is deterministic and does not mutate Group Lasso state.
    """
    cfg = store.cfg
    torch.manual_seed(seed)
    m = make_model(method, cfg).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    nb = len(x) // cfg["batch"]
    target = nb * cfg["epochs"]
    if nb < 1:
        raise ValueError("Training data smaller than batch")
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=target,
        pct_start=cfg["warmup"],
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )
    tx = torch.as_tensor(x, dtype=torch.float32)
    state = store.load(key)
    if state is None:
        state = dict(
            updates=0,
            weights=weights(m),
            best=None,
            best_score=None,
            best_update=None,
            history=[],
            permutation=None,
            opt=opt.state_dict(),
            scheduler=sched.state_dict(),
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if device.startswith("cuda")
            else None,
            seconds=0.0,
            method=method,
            seed=seed,
            lr=lr,
            target_updates=target,
            class_name=type(m).__module__ + "." + type(m).__name__,
            parameters=sum(p.numel() for p in m.parameters()),
        )
    else:
        m.load_state_dict(state["weights"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        if device.startswith("cuda") and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    print(key, "resume", state["updates"], "/", target, flush=True)
    stop = target if max_updates is None else min(target, max_updates)

    def checkpoint(drive=True):
        state.update(
            weights=weights(m),
            opt=copy.deepcopy(opt.state_dict()),
            scheduler=copy.deepcopy(sched.state_dict()),
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if device.startswith("cuda")
            else None,
        )
        store.save(key, state, drive)

    try:
        while state["updates"] < stop:
            u = state["updates"]
            epoch = u // nb
            bi = u % nb
            if bi == 0:
                state["permutation"] = torch.randperm(len(tx))
            ids = state["permutation"][bi * cfg["batch"] : (bi + 1) * cfg["batch"]]
            xb = tx[ids].to(device)
            m.train()
            t = time.monotonic()
            loss, info = m.loss(xb, target=xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            m.normalize_decoder()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native loss")
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            state["seconds"] += time.monotonic() - t
            state["updates"] += 1
            if state["updates"] % nb == 0:
                ep = epoch + 1
                if ep == 1 or ep == cfg["epochs"] or ep % cfg["log_epochs"] == 0:
                    m.eval()
                    sub = tx[torch.randperm(len(tx))[:20000]].to(device)
                    with torch.no_grad():
                        native_r2 = recon_r2(m, sub)
                        l0, dead = l0_dead(m, sub)
                    a = measure(m, val, device)
                    eligible = (
                        method != "group_lasso"
                        or abs(a["active_groups"] - cfg["active_factors"])
                        <= 0.1 * cfg["active_factors"]
                    )
                    row = dict(
                        update=state["updates"],
                        epoch=ep,
                        lr=float(opt.param_groups[0]["lr"]),
                        training_loss=float(loss.detach()),
                        native_subsample_r2=native_r2,
                        native_l0=l0,
                        native_dead=dead,
                        sparsity_matched=eligible,
                        **a,
                        native_loss_info=info,
                    )
                    state["history"].append(row)
                    if eligible and (
                        state["best_score"] is None or a["mse"] < state["best_score"]
                    ):
                        state.update(
                            best=weights(m),
                            best_score=a["mse"],
                            best_update=state["updates"],
                        )
                    store.event(
                        "native_training",
                        key=key,
                        epoch=ep,
                        target_epochs=cfg["epochs"],
                        **a,
                    )
                    store.write(key + "_history.json", dict(history=state["history"]))
            if state["updates"] % cfg["save_updates"] == 0 or state["updates"] == stop:
                checkpoint()
            elif state["updates"] % 100 == 0:
                checkpoint(False)
    except BaseException as exc:
        checkpoint(False)
        try:
            store.event(
                "failure", key=key, error=repr(exc), traceback=traceback.format_exc()
            )
            checkpoint()
        except Exception:
            pass
        raise
    return state


def choose_native(store, seed, method, states):
    eligible = [s for s in states if s["best"] is not None]
    if eligible:
        s = min(eligible, key=lambda s: s["best_score"])
        selected = s["best"]
        score = s["best_score"]
        matched = True
        field = "best"
    else:
        # Require matched L0 for replication comparisons.
        s = min(
            states,
            key=lambda s: abs(
                s["history"][-1]["active_groups"] - store.cfg["active_factors"]
            ),
        )
        selected = s["weights"]
        score = s["history"][-1]["mse"]
        matched = False
        field = "final_unmatched"
    out = dict(
        method=method,
        seed=seed,
        weights=selected,
        weights_sha=sha_weights(selected),
        lr=s["lr"],
        chosen_field=field,
        best_update=s["best_update"],
        selection_mse=score,
        sparsity_matched=matched,
        parameters=s["parameters"],
        class_name=s["class_name"],
        trials=[
            dict(
                lr=t["lr"],
                best_score=t["best_score"],
                best_update=t["best_update"],
                updates=t["updates"],
                final=t["history"][-1],
                seconds=t["seconds"],
            )
            for t in states
        ],
        factor_labels_used=False,
    )
    return store.save(f"s{seed}_{method}_selected", out)


def audit(store, key, w, method, ds, device):
    def run():
        m = make_model(method, store.cfg).to(device)
        m.load_state_dict(w)
        zc, D = encode(m, ds["packs"]["calibration"]["x"], device)
        zt, _ = encode(m, ds["packs"]["test"]["x"], device)
        out = metrics.evaluate(
            ds["packs"]["calibration"],
            ds["packs"]["test"],
            ds["truth"],
            zc,
            zt,
            D,
            ds["packs"]["train"]["x"].mean(0),
            heartbeat=lambda f, n: store.event(
                "factor_audit", key=key, factor=f, total=n
            ),
        )
        # Stricter one-fixed-group audit; keep separate from bounded greedy support recovery.
        cc = metrics.correlation_match(
            ds["packs"]["calibration"], zc, store.cfg["factors"]
        )
        count = 0
        collision = {}
        for r in out["factors"]:
            f = r["factor"]
            g = r["calibration_group"]
            ix, y = data.factor(ds["packs"]["calibration"], ds["truth"], f)
            cr = metrics.r2(y, zc[ix, g] @ D[g])
            r["calibration_locked_r2"] = cr
            compact = cr > 0.8 and r["heldout_matched_r2"] > 0.8
            r["compact_locked"] = compact
            if compact:
                count += 1
                collision.setdefault(g, []).append(f)
        out.update(
            compact_locked_count=count,
            collision_groups={str(g): fs for g, fs in collision.items() if len(fs) > 1},
            collision_excess=sum(max(0, len(fs) - 1) for fs in collision.values()),
            weights_sha=sha_weights(w),
        )
        if method == "sae":
            out["paper_sae_metric_note"] = (
                "Raw one-atom metric is not the paper ambient-rank atom-set metric; report separately."
            )
        store.write(key + "_audit.json", out)
        return out

    return store.result(key + "_audit", run)


def flatten_audit(seed, method, key, sc, extra=None):
    return dict(
        seed=seed,
        method=method,
        key=key,
        paper_mean_r2=sc["paper_mean_r2"],
        heldout_mean_r2=sc["heldout_mean_r2"],
        ve=sc["ve"],
        compact_count=sc["compact_locked_count"],
        collision_excess=sc["collision_excess"],
        mean_active_groups=sc["mean_active_groups"],
        bounded_multiple_fraction=sc["multiple_block_fraction"],
        **(extra or {}),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/manifold_zoo"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[2] / "configs/manifold_zoo.json",
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    store = Store(
        args.output, cfg, source_sha, args.output / "local_rescue", require_drive=False
    )
    ds = store.result(
        "manifold_zoo",
        lambda: data.generate(cfg, "goodfire", "independent", cfg["data_seed"]),
    )
    store.write("data_audit.json", data.audit(ds))
    rows = []
    for seed in cfg["seeds"]:
        chosen = {}
        for method in cfg["methods"]:
            states = [
                train_native(
                    store,
                    f"s{seed}_{method}_lr{i}",
                    method,
                    ds["packs"]["train"]["x"],
                    ds["packs"]["select"]["x"],
                    seed,
                    lr,
                    args.device,
                )
                for i, lr in enumerate(cfg["lrs"])
            ]
            chosen[method] = choose_native(store, seed, method, states)
        store.write(
            f"s{seed}_choices_frozen.json",
            {
                m: {k: v for k, v in r.items() if k != "weights"}
                for m, r in chosen.items()
            },
        )
        for method, r in chosen.items():
            key = f"s{seed}_{method}_selected"
            sc = audit(store, key, r["weights"], method, ds, args.device)
            rows.append(
                flatten_audit(
                    seed,
                    method,
                    key,
                    sc,
                    dict(
                        sparsity_matched=r["sparsity_matched"],
                        parameters=r["parameters"],
                    ),
                )
            )
        store.write(
            "summary.json",
            dict(
                rows=rows,
                scope="Reconstructed sampler; not byte-identical published data",
            ),
        )


if __name__ == "__main__":
    main()
