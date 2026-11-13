"""Development-only, paired 0.40 versus saved 0.20 loss-bound screen.

No new warm-up, extraction, mapping, Vanilla training, or test scoring.
Each seed's final 3,000-update endpoint is primary; failed guards stay visible.
"""

import copy
import csv
import gc
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity, sha_file
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import matched_baselines as matched, width_balance as previous
from .width_balance_fit import fit, state_identity, Balancer, encoder_diagnostics
from .drive_recovery import install_drive_recovery

PARENT_RUN = "e4aeabc90db740daf7d9fc59183d7ecfe5e4c9cece99acea0aa334e6016c073a"
ALLOWED_SEEDS = [21, 22, 23]


def seeds_checked(seeds):
    seeds = list(seeds)
    if (
        not seeds
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
        or any(s not in ALLOWED_SEEDS for s in seeds)
    ):
        raise ValueError("Choose distinct seeds from [21, 22, 23]")
    return sorted(seeds)


def settings():
    recipe = previous.settings()
    recipe.update(
        version=1,
        study="ggf_weight_bound_development",
        seeds=ALLOWED_SEEDS,
        upper_bound=0.4,
        endpoint_rule="Final 2500 + 500 updates; no development-best selection",
        test_access=False,
        reused_warmups=True,
    )
    recipe["balance"]["bounds"]["capture"][1] = 0.4
    recipe["balance"]["bounds"]["detection"][1] = 0.4
    return recipe


def prepare(
    parent_output,
    activation_output,
    output,
    local,
    cache_local,
    source_sha,
    seeds=(21,),
    require_drive=True,
):
    seeds = seeds_checked(seeds)
    local, cache_local = Path(local), Path(cache_local)
    roots = [Path(parent_output).resolve(), Path(activation_output).resolve()]
    dest = Path(output).resolve()
    if any(dest == r or dest in r.parents or r in dest.parents for r in roots):
        raise ValueError("Output overlaps a read-only source")
    parent = matched.discover(parent_output, local / "parent_read", run=PARENT_RUN)
    cfg = parent.config
    if cfg["recipe"] != previous.settings() or cfg["versions"] != matched.versions():
        raise ValueError("Saved parent recipe or numerical versions differ")
    summary_path = parent.results / "summary.json"
    summary = json.loads(summary_path.read_text())
    frozen = memory.required(parent, "all_models_frozen")
    if (
        not summary.get("complete")
        or not frozen.get("complete")
        or len(frozen["models"]) != 15
    ):
        raise ValueError("The five-arm parent must be complete")
    catalog = {(r["seed"], r["arm"]): r for r in frozen["models"]}
    if len(catalog) != 15 or set(catalog) != {
        (s, a) for s in ALLOWED_SEEDS for a in previous.ARMS
    }:
        raise ValueError("Parent catalog is incomplete or repeats endpoints")
    original = matched.discover(
        activation_output, local / "activation_read", run=cfg["source_runs"]["original"]
    )
    if original.run_id != memory.EXPECTED_PARENT:
        raise ValueError("Unexpected activation source")
    device = cfg["fit_config"]["device"]
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Select a CUDA GPU runtime. CPU is only used for the bundled checks."
        )
    parent_hardware = memory.required(parent, "preflight")["hardware"]
    if torch.cuda.get_device_name() != parent_hardware["gpu"]:
        raise RuntimeError(
            "Select the same GPU model as the paired control: "
            + parent_hardware["gpu"]
            + ". CUDA random streams may differ across GPU models; this screen requires identical saved noise hashes."
        )
    base = dict(cfg["base"], block_rank=4, local_save_updates=50)
    fit_cfg = copy.deepcopy(cfg["fit_config"])
    if (
        base["groups"],
        base["active_groups"],
        fit_cfg["images_per_update"],
        fit_cfg["lr"],
        fit_cfg["snr"],
    ) != (1024, 16, 8, 3e-4, 0.1):
        raise ValueError("Parent capacity or update recipe differs")
    norm = memory.required(original, "normalization")
    provenance = json.loads((original.results / "provenance.json").read_text())
    train, select = provenance["splits"]["train"], provenance["splits"]["select"]
    if (
        (len(train), len(select)) != (2000, 400)
        or len(set(train)) != 2000
        or len(set(select)) != 400
        or set(train) & set(select)
    ):
        raise ValueError("Unexpected training/development split")
    if (
        norm["mean"].shape != (196, 768)
        or not np.isfinite(norm["mean"]).all()
        or not np.isfinite(norm["scale"])
        or norm["scale"] <= 0
    ):
        raise ValueError("Invalid frozen normalization")
    if (
        provenance["splits"]
        != json.loads((original.results / "coverage_plan.json").read_text())["splits"]
    ):
        raise ValueError("Saved split identities differ")
    recipe = settings()
    payload = dict(
        recipe=recipe,
        parent_run=parent.run_id,
        activation_run=original.run_id,
        source_sha=source_sha,
        versions=matched.versions(),
        base=base,
        fit_config=fit_cfg,
        normalization_sha=identity(norm),
        train_ids=train,
        development_ids=select,
        gpu=parent_hardware["gpu"],
        baseline_bound=0.2,
        treatment_bound=0.4,
        test_access=False,
        policy="RUN_SEEDS controls execution only; unchanged per-seed identities permit later expansion",
    )
    store = inputs.JournalStore(output, payload, local / "rescue", require_drive)
    install_drive_recovery(store)
    stamps = memory.source_snapshot([parent, original])
    for path in [
        original.results / "provenance.json",
        original.results / "coverage_plan.json",
    ]:
        stamps.append(dict(path=str(path), sha256=sha_file(path)))
    ctx = dict(
        store=store,
        sources=dict(width_balance=parent, original=original),
        parent=parent,
        catalog=catalog,
        base=base,
        cfg=fit_cfg,
        recipe=recipe,
        device=device,
        local=local,
        seeds=seeds,
        stamps=stamps,
        checkpoint_stamps=[],
        packs={},
        source_sha=source_sha,
    )
    ctx["checkpoint_stamps"].append(matched.source_key_stamp(parent, "preflight"))
    # Verify all requested input phases before any model is trained.
    for seed in seeds:
        verify_seed(ctx, seed)
    memory.memory_safe_metrics()
    chunk = original.config["config"]["extract_chunk"]
    for split in ["train", "select"]:
        store.event(
            "loading_cached_split", split=split, images=len(provenance["splits"][split])
        )
        ctx["packs"][split] = memory.normalized_pack(
            original, split, provenance, norm, cache_local, chunk=chunk
        )
        for lo in range(0, len(provenance["splits"][split]), chunk):
            ctx["checkpoint_stamps"].append(
                matched.source_key_stamp(original, f"features_{split}_{lo:06d}")
            )
    store.write("protocol.json", payload)
    store.write("requested_seeds.json", dict(seeds=seeds, test_access=False))
    store.write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    matched.assert_sources(ctx)
    return ctx


def verify_seed(ctx, seed):
    parent = ctx["parent"]
    row = ctx["catalog"][(seed, "mask4_balanced")]
    warm_key = f"warm4_mask_s{seed}"
    p1_key = f"phase1_mask4_balanced_s{seed}"
    p2_key = f"phase2_mask4_balanced_s{seed}"
    warm, p1, p2 = [memory.required(parent, k) for k in [warm_key, p1_key, p2_key]]
    for st, n in [(warm, 7640), (p1, 2500), (p2, 500)]:
        matched.complete(st, n)
    if (
        warm["best"] is None
        or row["rank"] != 4
        or row["key"] != p2_key
        or not row["balanced"]
        or not row["supervised"]
    ):
        raise ValueError("Unexpected width-four parent endpoint")
    targets = row["targets"]
    matched.check_targets(targets, ctx["cfg"])
    if (
        identity(targets) != row["targets_sha"]
        or training.weight_sha(p2["weights"]) != row["weights_sha"]
        or training.weight_sha({"head": p2["head"]}) != row["head_sha"]
    ):
        raise ValueError("Frozen endpoint/readout identities differ")
    for st in [warm["best"], p1["weights"], p2["weights"]]:
        matched.valid_weights(st, "vanilla", ctx["base"])
    parent_recipe = previous.settings()["balance"]
    # The full fingerprints include IDs; use the activation provenance to reproduce them.
    original = ctx["sources"]["original"]
    provenance = json.loads((original.results / "provenance.json").read_text())
    for st, initial, n, rng_seed, head, balance in [
        (p1, warm["best"], 2500, 71000 + seed, None, None),
        (p2, p1["weights"], 500, 82000 + seed, p1["head"], p1["balance"]),
    ]:
        expected = identity(
            dict(
                initial=training.weight_sha(initial),
                targets=targets,
                updates=n,
                seed=rng_seed,
                balanced=True,
                supervised=True,
                reference_mse=row["reference_mse"],
                initial_head_sha=None
                if head is None
                else training.weight_sha({"head": head}),
                initial_balance=state_identity(balance),
                base=ctx["base"],
                cfg=ctx["cfg"],
                balance=parent_recipe,
                train_ids=provenance["splits"]["train"],
                select_ids=provenance["splits"]["select"],
            )
        )
        if st["fingerprint"] != expected:
            raise ValueError("Parent phase lineage/fingerprint differs")
    if p2["history"][-1]["update"] != 500:
        raise ValueError("Missing final development evaluation")
    baseline = p2["history"][-1]["validation"]
    validate_metrics(baseline)
    for key in [warm_key, p1_key, p2_key]:
        stamp = matched.source_key_stamp(parent, key)
        if stamp not in ctx["checkpoint_stamps"]:
            ctx["checkpoint_stamps"].append(stamp)
    ctx["store"].write(
        f"histories/reused_cap020_s{seed}_phase1.json",
        previous.json_safe(p1["history"]),
    )
    ctx["store"].write(
        f"histories/reused_cap020_s{seed}_phase2.json",
        previous.json_safe(p2["history"]),
    )
    return row, warm, p1, p2


def validate_metrics(v):
    if not np.isfinite(
        [
            v["macro_capture"],
            v["macro_ap"],
            v["reconstruction"]["ve"],
            v["reconstruction"]["mse"],
        ]
    ).all():
        raise FloatingPointError("Nonfinite development metric")


def compare(seed, baseline, treatment, reference_mse):
    validate_metrics(baseline)
    validate_metrics(treatment)
    limit = reference_mse * 1.02
    result = dict(
        seed=seed,
        split="development",
        capture_gain_pp=100 * (treatment["macro_capture"] - baseline["macro_capture"]),
        AP_gain_pp=100 * (treatment["macro_ap"] - baseline["macro_ap"]),
        VE_gain_pp=100
        * (treatment["reconstruction"]["ve"] - baseline["reconstruction"]["ve"]),
        original_native_MSE_limit=limit,
        reconstruction_guard_passed=bool(treatment["reconstruction"]["mse"] <= limit),
    )
    result["all_three_point_estimates_improve"] = all(
        result[k] > 0 for k in ["capture_gain_pp", "AP_gain_pp", "VE_gain_pp"]
    )
    result["passes_screen"] = (
        result["all_three_point_estimates_improve"]
        and result["reconstruction_guard_passed"]
    )
    return result


def run(ctx):
    results = []
    for seed in ctx["seeds"]:
        row, warm, baseline1, baseline2 = verify_seed(ctx, seed)
        targets = row["targets"]
        ref = row["reference_mse"]
        ctx["store"].event("higher_bound_seed_start", seed=seed, upper_bound=0.4)
        first = fit(
            ctx,
            f"phase1_cap040_s{seed}",
            warm["best"],
            targets,
            2500,
            71000 + seed,
            True,
            True,
            reference_mse=ref,
        )
        second = fit(
            ctx,
            f"phase2_cap040_s{seed}",
            first["weights"],
            targets,
            500,
            82000 + seed,
            True,
            True,
            reference_mse=ref,
            initial_head=first["head"],
            initial_balance=first["balance"],
        )
        for new, old, n in [(first, baseline1, 2500), (second, baseline2, 500)]:
            matched.complete(new, n)
            for field in ["sampled_batches_sha", "sampled_noise_sha"]:
                if new[field] != old[field]:
                    raise ValueError("Paired sampling/noise diverged: " + field)
        baseline = baseline2["history"][-1]["validation"]
        treatment = second["history"][-1]["validation"]
        comparison = compare(seed, baseline, treatment, ref)
        result = dict(
            seed=seed,
            baseline=baseline,
            treatment=treatment,
            comparison=comparison,
            baseline_weights_sha=row["weights_sha"],
            treatment_weights_sha=training.weight_sha(second["weights"]),
            targets_sha=row["targets_sha"],
            warm_weights_sha=training.weight_sha(warm["best"]),
            final_weights=second["history"][-1]["weights_next"],
            new_updates=3000,
            test_access=False,
            phase1_batch_sha=first["sampled_batches_sha"],
            phase2_batch_sha=second["sampled_batches_sha"],
            phase1_noise_sha=first["sampled_noise_sha"],
            phase2_noise_sha=second["sampled_noise_sha"],
        )
        ctx["store"].save(f"result_s{seed}", result)
        ctx["store"].write(f"reports/s{seed}.json", previous.json_safe(result))
        results.append(result)
        del first, second, warm, baseline1, baseline2
        gc.collect()
        torch.cuda.empty_cache()
    out = dict(
        complete=True,
        study_complete=ctx["seeds"] == ALLOWED_SEEDS,
        requested_seeds=ctx["seeds"],
        results=results,
        test_access=False,
        endpoint="Final 3000 guided updates per seed",
        all_requested_seeds_pass_screen=all(
            r["comparison"]["passes_screen"] for r in results
        ),
        interpretation="Development screen only. No independent confirmation or statistical significance claim.",
    )
    tables_and_plots(ctx, out)
    matched.assert_sources(ctx)
    ctx["store"].save("requested_screen_complete", out)
    ctx["store"].write("summary.json", previous.json_safe(out))
    ctx["store"].event(
        "development_screen_complete",
        seeds=ctx["seeds"],
        study_complete=out["study_complete"],
    )
    return out


def tables_and_plots(ctx, out):
    rows = []
    comparisons = []
    for r in out["results"]:
        comparisons.append(r["comparison"])
        for name, bound in [("baseline", 0.2), ("treatment", 0.4)]:
            v = r[name]
            rows.append(
                dict(
                    seed=r["seed"],
                    upper_bound=bound,
                    capture_pct=100 * v["macro_capture"],
                    patch_AP_pct=100 * v["macro_ap"],
                    VE_pct=100 * v["reconstruction"]["ve"],
                    MSE=v["reconstruction"]["mse"],
                    native_eligible=v["reconstruction"]["mse"]
                    <= r["comparison"]["original_native_MSE_limit"],
                )
            )
    s = ctx["store"]
    matched.csv_write(s.results / "development_comparison.csv", rows)
    matched.csv_write(s.results / "development_gains.csv", comparisons)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    for ax, key, title in zip(
        axes,
        ["capture_pct", "patch_AP_pct", "VE_pct"],
        ["Capture", "Patch AP", "Reconstruction VE"],
    ):
        for j, bound in enumerate([0.2, 0.4]):
            values = [r[key] for r in rows if r["upper_bound"] == bound]
            mean = float(np.mean(values))
            ax.bar(j, mean, color=["#667b99", "#008779"][j])
            ax.scatter(np.full(len(values), j), values, color="#222222", s=12)
            ax.text(j, mean + 1, f"{mean:.2f}%", ha="center", fontsize=10)
        ax.set_xticks([0, 1], ["Bound 0.20", "Bound 0.40"])
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Development only; seeds " + ", ".join(map(str, out["requested_seeds"]))
    )
    folder = s.results / "figures"
    folder.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(folder / ("weight_bound." + ext), dpi=220, bbox_inches="tight")
    plt.close(fig)
    pressure = []
    for seed in out["requested_seeds"]:
        for phase in [1, 2]:
            for label, bound in [("cap020", 0.2), ("cap040", 0.4)]:
                if label == "cap020":
                    path = (
                        s.results / f"histories/reused_cap020_s{seed}_phase{phase}.json"
                    )
                else:
                    path = s.results / f"histories/phase{phase}_cap040_s{seed}.json"
                for entry in json.loads(path.read_text()):
                    diagnostic = entry.get("diagnostics") or {}
                    adaptation = diagnostic.get("adaptation") or {}
                    used = entry.get("weights_used")
                    if used is None:
                        continue
                    target = adaptation.get("target_norms")
                    strengths = diagnostic.get("weighted_encoder_gradient_norms")
                    for j, key in [(1, "contrast"), (2, "detection")]:
                        pressure.append(
                            dict(
                                seed=seed,
                                upper_bound=bound,
                                phase=phase,
                                update=entry["update"],
                                loss=key,
                                weight_used=used[j],
                                at_upper_bound=bool(used[j] >= bound - 1e-6),
                                target_exceeds_current_strength=None
                                if target is None or strengths is None
                                else bool(target[j] > strengths[j]),
                            )
                        )
    matched.csv_write(s.results / "bound_pressure_snapshots.csv", pressure)


def preflight(ctx):
    """Exercise full-size GPU losses and adaptation without changing saved RNG."""
    device = ctx["device"]
    cfg = ctx["cfg"]
    store = ctx["store"]
    hardware = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    old = store.load("preflight")
    if old is not None and old["hardware"] == hardware:
        timing = old["seconds_per_update"]
        peak = old["peak_GiB"]
    else:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        try:
            row, warm, p1, p2 = verify_seed(ctx, ctx["seeds"][0])
            m = core.new_model("vanilla", ctx["base"], warm["best"], device)
            head = torch.nn.Parameter(torch.zeros(75, 2, device=device))
            opt = torch.optim.Adam(list(m.parameters()) + [head], lr=cfg["lr"])
            controller = Balancer(ctx["recipe"]["balance"], device)
            pack = ctx["packs"]["train"]
            idx = np.arange(cfg["images_per_update"])
            x = torch.as_tensor(np.array(pack["x"][idx], copy=True), device=device)
            xx = x.reshape(-1, 768)
            timings = []
            torch.cuda.reset_peak_memory_stats()
            for i in range(12):
                torch.cuda.synchronize()
                start = time.monotonic()
                rec, _ = m.loss(
                    xx + cfg["snr"] * xx.std() * torch.randn_like(xx), target=xx
                )
                cap, det, _ = core.supervised_loss(
                    m, x, [pack["masks"][j] for j in idx], row["targets"], head, cfg
                )
                losses = [rec, cap, det]
                if not all(torch.isfinite(v) for v in losses):
                    raise FloatingPointError("Nonfinite preflight loss")
                used = controller.weights().detach()
                if controller.reference is None:
                    controller.reference = np.asarray(
                        [float(v.detach()) for v in losses]
                    )
                if (i + 1) % 10 == 0:
                    norms, _ = encoder_diagnostics(m, losses)
                    controller.adapt(losses, norms)
                opt.zero_grad()
                (rec + used[1] * cap + used[2] * det).backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    list(m.parameters()) + [head], cfg["grad_clip"]
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite preflight gradient")
                opt.step()
                m.normalize_decoder()
                torch.cuda.synchronize()
                if i >= 2:
                    timings.append(time.monotonic() - start)
            timing = float(np.mean(timings))
            peak = torch.cuda.max_memory_allocated() / 2**30
            del m, head, opt, controller, x, xx, row, warm, p1, p2
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
    pending = {}
    for seed in ctx["seeds"]:
        pending[seed] = 0
        for phase, n in [(1, 2500), (2, 500)]:
            st = store.load(f"phase{phase}_cap040_s{seed}")
            pending[seed] += n - (0 if st is None else st["updates"])
            del st
    out = dict(
        hardware=hardware,
        seconds_per_update=timing,
        peak_GiB=peak,
        pending_updates=pending,
        estimated_gradient_hours=sum(pending.values()) * timing / 3600,
        note="Gradient time only; development evaluations and verified Drive writes add time. No new warm-up or test scoring.",
    )
    store.save("preflight", out)
    store.write("preflight.json", out)
    matched.assert_sources(ctx)
    return out


def export(ctx, source):
    path = matched.export(ctx, source)
    new = path.with_name(
        "ggf_weight_bound_" + ctx["store"].run_id[:12] + "_results.zip"
    )
    path.replace(new)
    return new


def failure(ctx, stage, exc, source):
    try:
        ctx["store"].write(
            "failure.json",
            dict(
                stage=stage,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                resume="Rerun unchanged. Incomplete updates replay the latest verified checkpoint.",
            ),
        )
        print("Partial evidence:", export(ctx, source), flush=True)
    except BaseException as err:
        print("Partial export failed:", type(err).__name__, str(err), flush=True)
