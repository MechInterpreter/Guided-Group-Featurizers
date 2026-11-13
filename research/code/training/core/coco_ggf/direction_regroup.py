"""Frozen successful GGF direction atlas and capacity-preserving exchanges.

COCO masks supervise analysis, proposals and recovery. Inference uses native
GGF TopK, with no annotation input, new routing or category reassignment.
"""

import copy
import csv
import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import core, inputs
from coco_baselines import training
from coco_baselines.storage import identity, serial
from coco_coordinated import run as memory
from . import owner_routing as old, bank_experiment as bank, study
from .owner_model import OwnerGGF, supervised_terms


def settings():
    return dict(
        version=1,
        seed=21,
        shard_images=25,
        top_incoming=5,
        frozen_screen_candidates=8,
        recovery_candidates=3,
        screening_images=512,
        recovery_updates=300,
        restricted_updates=100,
        evaluate_every=100,
        local_save_every=25,
        gradient_every=100,
        images_per_update=8,
        lr=3e-4,
        snr=0.1,
        lambda_capture=0.05,
        lambda_detection=0.01,
        grad_clip=5.0,
        training_seed=105061,
        bootstrap=200,
        max_ap_drop=0.001,
        max_ve_drop=0.001,
        max_mse_growth=0.02,
        minimum_target_gain=0.005,
        minimum_mean_gain=0.001,
        max_category_drop=0.01,
        scope="One saved seed, supervised development analysis. Historical confirmation is not a fresh test.",
    )


def prepare(parent, output, local, sha, versions, device, require_drive=True):
    cfg = settings()
    ctx = old.prepare(
        parent,
        Path(output) / "inputs",
        local,
        sha,
        versions,
        device=device,
        require_drive=require_drive,
    )
    ctx["cfg"] = dict(
        ctx["eval_cfg"],
        **{
            k: cfg[k]
            for k in (
                "lr",
                "snr",
                "lambda_capture",
                "lambda_detection",
                "grad_clip",
                "images_per_update",
            )
        },
    )
    ctx["protocol"] = cfg
    ctx["store"] = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            source_sha=sha,
            versions=versions,
            device=device,
            parent_run=ctx["source"].run_id,
            activation_run=ctx["original"].run_id,
        ),
        Path(local) / "direction_rescue",
        require_drive,
    )
    w, h, t, row = old.source_start(ctx, cfg["seed"])
    ctx["initial"] = bank.native_snapshot(ctx, w, t, h, 1024, 3, 16)
    ctx["source_weights_sha"] = row["weights_sha"]
    ctx["store"].write(
        "activation_identity.json",
        {
            k: dict(
                image_count=len(p["ids"]),
                shape=list(p["x"].shape),
                normalized_identity=p.get("identity"),
                raw_float32_sha=p.get("raw_float32_sha"),
            )
            for k, p in ctx["packs"].items()
        },
    )
    ctx["store"].write(
        "split_roles.json",
        dict(
            ids={k: p["ids"] for k, p in ctx["packs"].items()},
            train="Direction profiles, proposal ranking, frozen gate screening, recovery",
            select="Historical development profiles, checkpoint/acceptance selection; not a fresh test",
            final_test_opened=False,
            category_readouts=t,
            source_weights_sha=row["weights_sha"],
        ),
    )
    return ctx


def model(ctx, snap):
    m = bank.make(ctx, snap)
    m.total_atoms = m.n_groups * m.group_size
    return m


def permute_snapshot(snap, outgoing, incoming):
    """Move encoder column+bias and decoder row together. Do not change heads/readouts."""
    n = snap["weights"]["W_dec"].shape[0]
    if snap["spec"]["rank"] != 3:
        raise ValueError("This diagnostic keeps width three")
    if not (0 <= outgoing < n and 0 <= incoming < n) or outgoing // 3 == incoming // 3:
        raise ValueError(
            "Exchange requires distinct original groups and valid atom IDs"
        )
    result = copy.deepcopy(snap)
    p = torch.arange(n)
    p[outgoing], p[incoming] = p[incoming].clone(), p[outgoing].clone()
    w = result["weights"]
    w["W_enc"] = w["W_enc"][:, p].clone()
    w["W_dec"] = w["W_dec"][p].clone()
    w["b_enc"] = w["b_enc"][p].clone()
    previous = snap["spec"].get("atom_ids", list(range(n)))
    result["spec"]["atom_ids"] = [previous[j] for j in p.tolist()]
    return result


@torch.no_grad()
def preflight(ctx):
    m = model(ctx, ctx["initial"])
    m.eval()
    x = torch.as_tensor(
        np.array(ctx["packs"]["train"]["x"][0], copy=True), device=ctx["device"]
    )
    # Check against the native released class, not merely against this wrapper.
    native = core.new_model(
        "vanilla",
        ctx["base"],
        {k: ctx["initial"]["weights"][k] for k in ("W_enc", "W_dec", "b_enc")},
        ctx["device"],
    )
    base_error = float((m(x)[0] - native(x)[0]).abs().max())
    changed = permute_snapshot(ctx["initial"], 0, 3)
    edited = model(ctx, changed)
    a = x @ m.W_enc + m.b_enc
    b = x @ edited.W_enc + edited.b_enc
    before = a @ m.W_dec
    after = b @ edited.W_dec
    relative = float((before - after).norm() / before.norm().clamp_min(1e-8))
    restored = permute_snapshot(changed, 0, 3)
    same = all(
        torch.equal(restored["weights"][k], ctx["initial"]["weights"][k])
        for k in ("W_enc", "W_dec", "b_enc")
    )
    report = dict(
        native_max_error=base_error,
        ungated_exchange_relative_error=relative,
        swap_twice_restores_weights=same,
        groups=m.n_groups,
        width=m.group_size,
        active_groups=m.l0,
        active_coordinates=m.l0 * m.group_size,
        numerical_ok=base_error < 1e-5 and relative < 1e-5 and same,
        note="Ungated equality is checked separately; native TopK predictions can change after exchange.",
    )
    ctx["store"].write("preflight.json", report)
    del m, native, edited
    gc.collect()
    if not report["numerical_ok"]:
        raise RuntimeError(
            "Conversion/preflight failed; no recovery training permitted"
        )
    return report


def timing(ctx):
    """Read-only profile and transient update timing; never a promised ETA."""
    m = model(ctx, ctx["initial"])
    p = ctx["packs"]["train"]
    study.sync(ctx["device"])
    start = time.perf_counter()
    shard_stats(m, p, 0, 1, ctx["cfg"])
    study.sync(ctx["device"])
    profile = time.perf_counter() - start
    ix = np.arange(min(ctx["protocol"]["images_per_update"], len(p["ids"])))
    x = torch.as_tensor(np.array(p["x"][ix], copy=True), device=ctx["device"])
    masks = [p["masks"][i] for i in ix]
    opt = torch.optim.Adam(m.parameters(), lr=ctx["protocol"]["lr"])
    times = []
    for _ in range(3):
        study.sync(ctx["device"])
        start = time.perf_counter()
        m.train()
        xx = x.reshape(-1, m.d)
        rec, _ = m.loss(xx, target=xx)
        cap, det, _, _ = supervised_terms(m, x, masks, ctx["cfg"])
        loss = rec + 0.05 * cap + 0.01 * det
        opt.zero_grad()
        loss.backward()
        opt.step()
        study.sync(ctx["device"])
        times.append(time.perf_counter() - start)
    # This temporary benchmark model is discarded; the saved start is unchanged.
    updates = ctx["protocol"]["recovery_updates"] * (
        1 + 2 * ctx["protocol"]["recovery_candidates"]
    )
    result = dict(
        seconds_per_profile_image=profile,
        transient_seconds_per_gradient_update=float(np.median(times)),
        maximum_initial_recovery_updates=updates,
        conditional_search_budget_control_updates=updates,
        rough_profile_compute_minutes=profile
        * sum(len(v["ids"]) for v in ctx["packs"].values())
        / 60,
        rough_initial_gradient_compute_minutes=float(np.median(times)) * updates / 60,
        excludes="All validation, frozen screening, profiling saves, downloads, proposal construction, confirmation and Drive. Warm-cache transient measurements, not an ETA.",
    )
    ctx["store"].write("timing.json", result)
    del m, opt
    gc.collect()
    return result


def empty_stats(C, A):
    return dict(
        gain=np.zeros((C, A)),
        remove=np.zeros((C, 3)),
        cross=np.zeros((C, 3, A)),
        raw_fg=np.zeros((C, A)),
        raw_bg=np.zeros((C, A)),
        gated_fg=np.zeros((C, A)),
        signed_delta=np.zeros((C, A)),
        exclusion=np.zeros((C, A)),
        energy=np.zeros(C),
        counts=np.zeros(C, dtype=np.int64),
    )


@torch.no_grad()
def shard_stats(m, pack, lo, hi, cfg):
    C = len(m.targets)
    A = m.total_atoms
    out = empty_stats(C, A)
    lookup = {t["category_index"]: c for c, t in enumerate(m.targets)}
    D = m.W_dec.detach()
    norm = D.square().sum(1)
    for i in range(lo, hi):
        x = torch.as_tensor(np.array(pack["x"][i], copy=True), device=m.W_dec.device)
        raw = x @ m.W_enc + m.b_enc
        z = m.encode(x).reshape(len(x), A)
        ww, _, cc = core.object_weights([pack["masks"][i]], list(lookup), cfg)
        if not len(cc):
            continue
        wt = torch.as_tensor(ww, dtype=x.dtype, device=x.device)
        dx = wt @ x
        da = wt @ raw
        dz = wt @ z
        for k, cat in enumerate(cc):
            c = lookup[int(cat)]
            g = m.targets[c]["contrast_group"]
            ids = list(range(3 * g, 3 * g + 3))
            r = dx[k] - dz[k, ids] @ D[ids]
            v = da[k]
            v0 = dz[k, ids]
            out["gain"][c] += (
                (2 * v * (D @ r) - v.square() * norm).double().cpu().numpy()
            )
            out["remove"][c] += (
                (-2 * v0 * (D[ids] @ r) - v0.square() * norm[ids])
                .double()
                .cpu()
                .numpy()
            )
            out["cross"][c] += (
                (2 * v0[:, None] * v[None, :] * (D[ids] @ D.T)).double().cpu().numpy()
            )
            # Positive/negative normalized instance weights; no background semantics assumed.
            fw = wt[k].clamp_min(0)
            bw = (-wt[k]).clamp_min(0)
            fw = fw / fw.sum().clamp_min(1e-12)
            bw = bw / bw.sum().clamp_min(1e-12)
            out["raw_fg"][c] += (fw @ raw.abs()).double().cpu().numpy()
            out["raw_bg"][c] += (bw @ raw.abs()).double().cpu().numpy()
            out["gated_fg"][c] += (fw @ z.abs()).double().cpu().numpy()
            out["signed_delta"][c] += v.double().cpu().numpy()
            excluded = (raw.abs() > 1e-6) & (z.abs() <= 1e-6)
            exclusion_rate = (fw @ excluded.float()) / (
                fw @ (raw.abs() > 1e-6).float()
            ).clamp_min(1e-12)
            out["exclusion"][c] += exclusion_rate.double().cpu().numpy()
            out["energy"][c] += float(dx[k].square().sum())
            out["counts"][c] += 1
    return out


def profiles(ctx, split):
    s = ctx["store"]
    key = "profiles_" + split
    cached = s.load(key)
    if cached is not None:
        s.event("reused_profiles", split=split)
        return cached
    m = model(ctx, ctx["initial"])
    m.eval()
    pack = ctx["packs"][split]
    total = empty_stats(len(m.targets), m.total_atoms)
    for lo in range(0, len(pack["ids"]), ctx["protocol"]["shard_images"]):
        hi = min(lo + ctx["protocol"]["shard_images"], len(pack["ids"]))
        shard = s.load(f"profile_{split}_{lo:05d}")
        if shard is None:
            shard = shard_stats(m, pack, lo, hi, ctx["cfg"])
            shard["image_ids"] = pack["ids"][lo:hi]
            s.save(f"profile_{split}_{lo:05d}", shard)
        if shard["image_ids"] != pack["ids"][lo:hi]:
            raise ValueError("Profile image order changed")
        for k in total:
            total[k] += shard[k]
        s.event("direction_profiles", split=split, done=hi, total=len(pack["ids"]))
    if (total["counts"] == 0).any() or (total["energy"] <= 0).any():
        raise ValueError("A declared category has no usable contrast; do not drop it")
    for k in ("gain", "remove", "cross"):
        total[k] /= total["energy"].reshape((-1,) + (1,) * (total[k].ndim - 1))
    for k in ("raw_fg", "raw_bg", "gated_fg", "signed_delta", "exclusion"):
        total[k] /= total["counts"][:, None]
    total["category_indices"] = np.array([t["category_index"] for t in m.targets])
    total["readout_groups"] = np.array([t["contrast_group"] for t in m.targets])
    s.save(key, total)
    folder = s.results / "directions"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (split + "_profiles.npz"), **total)
    del m
    gc.collect()
    return total


def rank_proposals(train, targets, cfg):
    """Exact two-vector exchange algebra under RAW/additive approximation.

    Includes donor cost across all categories currently read from either group.
    Actual changed TopK must subsequently be measured, not inferred from this score.
    """
    groups = train["readout_groups"]
    A = train["gain"].shape[1]
    rows = []
    seen = set()
    for c, t in enumerate(targets):
        receiver = int(groups[c])
        outs = range(3 * receiver, 3 * receiver + 3)
        order = np.argsort(-train["gain"][c], kind="stable")
        incoming = [int(j) for j in order if int(j) // 3 != receiver][
            : cfg["top_incoming"]
        ]
        for j in incoming:
            donor = j // 3
            for i in outs:
                target_gain = (
                    train["gain"][c, j]
                    + train["remove"][c, i % 3]
                    + train["cross"][c, i % 3, j]
                )
                if target_gain <= 0:
                    continue
                pair = tuple(sorted((i, j)))
                changes = np.zeros(len(groups))
                for k, g in enumerate(groups):
                    if g == receiver:
                        changes[k] = (
                            train["gain"][k, j]
                            + train["remove"][k, i % 3]
                            + train["cross"][k, i % 3, j]
                        )
                    elif g == donor:
                        changes[k] = (
                            train["gain"][k, i]
                            + train["remove"][k, j % 3]
                            + train["cross"][k, j % 3, i]
                        )
                row = dict(
                    category=t["category"],
                    category_index=int(t["category_index"]),
                    outgoing=i,
                    incoming=j,
                    receiver_group=receiver,
                    donor_group=donor,
                    target_predicted_gain=float(target_gain),
                    mean_predicted_gain=float(changes.mean()),
                    worst_predicted_category_gain=float(changes.min()),
                    excluded_fraction=float(train["exclusion"][c, j]),
                    affected_categories=[
                        int(targets[k]["category_index"])
                        for k in np.flatnonzero(
                            (groups == receiver) | (groups == donor)
                        )
                    ],
                    instances=int(train["counts"][c]),
                )
                # Retain the target orientation with the strongest mean/target score for a duplicate pair.
                if pair in seen:
                    ix = next(
                        k
                        for k, r in enumerate(rows)
                        if tuple(sorted((r["outgoing"], r["incoming"]))) == pair
                    )
                    if (row["mean_predicted_gain"], row["target_predicted_gain"]) > (
                        rows[ix]["mean_predicted_gain"],
                        rows[ix]["target_predicted_gain"],
                    ):
                        rows[ix] = row
                else:
                    seen.add(pair)
                    rows.append(row)
    rows.sort(
        key=lambda r: (
            -r["mean_predicted_gain"],
            -r["target_predicted_gain"],
            r["outgoing"],
            r["incoming"],
        )
    )
    for j, r in enumerate(rows):
        r["id"] = f"exchange{j:03d}"
    return rows


def atlas(ctx, train, dev):
    s = ctx["store"]
    targets = ctx["initial"]["spec"]["targets"]
    rows = []
    for c, t in enumerate(targets):
        order = np.argsort(-train["gain"][c], kind="stable")[:12]
        for rank, j in enumerate(order, 1):
            rows.append(
                dict(
                    category=t["category"],
                    category_index=t["category_index"],
                    rank=rank,
                    atom=int(j),
                    current_group=int(j) // 3,
                    train_gain_pp=float(train["gain"][c, j] * 100),
                    development_gain_pp=float(dev["gain"][c, j] * 100),
                    train_raw_object_magnitude=float(train["raw_fg"][c, j]),
                    train_raw_comparison_magnitude=float(train["raw_bg"][c, j]),
                    train_actual_object_magnitude=float(train["gated_fg"][c, j]),
                    train_exclusion_percent=float(train["exclusion"][c, j] * 100),
                    train_instances=int(train["counts"][c]),
                    development_instances=int(dev["counts"][c]),
                )
            )
    folder = s.results / "directions"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "object_direction_rankings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates = rank_proposals(train, targets, ctx["protocol"])
    s.write("directions/train_ranked_exchanges.json", candidates)
    # Visualization is descriptive; development values never rerank training proposals.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = np.argsort(-train["gain"].max(1), kind="stable")[:16]
    atoms = sorted(set(int(train["gain"][c].argmax()) for c in cats))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    matrices = [p["gain"][np.ix_(cats, atoms)] * 100 for p in (train, dev)]
    lim = max(max(abs(a).max() for a in matrices), 1e-9)
    for ax, a, label in zip(axes, matrices, ("Training", "Historical development")):
        im = ax.imshow(a, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(atoms)), [str(j) for j in atoms], rotation=90)
        ax.set_yticks(range(len(cats)), [targets[c]["category"] for c in cats])
        ax.set_title(label)
        ax.set_xlabel("Direction ID (before edits)")
    fig.suptitle(
        "Predicted help to existing object groups: same directions on separate images"
    )
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        label="Predicted capture gain (percentage points)",
        shrink=0.8,
    )
    folder = s.results / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.19, top=0.89, wspace=0.08)
    fig.savefig(folder / "direction_atlas.png", dpi=160)
    plt.close(fig)
    return candidates


def frozen_screen(ctx, candidates):
    s = ctx["store"]
    cached = s.load("frozen_screen")
    if cached is not None:
        return cached
    targets = ctx["initial"]["spec"]["targets"]
    cfg = dict(old.settings(), assignment_images=ctx["protocol"]["screening_images"])
    ix, coverage = old.assignment_indices(
        ctx["packs"]["train"], [t["category_index"] for t in targets], cfg, ctx["cfg"]
    )
    p = ctx["packs"]["train"]
    screen = dict(
        x=np.array(p["x"][ix], copy=True),
        ids=[p["ids"][i] for i in ix],
        masks=[p["masks"][i] for i in ix],
    )
    view = dict(ctx, packs=dict(ctx["packs"], train_screen=screen))
    s.write("frozen_screen_coverage.json", coverage)
    ref = s.load("frozen_screen_reference")
    if ref is None:
        m = model(ctx, ctx["initial"])
        ref = bank.evaluate(view, m, "train_screen")
        del m
        s.save("frozen_screen_reference", ref)
    rows = []
    for proposal in candidates[: ctx["protocol"]["frozen_screen_candidates"]]:
        key = "frozen_" + proposal["id"]
        row = s.load(key)
        if row is None:
            start = permute_snapshot(
                ctx["initial"], proposal["outgoing"], proposal["incoming"]
            )
            s.write(
                "atom_layouts/" + proposal["id"] + ".json",
                dict(
                    slot_to_original_atom_id=start["spec"]["atom_ids"],
                    outgoing=proposal["outgoing"],
                    incoming=proposal["incoming"],
                    width=3,
                    groups=1024,
                    active_groups=16,
                ),
            )
            m = model(ctx, start)
            report = bank.evaluate(view, m, "train_screen")
            del m
            c = proposal["category_index"]
            a = {t["category_index"]: t for t in report["categories"]}
            b = {t["category_index"]: t for t in ref["categories"]}
            row = dict(
                proposal=proposal,
                report=report,
                mean_capture_delta=report["macro_capture"] - ref["macro_capture"],
                target_capture_delta=a[c]["capture"] - b[c]["capture"],
                ap_delta=report["macro_ap"] - ref["macro_ap"],
                ve_delta=report["reconstruction"]["ve"] - ref["reconstruction"]["ve"],
                worst_category_capture_delta=min(
                    a[k]["capture"] - b[k]["capture"] for k in a
                ),
                worst_category_ap_delta=min(a[k]["ap"] - b[k]["ap"] for k in a),
                actual_gate_scope="Exact frozen model TopK counterfactual on training screen, not additive approximation",
            )
            s.save(key, row)
            s.write("frozen_screen/" + proposal["id"] + ".json", row)
        rows.append(row)
        s.event(
            "frozen_exchange",
            id=proposal["id"],
            completed=len(rows),
            total=min(len(candidates), ctx["protocol"]["frozen_screen_candidates"]),
            gain=row["mean_capture_delta"],
        )
    # Keep recoverable initial spikes. Target/non-target trade-offs are retained in logs.
    rows.sort(
        key=lambda r: (
            -r["mean_capture_delta"],
            -r["target_capture_delta"],
            r["proposal"]["id"],
        )
    )
    chosen = []
    seen_targets = set()
    for row in rows:
        c = row["proposal"]["category_index"]
        if c not in seen_targets:
            chosen.append(row)
            seen_targets.add(c)
        if len(chosen) >= ctx["protocol"]["recovery_candidates"]:
            break
    out = dict(
        rows=rows,
        chosen=chosen,
        coverage=coverage,
        selection_rule="Training-only exact mean capture, target capture, ID; target diversity; no immediate spike rejection",
    )
    s.save("frozen_screen", out)
    s.write("frozen_screen.json", out)
    del screen
    gc.collect()
    return out


def fit(ctx, key, initial, affected=None, updates=None, max_updates=None):
    s = ctx["store"]
    cfg = ctx["protocol"]
    updates = cfg["recovery_updates"] if updates is None else updates
    fingerprint = identity(
        dict(
            initial_sha=training.weight_sha(initial["weights"]),
            spec=initial["spec"],
            affected=affected,
            updates=updates,
        )
    )
    st = s.load(key)
    m = model(ctx, initial)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    if st is None:
        st = dict(
            updates=0,
            fingerprint=fingerprint,
            history=[],
            seconds=0.0,
            batch_sha=hashlib.sha256().hexdigest(),
        )
    else:
        if st["fingerprint"] != fingerprint:
            raise ValueError("Recovery identity changed")
        m.load_state_dict(st["current"]["weights"])
        opt.load_state_dict(st["optimizer"])

    def save(drive=True):
        st.update(
            current=bank.snapshot(m, initial["spec"]),
            optimizer=copy.deepcopy(opt.state_dict()),
        )
        s.save(key, st, drive)

    def audit():
        report = bank.evaluate(ctx, m)
        u = st["updates"]
        snap = bank.snapshot(m, initial["spec"])
        name = key + f"_u{u}"
        s.save(name, snap)
        row = dict(update=u, report=report, checkpoint=name)
        st["history"].append(row)
        save()
        s.write("histories/" + key + ".json", st["history"])
        s.event(
            "recovery_audit", key=key, updates=u, total=updates, **bank.scalar(report)
        )

    if not st["history"]:
        audit()
    s.event("fit_resume", key=key, updates=st["updates"], total=updates)
    try:
        limit = updates if max_updates is None else min(updates, max_updates)
        while st["updates"] < limit:
            u = st["updates"] + 1
            rng = np.random.default_rng(cfg["training_seed"] + u)
            ix = rng.integers(
                len(ctx["packs"]["train"]["ids"]), size=cfg["images_per_update"]
            )
            torch.manual_seed(cfg["training_seed"] + u)
            if str(ctx["device"]).startswith("cuda"):
                torch.cuda.manual_seed_all(cfg["training_seed"] + u)
            x = torch.as_tensor(
                np.array(ctx["packs"]["train"]["x"][ix], copy=True),
                device=ctx["device"],
            )
            masks = [ctx["packs"]["train"]["masks"][i] for i in ix]
            m.train()
            t0 = time.perf_counter()
            xx = x.reshape(-1, m.d)
            rec, _ = m.loss(
                xx + cfg["snr"] * xx.std() * torch.randn_like(xx), target=xx
            )
            cap, det, objects, _ = supervised_terms(m, x, masks, ctx["cfg"])
            terms = dict(
                native=rec,
                capture=cfg["lambda_capture"] * cap,
                detection=cfg["lambda_detection"] * det,
            )
            diagnostics = (
                study.component_gradients(m, terms)
                if u % cfg["gradient_every"] == 0
                else None
            )
            loss = sum(terms.values())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite recovery loss")
            opt.zero_grad()
            loss.backward()
            restricted = affected is not None and u <= cfg["restricted_updates"]
            if restricted:
                allowed = torch.zeros(m.total_atoms, device=ctx["device"])
                allowed[affected["atoms"]] = 1
                for name, axis in [("W_enc", 1), ("W_dec", 0), ("b_enc", 0)]:
                    p = getattr(m, name)
                    p.grad.mul_(
                        allowed[None]
                        if name == "W_enc"
                        else allowed[:, None]
                        if name == "W_dec"
                        else allowed
                    )
                head_mask = torch.zeros_like(m.head)
                head_mask[affected["heads"]] = 1
                if m.head.grad is not None:
                    m.head.grad.mul_(head_mask)
            norms = {
                n: float(p.grad.norm())
                for n, p in m.named_parameters()
                if p.grad is not None
            }
            grad = float(
                torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["grad_clip"])
            )
            if not np.isfinite(grad):
                raise FloatingPointError("Nonfinite gradient")
            opt.step()
            if restricted:
                with torch.no_grad():
                    ids = affected["atoms"]
                    d = m.W_dec[ids]
                    m.W_dec[ids] = d / d.norm(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                m.normalize_decoder()
            study.sync(ctx["device"])
            st["seconds"] += time.perf_counter() - t0
            st["updates"] = u
            st["batch_sha"] = hashlib.sha256(
                st["batch_sha"].encode() + ix.tobytes()
            ).hexdigest()
            if u % cfg["evaluate_every"] == 0 or u == updates:
                audit()
                s.write(
                    "training/" + key + f"_u{u}.json",
                    dict(
                        update=u,
                        losses={k: float(v.detach()) for k, v in terms.items()},
                        objects=objects,
                        gradient_norm=grad,
                        parameter_gradient_norms=norms,
                        component_gradients=diagnostics,
                        restricted=restricted,
                    ),
                )
            elif u % cfg["local_save_every"] == 0:
                save(False)
                s.event(
                    "training_heartbeat",
                    key=key,
                    update=u,
                    total=updates,
                    seconds_per_update=st["seconds"] / u,
                )
        save()
    except BaseException:
        save(False)
        try:
            save()
        except BaseException:
            pass
        raise
    del m, opt
    gc.collect()
    if str(ctx["device"]).startswith("cuda"):
        torch.cuda.empty_cache()
    return st


def run_recovery(ctx, screen):
    s = ctx["store"]
    cached = s.load("recovery_decision")
    if cached is not None:
        return cached
    base = s.load("baseline_report")
    if base is None:
        m = model(ctx, ctx["initial"])
        base = bank.evaluate(ctx, m)
        del m
        s.save("baseline_report", base)
    full = fit(ctx, "continued_ggf", ctx["initial"])
    decisions = []
    eligible = []
    used = ctx["protocol"]["recovery_updates"]
    for index, row in enumerate(screen["chosen"]):
        p = row["proposal"]
        groups = {p["receiver_group"], p["donor_group"]}
        affected = dict(
            atoms=sorted(j for g in groups for j in range(3 * g, 3 * g + 3)),
            heads=[
                k
                for k, t in enumerate(ctx["initial"]["spec"]["targets"])
                if t["contrast_group"] in groups or t["detector_group"] in groups
            ],
        )
        control = fit(ctx, f"pair{index}_control", ctx["initial"], affected)
        start = permute_snapshot(ctx["initial"], p["outgoing"], p["incoming"])
        edited = fit(ctx, f"pair{index}_edited", start, affected)
        if control["batch_sha"] != edited["batch_sha"]:
            raise AssertionError("Paired sample/noise schedules differ")
        checks = []
        control_by_u = {r["update"]: r for r in control["history"]}
        full_by_u = {r["update"]: r for r in full["history"]}
        for e in edited["history"]:
            c = control_by_u[e["update"]]
            f = full_by_u[e["update"]]
            gate = bank.acceptance(
                e["report"],
                base,
                c["report"],
                base,
                ctx["protocol"],
                p["category_index"],
            )
            gate2 = bank.acceptance(
                e["report"],
                base,
                f["report"],
                base,
                ctx["protocol"],
                p["category_index"],
            )
            check = dict(
                update=e["update"],
                checkpoint=e["checkpoint"],
                against_paired=gate,
                against_full_continuation=gate2,
                actual_deltas={
                    k: bank.scalar(e["report"])[k] - bank.scalar(c["report"])[k]
                    for k in ("capture", "ap", "ve", "mse")
                },
            )
            checks.append(check)
            if gate["passed"] and gate2["passed"] and e["update"] > 0:
                eligible.append(
                    dict(
                        proposal=p,
                        checkpoint=e["checkpoint"],
                        report=e["report"],
                        update=e["update"],
                    )
                )
        decisions.append(
            dict(
                proposal=p,
                checks=checks,
                control=control["history"],
                edited=edited["history"],
                batch_sha=edited["batch_sha"],
            )
        )
        s.write("decisions/" + p["id"] + ".json", decisions[-1])
        used += 2 * ctx["protocol"]["recovery_updates"]
    # Cost accounting includes all attempted recoveries and controls, even rejected ones.
    nominated = (
        max(
            eligible,
            key=lambda r: (
                r["report"]["macro_capture"],
                r["report"]["macro_ap"],
                -r["report"]["reconstruction"]["mse"],
            ),
        )
        if eligible
        else None
    )
    budget_control = None
    budget_gate = None
    confirmation = None
    accepted = False
    if nominated:
        budget_control = fit(
            ctx, "search_budget_continued_ggf", ctx["initial"], updates=used
        )
        strong = max(
            budget_control["history"],
            key=lambda r: (
                r["report"]["macro_capture"] + 0.2 * r["report"]["macro_ap"],
                -r["report"]["reconstruction"]["mse"],
            ),
        )
        budget_gate = bank.acceptance(
            nominated["report"],
            base,
            strong["report"],
            base,
            ctx["protocol"],
            nominated["proposal"]["category_index"],
        )
        if budget_gate["passed"]:
            norm = memory.required(ctx["original"], "normalization")
            ctx["packs"]["confirm"] = memory.normalized_pack(
                ctx["original"], "confirm", ctx["provenance"], norm, ctx["local"], 20
            )
            sets = [set(p["ids"]) for p in ctx["packs"].values()]
            if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
                raise ValueError("Confirmation overlap")
            reports = {}
            for name, snap in [
                ("original", ctx["initial"]),
                ("edited", s.load(nominated["checkpoint"])),
                ("budget_control", s.load(strong["checkpoint"])),
            ]:
                m = model(ctx, snap)
                reports[name] = bank.evaluate(ctx, m, "confirm")
                del m
            gate = bank.acceptance(
                reports["edited"],
                reports["original"],
                reports["budget_control"],
                reports["original"],
                ctx["protocol"],
                nominated["proposal"]["category_index"],
            )
            confirmation = dict(
                reports=reports,
                gate=gate,
                scope="Historical disjoint development; not untouched final test",
            )
            accepted = gate["passed"]
    out = dict(
        complete=True,
        accepted=accepted,
        baseline=base,
        continued=full["history"],
        decisions=decisions,
        eligible_checkpoints=len(eligible),
        nominee=nominated,
        search_update_cost=used,
        search_budget_control=budget_control["history"] if budget_control else None,
        search_budget_gate=budget_gate,
        confirmation=confirmation,
        final_test_opened=False,
        scope=ctx["protocol"]["scope"],
        interpretation="A target gain is not an overall advantage; rejection margins and matched controls remain primary.",
    )
    s.save("recovery_decision", out)
    s.write("summary.json", out)
    memory.assert_immutable(ctx["stamps"])
    return out


def comparisons(ctx, out):
    rows = [
        dict(
            method="Saved GGF",
            checkpoint="source",
            accepted=False,
            **bank.scalar(out["baseline"]),
        ),
        dict(
            method="Continued GGF (final)",
            checkpoint=out["continued"][-1]["checkpoint"],
            accepted=False,
            **bank.scalar(out["continued"][-1]["report"]),
        ),
    ]
    for d in out["decisions"]:
        for kind in ("control", "edited"):
            r = d[kind][-1]
            rows.append(
                dict(
                    method=d["proposal"]["id"] + " " + kind + " (final)",
                    checkpoint=r["checkpoint"],
                    accepted=bool(
                        out["accepted"]
                        and out["nominee"]
                        and out["nominee"]["checkpoint"] == r["checkpoint"]
                    ),
                    **bank.scalar(r["report"]),
                )
            )
    if out["nominee"]:
        rows.append(
            dict(
                method="Selected nominee",
                checkpoint=out["nominee"]["checkpoint"],
                accepted=out["accepted"],
                **bank.scalar(out["nominee"]["report"]),
            )
        )
    if out["search_budget_control"]:
        r = max(
            out["search_budget_control"],
            key=lambda r: r["report"]["macro_capture"] + 0.2 * r["report"]["macro_ap"],
        )
        rows.append(
            dict(
                method="Total search-budget GGF",
                checkpoint=r["checkpoint"],
                accepted=False,
                **bank.scalar(r["report"]),
            )
        )
    folder = ctx["store"].results
    with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ("capture", "ap", "ve")):
        ax.barh(
            range(len(rows)),
            [r[metric] * 100 for r in rows],
            color=[
                "#6c829b"
                if "edited" not in r["method"] and r["method"] != "Selected nominee"
                else "#1c9f96"
                for r in rows
            ],
        )
        ax.set_yticks(range(len(rows)), [r["method"] for r in rows])
        ax.invert_yaxis()
        ax.set_title(
            dict(
                capture="Object contrast capture",
                ap="Detection AP",
                ve="Reconstruction VE",
            )[metric]
        )
        ax.set_xlabel("Percent")
    fig.suptitle(
        "Final checkpoints and selected nominee are labeled separately; green does not imply acceptance"
    )
    fig.tight_layout()
    (folder / "figures").mkdir(exist_ok=True)
    fig.savefig(folder / "figures/comparison.png", dpi=150)
    plt.close(fig)
    ctx["store"].write("comparison.json", rows)
    return rows


def paired_diagnostics(ctx, out):
    """Keep exact image/instance evidence for final candidate-versus-control audits."""
    s = ctx["store"]
    cached = s.load("paired_diagnostics")
    if cached is not None:
        return cached
    from coco_baselines import metrics

    mapping = dict(
        mappings=[
            dict(
                category_index=t["category_index"],
                detector=dict(group=t["detector_group"]),
                contrast=dict(groups=[t["contrast_group"]]),
            )
            for t in ctx["initial"]["spec"]["targets"]
        ]
    )

    def compact(key, snap):
        cached = s.load("compact_" + key)
        if cached is not None:
            return cached
        m = model(ctx, snap)
        p = ctx["packs"]["select"]
        m.eval()
        audit = metrics.audit_arrays(
            m, p["x"], p["masks"], p["ids"], ctx["cfg"], ctx["device"]
        )
        a = study.compact_audit(audit, mapping, ctx["cfg"])
        s.save("compact_" + key, a)
        folder = s.results / "image_audits"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / (key + ".npz"),
            **{k: v for k, v in a.items() if isinstance(v, np.ndarray)},
            image_ids=np.array(a["image_ids"]),
            category_indices=np.array(a["category_indices"]),
            annotation_ids=np.array(a["objects"]),
        )
        del m, audit
        gc.collect()
        return a

    cfg = dict(
        ctx["cfg"], bootstrap=ctx["protocol"]["bootstrap"], fresh_test_seed=105061
    )
    comparisons = []
    for i, d in enumerate(out["decisions"]):
        a = d["edited"][-1]
        b = d["control"][-1]
        aa = compact(a["checkpoint"], s.load(a["checkpoint"]))
        bb = compact(b["checkpoint"], s.load(b["checkpoint"]))
        ci = study.paired(aa, bb, cfg)
        ci.update(
            proposal=d["proposal"],
            edited_checkpoint=a["checkpoint"],
            control_checkpoint=b["checkpoint"],
            scope="Final horizon, fixed original readouts, reused historical development. Exploratory, not selection-corrected or seed uncertainty.",
        )
        comparisons.append(ci)
        s.event("paired_image_audit", completed=i + 1, total=len(out["decisions"]))
        del aa, bb
    s.save("paired_diagnostics", comparisons)
    s.write("paired_diagnostics.json", comparisons)
    return comparisons


def export(ctx):
    s = ctx["store"]
    path = s.output / f"agf_direction_regroup_{s.run_id[:12]}_results.zip"
    manifest = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(s.results.rglob("*")):
            if p.is_file():
                name = "results/" + p.relative_to(s.results).as_posix()
                z.write(p, name)
                manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        z.writestr("artifact_hashes.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise OSError("Export corruption")
    return path
