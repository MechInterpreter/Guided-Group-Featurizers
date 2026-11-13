"""Saved-start activity diagnostic and supervised owner/scalar AGF pilot.

Training and existing development only. A disjoint historical confirmation set
checks one nominee; this is NOT a new independent final-test claim.
"""

import copy
import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import core, inputs
from coco_baselines import metrics, training
from coco_baselines.storage import identity, serial
from coco_coordinated import run as memory
from . import owner_routing as old, study, wider_groups
from .owner_model import OwnerGGF, supervised_terms
from .banked_model import BankedAGF, edited_layout

WIDE_RUN = "367b64ceef5dd2c949bad756a4955cfb1139770e01fcf57e0f2ece944644ee26"
WIDE_SHA = "ff66e20b9c6a5731223f71da6668e5f16f02110aff13ebe12db75f1ba56a9f65"


def settings():
    return dict(
        version=1,
        seed=21,
        wide_active=[3, 6, 12],
        wide_updates=500,
        maturation_updates=500,
        recovery_updates=600,
        restricted_updates=100,
        maximum_candidates=3,
        total_budget=48,
        owner_budget=24,
        max_width=16,
        local_save_every=25,
        evaluate_every=100,
        gradient_every=100,
        training_seed=105021,
        lambda_capture=0.05,
        lambda_detection=0.01,
        lr=3e-4,
        snr=0.1,
        grad_clip=5.0,
        images_per_update=8,
        max_ap_drop=0.001,
        max_ve_drop=0.001,
        max_mse_growth=0.02,
        minimum_target_gain=0.005,
        minimum_mean_gain=0.001,
        max_category_drop=0.01,
        proposal_screen_images=256,
        bootstrap=200,
        scope="One-seed development diagnostic; supervised masks; no fresh test; no guaranteed success.",
    )


def prepare(
    source_output,
    wide_output,
    output,
    local,
    sha,
    versions,
    cfg,
    device,
    require_drive=True,
):
    if cfg != settings():
        raise ValueError(
            "Use the declared diagnostic settings; freeze a new protocol for changes"
        )
    # Verified lineage, read-only mmap activations, original successful GGF.
    ctx = old.prepare(
        source_output,
        Path(output) / "inputs",
        local,
        sha,
        versions,
        device=device,
        require_drive=require_drive,
    )
    oldcfg = ctx["cfg"]
    ctx["owner_assignment_cfg"] = dict(
        oldcfg, assignment_images=2000, assignment_pool_groups=1024
    )
    ctx["cfg"] = dict(
        ctx["eval_cfg"],
        **{
            k: v
            for k, v in cfg.items()
            if k
            in (
                "lr",
                "snr",
                "grad_clip",
                "images_per_update",
                "lambda_capture",
                "lambda_detection",
            )
        },
    )
    ctx["protocol"] = copy.deepcopy(cfg)
    wide = memory.FrozenStore(
        Path(wide_output) / "checkpoints" / WIDE_RUN[:20], Path(local) / "read_wide"
    )
    dest = Path(output).resolve()
    parent = Path(wide_output).resolve()
    if dest == parent or dest in parent.parents or parent in dest.parents:
        raise ValueError("Wider-group parent is read-only")
    if wide.run_id != WIDE_RUN:
        raise ValueError("Wrong wider-group parent")
    st = memory.required(wide, "ggf_b16_k3_s21")
    if training.weight_sha(st["best"]) != WIDE_SHA or st["best_update"] != 2250:
        raise ValueError("Wrong selected 16/3 checkpoint")
    ctx["wide"] = wide
    ctx["store"] = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            source_sha=sha,
            versions=versions,
            device=device,
            source_run=ctx["source"].run_id,
            wide_run=wide.run_id,
            wide_weights=WIDE_SHA,
            activation_run=ctx["original"].run_id,
        ),
        Path(local) / "bank_rescue",
        require_drive,
    )
    ctx["stamps"] += memory.source_snapshot([wide])
    s = ctx["store"]
    norm = memory.required(ctx["original"], "normalization")
    # Disjoint from this run's train/select. Previously used in research: label honestly.
    ctx["packs"]["confirm"] = memory.normalized_pack(
        ctx["original"], "confirm", ctx["provenance"], norm, local, 20
    )
    sets = [set(p["ids"]) for p in ctx["packs"].values()]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError("Split overlap")
    coverage = {}
    for split, pack in ctx["packs"].items():
        rows = []
        for c in ctx["cfg"]["category_indices"]:
            objects = pos = neg = 0
            for iid, mm in zip(pack["ids"], pack["masks"]):
                _, _, cats = core.object_weights([mm], [c], ctx["cfg"])
                objects += int(len(cats) > 0)
                pick = np.random.default_rng(
                    ctx["cfg"]["patch_sample_seed"] + int(iid)
                ).choice(196, ctx["cfg"]["detection_patches"], replace=False)
                f = mm["fraction"][pick, c] / 255.0
                known = mm["crowd"][pick] <= ctx["cfg"]["negative_fraction"]
                pos += int(((f >= ctx["cfg"]["positive_fraction"]) & known).any())
                neg += int(((f <= ctx["cfg"]["negative_fraction"]) & known).any())
            rows.append(
                dict(
                    category_index=c,
                    object_images=objects,
                    positive_images=pos,
                    negative_images=neg,
                )
            )
        coverage[split] = rows
    s.write("coverage.json", coverage)
    if any(
        min(r["object_images"], r["positive_images"], r["negative_images"])
        < ctx["cfg"]["decision_min_images"]
        for rows in coverage.values()
        for r in rows
    ):
        raise ValueError(
            "Declared category coverage insufficient; no training begun, no category dropped"
        )
    s.write(
        "split_roles.json",
        dict(
            ids={k: p["ids"] for k, p in ctx["packs"].items()},
            train="Weights, assignment, proposals and screening",
            select="Previously inspected development; checkpoint selection",
            confirm="Historical disjoint second development, one frozen nominee only; not fresh evidence",
            final_test_opened=False,
        ),
    )
    return ctx


def snapshot(model, spec):
    return dict(spec=copy.deepcopy(spec), weights=training.weights(model))


def make(ctx, snap):
    p = snap["spec"]
    w = snap["weights"]
    core_w = {k: w[k] for k in ("W_enc", "W_dec", "b_enc")}
    if p["kind"] == "hybrid":
        m = BankedAGF(
            core_w,
            p["layout"],
            p["targets"],
            w["head"],
            p["total_budget"],
            p["owner_budget"],
            p["max_width"],
        )
    else:
        m = OwnerGGF(
            core_w,
            p["groups"],
            p["rank"],
            p["active"],
            p["targets"],
            w["head"],
            "continued_ggf",
        )
    m.load_state_dict(w)
    return m.to(ctx["device"])


def native_snapshot(ctx, weights, targets, head, groups, rank, active):
    m = OwnerGGF(weights, groups, rank, active, targets, head, "continued_ggf")
    return snapshot(
        m, dict(kind="ggf", groups=groups, rank=rank, active=active, targets=targets)
    )


def scalar(r):
    return dict(
        capture=r["macro_capture"],
        ap=r["macro_ap"],
        ve=r["reconstruction"]["ve"],
        mse=r["reconstruction"]["mse"],
    )


def quality(r, refs, cfg):
    a = scalar(r)
    margins = {}
    for label, ref in refs.items():
        b = scalar(ref)
        margins[label + "_AP"] = a["ap"] - b["ap"] + cfg["max_ap_drop"]
        margins[label + "_VE"] = a["ve"] - b["ve"] + cfg["max_ve_drop"]
        margins[label + "_MSE"] = b["mse"] * (1 + cfg["max_mse_growth"]) - a["mse"]
    return dict(
        passed=all(np.isfinite(v) and v >= -1e-12 for v in margins.values()),
        margins=margins,
        failures=[k for k, v in margins.items() if not np.isfinite(v) or v < -1e-12],
    )


def acceptance(r, start, control, original, cfg, target):
    q = quality(
        r, dict(round_start=start, matched_control=control, original=original), cfg
    )
    margins = q["margins"]
    a = {t["category_index"]: t for t in r["categories"]}
    for label, ref in [("round_start", start), ("matched_control", control)]:
        b = {t["category_index"]: t for t in ref["categories"]}
        margins[label + "_target_capture"] = (
            a[target]["capture"] - b[target]["capture"] - cfg["minimum_target_gain"]
        )
        for c in a:
            if c != target:
                for metric in ("capture", "ap"):
                    margins[f"{label}_{c}_{metric}"] = (
                        a[c][metric] - b[c][metric] + cfg["max_category_drop"]
                    )
    margins["mean_capture_vs_control"] = (
        r["macro_capture"] - control["macro_capture"] - cfg["minimum_mean_gain"]
    )
    return dict(
        passed=all(np.isfinite(v) and v >= -1e-12 for v in margins.values()),
        margins=margins,
        failures=[k for k, v in margins.items() if not np.isfinite(v) or v < -1e-12],
    )


@torch.no_grad()
def evaluate(ctx, model, split="select", arrays=False):
    pack = ctx["packs"][split]
    cfg = ctx["cfg"]
    model.eval()
    report = core.validation(
        model, pack, model.targets, cfg, ctx["device"], with_arrays=arrays
    )
    wider_groups.checked_report(report, cfg["category_indices"])
    if isinstance(model, BankedAGF):
        sums = np.zeros(5)
        n = 0
        maximum = 0
        rawheat = []
        for i, iid in enumerate(pack["ids"]):
            x = torch.as_tensor(np.array(pack["x"][i], copy=True), device=ctx["device"])
            _, _, logits, d = model.route(x)
            counts = torch.stack(
                [
                    d["requested"].sum(1),
                    d["admitted"].sum(1),
                    d["owner_coordinates"],
                    d["scalar_coordinates"],
                    d["requested"].sum(1) - d["admitted"].sum(1),
                ],
                dim=1,
            )
            sums += counts.double().sum(0).cpu().numpy()
            n += len(x)
            maximum = max(
                maximum, int((d["owner_coordinates"] + d["scalar_coordinates"]).max())
            )
            pick = np.random.default_rng(cfg["patch_sample_seed"] + int(iid)).choice(
                196, cfg["detection_patches"], replace=False
            )
            rawheat.append(logits[pick].cpu().numpy())
        if maximum > model.total_budget:
            raise AssertionError("Active coordinate budget exceeded")
        report["routing"] = dict(
            zip(
                [
                    "requested_owners",
                    "admitted_owners",
                    "owner_coordinates",
                    "scalar_coordinates",
                    "budget_excluded_owners",
                ],
                (sums / n).tolist(),
            )
        )
        report["routing"].update(
            maximum_coordinates=maximum,
            core_parameters=sum(
                getattr(model, k).numel() for k in ("W_enc", "W_dec", "b_enc")
            ),
            deployed_head_parameters=model.head.numel(),
            scalar_atoms=len(model.scalar_ids),
            widths=[len(g) for g in model.layout],
        )
        # Raw-head AP is secondary; primary AP remains actual gated owner response.
        aa = report.get("_audit")
        if aa is not None:
            y, valid = metrics.binary_labels(aa, cfg)
            h = np.concatenate(rawheat)
            report["raw_head_AP"] = [
                dict(
                    category_index=t["category_index"],
                    ap=metrics.ap(
                        h[:, j],
                        y[:, t["category_index"]],
                        valid[:, t["category_index"]],
                    ),
                )
                for j, t in enumerate(model.targets)
            ]
    return report


def fit(ctx, key, initial, updates, refs=None, affected=None, max_updates=None):
    """Same deterministic per-update samples/noise, recovery horizon and fresh Adam."""
    s = ctx["store"]
    cfg = ctx["protocol"]
    device = ctx["device"]
    m = make(ctx, initial)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    st = s.load(key)
    fingerprint = identity(
        dict(
            start=training.weight_sha(initial["weights"]),
            spec=initial["spec"],
            updates=updates,
            affected=affected,
        )
    )
    if st is None:
        st = dict(
            updates=0,
            target=updates,
            fingerprint=fingerprint,
            history=[],
            seconds=0.0,
            batch_sha=hashlib.sha256().hexdigest(),
        )
    else:
        if st["fingerprint"] != fingerprint:
            raise ValueError("Fit identity changed")
        m.load_state_dict(st["current"]["weights"])
        opt.load_state_dict(st["optimizer"])
    limit = updates if max_updates is None else min(updates, max_updates)

    def save(drive=True):
        st.update(
            current=snapshot(m, initial["spec"]),
            optimizer=copy.deepcopy(opt.state_dict()),
        )
        s.save(key, st, drive)

    def audit():
        u = st["updates"]
        r = evaluate(ctx, m)
        row = dict(
            update=u,
            report=r,
            quality=quality(r, refs or {}, cfg),
            checkpoint=key + f"_u{u}",
        )
        st["history"].append(row)
        s.save(row["checkpoint"], snapshot(m, initial["spec"]))
        save()
        s.write("histories/" + key + ".json", st["history"])
        s.event(
            "checkpoint",
            key=key,
            update=u,
            total=updates,
            **scalar(r),
            eligible=row["quality"]["passed"],
        )

    if not st["history"]:
        audit()
    print(key, "resume", st["updates"], "/", updates, flush=True)
    try:
        while st["updates"] < limit:
            u = st["updates"] + 1
            # Index-addressable schedule survives retries and is identical in all arms.
            rng = np.random.default_rng(cfg["training_seed"] + u)
            ix = rng.integers(
                len(ctx["packs"]["train"]["ids"]), size=cfg["images_per_update"]
            )
            torch.manual_seed(cfg["training_seed"] + u)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(cfg["training_seed"] + u)
            x = torch.as_tensor(
                np.array(ctx["packs"]["train"]["x"][ix], copy=True), device=device
            )
            masks = [ctx["packs"]["train"]["masks"][i] for i in ix]
            m.train()
            t = time.monotonic()
            xx = x.reshape(-1, m.d)
            native, _ = m.loss(
                xx + cfg["snr"] * xx.std() * torch.randn_like(xx), target=xx
            )
            cap, det, objects, _ = supervised_terms(m, x, masks, ctx["cfg"])
            terms = dict(
                native=native,
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
                raise FloatingPointError("Nonfinite objective")
            opt.zero_grad()
            loss.backward()
            restricted = affected is not None and u <= cfg["restricted_updates"]
            if restricted:
                atom_mask = torch.zeros(m.total_atoms, device=device)
                atom_mask[affected["atoms"]] = 1
                m.W_enc.grad.mul_(atom_mask[None])
                m.W_dec.grad.mul_(atom_mask[:, None])
                m.b_enc.grad.mul_(atom_mask)
                hm = torch.zeros_like(m.head)
                hm[affected["owners"]] = 1
                m.head.grad.mul_(hm)
            norms = {
                name: float(p.grad.norm())
                for name, p in m.named_parameters()
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
            study.sync(device)
            st["seconds"] += time.monotonic() - t
            st["updates"] = u
            st["batch_sha"] = hashlib.sha256(
                st["batch_sha"].encode() + ix.tobytes()
            ).hexdigest()
            if (
                u in (50, 100, 250, 500)
                or u % cfg["evaluate_every"] == 0
                or u == updates
            ):
                st["last_training"] = dict(
                    update=u,
                    losses={k: float(v.detach()) for k, v in terms.items()},
                    objects=objects,
                    gradient_norm=grad,
                    gradient_norms=norms,
                    component_gradients=diagnostics,
                    restricted=restricted,
                )
                audit()
                s.write("training/" + key + f"_u{u}.json", st["last_training"])
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
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return st


def best(st, eligible=True):
    rows = [r for r in st["history"] if not eligible or r["quality"]["passed"]]
    return (
        max(
            rows,
            key=lambda r: (
                r["report"]["macro_capture"] + 0.2 * r["report"]["macro_ap"],
                -r["report"]["reconstruction"]["mse"],
                -r["update"],
            ),
        )
        if rows
        else None
    )


def reference(ctx):
    s = ctx["store"]
    saved = s.load("original_reference")
    if saved is not None:
        return saved
    w, h, t, row = old.source_start(ctx, 21)
    snap = native_snapshot(ctx, w, t, h, 1024, 3, 16)
    m = make(ctx, snap)
    out = dict(snapshot=snap, report=evaluate(ctx, m), weights_sha=row["weights_sha"])
    s.save("original_reference", out)
    s.write("original_reference.json", out["report"])
    return out


def run_activity(ctx):
    s = ctx["store"]
    cached = s.load("activity_complete")
    if cached is not None:
        return cached
    ref = reference(ctx)
    source = memory.required(ctx["wide"], "ggf_b16_k3_s21")
    targets = memory.required(ctx["wide"], "targets_b16_k3_s21")
    rows = []
    for k in ctx["protocol"]["wide_active"]:
        # Saved dictionary is selected at 2250; historical final head was not saved
        # at that point. Common fresh loss-only heads avoid mismatching checkpoints.
        snap = native_snapshot(
            ctx, source["best"], targets, torch.zeros(len(targets), 2), 192, 16, k
        )
        m = make(ctx, snap)
        before = evaluate(ctx, m)
        del m
        st = fit(
            ctx,
            f"wide16_active{k}",
            snap,
            ctx["protocol"]["wide_updates"],
            dict(original_ggf=ref["report"]),
        )
        choice = best(st) or best(st, False)
        quality_ok = best(st) is not None
        rows.append(
            dict(
                active_groups=k,
                active_coordinates=k * 16,
                total_directions=3072,
                initial=before,
                selected=choice,
                final=st["history"][-1],
                qualified=best(st) is not None,
                capture_gain_vs_original=choice["report"]["macro_capture"]
                - ref["report"]["macro_capture"],
                overall_screen_positive=quality_ok
                and choice["report"]["macro_capture"]
                >= ref["report"]["macro_capture"]
                + ctx["protocol"]["minimum_mean_gain"],
                batch_sha=st["batch_sha"],
                gradient_seconds=st["seconds"],
                updates=st["updates"],
            )
        )
    if len({r["batch_sha"] for r in rows}) != 1:
        raise AssertionError("Wide activity arms saw different samples")
    out = dict(
        complete=True,
        arms=rows,
        interpretation="Same dictionary start/readouts; activity and active compute differ. More activity is not a capacity-matched win.",
        final_test_opened=False,
    )
    s.save("activity_complete", out)
    s.write("activity_summary.json", out)
    return out


def hybrid_start(ctx):
    s = ctx["store"]
    cached = s.load("hybrid_initial")
    if cached is not None:
        return cached
    view = dict(ctx, cfg=ctx["owner_assignment_cfg"])
    plan = old.owner_plan(view, 21)
    w, h, targets, _ = old.source_start(ctx, 21)
    layout = [[3 * g + j for j in range(3)] for g in plan["owners"]]
    rt = copy.deepcopy(plan["router_targets"])
    for i, t in enumerate(rt):
        t["contrast_group"] = t["detector_group"] = i
    p = ctx["protocol"]
    m = BankedAGF(
        w,
        layout,
        rt,
        plan["router_head"],
        p["total_budget"],
        p["owner_budget"],
        p["max_width"],
    ).to(ctx["device"])
    spec = dict(
        kind="hybrid",
        layout=layout,
        targets=rt,
        total_budget=p["total_budget"],
        owner_budget=p["owner_budget"],
        max_width=p["max_width"],
    )
    x = torch.as_tensor(
        np.array(ctx["packs"]["train"]["x"][0], copy=True), device=ctx["device"]
    )
    original = make(ctx, reference(ctx)["snapshot"])
    original.eval()
    m.eval()
    with torch.no_grad():
        a = (x @ original.W_enc + original.b_enc) @ original.W_dec
        b = (x @ m.W_enc + m.b_enc) @ m.W_dec
        if not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
            raise AssertionError("Ungated conversion changed weights")
    owner_native = OwnerGGF(
        w, 1024, 3, 16, plan["targets"], plan["head"], "owner_topk"
    ).to(ctx["device"])
    initial = snapshot(m, spec)
    s.write(
        "conversion.json",
        dict(
            ungated_max_error=float((a - b).abs().max()),
            original=reference(ctx)["report"],
            unique_owner_native=evaluate(ctx, owner_native),
            hybrid=evaluate(ctx, m),
            total_atoms=m.total_atoms,
            owner_atoms=sum(map(len, layout)),
            scalar_atoms=len(m.scalar_ids),
            assignment=plan["assignment_sha"],
            layout=layout,
            source="successful 3/16 GGF",
        ),
    )
    s.save("hybrid_initial", initial)
    return initial


@torch.no_grad()
def selector_diagnostics(ctx, snap):
    m = make(ctx, snap)
    m.eval()
    p = ctx["packs"]["train"]
    rows = []
    # Label oracle affects this small TRAINING diagnostic only.
    for i in range(min(16, len(p["ids"]))):
        x = torch.as_tensor(np.array(p["x"][i], copy=True), device=ctx["device"])
        mm = p["masks"][i]
        oracle = torch.as_tensor(
            mm["fraction"][:, [t["category_index"] for t in m.targets]] / 255
            >= ctx["cfg"]["positive_fraction"],
            device=ctx["device"],
        )
        for name, kw in [
            ("actual", {}),
            ("more_owner_allowance", dict(owner_budget=m.total_budget)),
            ("mask_oracle", dict(oracle_requests=oracle)),
        ]:
            a, z, _, d = m.route(x, **kw)
            pred = a @ m.W_dec
            rows.append(
                dict(
                    image_id=int(p["ids"][i]),
                    selector=name,
                    mse=float((pred - x).square().mean()),
                    owners=float(d["admitted"].sum(1).float().mean()),
                    owner_coordinates=float(d["owner_coordinates"].float().mean()),
                    scalar_coordinates=float(d["scalar_coordinates"].float().mean()),
                )
            )
    ctx["store"].write(
        "selector_training_diagnostics.json", dict(rows=rows, not_method_results=True)
    )


@torch.no_grad()
def propose(ctx, snap):
    s = ctx["store"]
    cached = s.load("proposals")
    if cached is not None:
        return cached
    m = make(ctx, snap)
    m.eval()
    p = ctx["packs"]["train"]
    C = m.n_groups
    A = m.total_atoms
    gain = np.zeros((C, A))
    remove = np.zeros((C, A))
    den = np.zeros(C)
    counts = np.zeros(C, int)
    exclusion = np.zeros((C, A))
    positive = np.zeros((C, A))
    sample_ids = []
    cross = [np.zeros((len(g), A)) for g in m.layout]
    lookup = {t["category_index"]: j for j, t in enumerate(m.targets)}
    # Every training instance contributes; directions score as actual signed vectors.
    D = m.W_dec.detach()
    norm = D.square().sum(1)
    atom_inner = [D[ids] @ D.T for ids in m.layout]
    for i, iid in enumerate(p["ids"]):
        x = torch.as_tensor(np.array(p["x"][i], copy=True), device=ctx["device"])
        codes, z, _, diag = m.route(x)
        ww, ii, cc = core.object_weights([p["masks"][i]], list(lookup), ctx["cfg"])
        if len(cc):
            wt = torch.as_tensor(ww, device=x.device)
            dx = wt @ x
            da = wt @ diag["raw"]
            dz = torch.einsum("op,pgr->ogr", wt, z)
            gated = wt @ codes
            for k, c in enumerate(cc):
                j = lookup[int(c)]
                pred = dz[k, j] @ m.atoms()[j]
                r = dx[k] - pred
                v = da[k]
                g = 2 * v * (D @ r) - v.square() * norm
                gain[j] += g.cpu().numpy()
                den[j] += float(dx[k].square().sum())
                counts[j] += 1
                # Removing an admitted owner's atom: error(r+v) vs error(r).
                ids = m.layout[j]
                v0 = gated[k, ids]
                rr = -2 * v0 * (D[ids] @ r) - v0.square() * norm[ids]
                remove[j, ids] += rr.cpu().numpy()
                cross[j] += (2 * v0[:, None] * v[None, :] * atom_inner[j]).cpu().numpy()
                fg = torch.as_tensor(ww[k] > 0, device=x.device)
                if bool(fg.any()):
                    positive[j] += (diag["raw"][fg].abs() > 1e-6).sum(0).cpu().numpy()
                    exclusion[j] += (
                        ((codes[fg].abs() <= 1e-6) & (diag["raw"][fg].abs() > 1e-6))
                        .sum(0)
                        .cpu()
                        .numpy()
                    )
        sample_ids.append(int(iid))
        if (i + 1) % 100 == 0:
            s.event("proposal_statistics", images=i + 1, total=len(p["ids"]))
    if (den <= 0).any():
        raise ValueError("No contrast evidence for a category")
    gain /= den[:, None]
    remove /= den[:, None]
    cross = [v / den[c] for c, v in enumerate(cross)]
    frac = exclusion / np.maximum(positive, 1)
    scalars = set(m.scalar_ids.cpu().tolist())
    candidates = []
    for c in range(C):
        top = sorted(scalars, key=lambda j: (-gain[c, j], j))[:2]
        weak = max(m.layout[c], key=lambda j: (remove[c, j], -j))
        for j in top:
            if gain[c, j] <= 0:
                continue
            common = dict(
                owner=c,
                category_index=m.targets[c]["category_index"],
                category=m.targets[c]["category"],
                atom=j,
                raw_predicted_gain=float(gain[c, j]),
                excluded_fraction=float(frac[c, j]),
                instances=int(counts[c]),
            )
            if len(m.layout[c]) < m.group_size:
                candidates.append(dict(common, operation="grow"))
        # Exchange score includes removing the actual old contribution AND its
        # cross term with the incoming raw contribution; addition alone is wrong.
        corrections = remove[c, m.layout[c]][:, None] + cross[c]
        choices = corrections.argmax(0)
        exchange_gain = gain[c] + corrections[choices, np.arange(A)]
        for j in sorted(scalars, key=lambda j: (-exchange_gain[j], j))[:2]:
            if exchange_gain[j] <= 0:
                continue
            candidates.append(
                dict(
                    owner=c,
                    category_index=m.targets[c]["category_index"],
                    category=m.targets[c]["category"],
                    atom=j,
                    remove=m.layout[c][int(choices[j])],
                    operation="exchange",
                    raw_predicted_gain=float(exchange_gain[j]),
                    addition_only_gain=float(gain[c, j]),
                    excluded_fraction=float(frac[c, j]),
                    instances=int(counts[c]),
                )
            )
        if len(m.layout[c]) > 1 and remove[c, weak] > 0:
            candidates.append(
                dict(
                    owner=c,
                    category_index=m.targets[c]["category_index"],
                    category=m.targets[c]["category"],
                    operation="shrink",
                    remove=weak,
                    raw_predicted_gain=float(remove[c, weak]),
                    instances=int(counts[c]),
                )
            )
    # Cover grow/exchange/shrink if positive proposals exist. This avoids making
    # a dimension-adaptation test contain only equal-width exchanges by tie order.
    candidates.sort(
        key=lambda r: (
            -r["raw_predicted_gain"],
            r["owner"],
            r["operation"],
            r.get("atom", -1),
        )
    )
    pool = []
    for op in ("grow", "exchange", "shrink"):
        row = next((r for r in candidates if r["operation"] == op), None)
        if row is not None:
            pool.append(row)
    pool += [r for r in candidates if r not in pool]
    selected = []
    for r in pool[: ctx["protocol"]["maximum_candidates"]]:
        r = dict(r, id=f"proposal{len(selected)}")
        r["new_layout"] = edited_layout(m.layout, r, A, m.group_size)
        selected.append(r)
    folder = s.results / "fingerprints"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "training_direction_profiles.npz",
        gain=gain,
        remove_gain=remove,
        excluded_fraction=frac,
        counts=counts,
        category_indices=np.array(list(lookup)),
    )
    out = dict(
        shortlist=selected,
        rankings=candidates,
        training_image_ids=sample_ids,
        rule="Best positive grow, exchange and shrink if available; fill empty slots by remaining training score. Exact gated counterfactual follows before recovery.",
        owner_to_owner="Deferred; scalar grow/exchange/shrink only",
    )
    s.save("proposals", out)
    s.write("proposals.json", out)
    return out


def change_snapshot(ctx, start, proposal):
    spec = copy.deepcopy(start["spec"])
    spec["layout"] = proposal["new_layout"]
    oldm = make(ctx, start)
    w = {k: getattr(oldm, k).detach().cpu() for k in ("W_enc", "W_dec", "b_enc")}
    m = BankedAGF(
        w,
        spec["layout"],
        spec["targets"],
        oldm.head.detach().cpu(),
        spec["total_budget"],
        spec["owner_budget"],
        spec["max_width"],
    )
    # Retain learned calibration buffers and heads; movement changes the gate norm.
    m.heat_mean.copy_(oldm.heat_mean.cpu())
    m.heat_scale.copy_(oldm.heat_scale.cpu())
    return snapshot(m, spec)


def run_hybrid(ctx):
    s = ctx["store"]
    done = s.load("hybrid_complete")
    if done is not None:
        return done
    cfg = ctx["protocol"]
    ref = reference(ctx)
    initial = hybrid_start(ctx)
    selector_diagnostics(ctx, initial)
    cont = fit(
        ctx,
        "continued_original_ggf",
        ref["snapshot"],
        cfg["maturation_updates"],
        dict(original=ref["report"]),
    )
    controw = best(cont) or best(cont, False)
    mat = fit(
        ctx,
        "hybrid_maturation",
        initial,
        cfg["maturation_updates"],
        dict(original=ref["report"], continued=controw["report"]),
    )
    mrow = best(mat)
    if mat["batch_sha"] != cont["batch_sha"]:
        raise AssertionError("Maturation/control sample mismatch")
    out = dict(
        complete=True,
        adaptation_accepted=False,
        original=ref["report"],
        continued=controw,
        maturation=mrow or best(mat, False),
        final_maturation=mat["history"][-1],
        stopped_before_edits=mrow is None,
        final_test_opened=False,
    )
    if mrow is None:
        out["reason"] = (
            "No hybrid checkpoint preserves original/continued GGF AP and reconstruction; diagnose conversion and gating before transfers."
        )
        s.save("hybrid_complete", out)
        s.write("hybrid_summary.json", out)
        return out
    start = s.load(mrow["checkpoint"])
    props = propose(ctx, start)
    decisions = []
    eligible = []
    for p in props["shortlist"]:
        affected = dict(
            atoms=sorted(
                set(start["spec"]["layout"][p["owner"]] + p["new_layout"][p["owner"]])
            ),
            owners=[p["owner"]],
        )
        edited = change_snapshot(ctx, start, p)
        # Actual training counterfactual, using full training cohort for coverage.
        base_train = s.load("matured_training_report")
        if base_train is None:
            model = make(ctx, start)
            base_train = evaluate(ctx, model, "train")
            del model
            s.save("matured_training_report", base_train)
        model = make(ctx, edited)
        pre = evaluate(ctx, model, "train")
        del model
        s.write(
            "counterfactuals/" + p["id"] + ".json",
            dict(
                proposal=p,
                before_training=base_train,
                actual_training=pre,
                exact_gain={
                    k: scalar(pre)[k] - scalar(base_train)[k]
                    for k in ("capture", "ap", "ve")
                },
                note="Initial error spikes are recorded, not an automatic rejection.",
            ),
        )
        control = fit(
            ctx,
            p["id"] + "_control",
            start,
            cfg["recovery_updates"],
            dict(original=ref["report"]),
            affected,
        )
        cr = best(control) or best(control, False)
        cand = fit(
            ctx,
            p["id"] + "_edited",
            edited,
            cfg["recovery_updates"],
            dict(original=ref["report"]),
            affected,
        )
        if cand["batch_sha"] != control["batch_sha"]:
            raise AssertionError("Candidate/control sample mismatch")
        checks = []
        for r in cand["history"]:
            gate = acceptance(
                r["report"],
                mrow["report"],
                cr["report"],
                ref["report"],
                cfg,
                p["category_index"],
            )
            checks.append(
                dict(update=r["update"], gate=gate, checkpoint=r["checkpoint"])
            )
            if gate["passed"]:
                eligible.append(dict(proposal=p, row=r, control=cr))
        decisions.append(
            dict(
                proposal=p,
                checks=checks,
                control=cr,
                final=cand["history"][-1],
                batch_sha=cand["batch_sha"],
            )
        )
        s.write("decisions/" + p["id"] + ".json", decisions[-1])
    out["decisions"] = decisions
    out["eligible_checkpoints"] = len(eligible)
    if eligible:
        eligible.sort(
            key=lambda a: (
                -a["row"]["report"]["macro_capture"],
                -a["row"]["report"]["macro_ap"],
                a["row"]["report"]["reconstruction"]["mse"],
                a["proposal"]["id"],
            )
        )
        nominee = eligible[0]
        s.save("frozen_nominee", nominee)
        s.write("frozen_nominee.json", nominee)
        # Extra no-edit controls charged for ALL candidate + paired-control updates.
        search_updates = len(decisions) * 2 * cfg["recovery_updates"]
        budget = fit(
            ctx,
            "hybrid_search_budget_control",
            start,
            search_updates,
            dict(original=ref["report"]),
        )
        br = best(budget) or best(budget, False)
        gate = acceptance(
            nominee["row"]["report"],
            mrow["report"],
            br["report"],
            ref["report"],
            cfg,
            nominee["proposal"]["category_index"],
        )
        out["search_budget_control"] = br
        out["search_budget_gate"] = gate
        if gate["passed"]:
            reports = {}
            for name, snap in [
                ("nominee", s.load(nominee["row"]["checkpoint"])),
                ("round_start", start),
                ("matched", s.load(nominee["control"]["checkpoint"])),
                ("search_budget", s.load(br["checkpoint"])),
                ("original", ref["snapshot"]),
            ]:
                model = make(ctx, snap)
                reports[name] = evaluate(ctx, model, "confirm")
                del model
            gates = {
                key: acceptance(
                    reports["nominee"],
                    reports["round_start"],
                    reports[key],
                    reports["original"],
                    cfg,
                    nominee["proposal"]["category_index"],
                )
                for key in ("matched", "search_budget")
            }
            out.update(
                confirmation=reports,
                confirmation_gates=gates,
                adaptation_accepted=all(g["passed"] for g in gates.values()),
            )
        out["nominee"] = nominee
    out["interpretation"] = (
        "An accepted edit is a one-seed supervised development finding. Historical confirmation is not a fresh test. No-edit fallback is not AGF success."
    )
    s.save("hybrid_complete", out)
    s.write("hybrid_summary.json", out)
    memory.assert_immutable(ctx["stamps"])
    return out


def preflight(ctx):
    s = ctx["store"]
    oldp = s.load("preflight")
    if oldp is not None:
        return oldp
    # Cheap temporary two-owner model; no training mappings are consumed.
    ref = reference(ctx)
    w = ref["snapshot"]["weights"]
    targets = []
    for c in range(2):
        targets.append(
            dict(
                category_index=c,
                category=str(c),
                contrast_group=c,
                detector_group=c,
                heat_mean=0.0,
                heat_scale=1.0,
            )
        )
    m = BankedAGF(w, [[0, 1, 2], [3, 4, 5]], targets, torch.zeros(2, 2)).to(
        ctx["device"]
    )
    x = torch.as_tensor(
        np.array(ctx["packs"]["train"]["x"][:2], copy=True), device=ctx["device"]
    ).reshape(-1, m.d)
    times = []
    for _ in range(3):
        study.sync(ctx["device"])
        t = time.monotonic()
        m.train()
        loss, _ = m.loss(x)
        loss.backward()
        study.sync(ctx["device"])
        times.append(time.monotonic() - t)
        m.zero_grad()
    out = dict(
        numerical_ok=bool(torch.isfinite(loss)),
        temporary_owner_count=2,
        seconds_per_reconstruction_update=float(np.median(times)),
        guaranteed_max_core_updates=3 * 500 + 2 * 500 + 2 * 3 * 600,
        conditional_search_control_updates=2 * 3 * 600,
        timing_note="Small two-owner reconstruction smoke only, not full supervised runtime estimate. Real 75-owner timing appears in training heartbeats.",
        no_final_test=True,
    )
    s.save("preflight", out)
    s.write("preflight.json", out)
    return out


@torch.no_grad()
def compact_capture_audit(ctx, snap, key):
    model = make(ctx, snap)
    r = evaluate(ctx, model, arrays=True)
    a = r.pop("_audit")
    n = len(ctx["packs"]["select"]["ids"])
    cohort = ctx["cfg"]["category_indices"]
    lookup = {c: j for j, c in enumerate(cohort)}
    sse = np.zeros((n, len(cohort)))
    energy = np.zeros_like(sse)
    counts = np.zeros_like(sse)
    for t in model.targets:
        c = t["category_index"]
        take = np.array([o["category"] == c for o in a["objects"]], bool)
        dx = a["dx"][take]
        pred = a["dz"][take, t["contrast_group"]] @ a["atoms"][t["contrast_group"]]
        indices = np.array([o["image_index"] for o in a["objects"]])[take]
        np.add.at(sse[:, lookup[c]], indices, np.square(dx - pred).sum(1))
        np.add.at(energy[:, lookup[c]], indices, np.square(dx).sum(1))
        np.add.at(counts[:, lookup[c]], indices, 1)
    folder = ctx["store"].results / "image_audits"
    folder.mkdir(parents=True, exist_ok=True)
    y, valid = metrics.binary_labels(a, ctx["cfg"])
    np.savez_compressed(
        folder / (key + ".npz"),
        image_ids=np.array(ctx["packs"]["select"]["ids"]),
        category_indices=np.array(cohort),
        sse=sse,
        energy=energy,
        counts=counts,
        patch_image_index=a["image_index"],
        detector_heat=a["heat"][:, [t["detector_group"] for t in model.targets]],
        detector_category_indices=np.array(
            [t["category_index"] for t in model.targets]
        ),
        fraction=a["fraction"],
        crowd=a["crowd"],
    )
    ctx["store"].write("detailed/" + key + ".json", r)
    return sse, energy


@torch.no_grad()
def bank_contrasts(ctx, snap):
    model = make(ctx, snap)
    model.eval()
    pack = ctx["packs"]["select"]
    rows = []
    lookup = {t["category_index"]: j for j, t in enumerate(model.targets)}
    for i, iid in enumerate(pack["ids"]):
        x = torch.as_tensor(np.array(pack["x"][i], copy=True), device=ctx["device"])
        codes, z, _, d = model.route(x)
        ww, _, cc = core.object_weights([pack["masks"][i]], list(lookup), ctx["cfg"])
        if not len(cc):
            continue
        w = torch.as_tensor(ww, device=x.device)
        dx = w @ x
        dz = torch.einsum("op,pgr->ogr", w, z)
        sc = (w @ d["scalar_codes"]) @ model.W_dec
        full = (w @ codes) @ model.W_dec
        for k, c in enumerate(cc):
            owner = dz[k, lookup[int(c)]] @ model.atoms()[lookup[int(c)]]
            den = float(dx[k].square().sum())
            rows.append(
                dict(
                    image_id=int(iid),
                    category_index=int(c),
                    target_energy=den,
                    owner_sse=float((dx[k] - owner).square().sum()),
                    scalar_sse=float((dx[k] - sc[k]).square().sum()),
                    combined_owner_scalar_sse=float(
                        (dx[k] - owner - sc[k]).square().sum()
                    ),
                    full_sse=float((dx[k] - full[k]).square().sum()),
                    owner_scalar_dot=float(owner @ sc[k]),
                    owner_energy=float(owner.square().sum()),
                    scalar_energy=float(sc[k].square().sum()),
                )
            )
    ctx["store"].write(
        "bank_contrast_audit.json",
        dict(
            rows=rows,
            note="Signed contributions; captures do not add. Scalar-bank object information is allowed and measured, not assumed absent.",
        ),
    )


@torch.no_grad()
def gallery(ctx, snap):
    """Predetermined annotation-selected examples; image downloads are optional."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from coco_baselines import data
    from PIL import Image

    m = make(ctx, snap)
    m.eval()
    p = ctx["packs"]["select"]
    folder = ctx["store"].results / "gallery"
    folder.mkdir(parents=True, exist_ok=True)
    selected = []
    for j in np.linspace(0, len(m.targets) - 1, min(6, len(m.targets)), dtype=int):
        c = m.targets[j]["category_index"]
        ix = next(
            i
            for i in sorted(range(len(p["ids"])), key=lambda i: p["ids"][i])
            if any(o["category"] == c for o in p["masks"][i]["instances"])
        )
        selected.append(
            dict(
                owner=int(j),
                category=m.targets[j]["category"],
                category_index=c,
                image_id=int(p["ids"][ix]),
                index=ix,
            )
        )
    ctx["store"].write(
        "gallery/selection.json",
        dict(
            rows=selected,
            rule="Six category IDs evenly spaced in the frozen cohort; lowest image ID with an annotation; no model-score selection.",
        ),
    )
    for row in selected:
        i = row["index"]
        iid = row["image_id"]
        x = torch.as_tensor(np.array(p["x"][i], copy=True), device=ctx["device"])
        _, z, _, d = m.route(x)
        heat = z[:, row["owner"]].norm(dim=1).cpu().numpy().reshape(14, 14)
        scalar = d["scalar_codes"]
        winner = scalar.abs().argmax(1).cpu().numpy().reshape(14, 14)
        values = scalar.abs().max(1).values.cpu().numpy().reshape(14, 14)
        mask = (p["masks"][i]["fraction"][:, row["category_index"]] / 255.0).reshape(
            14, 14
        )
        np.savez_compressed(
            folder / f"{iid}_owner{row['owner']}.npz",
            owner_heat=heat,
            mask_fraction=mask,
            top_scalar_atom=winner,
            top_scalar_amount=values,
        )
        fig = Figure(figsize=(10, 3))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3)
        image = None
        try:
            image_path = data.download(
                data.COCO_BUCKET + f"val2017/{iid:012d}.jpg",
                Path(ctx["local"]) / "gallery_images" / f"{iid}.jpg",
            )
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                image = np.asarray(data.transform_image(im, data.geometry(*im.size)))
        except Exception as exc:
            ctx["store"].event(
                "gallery_image_unavailable", image_id=iid, error=str(exc)
            )
        for ax, v, title in zip(
            axes,
            [mask, heat, values],
            [
                "COCO annotation",
                f"Owner {row['owner']}: {row['category']}",
                "Scalar-bank response",
            ],
        ):
            if image is not None:
                ax.imshow(image, extent=(0, 14, 14, 0))
            ax.imshow(
                v,
                extent=(0, 14, 14, 0),
                cmap="magma",
                alpha=0.6 if image is not None else 1,
            )
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Image {iid}; top scalar atom IDs saved in companion NPZ")
        fig.tight_layout()
        fig.savefig(folder / f"{iid}_owner{row['owner']}.png", dpi=150)


def finish(ctx, activity, hybrid):
    s = ctx["store"]
    cached = s.load("final_report")
    if cached is not None:
        return cached
    ref = reference(ctx)
    rows = [
        dict(method="Saved GGF 3/16", **scalar(ref["report"]), active_coordinates=48)
    ]
    for r in activity["arms"]:
        rows.append(
            dict(
                method=f"GGF 16/{r['active_groups']}",
                **scalar(r["selected"]["report"]),
                active_coordinates=r["active_coordinates"],
                checkpoint=r["selected"]["checkpoint"],
                quality_pass=r["qualified"],
            )
        )
    for label, key in [
        ("Continued GGF", "continued"),
        ("Hybrid before edits", "maturation"),
    ]:
        r = hybrid[key]
        rows.append(
            dict(
                method=label,
                **scalar(r["report"]),
                active_coordinates=48,
                checkpoint=r["checkpoint"],
                quality_pass=r["quality"]["passed"],
            )
        )
    nominee = hybrid.get("nominee")
    if nominee:
        rows.append(
            dict(
                method="AGF nominee",
                **scalar(nominee["row"]["report"]),
                active_coordinates=48,
                checkpoint=nominee["row"]["checkpoint"],
                accepted=hybrid["adaptation_accepted"],
            )
        )
    selected = nominee["row"] if nominee else hybrid["maturation"]
    snap = s.load(selected["checkpoint"])
    a = compact_capture_audit(
        ctx, s.load(hybrid["continued"]["checkpoint"]), "continued_ggf"
    )
    b = compact_capture_audit(ctx, snap, "hybrid_or_nominee")
    if not np.allclose(a[1], b[1], rtol=1e-5, atol=1e-6):
        raise AssertionError("Paired capture denominator differs")
    rng = np.random.default_rng(105999)
    draws = []
    for _ in range(ctx["protocol"]["bootstrap"]):
        ii = rng.integers(len(a[0]), size=len(a[0]))
        den = a[1][ii].sum(0)
        if (den <= 0).any():
            continue
        draws.append(float(((a[0][ii].sum(0) - b[0][ii].sum(0)) / den).mean()))
    interval = np.quantile(draws, [0.025, 0.975]).tolist() if draws else None
    bank_contrasts(ctx, snap)
    gallery(ctx, snap)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(14, 4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    for ax, k, title in zip(
        axes,
        ["capture", "ap", "ve"],
        ["Single-group object contrast capture", "Detection AP", "Reconstruction VE"],
    ):
        ax.bar([r["method"] for r in rows], [100 * r[k] for r in rows])
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", labelrotation=40)
    fig.tight_layout()
    folder = s.results / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "comparison.png", dpi=160)
    out = dict(
        complete=True,
        table=rows,
        adaptation_accepted=hybrid["adaptation_accepted"],
        paired_capture_interval_vs_continued=interval,
        bootstrap_replicates=len(draws),
        bootstrap_scope="Paired images, fixed selected models and readouts; exploratory reused development, no selection correction or seed uncertainty.",
        scope=ctx["protocol"]["scope"],
        confirmation_scope="Historical disjoint development; no fresh test",
        next_decision="Activity helps: freeze activity tradeoff and confirm. Hybrid fails conversion: inspect routing/banks. Hybrid works but edits fail: inspect predicted/exact gain and recovery. Accepted repair: independent seeds and reserved images needed.",
    )
    s.save("final_report", out)
    s.write("summary.json", out)
    memory.assert_immutable(ctx["stamps"])
    return out


def export(ctx):
    s = ctx["store"]
    root = Path("/content") if s.require_drive else s.output
    files = [
        p
        for p in s.results.rglob("*")
        if p.is_file() and p.name != "artifact_manifest.json"
    ]
    s.write(
        "artifact_manifest.json",
        dict(
            run_id=s.run_id,
            checkpoint_root=str(s.root),
            files=[
                dict(
                    path=p.relative_to(s.results).as_posix(),
                    bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
                for p in files
            ],
            final_test_opened=False,
        ),
    )
    path = root / f"agf_banks_activity_{s.run_id[:12]}_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in s.results.rglob("*"):
            if p.is_file():
                z.write(p, "results/" + p.relative_to(s.results).as_posix())
    return path
