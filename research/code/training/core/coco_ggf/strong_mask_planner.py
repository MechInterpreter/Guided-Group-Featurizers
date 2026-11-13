"""Mask-only fresh planning with separate detection/contrast coverage quotas.

The three evidence flags need not occur in the same image. Each split still
requires the declared minimum for EACH flag in EVERY retained category.
"""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import block_diag, csc_matrix, eye, hstack, vstack
from coco_adaptation import inputs, coverage
from coco_baselines import data
from coco_baselines.storage import identity, sha_file
from coco_coordinated import run as memory

OLD_SOURCE_SHAS = {
    "da375f35f6ec9f74a01da4fbafceaee1d653633d581103ca383e4cba81635879",
    "b6c664601657eb087ebda56b3195773f82233d1da50ba915e9c93ae6858db2c9",
    "acf9d447b4a012ac6969b11711114f7a2e908054be3c91be523be45ff49ce491",
}
EVIDENCE_NAMES = (
    "positive_patch_images",
    "negative_patch_images",
    "object_contrast_images",
)


def validate(plan, used, cohort, cfg):
    if plan["coverage"]["category_indices"] != cohort:
        raise ValueError("Fresh cohort changed")
    if plan["excluded_ids_sha"] != identity(sorted(used)):
        raise ValueError("Historical exclusions changed")
    if plan.get("coverage_protocol") != cfg["coverage_protocol"]:
        raise ValueError("Wrong coverage protocol")
    confirm = plan["splits"]["confirm"]
    test = plan["splits"]["test"]
    if (
        len(confirm) != len(set(confirm))
        or identity(confirm) != plan["saved_development_ids_sha"]
    ):
        raise ValueError("Historical development identity failed")
    if not set(confirm) <= set(used):
        raise ValueError("Development must be a declared historical split")
    if (
        len(test) != cfg["test_images"]
        or len(set(test)) != len(test)
        or set(test) & (set(used) | set(confirm))
    ):
        raise ValueError("Fresh test size/identity/overlap failed")
    counts = np.asarray(plan["coverage"]["split_evidence_counts"]["test"])
    if (
        counts.ndim != 2
        or counts.shape[1] != 3
        or np.any(counts[cohort] < cfg["test_min_images"])
    ):
        raise ValueError("Fresh test evidence quota missing")


def allocate(ids, evidence, cohort, cfg, event=lambda *a, **k: None):
    ids = np.asarray(ids)
    ev = np.asarray(evidence, dtype=bool)
    sizes = [cfg["development_images"], cfg["test_images"]]
    quotas = [cfg["development_min_images"], cfg["test_min_images"]]
    if ev.ndim != 3 or ev.shape[0] != len(ids) or ev.shape[2] != 3:
        raise ValueError("Expected positive/negative/contrast evidence per image")
    if len(ids) != len(set(ids.tolist())) or len(ids) < sum(sizes):
        raise ValueError("Insufficient unique mask pool for both fresh sets")
    bits = ev[:, cohort, :].reshape(len(ids), -1).astype(np.int32)
    if np.any(bits.sum(0) < sum(quotas)):
        raise ValueError(
            "Insufficient per-evidence supply; no quota or category is changed"
        )
    rng = np.random.default_rng(cfg["plan_seed"])
    order = rng.permutation(len(ids))
    priority = np.empty(len(ids), int)
    priority[order] = np.arange(len(ids))
    patterns = {}
    for i in order:
        row = bits[i]
        if row.any():
            bucket = patterns.setdefault(row.tobytes(), [])
            if len(bucket) < sum(quotas):
                bucket.append(int(i))
    kernel = np.array([i for v in patterns.values() for i in v], int)
    n = len(kernel)
    cover = csc_matrix(bits[kernel].T.astype(float))
    qcount = cover.shape[0]
    matrix = vstack(
        [
            block_diag((cover, cover)),
            hstack([eye(n), eye(n)]),
            block_diag((csc_matrix(np.ones((1, n))), csc_matrix(np.ones((1, n))))),
        ],
        format="csc",
    )
    lower = np.r_[np.repeat(quotas, qcount), np.zeros(n), np.zeros(2)]
    upper = np.r_[np.full(2 * qcount, np.inf), np.ones(n), sizes]
    event(
        "joint_evidence_solver_start",
        candidates=len(ids),
        kernel_images=n,
        quotas=quotas,
        sizes=sizes,
        evidence_names=EVIDENCE_NAMES,
    )
    result = milp(
        np.ones(2 * n),
        integrality=np.ones(2 * n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options=dict(time_limit=cfg["solver_seconds"], mip_rel_gap=0.02),
    )
    event(
        "joint_evidence_solver_result",
        status=int(result.status),
        message=str(result.message),
    )
    if result.x is None:
        raise ValueError(
            "No joint feasible evidence assignment; quotas remain unchanged"
        )
    chosen = [
        kernel[np.flatnonzero(result.x[j * n : (j + 1) * n] > 0.5)].tolist()
        for j in range(2)
    ]
    free = np.ones(len(ids), bool)
    for take, q, size in zip(chosen, quotas, sizes):
        if (
            len(take) > size
            or (q > 0 and not take)
            or np.any(bits[take].sum(0) < q)
            or not free[take].all()
        ):
            raise ValueError("Invalid joint solver incumbent")
        free[take] = False
    # Reserve both quota cores before filling either set.
    target = bits.mean(0)
    for j, take in enumerate(chosen):
        if sizes[j] == 0:
            continue
        have = bits[take].sum(0)
        goal = np.maximum(target, quotas[j] / sizes[j]) * sizes[j]
        while len(take) < sizes[j]:
            pool = np.flatnonzero(free)
            pool = pool[np.argsort(priority[pool])[:1024]]
            deficit = goal - have
            cost = np.sum(
                ((deficit - bits[pool]) ** 2 - deficit**2) / np.maximum(goal, 1), axis=1
            )
            i = int(pool[np.lexsort((priority[pool], cost))[0]])
            take.append(i)
            free[i] = False
            have += bits[i]
    splits = {
        name: ids[rng.permutation(np.asarray(take, dtype=int))].tolist()
        for name, take in zip(("confirm", "test"), chosen)
    }
    report = dict(
        category_indices=cohort,
        evidence_names=list(EVIDENCE_NAMES),
        split_evidence_counts={
            name: ev[take].sum(0).tolist()
            for name, take in zip(("confirm", "test"), chosen)
        },
        # Keep the original intersection counts for extraction equality/audit.
        split_counts={
            name: ev[take].all(2).sum(0).tolist()
            for name, take in zip(("confirm", "test"), chosen)
        },
        minimums=dict(confirm=quotas[0], test=quotas[1]),
        sizes=dict(confirm=sizes[0], test=sizes[1]),
        candidate_evidence_counts=ev.sum(0).tolist(),
        allocator="joint_separate_evidence_binary_cover_then_prevalence_fill",
        solver_status=int(result.status),
    )
    return splits, report


def legacy_settings(cfg, version):
    expected = dict(cfg, development_images=2000, development_min_images=40)
    expected.pop("development_source", None)
    expected["scope"] = (
        "One remeasured training graph/pass per seed; eight trials, at most five nonconflicting exchanges; no gradients or baseline retraining. Fresh development selects; all seeds freeze before untouched test. Legacy native +2% MSE eligibility remains a separate required flag for an official upgrade."
    )
    if version == 2:
        expected.update(
            version=2, coverage_protocol="separate_three_evidence_quotas_v2"
        )
    else:
        expected["version"] = 1
        expected.pop("coverage_protocol", None)
        expected.pop("strict_evidence_quotas", None)
    return expected


def old_scans(ctx):
    """Import only annotation computations, never an old model or selection."""
    s = ctx["store"]
    cfg = ctx["strong_cfg"]
    for path in sorted((s.output / "results").glob("*/manifest.json")):
        rec = json.loads(path.read_text())
        c = rec["config"]
        if c.get("source_sha") not in OLD_SOURCE_SHAS:
            continue
        expected = legacy_settings(
            cfg,
            2
            if c["source_sha"]
            == "acf9d447b4a012ac6969b11711114f7a2e908054be3c91be523be45ff49ce491"
            else 1,
        )
        if c.get("config") != expected:
            raise ValueError("Prior mask-scan protocol changed")
        for key in ("recovery_run", "activation_run", "versions", "device"):
            if c.get(key) != s.config.get(key):
                raise ValueError("Prior mask scan lineage differs: " + key)
        prior = inputs.ReadStore(
            s.output / "checkpoints" / rec["run_id"][:20],
            ctx["local"] / "old_mask_scans",
        )
        for folder in sorted(
            list(prior.root.glob("mask_extra_c*"))
            + list(prior.root.glob("mask_evidence_c*"))
        ):
            yield prior, folder.name, memory.required(prior, folder.name)


def plan(ctx, coco):
    s = ctx["store"]
    cfg = ctx["fresh_cfg"]
    used = ctx["excluded_ids"]
    old = s.load("fresh_plan")
    if old is not None:
        validate(old, used, ctx["cohort"], cfg)
        if old["splits"]["confirm"] != ctx["packs"]["confirm"]["ids"]:
            raise ValueError("Saved development changed")
        return old
    source = memory._discover(
        ctx["original"].output / "coverage",
        lambda c: (
            c["source_sha"] == ctx["original"].config["source_sha"]
            and c["annotation_sha"] == ctx["provenance"]["annotation_sha"]
        ),
        ctx["local"] / "read_coverage",
    )
    table = {}
    ncat = len(ctx["provenance"]["categories"])

    def add(part):
        ids = list(map(int, part["ids"]))
        ev = np.asarray(part["evidence"], dtype=bool)
        if ev.shape != (len(ids), ncat, 3) or len(ids) != len(set(ids)):
            raise ValueError("Invalid annotation evidence shard")
        for i, e in zip(ids, ev):
            if i in used:
                continue
            if i in table and not np.array_equal(e, table[i]):
                raise ValueError("Conflicting mask evidence")
            table[i] = e

    folders = sorted(source.root.glob("coverage_scan_*"))
    for index, folder in enumerate(folders):
        add(memory.required(source, folder.name))
        if index % 20 == 0:
            s.event(
                "mask_pool_read", done=index + 1, total=len(folders), unused=len(table)
            )
    for prior, key, part in old_scans(ctx):
        add(part)
        s.event(
            "annotation_scan_reused",
            key=key,
            source_run=prior.run_id,
            unused=len(table),
        )
    for folder in sorted(s.root.glob("mask_evidence_c*")):
        add(memory.required(s, folder.name))
    if not table:
        raise ValueError("No unused annotation pool")
    need = cfg["development_min_images"] + cfg["test_min_images"]
    desired = int(np.ceil(need * cfg["coverage_reserve"]))
    supply = np.stack(list(table.values())).sum(0)
    seen = set(table) | used
    extension = []
    for c in ctx["cohort"]:
        if np.all(supply[c] >= desired):
            continue
        cat = ctx["provenance"]["categories"][c]["id"]
        candidates = (
            np.random.default_rng(cfg["plan_seed"] + c + 100)
            .permutation(sorted(set(coco.getImgIds(catIds=[cat])) - seen))[
                : cfg["extension_limit_per_category"]
            ]
            .tolist()
        )
        for lo in range(0, len(candidates), 25):
            chosen = candidates[lo : lo + 25]
            key = "mask_evidence_c%d_%s" % (c, identity(chosen)[:16])
            part = s.load(key)
            if part is None:
                ev = []
                for i in chosen:
                    im = coco.imgs[i]
                    mask = data.masks_for_image(
                        coco,
                        i,
                        data.geometry(im["width"], im["height"]),
                        sorted(coco.cats),
                    )
                    ev.append(
                        coverage.evidence(dict(ids=[i], masks=[mask]), ctx["cfg"])[0]
                    )
                part = dict(ids=chosen, evidence=np.stack(ev))
                s.save(key, part)
            if part["ids"] != chosen:
                raise ValueError("Mask extension identity changed")
            add(part)
            seen.update(chosen)
            supply += np.asarray(part["evidence"]).sum(0)
            s.event(
                "mask_evidence_extension",
                category_index=c,
                done=lo + len(chosen),
                counts=supply[c].tolist(),
                required_each=need,
                desired_reserve=desired,
            )
            if np.all(supply[c] >= desired):
                break
        extension.append(
            dict(
                category_index=c,
                checked=len(candidates),
                remaining_in_catalog=len(set(coco.getImgIds(catIds=[cat])) - seen),
            )
        )
    rows = np.stack(list(table.values()))
    supply = rows.sum(0)
    diagnostic = dict(
        protocol="Separate evidence quotas; no overlap, category dropping or quota reduction",
        excluded_images=len(used),
        unused_scanned_images=len(table),
        minimum_each=need,
        reserve_goal_each=desired,
        categories=[
            dict(
                category_index=c,
                category=ctx["provenance"]["categories"][c]["name"],
                counts=dict(zip(EVIDENCE_NAMES, map(int, supply[c]))),
                intersection_images=int(rows[:, c].all(1).sum()),
                quota_feasible=bool(np.all(supply[c] >= need)),
                reserve_met=bool(np.all(supply[c] >= desired)),
            )
            for c in ctx["cohort"]
        ],
        extensions=extension,
    )
    s.write("fresh_coverage_diagnostic.json", diagnostic)
    failed = [r for r in diagnostic["categories"] if not r["quota_feasible"]]
    if failed:
        s.event("fresh_coverage_infeasible", categories=failed)
        raise ValueError(
            "True per-evidence shortage before GPU extraction: "
            + json.dumps(failed)
            + ". See fresh_coverage_diagnostic.json. No quotas/categories or historical exclusions were changed."
        )
    if not all(r["reserve_met"] for r in diagnostic["categories"]):
        s.event(
            "optional_reserve_unavailable",
            note="Exact joint assignment still must meet every hard quota",
        )
    ids = list(table)
    splits, report = allocate(ids, rows, ctx["cohort"], cfg, s.event)
    historical = ctx["packs"]["confirm"]
    dev_ev = coverage.evidence(historical, ctx["cfg"])
    splits["confirm"] = list(historical["ids"])
    report["split_evidence_counts"]["confirm"] = dev_ev.sum(0).tolist()
    report["split_counts"]["confirm"] = dev_ev.all(2).sum(0).tolist()
    report["sizes"]["confirm"] = len(historical["ids"])
    report["development_role"] = (
        "Previously inspected original confirmation split; selection only, not an independent new confirmation"
    )
    result = dict(
        splits=splits,
        coverage=report,
        coverage_protocol=cfg["coverage_protocol"],
        saved_development_ids_sha=identity(splits["confirm"]),
        excluded_ids_sha=identity(sorted(used)),
        annotation_sha=ctx["provenance"]["annotation_sha"],
        rule="Reuse saved development for selection; masks-only fresh test with unchanged20-per-evidence quotas in all75 categories.",
    )
    validate(result, used, ctx["cohort"], cfg)
    s.save("fresh_plan", result)
    s.write("fresh_split_plan.json", result)
    return result
