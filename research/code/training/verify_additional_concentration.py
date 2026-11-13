"""Independent arithmetic/denominator verification of frozen scorer outputs."""

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np

BASE = Path(__file__).resolve().parents[2]
REVIEW = BASE / "output/ggf_all75_presence_review_20261028"


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, float):
        assert b is not None and math.isclose(a, b, abs_tol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def independent_metrics(pred, pos, units):
    total = pred.sum(1)
    union = total > 0
    np_, nn = int(pos.sum()), int((~pos).sum())
    tp, fp = int((union & pos).sum()), int((union & ~pos).sum())
    increments, unique = [], []
    first = np.where(union, pred.argmax(1), -1)
    for j, u in enumerate(units):
        added, cumulative = first == j, (first >= 0) & (first <= j)
        ap, an = int((added & pos).sum()), int((added & ~pos).sum())
        increments.append(
            dict(
                unit=u,
                rank=j + 1,
                new_positive=ap,
                new_negative=an,
                positive_increment=ap / np_,
                negative_increment=an / nn,
                cumulative_positive=int((cumulative & pos).sum()),
                cumulative_negative=int((cumulative & ~pos).sum()),
            )
        )
        exclusive = (total == 1) & pred[:, j]
        ep, en = int((exclusive & pos).sum()), int((exclusive & ~pos).sum())
        unique.append(
            dict(
                unit=u,
                positive_count=ep,
                negative_count=en,
                positive_rate=ep / np_,
                negative_rate=en / nn,
            )
        )

    def overlap(mask):
        xs, zero = [], 0
        for j in range(len(units)):
            for k in range(j):
                union_n = int((pred[mask, j] | pred[mask, k]).sum())
                if union_n:
                    xs.append(int((pred[mask, j] & pred[mask, k]).sum()) / union_n)
                else:
                    zero += 1
        return dict(
            mean_jaccard=float(np.mean(xs)) if xs else None,
            informative_pairs=len(xs),
            zero_union_pairs=zero,
            planned_pairs=len(units) * (len(units) - 1) // 2,
        )

    return dict(
        positive_count=np_,
        negative_count=nn,
        true_positive=tp,
        false_positive=fp,
        OR_coverage=tp / np_,
        OR_FPR=fp / nn,
        positive_prediction_precision=tp / (tp + fp) if tp + fp else None,
        positive_overlap=overlap(pos),
        negative_overlap=overlap(~pos),
        incremental_coverage=increments,
        unique_coverage=unique,
    )


def verify(root):
    collection, results = (
        read(root / "COLLECTION_COMPLETE.json"),
        read(root / "CONCENTRATION_COMPLETE.json"),
    )
    inputs = read(root / "INPUTS.json")
    cats = inputs["categories"]
    assert len(cats) == len(set(cats)) == 75
    banks = {x["bank_id"]: x for x in collection["banks"]}
    assert len(results["results"]) == len(banks)
    assert {r["bank_id"] for r in results["results"]} == set(banks)
    locks = {
        role: read(root / "presence" / f"{role}_LOCK.json")
        for role in ["calibration", "audit"]
    }
    assert set(locks["calibration"]["ids"]).isdisjoint(locks["audit"]["ids"])
    assert (
        len(locks["calibration"]["ids"]) == 400 and len(locks["audit"]["ids"]) == 1500
    )
    folds = {f["fold"]: f for f in inputs["folds"]}
    aggregates, bank_aggregates, support_rows = defaultdict(list), defaultdict(list), []
    curve_count = complete_count = 0
    for result in results["results"]:
        bid = result["bank_id"]
        bank = banks[bid]
        selection_path = root / "concentration" / bid / "SELECTION_LOCK.json"
        assert sha(selection_path) == result["selection_lock_sha256"]
        selection = read(selection_path)
        assert selection["calibration_only"]
        assert len(result["rows"]) == 75 and {
            r["category_index"] for r in result["rows"]
        } == set(cats)
        selected = {r["category_index"]: r["selection"] for r in selection["rows"]}
        portable = root / "portable" / bid / "POOLED.npz"
        assert sha(portable) == read(portable.with_suffix(".json"))["sha256"]
        with np.load(portable) as z:
            xp = {role: z[role + "_pooled"] for role in locks}
            for role in locks:
                assert z[role + "_ids"].tolist() == locks[role]["ids"]
                assert xp[role].shape == (len(locks[role]["ids"]), bank["groups"])
        for row in result["rows"]:
            c = row["category_index"]
            sel = selected[c]
            states = {
                role: [
                    locks[role]["labels"][str(c)][str(i)] for i in locks[role]["ids"]
                ]
                for role in locks
            }
            support = {role: Counter(v) for role, v in states.items()}
            support_rows.append(
                dict(
                    bank_id=bid,
                    category_index=c,
                    calibration=dict(support["calibration"]),
                    audit=dict(support["audit"]),
                )
            )
            eligible = all(
                support[r]["yes"] >= 5 and support[r]["no"] >= 10 for r in locks
            )
            assert len(row["curves"]) == 8
            assert {(v["negative_quantile"], v["budget"]) for v in row["curves"]} == {
                (q, b) for q in [0.95, 0.90] for b in [1, 2, 4, 8]
            }
            for qrow in sel["quantiles"]:
                if qrow["status"] != "complete":
                    assert (
                        support["calibration"]["yes"] < 5
                        or support["calibration"]["no"] < 10
                    )
                    continue
                cal_no = np.array(states["calibration"]) == "no"
                cal_yes = np.array(states["calibration"]) == "yes"
                neg = np.sort(xp["calibration"][cal_no], axis=0)
                thresholds = neg[math.ceil(qrow["negative_quantile"] * (len(neg) - 1))]
                same(thresholds.tolist(), qrow["thresholds"])
                detect = xp["calibration"] > thresholds
                quality = (detect[cal_yes].mean(0) + (~detect[cal_no]).mean(0)) / 2
                np.testing.assert_allclose(
                    quality, qrow["balanced_accuracy_by_unit"], rtol=0, atol=1e-12
                )
                # Match the frozen floating-point operation order for exact tie ordering.
                frozen_quality = 0.5 * (
                    detect[cal_yes].mean(0) + 1 - detect[cal_no].mean(0)
                )
                order = sorted(
                    range(bank["groups"]), key=lambda u: (-float(frozen_quality[u]), u)
                )
                assert order == qrow["ranked_units"]
            for curve in row["curves"]:
                curve_count += 1
                assert (curve["metrics"] is not None) == eligible
                if eligible:
                    assert curve["status"] == "complete"
                    complete_count += 1
                    qrow = next(
                        q
                        for q in sel["quantiles"]
                        if q["negative_quantile"] == curve["negative_quantile"]
                    )
                    units = qrow["ranked_units"][: curve["budget"]]
                    assert units == curve["groups"]
                    known = np.isin(states["audit"], ["yes", "no"])
                    pos = np.array(states["audit"])[known] == "yes"
                    pred = (
                        xp["audit"][known][:, units]
                        > np.array(qrow["thresholds"])[units]
                    )
                    same(independent_metrics(pred, pos, units), curve["metrics"])
                for f in bank["evaluation_folds"]:
                    stratum = (
                        "all_categories"
                        if bank["regime"] == "full_category"
                        else ("guided" if c in folds[f]["guide"] else "excluded")
                    )
                    key = (
                        bank["regime"],
                        bank["kind"],
                        bank["capacity_matched"],
                        stratum,
                        curve["negative_quantile"],
                        curve["budget"],
                    )
                    aggregates[key].append(curve)
                    bank_aggregates[
                        (bid, f, stratum, curve["negative_quantile"], curve["budget"])
                    ].append(curve)

    def summarize(key, rows, fields):
        valid = [r["metrics"] for r in rows if r["metrics"] is not None]
        out = dict(zip(fields, key), planned=len(rows), supported=len(valid))
        for metric in ["OR_coverage", "OR_FPR", "positive_prediction_precision"]:
            vals = [m[metric] for m in valid if m[metric] is not None]
            out[metric] = float(np.mean(vals)) if vals else None
        for side in ["positive", "negative"]:
            vals = [
                m[side + "_overlap"]["mean_jaccard"]
                for m in valid
                if m[side + "_overlap"]["mean_jaccard"] is not None
            ]
            out[side + "_overlap_mean_Jaccard"] = float(np.mean(vals)) if vals else None
        return out

    summary = [
        summarize(
            k,
            v,
            [
                "regime",
                "kind",
                "capacity_matched",
                "stratum",
                "negative_quantile",
                "budget",
            ],
        )
        for k, v in sorted(aggregates.items())
    ]
    per_bank = [
        summarize(k, v, ["bank_id", "fold", "stratum", "negative_quantile", "budget"])
        for k, v in sorted(bank_aggregates.items())
    ]
    qa = read(root / "PRESENCE_QA.json")
    cells = qa["cells"]
    assert qa["planned_cases"] == 40 and qa["judged_cells"] == len(cells) == 3000
    cp = [x for x in cells if x["canonical"] == "yes"]
    sp = [x for x in cells if x["second"] == "yes"]
    same(sum(x["canonical"] == x["second"] for x in cells) / 3000, qa["raw_agreement"])
    same(
        sum(x["second"] == "yes" for x in cp) / len(cp),
        qa["canonical_positive_agreement"],
    )
    same(
        sum(x["canonical"] == "yes" for x in sp) / len(sp),
        qa["second_positive_agreement"],
    )
    dashboard = read(root / "COMPARISON_RESULTS.json")
    assert dashboard["stages"]["concentration"] == "complete"
    for row in dashboard["concentration_q95"]:
        found = next(
            x
            for x in summary
            if all(
                x[k] == row[k]
                for k in ["regime", "kind", "capacity_matched", "stratum", "budget"]
            )
            and x["negative_quantile"] == 0.95
        )
        same(found["OR_coverage"], row["coverage"])
        same(found["OR_FPR"], row["FPR"])
        assert (
            found["planned"] == row["planned"]
            and found["supported"] == row["supported"]
        )
    return dict(
        complete=True,
        root=str(root),
        banks=len(banks),
        category_bank_rows=75 * len(banks),
        curves=curve_count,
        supported_curves=complete_count,
        thresholds_rankings_and_all_OR_overlap_incremental_unique_metrics_independently_verified=True,
        raw_results_sha256=sha(root / "CONCENTRATION_COMPLETE.json"),
        summary=summary,
        per_bank_and_fold=per_bank,
        support=support_rows,
        QA={k: v for k, v in qa.items() if k != "cells"},
        QA_disagreements=[x for x in cells if x["canonical"] != x["second"]],
        limits=[
            "AI presence reference, not independent human ground truth",
            "Same source images; not an unseen-image generalization allocation",
            "Concentration around 75 predefined targets, not every discovered abstract concept",
            "Image-level presence and response; spatial target alignment remains pending",
            "Runs24–27 share initialization and source images; descriptive training-order variation only",
        ],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    root = Path(parser.parse_args().root)
    result = verify(root)
    assert (
        result["banks"] == 36
        and result["category_bank_rows"] == 2700
        and result["curves"] == 21600
    )
    (root / "CONCENTRATION_INDEPENDENT_VERIFICATION.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["complete", "banks", "category_bank_rows", "curves"]
            }
        )
    )
