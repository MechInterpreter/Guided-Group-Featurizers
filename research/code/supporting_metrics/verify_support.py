"""Recount the retained entropy ranks and headroom panel from saved evidence.

Run from the bundle root: python code/supporting_metrics/verify_support.py
No training, cache access, network requests, or evidence writes are performed.
"""

from pathlib import Path
from hashlib import sha256
import ast
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    "evidence/voc_heldout/selected_ranks.json": "c18ea8b390ca4cd8a74ab7163b260ec2088c648731f88a66a752f33f19404511",
    # Original restored headroom result, not a reconstructed display table.
    "supporting_metrics/wider_geometric_headroom.json": "6eeb9c15a1772f9f326263d211a303c9858047a70e4972fbaea72a522a3a268b",
    "training/core/coco_ggf/wider_groups.py": "a2236b19e2167396a71cb4e3b8861034ec10fc2c91c7d7eee891cf346e4aeef9",
}


def spectral_measures(eigenvalues):
    values = [float(value) for value in eigenvalues]
    assert len(values) == 12 and all(math.isfinite(value) for value in values)
    assert min(values) >= -1e-9 * max(max(abs(value) for value in values), 1e-12)
    values = [max(value, 0) for value in values]
    total = sum(values)
    if total <= 1e-12:
        return {
            "entropy_rank": None,
            "participation_ratio": None,
            "coefficient_matrix_stable_rank": None,
            "covariance_stable_rank": None,
        }
    positive = [value / total for value in values if value > 0]
    squared = sum(value**2 for value in values)
    return {
        "entropy_rank": math.exp(-sum(p * math.log(p) for p in positive)),
        "participation_ratio": total**2 / squared,
        "coefficient_matrix_stable_rank": total / max(values),
        "covariance_stable_rank": squared / max(values) ** 2,
    }


def verify_ranks():
    records = json.loads(
        (ROOT / "evidence/voc_heldout/selected_ranks.json").read_bytes()
    )
    expected = {
        "released_vanilla": ((6.58, 0.07), (4.99, 0.15)),
        "supervised_vanilla": ((5.98, 0.12), (4.21, 0.11)),
        "agf_g": ((6.40, 0.03), (6.28, 0.03)),
    }
    assert len(records) == 9
    identities = set()
    per_kind = {kind: [] for kind in expected}
    checked = 0
    for model in records:
        identity = (model["kind"], model["seed"])
        assert identity not in identities and model["kind"] in expected
        identities.add(identity)
        groups = model["selected"]
        assert len(groups) == 19 and len({g["category"] for g in groups}) == 19
        assert len(model["cohort_ids"]) == 256
        ranks = [[], []]
        for group in groups:
            assert group["allocated_width"] == 12 and group["active_samples"] >= 2
            for column, key in enumerate(("coefficient", "decoded_contribution")):
                saved = group[key]
                measures = spectral_measures(saved["spectrum"])
                assert (
                    not saved["zero_variance"] and measures["entropy_rank"] is not None
                )
                for metric, value in measures.items():
                    assert math.isclose(
                        value, saved[metric], rel_tol=1e-10, abs_tol=1e-10
                    ), (identity, group["category"], key, metric)
                ranks[column].append(measures["entropy_rank"])
                checked += 1
            assert math.isclose(
                group["entropy_utilization"],
                ranks[0][-1] / 12,
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        per_kind[model["kind"]].append([statistics.fmean(values) for values in ranks])
    assert identities == {(kind, seed) for kind in expected for seed in (21, 22, 23)}
    table = {}
    for kind, values in per_kind.items():
        mean = [statistics.fmean(row[column] for row in values) for column in range(2)]
        sd = [statistics.stdev(row[column] for row in values) for column in range(2)]
        for column in range(2):
            assert f"{mean[column]:.2f}" == f"{expected[kind][column][0]:.2f}"
            assert f"{sd[column]:.2f}" == f"{expected[kind][column][1]:.2f}"
        table[kind] = {
            "coefficient_mean": float(mean[0]),
            "coefficient_sample_SD": float(sd[0]),
            "contribution_mean": float(mean[1]),
            "contribution_sample_SD": float(sd[1]),
        }
    return {
        "model_orders": 9,
        "selected_category_rows": 171,
        "spectra_recounted": checked,
        "published_two_decimal_cells_match": True,
        "table_S27_ranks": table,
    }


def verify_headroom(check_source=False):
    record = json.loads(
        (ROOT / "supporting_metrics/wider_geometric_headroom.json").read_bytes()
    )
    categories = record["categories"]
    assert (
        len(categories) == 75 and len({c["category_index"] for c in categories}) == 75
    )
    widths = [3, 4, 8, 16]
    fields = [
        "train_optimistic_ceiling",
        "development_optimistic_ceiling",
        "development_capture_with_training_basis",
    ]
    rows = []
    for category in categories:
        assert [r["width"] for r in category["rows"]] == widths
        assert (
            min(
                category[k]
                for k in (
                    "train_objects",
                    "development_objects",
                    "train_energy",
                    "development_energy",
                )
            )
            > 0
        )
        for field in fields:
            values = [row[field] for row in category["rows"]]
            assert all(-1e-12 <= value <= 1 + 1e-12 for value in values)
            assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
        for row in category["rows"]:
            assert row[fields[2]] <= row[fields[1]] + 1e-10
    for width, saved in zip(widths, record["macro"]):
        assert saved["width"] == width
        row = {"width": width}
        for field in fields:
            value = statistics.fmean(
                next(r[field] for r in c["rows"] if r["width"] == width)
                for c in categories
            )
            assert math.isclose(value, saved[field], rel_tol=1e-12, abs_tol=1e-12)
            row[field + "_percent"] = 100 * value
        rows.append(row)
    assert f"{rows[0][fields[2] + '_percent']:.2f}" == "41.15"
    assert f"{rows[-1][fields[2] + '_percent']:.2f}" == "52.86"
    if check_source:
        source_formula_check(fields)
    return {
        "categories": 75,
        "per_category_width_rows": 300,
        "macro_rows": rows,
        "original_source_formula_checked": check_source,
        "scope": "Recounts saved category-level results; does not recompute source-image contrasts.",
    }


def source_formula_check(fields):
    # Optional NumPy/SciPy check of the original calculation, without the training stack.
    import numpy as np
    from scipy.linalg import eigh

    path = ROOT / "training/core/coco_ggf/wider_groups.py"
    tree = ast.parse(path.read_text())
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "spectrum"
    ]
    assert len(functions) == 1
    scope = {"np": np, "eigh": eigh}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    a = np.array([[3.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, 2.0]])
    probe = scope["spectrum"](a, b, [1, 2])
    first, full = probe["rows"]
    assert math.isclose(first["train_optimistic_ceiling"], 0.9)
    assert math.isclose(first["development_optimistic_ceiling"], 0.8)
    assert math.isclose(first["development_capture_with_training_basis"], 0.2)
    assert all(math.isclose(full[k], 1.0) for k in fields)


def verify(check_source=False):
    for name, expected in SOURCE_HASHES.items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    result = {
        "entropy_ranks": verify_ranks(),
        "headroom": verify_headroom(check_source),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify(check_source="--check-source" in sys.argv)
