"""Recount published statistics from the bundled per-order and per-seed records."""

import csv
import json
from math import isclose
from pathlib import Path
from statistics import mean, stdev


root = Path(__file__).resolve().parent / "evidence"


def load(path):
    return json.loads((root / path).read_bytes())


def check(values, expected_mean, expected_sd):
    assert isclose(mean(values), expected_mean, rel_tol=1e-9, abs_tol=1e-12)
    assert isclose(stdev(values), expected_sd, rel_tol=1e-9, abs_tol=1e-12)


numeric = load("matched75/MATCHED75_NUMERIC_SEEDS21_27.json")
matched = load("matched75/RESULTS_SEEDS21_27.json")
seeds = {str(seed) for seed in range(21, 28)}
assert numeric["matched_banks"] == 63
assert len(numeric["rows"]) == 32 and len(matched["concentration"]) == 80
concentration_cells = 0
for row in numeric["rows"]:
    assert row["per_seed"].keys() == seeds
    check(list(row["per_seed"].values()), row["mean"], row["sample_sd"])
for row in matched["concentration"]:
    for cell in row["metrics"].values():
        assert cell["per_seed"].keys() == seeds
        values = [value for value in cell["per_seed"].values() if value is not None]
        if values:
            check(values, cell["mean"], cell["sd"])
            concentration_cells += 1
        else:
            assert cell["mean"] is None and cell["sd"] is None

voc = root / "voc_slics_final"
with (voc / "VOC_completed_per_seed.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert {(r["seed"], r["method"]) for r in rows} == {
    (str(seed), method) for seed in (21, 22, 23) for method in ("GGF", "SLiCS")
}
summary = load("voc_slics_final/verified_analysis.json")
for method, metrics in summary["summary"].items():
    for name, cell in metrics.items():
        check(
            [float(r[name]) for r in rows if r["method"] == method],
            cell["mean"],
            cell["sd"],
        )

profiles = load("voc_slics_final/compute_profiles.json")
coverage = load("voc_slics_final/FLOP_COVERAGE.json")
assert coverage["full_FLOPs_available"] is False
assert len(profiles) == 17
for profile in profiles:
    assert profile["images"] in {8, 64}
    assert len(profile["seconds"]) == 3 and profile["flop_probes"]
    check(profile["seconds"], profile["seconds_mean"], profile["seconds_SD"])
    for probe in profile["flop_probes"]:
        assert probe["registered_operator_FLOPs"] == sum(
            probe["operator_FLOPs"].values()
        )

print(
    f"Verified 32 matched numeric rows, {concentration_cells} concentration summaries, "
    f"six VOC endpoints and {len(profiles)} execution profiles."
)
