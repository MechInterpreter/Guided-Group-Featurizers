"""Verify the anonymous export of GPT-majority and Gemini coherence judgments."""

from collections import Counter, defaultdict
import json
from math import isclose
from pathlib import Path
from statistics import mean, stdev

from verify_review_records import recount

root = Path(__file__).resolve().parent
votes = json.loads((root / "coherence_index.json").read_bytes())
frozen = json.loads(
    (root / "evidence/matched75/GEMINI_CROSS_MODEL_RESULTS.json").read_bytes()
)
reference = {b["bank_id"]: b for b in frozen["banks"]}
identities = {b["bank_id"]: b for b in votes["banks"]}
assert len(reference) == len(identities) == 63
counts = defaultdict(Counter)
agreement = defaultdict(Counter)
cases = defaultdict(Counter)
missing = {"provider_blocked", "invalid_rubric"}
judgments, _ = recount()
for (case, bank, endpoint, gpt, gemini), count in judgments.items():
    cases[case, bank][endpoint] += count
    counts[bank][endpoint, gemini] += count
    counts[bank]["gpt_" + endpoint, gpt] += count
    agreement[endpoint][gpt, gemini] += count
assert len(cases) == 4725
assert all(
    endpoints
    == {
        "semantic": 3,
        "visual": 3,
        "coherence": 1,
        "specificity": 1,
        "clear_discriminating": 1,
    }
    for endpoints in cases.values()
)
assert Counter(bank for _, bank in cases) == {bank: 75 for bank in reference}
for endpoint, matrix in agreement.items():
    expected = frozen["agreement"][endpoint]
    assert {f"{g}/{a}": n for (g, a), n in matrix.items()} == expected["confusion"]

definitions = {
    "semantic": ("yes", 225),
    "visual": ("yes", 225),
    "coherence": ("clear", 75),
    "clear_discriminating": ("yes", 75),
}
grouped = defaultdict(list)
for bank, votes_count in counts.items():
    assert {key: reference[bank][key] for key in identities[bank]} == identities[bank]
    for endpoint, tally in reference[bank]["counts"].items():
        assert {
            label: n for (kind, label), n in votes_count.items() if kind == endpoint
        } == tally
    info = identities[bank]
    group = (info["regime"], info["method"], info["seed"])
    metrics = {}
    for endpoint, (positive, planned) in definitions.items():
        assert (
            sum(n for (kind, _), n in votes_count.items() if kind == endpoint)
            == planned
        )
        positives = votes_count[endpoint, positive]
        invalid = sum(votes_count[endpoint, label] for label in missing)
        metrics[endpoint] = {
            "lower": positives / planned,
            "upper": (positives + invalid) / planned,
            "valid_only": positives / (planned - invalid),
            "missing": invalid / planned,
            "gpt": votes_count["gpt_" + endpoint, positive] / planned,
        }
        assert metrics[endpoint] == reference[bank]["metrics"][endpoint]
    grouped[group].append(metrics)

for row in frozen["methods"]:
    seed_rows = row["per_seed"]
    assert set(seed_rows) == {str(seed) for seed in range(21, 28)}
    for seed, summary in seed_rows.items():
        banks = grouped[row["regime"], row["method"], int(seed)]
        for endpoint in definitions:
            for field, value in summary[endpoint].items():
                assert isclose(
                    mean(b[endpoint][field] for b in banks), value, abs_tol=1e-12
                )
    for endpoint in definitions:
        for field, value in row["summary"][endpoint].items():
            actual = (
                stdev(seed[endpoint]["lower"] for seed in seed_rows.values())
                if field == "sample_sd"
                else mean(seed[endpoint][field] for seed in seed_rows.values())
            )
            assert isclose(actual, value, abs_tol=1e-12)
print(
    "Verified 4,725 groups, 63 banks, seven orders, GPT/Gemini agreement, and mean/sample SD."
)
