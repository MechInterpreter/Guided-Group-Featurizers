"""Prepare and score whole-group judgments for freshly generated blinded cards."""

import argparse
import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

CONTEXTS = ("context1", "context2", "context3")


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Frozen review content changed: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)


def setup(code):
    sys.path[:0] = [
        str(code / "training"),
        str(code / "training/core"),
        str(code / "review_records"),
    ]


def instructions(code):
    path = code / "review_records/original_scripts/group_review.py"
    tree = ast.parse(path.read_bytes())
    values = [
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "INSTRUCTIONS" for t in n.targets)
    ]
    if len(values) != 1:
        raise ValueError("Missing frozen whole-group rubric")
    return values[0], digest(path)


def prepare(root, code):
    from prepare_evaluation_reviews import verify_pack, assets
    from eval_common import verified_views

    out = root / "evaluation"
    pair = out / "reviews/coherence"
    folder = out / "reviews/whole"
    lock = verify_pack(pair)
    public = read(pair / "PUBLIC.json")
    private = read(pair / "PRIVATE.json")
    collection = read(out / "COLLECTION_COMPLETE.json")
    banks = collection["banks"]
    if (
        len(banks) != 63
        or any(b["groups"] != 75 for b in banks)
        or len(public["cases"]) != 4725
    ):
        raise ValueError(
            "Freeze all seven orders and every matched group before whole-group review"
        )
    records = assets(out)
    views = {b["bank_id"]: verified_views(out, b["bank_id"]) for b in banks}
    cases = []
    for case in private["cases"]:
        source = views[case["bank_id"]]["roles"]["audit"][str(case["unit"])]
        valid = []
        for display, j in enumerate(case["source_slots"], 1):
            view = source[j] if j < len(source) else {}
            if (
                view.get("patches")
                and records.get(view.get("image_id"), {}).get("status") == "ready"
            ):
                valid.append(display)
        cases.append(
            dict(
                case_id=case["case_id"],
                card="../coherence/cards/" + case["case_id"] + ".png",
                valid_high_views=valid,
                valid_high_count=len(valid),
            )
        )
    rubric, source_sha = instructions(code)
    freeze(
        folder / "PUBLIC.json",
        dict(
            schema="fresh-seven-order-whole-group-v1", instructions=rubric, cases=cases
        ),
    )
    freeze(folder / "PRIVATE.json", dict(cases=private["cases"], banks=banks))
    freeze(
        folder / "REVIEW_LOCK.json",
        dict(
            public_sha256=digest(folder / "PUBLIC.json"),
            private_sha256=digest(folder / "PRIVATE.json"),
            pair_lock_sha256=digest(pair / "REVIEW_LOCK.json"),
            rubric_source_sha256=source_sha,
            cards=lock["cards"],
        ),
    )
    for name in CONTEXTS:
        template = dict(
            public_sha256=digest(folder / "PUBLIC.json"),
            method_blind=True,
            reviewer="",
            cases=[],
        )
        for case in cases:
            template["cases"].append(
                dict(
                    case_id=case["case_id"],
                    theme="",
                    fit_views=[],
                    high_fit_count=None,
                    exceptions=[],
                    coherence="",
                    specificity="",
                    confidence="",
                    high_low_contrast="not_assessed",
                    observations="",
                )
            )
        freeze(folder / "templates" / (name + ".json"), template)
        for lo in range(0, len(cases), 12):
            freeze(
                folder / "packets" / name / (f"B{lo // 12:03d}.json"),
                dict(
                    context=name,
                    instructions=rubric,
                    public_sha256=template["public_sha256"],
                    card_root="../../",
                    cases=cases[lo : lo + 12],
                    output_template=dict(
                        template, cases=template["cases"][lo : lo + 12]
                    ),
                ),
            )
    print("Whole-group templates ready:", folder)


def majority(values):
    counts = Counter(values)
    return next((code for code, n in counts.items() if n >= 2), "unresolved_vote")


def score(root, code):
    import rubric_validation as rubric
    from score_evaluation import coherence
    from prepare_evaluation_reviews import verify_pack

    out = root / "evaluation"
    folder = out / "reviews/whole"
    lock = read(folder / "REVIEW_LOCK.json")
    verify_pack(out / "reviews/coherence")
    if (
        digest(folder / "PUBLIC.json") != lock["public_sha256"]
        or digest(folder / "PRIVATE.json") != lock["private_sha256"]
        or digest(out / "reviews/coherence/REVIEW_LOCK.json")
        != lock["pair_lock_sha256"]
        or instructions(code)[1] != lock["rubric_source_sha256"]
    ):
        raise ValueError("Whole-group allocation or card lock changed")
    public = read(folder / "PUBLIC.json")
    private = read(folder / "PRIVATE.json")
    cases = {c["case_id"]: c for c in public["cases"]}
    if (
        len(cases) != len(public["cases"])
        or len(private["cases"]) != len(cases)
        or {c["case_id"] for c in private["cases"]} != cases.keys()
    ):
        raise ValueError("Whole-group public/private case inventory differs")
    contexts = []
    sources = []
    for name in CONTEXTS:
        path = folder / "labels" / (name + ".json")
        raw = read(path)
        if (
            raw.get("public_sha256") != lock["public_sha256"]
            or raw.get("method_blind") is not True
            or not raw.get("reviewer", "").strip()
        ):
            raise ValueError("Missing whole-group blinding and reviewer provenance")
        rows = {r["case_id"]: r for r in raw["cases"]}
        if len(rows) != len(raw["cases"]) or rows.keys() != cases.keys():
            raise ValueError("Every planned whole-group judgment is required")
        for cid, row in rows.items():
            rubric.validate_case(row, cases[cid])
        for source in raw.get("sources", []):
            path = (out / source["path"]).resolve()
            if (
                not path.is_relative_to(out.resolve())
                or digest(path) != source["sha256"]
            ):
                raise ValueError("Raw review source changed")
        contexts.append(rows)
        sources.append(
            dict(
                path=str((folder / "labels" / (name + ".json")).relative_to(out)),
                sha256=digest(folder / "labels" / (name + ".json")),
                reviewer=raw["reviewer"],
            )
        )
    if len({r["reviewer"] for r in sources}) != 3:
        raise ValueError("Use three distinct fresh reviewer contexts")
    groups = {}
    outcomes = []
    for case in private["cases"]:
        ratings = [context[case["case_id"]] for context in contexts]
        codes = [r["coherence"] for r in ratings]
        specificity = [r["specificity"] for r in ratings]
        outcome = dict(
            case_id=case["case_id"],
            bank_id=case["bank_id"],
            unit=case["unit"],
            coherence=majority(codes),
            specificity=majority(specificity),
            clear_and_discriminating=codes.count("clear") >= 2
            and sum(v in ["specific", "broad_but_discriminating"] for v in specificity)
            >= 2,
            disagreement=len(set(codes)) > 1 or len(set(specificity)) > 1,
            ratings=ratings,
        )
        outcomes.append(outcome)
        groups.setdefault(case["bank_id"], []).append(outcome)
    banks = []
    for bank in private["banks"]:
        rows = groups[bank["bank_id"]]
        if sorted(r["unit"] for r in rows) != list(range(75)):
            raise ValueError("Whole-group denominator changed")
        counts = Counter(r["coherence"] for r in rows)
        specificity = Counter(r["specificity"] for r in rows)
        banks.append(
            dict(
                bank_id=bank["bank_id"],
                seed=bank["seed"],
                regime=bank["regime"],
                method=bank["kind"],
                fold=bank["fold"],
                groups=75,
                counts={k: counts[k] for k in [*rubric.WHOLE_CODES, "unresolved_vote"]},
                specificity_counts={
                    k: specificity[k] for k in [*rubric.WHOLE_SPEC, "unresolved_vote"]
                },
                clear_rate=counts["clear"] / 75,
                clear_and_discriminating_rate=sum(
                    r["clear_and_discriminating"] for r in rows
                )
                / 75,
                disagreements=sum(r["disagreement"] for r in rows),
            )
        )
    coherence(out)
    freeze(
        out / "WHOLE_COHERENCE_COMPLETE.json",
        dict(
            complete=True,
            banks=banks,
            group_outcomes=outcomes,
            sources=sources,
            public_sha256=lock["public_sha256"],
            scope="Separate three-context whole-group endpoint; all planned groups retained",
        ),
    )
    print("Pair and whole-group scores complete; source judgments preserved.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--phase", choices=["prepare", "score"], required=True)
    a = p.parse_args()
    setup(a.code_root.resolve())
    (prepare if a.phase == "prepare" else score)(
        a.root.resolve(), a.code_root.resolve()
    )


if __name__ == "__main__":
    main()
