"""Recount every matched judgment from anonymous raw notes and API bodies offline."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "review_records"))
from rubric_validation import paid_schema, validate


def majority(labels, fallback):
    return next(
        (label for label, count in Counter(labels).items() if count >= 2), fallback
    )


def recount():
    manifest = json.loads((ROOT / "review_records/MANIFEST.json").read_bytes())
    archive = ROOT / "review_records/judgments.zip"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    )
    recounted = Counter()
    statuses = Counter()

    with ZipFile(archive) as z:
        assert z.testzip() is None
        for record in manifest["files"]:
            assert (
                hashlib.sha256(z.read(record["path"])).hexdigest()
                == record["exported_sha256"]
            )
        load = lambda path: json.loads(z.read(path))
        sources = load("case_sources.json")
        lookup = {
            (s["era"], s["family"], s["context"], s["case_id"]): s for s in sources
        }
        assert len(lookup) == len(sources) == 28575
        grouped_sources = {}
        for source in sources:
            name = source["raw_file"]
            if name not in grouped_sources:
                grouped_sources[name] = {r["case_id"]: r for r in load(name)["cases"]}
            assert source["original_case_id"] in grouped_sources[name]
        production = load("PRODUCTION_MANIFEST.json")
        identities = set()
        for group in production["groups"]:
            cid, bank, era = (group[k] for k in ("case_id", "bank_id", "era"))
            for family in ("pair", "whole"):
                refs = []
                for context in ("context1", "context2", "context3"):
                    source = lookup[era, family, context, cid]
                    refs.append(
                        grouped_sources[source["raw_file"]][source["original_case_id"]]
                    )
                record = load(f"gemini/{era}/{family}/{cid}.json")
                assert (
                    record["group"],
                    record["era"],
                    record["family"],
                    record["evidence_sha256"],
                ) == (cid, era, family, group["evidence"][family]["sha256"])
                response = json.loads(record["raw_response"])
                label = None
                if response.get("error"):
                    assert (
                        response["error"]["metadata"]["error_type"]
                        == "content_policy_violation"
                    )
                    status = "provider_blocked"
                else:
                    assert (
                        response["provider"] == "Google AI Studio"
                        and response["service_tier"] == "flex"
                    )
                    assert (
                        response["id"].startswith("anonymous-sha256:")
                        and response["id"] not in identities
                    )
                    identities.add(response["id"])
                    case = group["evidence"][family]["case"]
                    props = paid_schema(case, family)["properties"]
                    template = dict.fromkeys(props)
                    if family == "pair":
                        template["pairs"] = [
                            dict.fromkeys(props["pairs"]["items"]["properties"])
                        ]
                    try:
                        label = validate(
                            response, case, template, family, paid=True, stronger=True
                        )
                        status = "valid"
                    except (ValueError, TypeError, KeyError, AttributeError):
                        status = "invalid_rubric"
                statuses[family, status] += 1

                def add(endpoint, gpt, gemini):
                    recounted[cid, bank, endpoint, gpt, gemini] += 1

                if family == "pair":
                    model = {p["pair"]: p for p in label["pairs"]} if label else {}
                    for endpoint in ("semantic", "visual"):
                        for pair in range(3):
                            gpt = majority(
                                [
                                    next(p for p in r["pairs"] if p["pair"] == pair)[
                                        endpoint
                                    ]
                                    for r in refs
                                ],
                                "unclear",
                            )
                            add(
                                endpoint,
                                gpt,
                                model[pair][endpoint] if label else status,
                            )
                else:
                    gpt = majority([r["coherence"] for r in refs], "unresolved_vote")
                    spec = majority([r["specificity"] for r in refs], "unresolved_vote")
                    disc = (
                        gpt == "clear"
                        and sum(
                            r["specificity"] in ("specific", "broad_but_discriminating")
                            for r in refs
                        )
                        >= 2
                    )
                    add("coherence", gpt, label["coherence"] if label else status)
                    add("specificity", spec, label["specificity"] if label else status)
                    add(
                        "clear_discriminating",
                        "yes" if disc else "no",
                        (
                            "yes"
                            if label["coherence"] == "clear"
                            and label["specificity"]
                            in ("specific", "broad_but_discriminating")
                            else "no"
                        )
                        if label
                        else status,
                    )
        assert len(production["groups"]) == 4725
    index = json.loads((ROOT / "coherence_index.json").read_bytes())
    digest = hashlib.sha256(
        json.dumps(sorted(recounted.items()), separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == index["judgment_multiset_sha256"], (
        "Raw judgments differ from the paper recount export"
    )
    assert sum(recounted.values()) == index["judgment_count"]
    return recounted, statuses


if __name__ == "__main__":
    recounted, statuses = recount()
    print(
        json.dumps(
            dict(
                verified=True,
                groups=4725,
                gpt_contexts=3,
                gemini_passes=1,
                normalized_endpoint_records=sum(recounted.values()),
                response_status={
                    f"{family}/{status}": n
                    for (family, status), n in sorted(statuses.items())
                },
                note="All rubric-invalid/provider-blocked judgments remain in the planned denominators. This checks saved labels and aggregation, not a fresh visual review.",
            ),
            indent=2,
        )
    )
