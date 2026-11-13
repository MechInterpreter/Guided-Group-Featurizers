# Maintenance and verification

## Scope of this cleanup

Maintained Python sources use consistent four-space indentation and standard
formatting. Statements previously compressed onto one line are expanded. Documentation uses GGF for the paper method. Historical AGF labels in inherited
structural-adaptation studies are retained because they describe different
variants rather than a misspelling of GGF.
Executable identifiers, historical directory names, run IDs, model settings,
scientific definitions and result records are preserved.

Retained historical numerical modules, upstream vendor code, raw review scripts
and measured evidence remain unchanged. The unrelated procedural-lighting
modules are excluded, as recorded in `code/PACKAGE_SCOPE.json`. Their older style and storage identifiers
are retained for provenance. They are separated from maintained entry points
and are browsable without unpacking an internal source archive.

`code/source_provenance/PRESENTATION.json` and its original-source archive record
the cleanup. `scripts/verify_source_presentation.py` checks every changed source
against its prior version: Python ASTs must match after normalizing only docstring indentation, except for explicitly recorded functional revisions with separate hashes and regression tests. Comments and whitespace do not affect
this comparison. Changes to fresh-evaluation stage wiring are separate, tested
functional additions, not presented as formatting-only changes. Disk-backed feature
assembly and normalization are also a declared functional revision. Tests compare
the mean, scale, normalized file bytes and hashes against the frozen reference.
The original normalizer and numerical research source remain unchanged.

Two text-sensitive modules, `code/training/core/supervised_study.py` and
`object_schedule.py`, retain their exact source text because the original
precision-recovery code patches function source strings. Formatting them would
change that behavior even with an equal Python AST. They are explicitly excluded
from formatting; the actual precision-overlay CPU contract covers their use.

The headroom source `code/training/core/coco_ggf/wider_groups.py` also retains
its exact bytes: the retained supporting-metric audit checks its original hash
as part of the published formula provenance.

The upstream `code/training/concept_sae_native.py` also retains its exact bytes.
The Concept-SAE loader checks the pinned upstream SHA before importing it, so
formatting is excluded for this file.

## Local checks

```sh
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
python -m ruff format --check .
```

Formatting excludes immutable historical/evidence/vendor trees. The manifest and
evidence audit check their original bytes instead. The unit tests cover package
membership, integrity and the fresh presence-scoring dispatch without training
or external requests. Recorded validation results are in `docs/VERIFICATION.json`.

Formatting changes create a new fresh-run source identity. Start a new fresh-run
directory for this edition. Do not edit a saved run identity to resume with
changed source. Historical identities and source containers are unchanged.

## Reproduction limits

The offline checks verify the delivered evidence and arithmetic, source integrity,
notebook syntax and CPU contracts. They do not establish numerical equivalence
of a complete retraining run on a GPU. Public data, licensed model access,
recorded numerical versions, substantial GPU time and new visual judgments are
separate requirements. The fresh historical workers rebuild their prerequisite stores locally. See
`REPRODUCING.md` for the public inputs, stage order and validation status.
