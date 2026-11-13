# Guided Group Featurizers

Anonymous research code and evidence for the CVPR submission. The main study
compares 75-group methods across seven training orders (seeds 21–27). The bundle
includes the saved GPT-majority reviews, a separate Gemini check, numeric results,
and implementations of GGF and its baselines.

## Browse the methods

All runtime Python files are included as ordinary files. Extraction is enough;
no restore command is needed. Start with [the method guide](docs/METHODS.md).

- `code/training/cd_train.py`: GGF, supervised Vanilla and SLiCS training.
- `code/training/core/`: shared BSF models, GGF objectives and adaptive weighting.
- `code/vendor/goodfire_bsf/`: the pinned, unmodified upstream BSF implementation.
- `code/training/concept_sae*.py`: the Concept-SAE architecture and training adapter.
- `code/sources/`: exact historical source profiles, directly browsable.

GGF retains the Vanilla BSF architecture. Its object-guided initialization and
training objectives are implemented in the shared training modules.

## Verify the saved results

From this directory, with Python 3.11 or newer:

```sh
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
python code/workflows.py list
```

The evidence audit uses the standard library, requires no model/API access, and
checks hashes, original review records, denominators, per-order aggregates,
Appendix paths and historical source profiles. Fresh training and visual review
are described in the experiment instructions.

## Run experiments

Open `notebooks/reproduce_paper.ipynb` in Colab, select a GPU runtime and run all
cells. The default `quick` mode performs a small integration check of each pipeline
section and recounts the supplied evidence. Set `RUN_MODE="evidence"` for an offline
recount, or `RUN_MODE="full"` to select full-budget experiment stages. See
[the quick-check guide](docs/CLEAN_ROOM_VALIDATION.md).

Read [REPRODUCING.md](docs/REPRODUCING.md) before installing dependencies or
running a GPU stage. It specifies the recorded numerical versions, public-data
and DINOv3 access requirements, stage order, required labels and remaining
validation limits. The fresh planner rebuilds the matched study and historical prerequisites from
public inputs without the original Drive caches or checkpoints. A complete
fresh CUDA run of every experiment has not been validated. New visual reviews
require external reviewer access.

The default workflow command prints a plan. Add `--execute` to launch a selected
stage after checking its inputs. Saved-evidence verification uses the included records.

## Code organization and provenance

- `code/training/`, `bootstrap/`, `figure_sources/`: maintained implementations.
- `code/sources/`: exact historical editions used by the retained study profiles.
- `code/evidence/`, `review_records/`: fixed measurements and review provenance.
- `code/frozen_sources/`, `frozen_legacy/`: verified historical source containers.
- `docs/PAPER_MAP.md`: links paper claims to evidence and source files.
- `docs/CODE_QUALITY.md`: cleanup guarantees, checks and validation limits.
- `scripts/`: package construction and offline verification.
- `tests/`: release and fresh-evaluation contracts without GPU or API calls.

The supplementary ZIP includes these reproduction files and the anonymous
Appendix PDF. The standalone code ZIP includes the same reproduction files.
All paper evidence paths are relative to `code/`. Local authoring history and
publication build outputs are excluded from the anonymous code payload.

Original and third-party licenses are retained in `LICENSE` and `LICENSES.md`.
Do not remove required attribution from upstream sources.
