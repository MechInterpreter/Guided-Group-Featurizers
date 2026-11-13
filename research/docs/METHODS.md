# Method implementation guide

Paths are relative to the repository root. All listed modules are ordinary files
in both release ZIPs. The notebooks are launchers; they do not contain hidden
model implementations.

## Matched 75-group comparison

| Method or component | Implementation | What to inspect |
|---|---|---|
| GGF | `code/training/cd_train.py` | `fit_bsf(..., kind="ggf")`: object-guided initialization, warm-up and guided continuation. |
| GGF initialization | `code/training/core/scale_data.py` | `mask_dictionary`: annotation-derived dictionary construction. |
| GGF objectives | `code/training/core/coco_adaptation/core.py` | `supervised_loss`: object-level guidance terms. |
| GGF adaptive weighting | `code/training/core/coco_ggf/capacity_fit.py` | `Balancer`: relative weighting of the reconstruction and guidance terms. |
| Shared training loop | `code/training/core/scale_fit.py` | `fit`: optimizer/RNG checkpointing, objective construction and updates. |
| Vanilla BSF | `code/vendor/goodfire_bsf/bsf/vanilla.py` | `VanillaBSF`: the upstream architecture. `code/training/core/scale_models.py` selects the model. |
| Supervised Vanilla BSF | `code/training/cd_train.py` | `fit_bsf(..., kind="supervised_vanilla")`: shared label-free warm-up and development-selected fixed guidance weights. |
| SLiCS | `code/training/cd_train.py` | `fit_slics`: constrained dictionary fitting. `core/coco_ggf/slics_repair.py` implements the NNLS solver; `training_precision.py` implements precision retry. |
| Concept-SAE | `code/training/concept_sae_native.py`, `concept_sae.py`, `concept_sae_v2.py` | Released architecture, adapter and development-selected staged training. Concept-SAE uses its own native interface; it is not an extra matched 75-group bank. |

GGF changes initialization and training while retaining the BSF architecture.
The full-category and category-split schedules, groups and seed assignments are
in `code/bootstrap/SPECIFICATION.json` and the frozen training protocols. Formatting
has not changed these scientific settings.

## Other BSF variants and supporting studies

`code/vendor/goodfire_bsf/bsf/grassmannian.py` and `group_lasso.py` contain the
upstream variants. `code/training/core/scale_models.py` also contains explicitly
labelled paper-equation adapters. An adapter must not be mistaken for an
unpublished upstream checkpoint or an upstream training schedule.

The six historical profiles are immediately browsable in `code/sources/`:
`components`, `controlled_variation`, `geometry_consistency`,
`matched75_original`, `supervised_five_seed` and `voc_slics_compute`.
`code/frozen_sources/profiles.json` records each member's original/exported hash.
These editions retain their recorded code and historical storage identifiers.
They are not duplicate independent experiments. Workflows check the exact source
before execution and require the inputs described in `REPRODUCING.md`.

## Evaluation

- `code/training/collect_evaluation.py`: native response collection.
- `code/training/score_evaluation.py`: numeric and presence-based endpoints.
- `code/training/original_concentration.py`: original concentration definitions.
- `code/training/prepare_evaluation_reviews.py`: blinded card/presence packaging.
- `code/review_records/`: run-used review construction, validators and saved notes.
- `code/bootstrap/fresh_evaluation.py`: fresh-run collection and presence-scoring stages.

Concentration, coverage and false positives are numerical calculations on response
arrays and crop-presence labels. They do not require semantic-coherence judgments.
Newly collected cards still require the stated presence QA and visual review;
saved review scores are not automatically assigned to new model outputs.

## Historical terminology

Some inherited modules discuss AGF structural-adaptation variants or use
`AGF_COCO_*` storage keys. Those labels are retained intentionally: replacing
them with GGF would misdescribe those studies or break recorded input paths.
The paper method in this repository is Guided Group Featurizers (GGF).
