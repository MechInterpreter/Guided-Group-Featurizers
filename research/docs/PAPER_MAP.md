# Paper evidence and reproduction map

Paths below are relative to the `code/` directory. The same experimental evidence supports the anonymous CVPR paper and the corresponding preprint. Study labels are used instead of page numbers, which can change with typesetting.

## What can be verified offline

Run `python verify_review_records.py`, `python verify_coherence.py`, and `python verify_numeric_evidence.py`. The first script revalidates the saved Gemini response bodies and reconstructs all three-context GPT majorities from the original per-reviewer notes. It regenerates 42,525 endpoint records for all 4,725 matched groups and checks them against the checksum of the paper's normalized export. The other scripts check the reported coherence aggregates and the numerical means/sample standard deviations from the supplied per-order records.

The seven-order comparison uses training orders 21-27, 63 matched banks, and 75 groups in each bank. The same Vanilla bank is shared across the two guidance folds and is counted once. The GPT and Gemini results remain separate. Seed-21 reviews are reused with their original identities and hashes, not treated as new independent reviews. The raw archive additionally retains the 75 unmatched native SLiCS seed-21 pair groups as reference material; they are excluded from the matched-75 claims.

These checks verify saved evidence, denominators, aggregation, and provenance. They do not retrain a backbone or featurizer, regenerate all response tensors, run new API calls, or repeat the visual review.

## Main-paper claims and Appendix evidence

| Paper material / source label | Included evidence and plots | Experimental source | Additional inputs for a fresh run |
|---|---|---|---|
| Figure 1: shattering, dilution and compact capture; `fig:tiling-schematic` | `figure_sources/inputs/tiling/ggf_tiling_inline.pdf` | Conceptual vector illustration adapted from Bhalla et al. (2026), *Do Sparse Autoencoders Capture Concept Manifolds?*, Figure 3; attribution is in the paper caption | No empirical input or model inference; the publication asset is supplied directly. |
| Semantic/visual pair coherence, whole-group themes, specificity; `sec:matched75-coherence`, `app:matched75`, `app:cross-model` | `coherence_index.json`; `review_records/judgments.zip`; `evidence/matched75/RESULTS_SEEDS21_27.json`, `GEMINI_CROSS_MODEL_RESULTS.json`; `figure_sources/make_matched75_compact.py` | Frozen pair/whole protocols in `training/`; original Gemini runner/scorer and request construction in `review_records/`; public-card generation in `training/prepare_evaluation_reviews.py` | `bootstrap/full_pipeline.py` stages `matched.*` rebuild features, fits and cards from public inputs. Fresh visual labels require external reviewer access; existing notes need none. |
| Concentration, coverage, false positives, overlap; `sec:matched75-concentration`, `app:presence-formulas` | Per-order cells and thresholds in `evidence/matched75/RESULTS_SEEDS21_27.json`; 80 concentration rows retained; Figure 4 renderer above | `training/original_concentration.py`, `score_evaluation.py`, `verify_additional_concentration.py`; exact original 21-23 source in `sources/matched75_original` | `matched.concentration` uses freshly collected arrays, completed crop-presence labels and independent QA. These numerical metrics do not use coherence judgments. |
| Seven-order 75-group Capture, Patch AP and reconstruction; `tab:matched75-AP`, `tab:matched75-reconstruction` | `evidence/matched75/MATCHED75_NUMERIC_SEEDS21_27.json` (32 rows, seven values per row) | Original 21-23 collection source and separate 24-27 collection source in `training/`; use the corresponding workflow profile | `matched.*` downloads the declared public allocations and creates normalization, activations, masks and checkpoints locally. |
| Figure 2 and `app:coherence-pca` | `figure_sources/inputs/coherence_geometry/`: measured coefficients/PCA, patch IDs, original image IDs, 24 displayed crops, selection and review metadata; `make_coherence_group_geometry.py` | Original frozen model responses and saved PCA input, retained in the input provenance | No additional data for rendering the figure; original model/checkpoint inputs are needed to repeat coefficient extraction. |
| Figure 3 architecture | `make_architecture.py`, `architecture_glyphs.py`, exact toy initialization array/provenance | Diagram and illustration source | No external input for rendering. The toy glyphs are explanatory, not an empirical result. |
| COCO/VOC comparisons and SLiCS/Concept-SAE details; `app:supervised-five`, `app:voc-slics`, `app:concept-comparison`, `app:voc-results` | `evidence/supervised_five_seed`, `evidence/voc_slics_final`, `evidence/voc_heldout`, `evidence/voc_final_compute`; exact comparison CSVs under `figure_sources/inputs/final_comparisons`; `make_voc_slics_final.py` | Frozen profiles `supervised_five_seed` and `voc_slics_compute`; retained source layers in `frozen_sources` | `bootstrap/fresh_supervised_five.py`, `fresh_voc.py`, `fresh_controls.py` and `fresh_concept.py` build the required stores from public inputs; the original precision-resume entrypoints remain source references. |
| Initialization, losses and native-baseline controls; `app:components-current`, `app:matched-baselines`, `app:widthbalance`, `app:cap040test` | `evidence/components_current`, `matched_baselines`, `width_balance`, `cap040_test`, `current_control`; `make_components.py`, `make_compact_figures.py`, `make_grouped_baselines.py` | Frozen `components` profile and its original source layers | `bootstrap/fresh_components.py` and `fresh_legacy.py` build their original ancestor graph. Group counts and cohorts remain separate from matched-75 results. |
| Capacity, continuation and group-count controls; `app:capacitylatest`, `app:capacity`, `app:groupcount` | `evidence/dictionary_size`, `parameter_efficiency`, `structural12`, `width5`, `continuation`, `continuation5`, `pilot4096_test`; `make_capacity_overview_named.py`, `make_latest.py`, `make_compact_figures.py` | Saved intermediate-study records and the component profile where applicable | `bootstrap/fresh_legacy.py` rebuilds native, width, continuation and capacity ancestors with the original study-specific budgets and readout rules. |
| Known-factor synthetic recovery; `app:manifoldzoo` | `evidence/zoo_guided`, `evidence/zoo_baselines`; exact projected reconstructions and generator under `figure_sources/inputs/zoo_*`; `make_zoo_guided.py`, `make_zoo_reconstructions.py` | Frozen core benchmark modules and synthetic-study source in `evidence/zoo_guided/` | No original images. Fresh training requires the pinned synthetic generator/settings and model seeds; saved plot inputs render offline. |
| Readouts, subset/removal controls, VOC organization; `app:planned-consolidation`, `app:voc-group-organization` | `evidence/voc_heldout`, `evidence/voc_final_compute`, archived analyses and protocol records | `geometry_consistency`/`controlled_variation` frozen profiles and their core source versions | `bootstrap/fresh_analysis.py` consumes newly fitted controls, SLiCS and Concept-SAE models. Each analysis keeps its original allocation and fixed readouts. |
| Fingerprints, stable category preferences, direction agreement; `app:category-fingerprints`, `fig:direction-fingerprints` | `evidence/fingerprints` contains per-seed/per-category results and summaries; `figure_sources/inputs/fingerprint_plot_arrays` contains exact AP/prevalence/readout arrays for all 15 method-order combinations | Frozen `geometry_consistency` profile; `make_fingerprints.py`, `make_fingerprints_robustness.py` | No external input to redraw the supplied plots. `analysis.geometry` regenerates response/direction tensors from newly fitted checkpoints. |
| Controlled 2D transformation consistency; `app:controlled-robustness` | `evidence/controlled_robustness`, `figure_sources/inputs/controlled_robustness`; `make_controlled_robustness.py`, `make_fingerprints_robustness.py` | Original `controlled_variation` numerical functions called by `bootstrap/fresh_analysis.py --phase controlled` | `analysis.controlled` uses fresh frozen models and public COCO inputs for 2D rotation, scale and brightness changes. It does not render 3D scenes. |
| Bird/doughnut and other qualitative examples; `app:matched-blinded-examples`, `app:heatmaps` | Paper figure files; `evidence/matched75/BLINDED_GALLERY_PROVENANCE.json`; all matched GPT/Gemini notes in `review_records`; native and natural-panel inputs in `figure_sources/inputs` | Public-card generator and exact group/patch provenance; qualitative renderers | Source images and original response maps for re-extracting a new gallery. Selected examples are illustrative; aggregate results use all planned groups. |
| Compute and hardware claims; `app:compute-reporting`, `app:flop-coverage` | `evidence/voc_slics_final/compute_profiles.json`, per-call times and operator counts, `FLOP_COVERAGE.json`; `make_voc_slics_final.py` | `voc_slics_compute` profile and included numerical-precision modules | `bootstrap/fresh_compute.py` profiles newly fitted VOC checkpoints. Timings remain hardware-specific; registered operator FLOPs are not full FLOPs. |

`APPENDIX_PATHS.json` records literal supplementary-code paths cited by the paper. `frozen_sources/profiles.json` identifies the six retained study profiles and their source member hashes. The exact container store retains only sources referenced by those profiles or their parameter references.

### Retained rank and headroom support

`supporting_metrics/README.md` supplies the definitions, aggregation rules,
source identities and limits for Figure S6(b) (`tab:headroom`) and Table S27's
entropy ranks (`tab:voc-geometry`). The original 75-category headroom export is
`supporting_metrics/wider_geometric_headroom.json`; the VOC covariance spectra
are already retained in `evidence/voc_heldout/selected_ranks.json`.
`supporting_metrics/verify_support.py` recounts the published summaries and is
included in the normal bundle verification. No superseded gallery is required.

## Historical sources and maintained code

`sources/<profile>/` contains every retained historical profile as ordinary files. No restoration is needed to inspect them. `python tools/materialize_frozen.py --verify-all` additionally checks their source vaults. Historical workflows verify the included source before execution; supply the recorded configuration with local input paths. `frozen_sources/profile_parameters/` preserves parameter/execution cells without embedded payloads or outputs. These text files are references, not additional notebooks or a promise of a ready-to-run dataset download.

The source vault shares identical files across versions. Frozen historical editions retain the recorded anonymity/path changes and their original/exported hashes. Maintained sources also receive behavior-preserving formatting, documented in `../docs/CODE_QUALITY.md`; the historical editions and measured evidence remain unchanged. The original numerical archive required by the historical precision-resume check is retained byte-for-byte in `frozen_sources/exact_required_archives/`. Historical manifests under `training/` describe their original larger source exports; the deliverable's `SHA256SUMS` verifies the packaged files.

## Rendering checks and remaining dependencies

Install `requirements-figures.txt` for offline plots. `figure_sources/RENDERER_VERIFICATION.json` records the renderer checks. Renderer source files and inputs are directly present after ZIP extraction.

The renderer verification record identifies the scripts exercised in the verification workspace. Use the workflow registry for supported main entry points. Public datasets and access to the pinned DINOv3 backbone are required. The maintained workers generate checkpoints, feature/response arrays and review cards locally. The bundle retains source hashes, image identities, protocols and saved numeric evidence.

## Raw review audit

`review_records/judgments.zip` contains 2,388 GPT packets, including the original seed-21 notes, and all 9,450 Gemini API response bodies. It preserves unclear judgments and rubric-invalid responses. API generation identifiers are one-way hashed, local author paths are made relative, timestamps are removed from Gemini transport metadata, and opaque encrypted provider reasoning tokens are omitted; response message contents and judgment/evidence fields are not rewritten. `review_records/MANIFEST.json` records original and exported hashes. The exact run-used Gemini request builder, validators, runner, scorer, and dependency modules are retained in `review_records/` and `review_records/original_scripts/` without keys.

The original runner expects its original workspace-shaped frozen input tree, which is not entirely present in the upload; it is supplied for protocol inspection and reconstruction, not advertised as an immediate paid rerun. Fresh API review needs the recorded card images, model/provider availability, an API key supplied outside the repository, and a separate spending decision. None of the verification commands sends a request or spends credits.

## Fresh public-input stages

`../docs/FRESH_SYNTHETIC.md` documents synthetic generation and training.
`../docs/FRESH_VOC.md` documents VOC2012 extraction and the recorded 19-group
GGF/SLiCS fits. `../docs/REPRODUCING.md` lists the full public-input dependency planner for the
1,024/4,096-group, Concept-SAE, component, geometry and controlled-view studies.
It rebuilds prerequisite stores; complete full-budget GPU validation is pending.
