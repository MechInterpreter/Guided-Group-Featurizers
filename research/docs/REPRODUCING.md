# Reproducing the paper

## Verify the supplied results

Run `python scripts/verify_bundle.py` from the extracted bundle root. This checks
file hashes, cited Appendix paths, notebook cells, original reviews, separate GPT
and Gemini recounts, and numerical summaries. The audit reconstructs 42,525
normalized endpoint records for 4,725 groups in 63 banks across seven orders.
It retains unclear judgments, invalid Gemini responses and original seed-21
provenance. It also recounts 32 matched Capture/AP/reconstruction summaries,
360 concentration summaries, six VOC endpoints and 17 execution profiles.

The seven-order records are in `code/evidence/matched75/RESULTS_SEEDS21_27.json`
and `MATCHED75_NUMERIC_SEEDS21_27.json`. `RESULTS_MEAN_SD.json` is the preserved
three-order source record. The recorded `seeds` field identifies each scope.
Saved-evidence verification needs no GPU, model access or API key.

## One notebook, three modes

Open `notebooks/reproduce_paper.ipynb` in Colab and select a GPU runtime.
Upload either delivered ZIP and enable the `HF_TOKEN` secret for the gated
DINOv3 weights. The modes are:

- `quick`: small integration workloads for every software family, plus evidence verification.
- `evidence`: the supplied evidence audit without installing GPU dependencies.
- `full`: plan or execute full-budget studies and their prerequisites.

Use `docs/CLEAN_ROOM_VALIDATION.md` for the quick-check workloads and receipts.
Choose a separate `RUN_NAME` for full studies. Retain that directory to resume;
Colab resets erase local results unless copied to persistent storage.

## Environment and public inputs

Use Python 3.11 or newer with a compatible CUDA driver. The notebook installs
matching Torch-family wheels and checks compiled image operators before model
loading. For an isolated local CUDA environment:

```sh
python -m pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt numpy==2.1.3 scipy==1.16.3 scikit-learn==1.6.1 einops==0.8.2
```

The workers download official COCO/VOC inputs and the pinned DINOv3 model,
verify identifiers and hashes, and generate normalization, features and model
checkpoints inside the selected run directory. Original Drive caches and trained
checkpoints are not required by these workers. Public dataset downloads, gated
model access, disk space and GPU compute remain required. The Manifold Zoo
baselines use their separately recorded CUDA 12.8 environment, which the full
planner installs in an isolated environment without changing the main one.

## Full experiment planner

In the notebook, set `RUN_MODE = "full"`. Leave `EXECUTE_FULL = False` to inspect
the plan; set it to `True` to execute. `FULL_STAGES = ()` selects all numerical
studies and prepares review cards. A subset automatically includes prerequisites.
The equivalent command line starts by printing the plan:

```sh
python code/bootstrap/full_pipeline.py --root ../ggf_full_reproduction
python code/bootstrap/full_pipeline.py --root ../ggf_full_reproduction --stages controls.coco.evaluate
```

Add `--execute` to run the selected plan. Each phase uses a separate process,
streams a log and writes a progress receipt only after the worker succeeds.
A rerun verifies the source edition and saved output receipts before skipping
completed phases. Modified sources or unrelated output directories are rejected.

Feature assembly and normalization use disk-backed arrays with bounded temporary
allocations. The recorded mean, scale and normalized values retain the original
arithmetic. To reuse extraction shards after an interrupted compatible edition,
choose a new run directory and set `FEATURE_CACHE` in the notebook, or pass
`--feature-cache /path/to/previous/matched75` to the planner. The importer checks
the original specification, scientific source, journal checksums, image order and
feature dimensions. It copies only development feature shards and records their
source lineage. It does not import fitted models or review labels.


| Stage family | Rebuilt experiments |
|---|---|
| `matched.*` | Seven orders, 28 full-category and 35 category-split 75-group banks; gated audit extraction, native response collection and blinded cards. |
| `components.*` | Three-order 4,096-group pilot and initialization/loss controls. |
| `controls.coco.*`, `controls.voc.*` | Original 1,024-group comparisons, numerical scoring and supporting analyses. |
| `supervised5.*` | Five-order GGF/SLiCS comparison and its separately reserved holdout. |
| `voc19.*` | Three-order 19-group GGF/SLiCS fits and evaluation. |
| `concept.coco.*`, `concept.voc.*` | Development-only tokenizer selection, COCO/VOC Concept-SAE v2 stages and evaluation. |
| `legacy.*` | Native and mask-initialized ancestors, matched baselines, bounded-weight/PCGrad continuations, 4D/5D/6D/8D/12D and group-count/direction-budget controls. |
| `analysis.geometry`, `analysis.controlled` | Frozen-model geometry, fingerprints, subset/removal analyses and controlled 2D transformations. |
| `compute` | Original 17 disposable-model VOC execution profiles and registered-operator coverage. |
| `zoo.*` | Public synthetic generation, baseline/GGF fitting and known-factor audits. |

The historical adapters call the retained original numerical functions. New
journals record their own source, input, normalization and checkpoint identities.
They validate newly fitted ancestors instead of substituting a historical run ID.
The allocation in `LEGACY_ALLOCATIONS.json` preserves the separate 400-image
mapping cohort. The five-order comparison keeps its distinct 1,500-image holdout.
Shared Vanilla banks and original fixed denominators are preserved.

## New blinded reviews and concentration labels

After `matched.cards`, supply fresh judgments for the newly generated cards.
Only public packets and referenced images go to reviewers; keep `PRIVATE.json`
and bank metadata separate. Review every full PNG and its marked zoom strips.
Use three distinct fresh contexts for the fixed pair and whole-group rubrics.

- Pair templates and packets: `matched75/evaluation/reviews/coherence/`.
  Completed context files go in its `labels/` directory using the provided schema.
- Whole-group templates and packets: `matched75/evaluation/reviews/whole/`.
  Completed files go in `labels/context1.json`, `context2.json`, and `context3.json`.
- Crop-presence templates: `matched75/evaluation/reviews/presence/`.
  Complete its templates in `labels/*.json` for all fixed category definitions.
- Independent presence QA: `matched75/evaluation/reviews/presence_qa/TEMPLATE.json`.
  Save the completed response as `labels.json` in that directory.

Run `--stages matched.coherence --execute` after completing both coherence
rubrics. Run `--stages matched.concentration --execute` after completing presence
labels and QA. The planner runs the presence locks and calibration-only selection
before numerical concentration scoring. Coherence judgments do not determine
coverage or false-positive rates. Unclear judgments remain in planned totals;
missing reviews stop scoring. Raw judgments and their hashes are retained.

The notebook makes no API requests. New visual reviews require reviewer/model
access. A separate model check must retain its model identity and be reported
separately from a three-context majority. Saved GPT and Gemini reviews audit the
published models; they are not labels for newly fitted response cards.

## Sources, figures and validation

`docs/PAPER_MAP.md` connects paper claims, evidence paths and fresh workers.
`code/sources/` exposes original source editions directly. The historical
workspace entrypoints remain reference material; use the maintained full planner
for public-input reproduction. Hash-protected numerical kernels, original
reviews, evidence and upstream implementations retain their recorded bytes.

`code/figure_sources/` supplies the renderers and saved plotting inputs. Figure 1
is an attributed vector illustration; Figure 2 includes measured PCA/crop inputs;
Figure 3 is the architecture schematic; Figure 4 uses the seven-order records.
The rank/headroom audit is documented in `code/supporting_metrics/README.md`.
Procedural 3D lighting and Blender are outside the paper workflow and excluded
from the delivered study modules.

The prior quick notebook passed all its small Colab sections in the supplied
execution log. This release additionally tests historical fit kernels on CPU,
plus dependency ordering, locks, failure/resume behavior and fresh-review scoring.
Full-budget GPU execution and numerical equivalence of every fresh study have
not been established by these checks. Completion receipts describe executed
stages; they do not certify reproduction of the paper's numerical outcomes.
