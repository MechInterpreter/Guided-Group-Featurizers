# Minimum-work pipeline check

Open `notebooks/reproduce_paper.ipynb` in Colab, select a GPU runtime, leave
`RUN_MODE = "quick"`, and run all cells. Upload `ggf_anonymous_repository.zip`
or `GGF_CVPR2027_supplementary.zip` when prompted. Enable the `HF_TOKEN` secret
for the gated DINOv3 weights.

The setup installs matching Torch 2.11.0, torchvision 0.26.0 and torchaudio 2.11.0
CUDA 13.0 wheels and tests the compiled image operators before loading the model.
The evidence check recounts all supplied review and numerical results.

## What the quick run exercises

| Section | Minimum workload |
|---|---|
| Public backbone | Two COCO training images, pinned DINOv3 tensor verification, feature extraction and a 75-group BSF backward pass. |
| VOC inputs | Four real images with class/instance masks and normalization from two FIT images. The official archive must still be downloaded and checksum-verified. |
| BSF training | Two warm and two continuation updates per arm, supervised and adaptive losses, exact optimizer/RNG resume, category-split heads, six baseline objectives, and 19/75/1,024/4,096-group shapes. |
| SLiCS | One atom-update sweep and a strict NNLS certificate at the recorded tolerance. |
| Concept-SAE | One optimizer step in each of its three stages, using the pinned upstream architecture. |
| Geometry and transformations | Small numerical fixtures, four image transformations, PCA and frozen-probe functions. |
| Concentration | A freshly fitted tiny model, separate calibration/audit fixtures, known presence labels, coverage and false positives. |
| Synthetic recovery | Generated tiny datasets, short fits of all four methods and baseline audits. |
| Historical kernels | Short original guided, recovery, bounded-weight, PCGrad and capacity updates; warm-up resume; 5D-to-8D expansion with preserved initial reconstruction and Adam state. |
| Execution profiles | Two disposable timing repeats and one registered-operator FLOP probe. |
| Review cards | Six public images, marked regions and original zoom-strip layout. |
| Figures | A plot from the quick model outputs and the four supplied paper figure families. |

Most optimizer checks use low-dimensional fixtures to keep the work small. Real
public-image extraction runs separately. These tests exercise software paths;
their values are not estimates of the paper's results. Semantic judgments require
an independent review of newly generated response cards.

## Outputs and resuming

Commands stream their output and save full logs under `ggf_logs/` and the quick
run's `logs/` directory. A failure includes the last 60 output lines and log path.
Each successful section writes `checks/<section>.json`. Only success in every
section produces `QUICK_COMPLETE.json`.

Keep `RUN_NAME` to resume completed sections from the same source edition. Use a
new name after changing the code or environment. The driver refuses to mix source
editions or replace an unrelated output directory. Colab runtime resets erase
local outputs unless they are downloaded or copied to persistent storage.

First-run duration depends on model access, network speed and the VOC archive download. Full research budgets are selected separately in full mode.
No paid review requests are submitted by this notebook.

## Validation status

Local CPU runs have exercised the tiny training, resume, strict NNLS,
Concept-SAE, geometry, concentration, synthetic, profiling and plotting sections.
The original public-input GPU smoke passed in Colab with the pinned DINOv3 tensor
hash. The expanded CUDA/VOC sequence passed in the user-supplied Colab log. Full GPU
retraining of every paper experiment has not been validated.

The historical workers and their public-input prerequisite graph are listed in
`REPRODUCING.md` and `PAPER_MAP.md`. Completing a quick check does not certify
full-budget execution or reproduce the paper's numerical outcomes. Blender is
not part of the paper workflow.
