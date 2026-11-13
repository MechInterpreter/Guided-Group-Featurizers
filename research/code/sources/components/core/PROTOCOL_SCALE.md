# AGF group-count, Top-k, adaptive-cap and seven-seed comparison

This is a new, paired, from-scratch COCO study. It reuses the verified cached
2,000 training, 400 selection and 400 confirmation images, masks, normalized
DINOv3 activations and numerical package versions. It does not retrain DINOv3.
Prior 12D results are retained as historical references, not substituted for
newly initialized 4,096-group controls. Inputs remain read-only.

## Fixed architecture and exposure

Every main configuration has 12 signed directions per block. The factorial is
G in {1,024, 4,096} by Top-k in {16, 32}. AGF initializes its first 75 blocks
with the leading 12 eigenvectors of the uncentered second moment of normalized
TRAIN patches with at least 50% category coverage and at most 10% crowd coverage.
Other blocks retain same-seed random released Vanilla initialization. This is
different from the earlier 8D-to-12D checkpoint expansion.

Every model receives 7,640 reconstruction warm-up updates at learning rate
0.0003, followed by 12,000 additional updates. All updates sample eight TRAIN
images with replacement; every image supplies all 196 patches. Paired methods
use identical saved image and noise generators, independent of model construction.
Noise scale is 0.1, gradient clipping 5, Adam, and fixed final checkpoints.
The additional phase starts a fresh model Adam in every arm. AGF starts fresh
logistic heads and the existing bounded shared-encoder controller. Reconstruction
weight is one; object-weight floors are 0.0125 and 0.0025, reference batches 20,
controller Adam rate 0.01, exponent 1.5 and adjustment cadence every ten updates.

The 3,000-update development screen is an exact prefix of the 12,000-update phase.
Its compatible models, Adam/controller state and mappings resume in the final
study. Completed fits are not rewritten or retrained. Warm-ups are shared by all
caps and additional-phase learning-rate trials for the same method/configuration/
seed; mask-only and fixed-weight arms share AGF's reconstruction warm-up.

## Automatic stage order

1. Validate prior catalogs, cached features, source hashes and numerical versions.
   Reserve a new annotation-only holdout and check DINO access and cached TRAIN
   image replay before expensive training. Run disposable full-batch T4 checks.
2. Three seeds (21, 22, 23), all four G/Top-k combinations, AGF-G cap 0.40 and
   matched released Vanilla: 24 development endpoints.
3. At the development-selected configuration, compare caps 0.40, 0.60 and 1.0
   from identical mask reconstruction warm-ups. Reuse the completed 0.40 prefixes.
4. Tune each of six baseline implementations using three additional-phase
   learning rates: 0.0001, 0.0003 and 0.001. Warm-up is held fixed. Paper Group
   Lasso trials pair these rates with L2,1 coefficients 0.0001, 0.001 and 0.01;
   this is three explicit joint candidates, not a full 3-by-3 sweep.
5. Freeze selected configuration, cap and baseline settings, then finish seven
   seeds (21 through 27). The first three are reused development starts; seeds
   24 through 27 are additional starts, not previously screened.
6. Compare AGF-G, all six baselines, mask-initialization-only reconstruction,
   fixed-weight guided learning (0.05 contrast, 0.01 detection), and AGF-X.
   There are 70 final endpoints: 63 trained paths and seven exchange descendants.
7. Freeze every endpoint/readout before extracting or scoring the new holdout.
8. Analyze effective rank, coding-cost proxy and deployment costs; export evidence.

## Predeclared development selection

Choose configurations using only three-seed selection-set means, with AGF at
1,024 groups / Top-16 as the reference. Feasible candidates preserve patch AP
within 0.5 percentage points and VE within 0.1 points of the reference. Maximize
capture + 0.2 AP, preferring lower G times width times Top-k among scores within
0.1 points of the maximum. This is a declared cost heuristic, not measured latency.
Cap selection uses the same rule against cap 0.40 and favors the lower cap within
the score margin. These margins are experimental choices, not literature-derived
universal thresholds. If no larger configuration earns a benefit, the reference
can be selected; the notebook does not force 4,096 groups or cap 1.0.

Baseline additional-phase settings minimize three-seed development MSE. Lasso
requires mean activity within 20% of the target in each seed. If no candidate
meets that tolerance, retain and explicitly flag the best-MSE candidate; never
call it exact Top-k or drop that method from the reported cohort.

## Baseline audit

Pinned released source: goodfire-ai/block-sparse-featurizer,
commit 219f121ea82d2b19200d1dac918396e6058d7eb9. All files in `bsf/` are unchanged.

* Released Vanilla: free maps, unit decoder atoms, hard Top-k, reconstruction loss.
* Released Grassmannian: QR each forward, one unconstrained gain per group.
* Released Group Lasso: hard signed BlockJumpReLU, kernel backward, learned
  thresholds, dual mean-L0 penalty. It is not the paper's soft shrinkage.
* Paper-spec Vanilla: free maps, Frobenius unit-ball blocks, hard Top-k, AuxK.
* Paper-spec Grassmannian: tied maps, one shared positive gain, QR retraction
  every 20 updates, AuxK. Initial gain is a documented TRAIN-batch least-squares
  calibration because exact initialization details are not fully specified.
* Paper-spec Group Lasso: learned positive thresholds, soft block shrinkage,
  excess-activity L2,1 penalty, AuxK. Initial threshold uses the first TRAIN-batch
  norm quantile; penalty settings are documented development choices.

AuxK uses runner-up nonselected blocks, a detached reconstruction residual and
coefficient 1/Top-k. Objectives use mean squared coordinate error consistently.
The paper adapters implement its architecture/loss descriptions, not unpublished
author checkpoints. This study uses our shared COCO normalization, denoising,
data exposure and optimizer protocol. It does not reproduce the paper's original
ImageNet activation population, three-epoch cosine schedule or reported numbers.

## Mappings and Exchanges

The existing TRAIN-only mapping policy searches every group after reconstruction
warm-up. Detection and capture mappings can differ. Disk-backed arrays prevent
large 4,096-group mapping concatenations. Fixed readouts and heat statistics are
retained during additional fitting. Evaluation streams only selected readout
groups while preserving the complete model's original gating, reducing RAM.

AGF-X uses the previously validated width-aware dependency and two-direction
exchange implementation. TRAIN screening uses 512 coverage-balanced images,
candidate pool 64, dependency shortlist eight, maximum eight attempts and five
accepted exchanges. No gradient updates or readout refitting occur. Exact edited
models must gain capture and preserve AP, VE, MSE and per-category outcomes on
TRAIN screening and both historical development splits. Selection also checks
the common selected released Vanilla warm-up MSE allowance of 2%. A parent that
fails that common allowance remains unchanged and its reason is reported.
Rejected/empty exchanges are retained. Heads and direction inventory are preserved.

## Independent final evaluation

Default holdout: 1,500 unused COCO train2017 images, selected by cropped annotation
coverage only, disjoint from every original provenance split and saved strong-run
split. All 75 predeclared categories require at least ten jointly usable images.
This is a coverage-enriched benchmark, not natural COCO prevalence. No category
is removed based on model performance. Holdout features remain unopened until
all final checkpoints freeze.

Pinned DINO weights are verified by tensor digest. Before training, a cached
TRAIN image is re-extracted to check the pipeline within explicitly recorded
float16-cache tolerances (maximum 0.005 and RMS 0.0005 raw activation error).
The backbone is cached to Drive and feature shards are resumable. HF_TOKEN is
read from Colab Secrets once at setup, never printed or saved in source/config.

Report all endpoints and category scores. Per-seed paired image intervals use
1,000 whole-image bootstrap replicates. Separate Student-t intervals summarize
paired differences over seven training starts, conditional on this fixed cohort.
These are different uncertainty sources, not a combined hierarchical interval.
Primary contrasts are AGF-G versus released Vanilla and AGF-X versus AGF-G;
other contrasts are secondary, without multiplicity adjustment.

## Rank, coding cost and deployment

Compute centered coefficient covariance conditional on nonzero gated activity on
the same 392,000 TRAIN patches. Report spectral-entropy effective rank,
participation ratio, both covariance and coefficient-matrix stable ranks,
contribution spectra, utilization, activity counts and zero/insufficient cases.
The first 75 groups are semantic initial slots only for masked arms. Reserve
counts are 949 or 4,021 according to selected G. Ranks are not ground-truth
concept dimensions. Variable-activity Lasso is never checked against hard Top-k.

The MDL-related output is explicitly a **Gaussian coding-cost proxy**, not a
realized compressed bitstream or unmodified Goodfire MDL estimator. On the common
historical confirmation cohort, it charges enumerative supports, per-patch K,
conditional decoded-contribution Gaussian covariance rates, residual Gaussian
rates, full stored encoder/decoder/buffer/head precision, and mean/basis side
information. Canonical contribution spectra avoid arbitrary block scaling in the
code term. Distortions are fixed at 1%, 5%, 10% and 20% of TRAIN mean squared
coordinate magnitude. Model and side-information overhead is amortized over
392,000 unique TRAIN patches, never duplicated training updates. See each report
for exact terms and interpretation. This estimate alone cannot establish actual
compression efficiency or useful object semantics.

Measure dictionary forward latency on one 196-patch image with five warm-up and
30 synchronized timed passes; report median, p95 and peak GPU memory. Backbone,
data loading and masks are excluded. Report training parameter counts separately
from inference cost. No model/readout is fitted during analysis.

## Recovery and storage

Select T4, grant Drive and HF_TOKEN secret access up front, and Run All. Numerical
and extraction dependency versions are persisted and restored on later runtimes.
Training saves valid local generations every 50 updates and verified Drive
generations every 250 updates, phase limits and endpoints. Graceful Stop flushes
the latest complete local state. Partial optimizer/controller updates are never
committed; abrupt runtime loss can replay up to 249 updates. Completed phases
skip. Rank/code accumulation is separately resumable. Use one writer per output.

This is substantially larger than the earlier continuation notebooks and will
likely require multiple T4 sessions. All model/optimizer checkpoints remain
separate in Drive for future continuation. Ensure substantial Drive storage is
available, particularly if G=4,096 wins. The final ZIP contains source, hashes,
metrics, decisions, spectra and figures, not model/optimizer checkpoints, DINO
weights, images, cached features or analysis accumulators.
