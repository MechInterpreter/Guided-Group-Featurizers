# AGF: 5D continuation, three arms and three seeds

Finish Section 4 of agf_prioritized_experiments_colab.ipynb, including all seeds
and both 5D arms and their evaluations. Stop the previous notebook before Section
5 and do not run it concurrently with this notebook. Select Tesla T4 here and
click Run all. Existing 5D source run is pinned to full identity
95fbab1d48bff58545543e390b89e28c87c8cc8612db4fc25e9759489407befb.
The saved completed 4D continuation run is
32dcb7bd6e9ccc3ad24fa9744a9e390229af6b2eea773fff813a4efdeabe0f41.
Missing, partial, ambiguous or corrupt inputs stop without retraining fallback.

Original 4D cap-0.40 and Vanilla audits are also recovered from the completed,
pinned continuation source. Verify their model/readout and test identities,
completed summary and compact cohort before copying; never recompute controls.

## Models and execution order
Seeds 21, 22, 23 for each arm. First run ordinary 5D AGF-G continuation for all
three seeds, then matched 5D Vanilla continuation for all three, then 5D PCGrad
on encoder and decoder for all three. Each arm adds exactly 3,000 updates from
its saved 10,640-scheduled-update endpoint, finishing at 13,640. Both guided
branches use the SAME final saved mask-initialized 5D checkpoint, Adam moments,
head and controller; PCGrad does not start from ordinary continuation. The
Vanilla control starts from the paired final 5D Vanilla checkpoint and carries
its Adam state. No new warm-ups, initialization, mapping, extraction, exchanges,
HF token or dataset downloads. New work totals 27,000 gradient updates.

All models have 1,024 groups, five directions per group and Top-16 gating:
5,120 directions and up to 80 active coefficient slots. All guided arms use
reconstruction weight 1 and bounded adaptive object weights, contrast
[0.0125, 0.40], detection [0.0025, 0.40]. The cap is not a fixed weight. Carry
the controller's reference losses, counters and Adam state. Controller uses
raw shared encoder gradient norms, alpha 1.5, weight learning rate 0.01 and
updates every 10 steps. This is GradNorm-inspired, not canonical GradNorm or
a meta-gradient through the model optimizer. Vanilla uses reconstruction only.
Model learning rate stays 0.0003, noise 0.1, eight images per update and global
gradient clipping 5. Reuse the same 2,000 TRAIN and 400 development images.
There is no reset of model Adam, heads or controller at this continuation.

## PCGrad and matching
For PCGrad, project conflicting weighted task gradients separately over encoder
parameters shared by reconstruction/contrast/detection and decoder parameters
shared by reconstruction/contrast. Detection has no decoder gradient. Independent
saved NumPy RNGs choose the projection order in each domain. Private classifier
heads keep their ordinary weighted gradients; clip globally then step Adam and
normalize decoder directions as before. The controller measures raw encoder
gradients before projection. Diagnostics include task-gradient cosines for both
encoder and decoder. This is our weighted parameter-domain adaptation of PCGrad,
not a guaranteed improvement. Reference: Yu et al., Gradient Surgery for
Multi-Task Learning, NeurIPS 2020:
https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf

Use 93000 + seed for all new image/noise streams, matching saved 4D continuation.
Verify sampled image and noise hashes across all new arms and against saved 4D
ordinary/Vanilla continuation. Reuse those controls and saved 4D encoder-only
PCGrad reports; no 4D training. The 4D encoder-only PCGrad comparison changes
both width and gradient domains, so it does not isolate either change. Only
5D PCGrad versus 5D ordinary isolates gradient projection. Equal scheduled
updates across widths do not imply equal parameter capacity or FLOPs. Warm-up
development-selected update is recorded separately from scheduled total.
TRAIN-fixed category readouts remain unchanged within each model lineage.

## Endpoints, evaluation and reporting
Use final fixed-budget endpoints. Do not choose development-best or test-best
checkpoints. Freeze and verify ALL NINE new endpoints before opening test
features. Reuse the same previously inspected 1,500 test images and cached
starting-model/4D control audits. Perform new inference only for the nine new
models. All 75 category outcomes remain included, as do original native
development-MSE x 1.02 failures. Do not compare test MSE with a development
threshold or change the threshold. Save per-seed and three-seed mean capture,
patch AP and VE, reconstruction MSE, native eligibility, model/readout/optimizer/
controller hashes, starting budget and batch/noise hashes. Compare continuation
with starting models, matched Vanilla and saved 4D continuation, and compare
5D PCGrad with 5D ordinary. Conditional paired whole-image bootstrap intervals
use 200 draws and do not quantify seed uncertainty or correct multiple tests.
The test cohort is exploratory because it has been inspected; use an untouched
cohort for independent confirmation of a selected architecture.

## Resume and outputs
Local two-slot saves every 50 updates. Drive saves every 250 and at endpoints,
SHA256 read-back verified with retries for transient I/O errors. Restore model,
Adam, head, controller, sample RNG, projection RNGs and Torch/CUDA RNGs; runtime
loss may replay only unsaved updates. Finished fits and audits reuse their
fingerprints, not newest-folder discovery. Stop signals terminate the worker;
never run two writers for the output. Partial results ZIPs are saved after each
completed three-seed arm; the final ZIP is saved before automatic download.
Sources are read-only. Initial Drive authorization may be required. Packages
match saved numerical versions and run in a fresh child process, with no
kernel restart or later token prompts.
Output: LOCAL_INPUTS/AGF_COCO_5D_continuation_v1.
Compact ZIPs include source, protocol, tables, plots, per-category results,
paired intervals and checkpoint locations. Large optimizer/activation files
remain on Drive. CPU synthetic tests cover exact resume, gradient domains,
branch pairing, source integrity, freeze barriers and control audit reuse.
Full-size cached-data CUDA training is executed in Colab, not locally.
