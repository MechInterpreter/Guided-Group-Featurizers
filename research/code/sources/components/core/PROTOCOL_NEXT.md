# AGF: prioritized three-experiment notebook

Select Tesla T4, then Run all. Order is fixed: 5D first, broader PCGrad second,
longer ordinary continuation third. Each experiment completes all three seeds,
freezes its endpoints, evaluates them, and exports results before the next one.
The notebook displays three-seed means and pauses 30 seconds between experiments
so the user can stop execution. Rerun unchanged to resume. There is no automatic
test-driven branch or checkpoint choice. A stopped pipeline is partial, not a
completed three-experiment result.

## Shared method and controls
All guided arms use bounded reconstruction-anchored adaptive weights with
contrast bounds [0.0125, 0.40], detection bounds [0.0025, 0.40], and reconstruction
weight exactly 1. The upper bound is a cap, not a permanently fixed weight.
The controller matches the saved cap-0.40 implementation: raw shared-encoder
gradient norms, alpha 1.5, weight Adam learning rate 0.01, adaptation every 10
updates, 20 valid reference batches for a fresh controller. Carried controllers
retain reference losses and optimizer state. This is GradNorm-inspired, not
canonical GradNorm or a meta-gradient through the encoder optimizer.
Vanilla BSFs use reconstruction only; no adaptive object weights or labels in
their training. Training-only readouts provide the same object metrics.
Seeds are 21, 22 and 23. Groups are 1,024 and Top-16 throughout. No Exchanges
are performed; these are AGF-G experiments, with legacy checkpoint IDs retained.
Guided learning rate is 0.0003, noise 0.1, eight sampled images per update,
global gradient clipping at 5. Numerical packages and the Tesla T4 match sources.
Existing 2,000 training and 400 development images, normalization and backbone
activations are reused. No HF token, image download or extraction occurs.

## 1. Five directions per group
Append the fifth eigen-direction of the TRAIN foreground uncentered Gram matrix
to each of the 75 saved four-direction category initializers. Preserve their
first four axes byte-for-byte, including orientation. Use foreground coverage
at least 0.50 and crowd coverage at most 0.10, exactly as in the saved initializer.
Check eigenspace agreement and orthogonality. Each category direction is cached.
COCO category indices contain gaps; group slots follow the retained category
order and are not interchangeable with raw category indices.

Train mask-initialized 5D AGF-G and random-initialized 5D Vanilla for all seeds.
The 5D architecture has 5,120 directions and at most 80 active coefficient slots,
versus 4,096 and 64 for 4D. The new 5D Vanilla matches group width, sparsity and
scheduled gradient budget; masks, supervision and head parameters still differ.
This controls capacity without claiming identical computation or supervision.
New warm-ups are necessary for both 5D arms; 4D checkpoints cannot be resumed
as 5D models. Each reconstruction warm-up runs 40 epochs / 7,640 updates with
the original OneCycle schedule and development-minimum-MSE checkpoint choice.
TRAIN-only readouts are chosen once after warm-up and frozen. Then train 2,500
+ 500 updates, resetting model Adam at that boundary as in the saved 4D study;
carry trained heads and adaptive controller between guided phases. Vanilla
continues reconstruction only in both phases. Paired epoch permutations match
the saved 4D warm-up; phase image/noise streams match saved 4D controls at
71000 + seed and 82000 + seed. Reject differences rather than rerun controls.
Reuse saved cap-0.40 4D AGF-G and 4D Vanilla test reports. Three-seed comparisons:
5D AGF-G versus 5D Vanilla, 5D AGF-G versus 4D AGF-G, and 5D versus 4D Vanilla.
New work: six warm-ups and six 3,000-update fits = 63,840 scheduled updates.

## 2. PCGrad on encoder and decoder
Start at the SAME final saved 4D cap-0.40 checkpoint used in the completed
ordinary and encoder-only PCGrad experiment. Carry model Adam, trained heads
and controller; add 3,000 updates, finishing at scheduled budget 13,640.
Reuse completed ordinary continuation, encoder-only PCGrad and matched Vanilla
continuation. Do not retrain these controls. Use identical sampled images/noise
(93000 + seed) and verify their saved hash chains.

Encoder parameters receive separate weighted gradients for reconstruction,
contrast and detection. For each task, shuffle other task order and project a
conflicting gradient against the ORIGINAL other gradient before summing projected
task gradients. Decoder parameters receive the corresponding two-task projection
for reconstruction and contrast; detection has no decoder path. Encoder and
decoder are separate projection domains with independent saved RNGs. The encoder
RNG matches the prior encoder-only experiment, so adding decoder projection does
not silently change encoder projection order. Classifier heads keep their ordinary
weighted detection gradients. Global clipping and Adam follow projection.
The controller still measures raw shared-encoder gradients before PCGrad.
Report encoder and decoder gradient conflicts and all metric/guard outcomes.
This is a weighted, parameter-domain adaptation of PCGrad, not guaranteed
improvement or unchanged canonical PCGrad. Yu et al., NeurIPS 2020:
https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
New work: three 3,000-update fits = 9,000 updates. Twelve paired comparisons.

## 3. Longer ordinary continuation
Start from the completed ordinary 4D guided endpoint and its completed matched
4D Vanilla endpoint, both at scheduled budget 13,640. Add 3,000 updates to each,
finishing at 16,640. Carry Adam, heads and controller; preserve cap 0.40 and learning
rate. Use new paired batches/noise at 94000 + seed. No PCGrad in either arm.
Compare guided versus equally continued Vanilla and changes from both starting
endpoints. This is a bounded diagnostic of continued capture gains, not a claim
that the capture ceiling has been reached or that more training restores VE.
New work: six 3,000-update fits = 18,000 updates. Nine paired comparisons.

## Evaluation and resuming
Total planned new work: 90,840 scheduled updates, including necessary 5D warm-ups.
All endpoints are final fixed-budget endpoints, with warm selected update recorded.
Scoring uses the identical cached 1,500 previously inspected test images. Every
experiment's complete set of seeds and arms is frozen and identity-checked before
that experiment's test scoring. Test features are closed again before any next
training stage. No fitting, calibration, remapping or selection on test occurs.
If the user stops after inspecting results and launches a different study, the
cohort remains exploratory; use an untouched cohort for independent confirmation.
Report capture, patch AP and reconstruction VE, all 75 category outcomes, and
unchanged original native-development-MSE times 1.02 guards. Retain every failed
guard. Intervals are conditional paired whole-image bootstraps with 200 draws;
they do not measure seed-level uncertainty or correct for multiple comparisons.

Checkpoint local model/Adam/head/controller and sample, projection, Torch/CUDA RNG
states every 50 updates; checksum-verify Drive writes every 250 and at endpoints.
Warm-up additionally saves its epoch permutation and learning-rate scheduler.
Runtime loss can replay up to the unsaved interval; completed warm-ups, fits,
readouts and audits are reused. Source stores are read-only. Source discovery uses
exact full run identities, never the newest folder or ambiguous manifest counts.
Incompatible, missing or corrupt sources stop without a retraining fallback.
Stopping a cell signals the worker to exit; do not run two writers concurrently.

Initial Drive authorization may be required. Saved exact package versions install
before a fresh child process, avoiding a kernel restart. Per-stage compact ZIPs
contain sources, protocols, scores, per-category results, plots, paired intervals
and hashes. Large activation/optimizer checkpoints remain on Drive. The final
cell downloads the ZIP; after an early stop the most recent ZIP is already on
Drive at MyDrive/AGF_COCO_prioritized_width5_pcgrad_continuation_v1.
Synthetic CPU tests verify projection math, gradient domains, exact interrupted
resume, schedule pairing, freeze barriers and inference-only control reuse.
Full-size saved-data CUDA execution is performed in Colab, not locally.
