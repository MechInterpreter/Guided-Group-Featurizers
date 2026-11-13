# Saved 0.40 test evaluation and higher-cap development sweep

Both notebooks require the completed source runs already saved on Drive:
- GGF_COCO_width_balance_v1, run e4aeabc90db7 (all five arms, three seeds).
- GGF_COCO_weight_bound_v1, run 842bd916b90c (0.40, all three seeds).
- AGF_COCO_stratified_confirmation_v1 (original normalization and caches).
The test notebook additionally requires AGF_COCO_strong_mask_regroup_v1,
which stores the previously scored 1,500-image test features and masks.

No inputs are retrained or replaced. Missing/corrupt/ambiguous inputs stop.
All saved sources use read-only stores, complete phase checks, SHA256 payload
verification, frozen readouts and exact endpoint identities.

## Evaluation notebook
Seeds 21, 22, 23; all final 0.40 endpoints after 2,500 + 500 updates.
Reuses exact cached test reports and compact paired-audit arrays for the 0.20
GGF, Vanilla BSF 4D, and original fixed-loss GGF 3D. Only the three 0.40 models
receive new inference. No fitting, mapping, extraction, optimizer, or exchanges.
The same 75 training-fixed category readouts and the same 1,500 cached test
images are used. The original development native-MSE x 1.02 guard is unchanged;
all outcomes, including failed guards, remain visible. Recompute development
reconstruction to verify that the saved treatment checkpoint reproduces.
Nine comparisons receive the original 200 whole-image paired bootstrap draws.
Intervals are conditional on fixed trained models and shared images, not seed
or population confidence guarantees; no multiple-comparison correction.
The test has been inspected in earlier experiments, so it is not a new untouched
holdout. Do not use its scores to choose future caps or checkpoints.

## Development sweep notebook
Default cap 0.60 for seeds 21, 22, 23: 9,000 new guided updates. Expand RUN_CAPS
to [0.60, 0.80, 1.00] to add 18,000 updates, reusing all completed 0.60 trials.
The canonical supported-cap protocol remains fixed, so requested scope does
not change checkpoint identity. Testing all nine trials totals 27,000 updates.
Every trial starts from the same saved 4D mask reconstruction warm-up. It does
not continue from a trained 0.20 or 0.40 endpoint. G=1,024, b=4, k=16, 64 maximum
active coordinates. Reconstruction coefficient remains 1; contrast/detection
start at 0.05/0.01, lower bounds 0.0125/0.0025. Only their shared upper cap varies.
Other controller settings and all training settings remain unchanged.
This is bounded reconstruction-anchored GradNorm-inspired balancing, not
canonical GradNorm (https://proceedings.mlr.press/v80/chen18a.html).
Changing a cap does not force either loss weight to that value.
Model Adam resets at the 2,500 boundary; trained classifier heads and controller
state carry forward. Batch/noise hash chains must match the saved 0.20 controls.
Use the same Tesla T4 as the paired source. No new warm-up, Vanilla training,
backbone/image extraction, HF token, AGF exchanges, or test feature access.
Evaluate every 250, use final endpoints only. Compare development scores with
both saved 0.20 and 0.40 controls; retain all reconstruction failures.
Select tradeoffs on development only, preferably examining the Pareto frontier
of capture, patch AP and VE among models satisfying the unchanged native guard.
Larger caps are exploratory; the 0.40 screen improved capture/AP but reduced VE
in all seeds. No cap is asserted to improve all three metrics.

## Resume and packaging
Source and every artifact have checksums. Completed reports/trials are reused.
Training: local saves every 50 updates; verified Drive saves every 250 and at
both phase endpoints. After runtime loss, at most the unsaved interval replays.
Stop at a completed checkpoint before disconnecting. Compact exports contain
source, protocol, reports, category metrics, identities, plots, and audit
locations. Large model/optimizer and detection-score checkpoints remain on Drive.
Setup can install exact source versions and require one explicit runtime restart;
there are no later HF-token or image-download prompts.

CPU synthetic protocol/inference/resume checks do not execute the full saved-data
CUDA experiment, which must run in Colab against the user's Drive.
