# Completed-cap test audit and three-arm PCGrad continuation

Both notebooks are self-contained. Select Tesla T4 and Run all. Initial Drive
authorization may be required; no later HF token, package-restart, image-download,
backbone-extraction or manual-selection prompts. Scientific computation runs in
a fresh child Python process after exact saved package versions are installed.

## Test-only scope
Evaluate final saved caps 0.40 and 0.60 for seeds 21, 22 and 23. Verify and freeze
all six endpoints before reading test features. The 0.60 engine/phase lineage,
warm-up, heads, readouts and paired batch/noise hashes must match saved controls.
Caps 0.80 and 1.00 are explicitly outside this audit; their unfinished checkpoints
are not read or changed. Reuse saved 0.20 GGF, Vanilla BSF 4D and original GGF 3D
control reports and compact arrays. Inference only: zero training updates.
All five arms have three seeds; 21 conditional paired whole-image comparisons.
Output: MyDrive/GGF_COCO_completed_caps_test_v1.

## Continuation scope
Seeds 21, 22 and 23; all three arms receive 3,000 additional updates:
1. Ordinary adaptive GGF continuation from its saved final 0.40 checkpoint.
2. GGF from that identical checkpoint with weighted PCGrad on the encoder.
3. Vanilla BSF continuation from its saved width-four endpoint.
The two GGF arms carry identical source weights, trained heads, Adam moments and
bounded reconstruction-anchored controller states. Vanilla carries its own
source weights and Adam moments. No warm-up or previous fit is rerun. All start
after 7,640 warm-up + 3,000 continuation updates, and finish after 13,640 total
updates. Warm-ups retain the previously development-selected checkpoint.

Capacity: 1,024 groups, four directions per group, Top-16, 64 maximum active
coefficient slots. Learning rate 0.0003 and noise 0.1 remain unchanged. The
GGF controller retains cap 0.40, lower contrast/detection bounds 0.0125/0.0025,
and reconstruction coefficient 1. Fresh identical batches/noise are seeded
93000 + seed across all three arms. PCGrad projection order uses an independent
saved NumPy RNG and cannot change image sampling or Torch noise streams.

PCGrad modification: compute the three weighted task gradients. On parameters
shared by all three tasks (the encoder), project conflicting gradients against
the original other task gradients in shuffled order and sum the projected
vectors. The decoder retains the weighted reconstruction-plus-contrast gradient;
the private logistic-classifier heads retain their ordinary detection gradient.
This is an encoder-only weighted adaptation of PCGrad, not a claim of unchanged
canonical PCGrad or guaranteed improvement. See Yu et al., NeurIPS 2020:
https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
The adaptive controller remains GradNorm-inspired, not canonical GradNorm.

Use final 3,000-update endpoints, not a test-selected checkpoint. Freeze all
nine endpoints before opening test features. Retain failed original native-MSE
x 1.02 guards and every category. Report capture, patch AP and reconstruction
VE for the three continued arms and both starting models. Compare changes from
each starting checkpoint, PCGrad with ordinary continuation, and both GGF arms
with continued Vanilla. There are 18 paired comparisons. Primary evidence of
success requires improved three-seed mean capture, AP and VE over ordinary
continuation and the starting GGF, with per-seed outcomes and guards visible.
Output: MyDrive/GGF_COCO_pcgrad_continuation_v1.

## Shared evaluation, resume and evidence
Training uses existing 2,000 images and development 400. Test uses the identical
cached 1,500 previously inspected images, normalized with the original frozen
training statistics. No test calibration, label remapping or fitting occurs.
Readouts are the source training-selected assignments. Intervals use 200 paired
whole-image bootstrap draws, conditional on fitted models, with no seed-level
or multiple-comparison correction. This is a reused cohort, not independent
confirmation. No structural AGF swaps occur in either workflow.

Continuation saves locally every 50 updates and verifies Drive writes every
250 updates and at the final endpoint. Model and Adam states, heads, controller,
image/projection RNGs and Torch/CUDA RNGs resume together. Completed fits/audits
are reused; after runtime loss only unsaved updates can replay. Missing,
ambiguous, corrupt or incompatible source data stop without retraining fallback.
Source inputs are read-only. Do not run two copies writing the same output.
Stop the current all-cap training cell before running the test-only notebook.

Both workflows export a compact ZIP of protocols, plots, per-seed and category
results, paired intervals, source and checksums. Large optimizer checkpoints,
activation caches and audit arrays remain on Drive. The final cell downloads
the ZIP. Synthetic CPU tests and notebook setup smoke checks are local
validation; full saved-data CUDA execution occurs in Colab.
