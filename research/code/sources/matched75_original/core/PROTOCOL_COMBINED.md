# All caps in one run: development, freeze, test, compact export

Select a Tesla T4, matching the saved paired training controls, and Run all.
Drive authorization may be required at the initial mount. There are no later
HF-token prompts, new image downloads, backbone extraction, or manual model
selection pauses. Exact source packages are installed before computation.
The numerical experiment runs in a fresh child Python process so package
changes do not require restarting the notebook kernel.

## Fixed execution order
1. Verify source, versions, saved completed runs, warm-ups, normalization,
   control reports and the existence/checksums of required test shards. No
   experimental test features are opened at this stage.
2. Run CPU checks and a disposable full-size GPU loss/controller preflight.
3. Run caps 0.60, 0.80 and 1.00 across seeds 21, 22 and 23 on training and
   development data only. Each receives 2,500 + 500 guided updates from the
   same saved 4D mask reconstruction warm-up. Total maximum new updates: 27,000.
4. Verify and freeze every final endpoint. Before any new test features open,
   save a development recommendation using the existing score: capture plus
   0.20 times patch AP (both fractions). Only caps passing the original native
   reconstruction-MSE x 1.02 guard in all three seeds can be recommended.
   Ties prefer higher VE and then a smaller cap. This is a transparent fixed
   development rule, not evidence that one cap improves every metric.
5. Load the identical cached 1,500-image test and score caps 0.40, 0.60, 0.80
   and 1.00 for all three seeds. Reuse the already saved 0.20 GGF, Vanilla BSF
   4D and original fixed-loss GGF 3D control reports and paired arrays. If the
   earlier 0.40 evaluation notebook has completed exact matching audits,
   those audits are reused too. No controls or endpoints are trained here.
6. Report all caps, including failures of the original reconstruction guard.
   Each treatment is compared with all three controls. Higher caps are also
   compared with 0.40: 45 conditional paired whole-image comparisons, using
   the original 200 bootstrap resamples per comparison. No correction for
   multiple comparisons or seed-level confidence claim is made.
7. Export one compact ZIP with separate development and test tables, category
   results, confidence intervals, plots, source, checksums, checkpoint locations
   and the pre-test development recommendation. Large checkpoints and feature
   caches remain on Drive. The final notebook cell downloads the ZIP.

The test cohort has already been inspected in previous experiments. Its reuse
provides common-cohort evaluation, not a new independent holdout. The test does
not choose the preferred cap or checkpoint. These are GGF trials; no structural
AGF exchanges occur.

## Capacity and training recipe
G=1,024; b=4; k=16; maximum 64 active coefficient slots. Reconstruction weight
stays at 1; contrast/detection start at 0.05/0.01 with lower bounds 0.0125/0.0025.
Only their shared adaptive upper cap changes. Actual weights need not reach it.
The unchanged bounded reconstruction-anchored controller is GradNorm-inspired,
not canonical GradNorm (https://proceedings.mlr.press/v80/chen18a.html).
The same 2,000 training images and 400 development images, learning rate,
sampling/noise streams, 2,500-boundary Adam reset, head carryover and controller
carryover are reused. The original training-selected category readouts remain
fixed; no test calibration is performed.

## Reuse and resume
The previous higher-cap notebook's entire numerical engine is byte-identical:
76 implementation files are checked against its published source snapshot.
Its source identity is retained for the training store only, allowing exact
reuse of any existing higher-cap trials. The new coordinator and evaluation
have separate new source identities, all recorded in the evidence.

Training output: MyDrive/GGF_COCO_higher_caps_v1
Combined output: MyDrive/GGF_COCO_caps_complete_v1
Optional prior 0.40 test output: MyDrive/GGF_COCO_cap040_test_v1

Completed trials and audits are reused on Run all. Local training saves occur
every 50 updates; verified Drive saves every 250 and at both phase endpoints.
After runtime loss only unsaved updates may replay. Never run two notebooks
against the same training/output store simultaneously.

All source inputs are read-only. Missing, ambiguous, incomplete or corrupt
inputs stop without a retraining/extraction fallback. Validation here uses
small CPU synthetic data; full saved-data GPU execution occurs in Colab.
