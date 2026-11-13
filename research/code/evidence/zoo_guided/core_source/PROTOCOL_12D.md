# AGF 12D follow-up from the completed 8D checkpoints

This experiment tests whether increasing block width from eight to twelve
directions improves object capture, patch AP and reconstruction beyond another
3,000 updates of unchanged 8D training. It also measures the new effective ranks.

All arms have 1,024 groups and raw coefficient-norm Top-16 gating. AGF uses the
existing shared-encoder adaptive controller, reconstruction coefficient 1,
contrast bounds [0.0125, 0.40] and detection bounds [0.0025, 0.40]. No PCGrad or
Exchanges is used. No warm-up, backbone extraction or new mask initialization
is run. The test cohort is the same previously inspected 1,500-image cohort.

## Starting points and new arms

Sources are the completed 8D/top-16 AGF and 8D/top-16 Vanilla checkpoints for
seeds 21, 22 and 23 in `LOCAL_INPUTS/AGF_COCO_capacity_sweep_v1`:

- Run `2921b057f824362ab83e76e7fb35645f94c376fadf8b86f4f22d1a2311105b9e`.
- Source SHA `6f9138f8885fdbb2d5971f30ab0c5d744d4c8fa914e1c191491c895d5ca8cd79`.
- Each source has 16,640 scheduled updates. Its weights, Adam moments, heads,
  controller, reference losses and fixed TRAIN readouts are verified.

| Order | Arm | Seeds | New updates per seed | Directions | Active coefficient slots |
|---|---|---|---:|---:|---:|
| 1 | 12D AGF | 21, 22, 23 | 3,000 | 12 | 192 |
| 2 | Matched 12D Vanilla BSF | 21, 22, 23 | 3,000 | 12 | 192 |
| 3 | Continued 8D AGF budget control | 21, 22, 23 | 3,000 | 8 | 128 |
| 4 | Continued 8D Vanilla budget control | 21, 22, 23 | 3,000 | 8 | 128 |

There are 12 new endpoints and 36,000 planned new gradient updates. Every endpoint
reaches 19,640 scheduled updates. The six saved 8D source models and their test
audits are reused without retraining. New 8D continuations are needed as equal
additional-budget controls. Each branch starts independently from its own
finished 8D source; no branch starts from another newly fitted model.

AGF and Vanilla differ in supervision. Matching covers width, gating, scheduled
budget, image/noise schedule, dataset and additional updates, not identical
supervision or wall-clock cost. AGF has 150 additional classifier-head parameters.
Core parameter counts are 18,886,656 for 12D and 12,591,104 for 8D. The wider model
has 50% more core parameters and maximum active coefficient slots.

## Width expansion and continuation

All eight learned encoder columns, encoder biases and decoder rows stay in their
existing groups. Four seeded random decoder directions are projected orthogonal
to the group's learned span and previous additions, then normalized to unit
length. New encoder columns and biases start at zero. Thus, before training, the
Top-16 code and reconstruction match the source up to floating-point roundoff.
Disposable T4 preflight checks verify this property on the saved data.

Old Adam moments, scalar step counters, heads and controller state are retained.
New-coordinate first and second moments are zero. The new coordinates share the
existing Adam step counter; there is no independently aged optimizer for them.
All directions may fine-tune. Existing slots are preserved structurally, not frozen.

The unchanged fit engine uses eight sampled training images with replacement,
learning rate 0.0003, noise scale 0.1, gradient clipping at 5, and unit decoder-row
normalization. Model construction is followed by a reset of the training-noise
generator so width-dependent initialization does not change the paired streams.
All four arms use phase seed 123000 + seed. Batch and noise hashes must match.

The final 3,000-update checkpoint is evaluated. No best checkpoint is chosen from
test results. The original TRAIN category capture/detection readouts and heat
normalization remain fixed. AGF's logistic heads continue training. Vanilla is
reconstruction-only; its intermediate capture/AP log fields are intentionally
None, and complete object metrics are computed at endpoints.

## Evaluation

Reuse the same 2,000 TRAIN, 400 selection and 400 confirmation images, their saved
patch features, masks and normalization. Verify the disjoint split identities.
All 12 endpoints must freeze before opening test features. Report all endpoints,
including original native-MSE guard failures. The 2% native-MSE limit is applied
to its original selection split, not to test MSE from a different distribution.

Score 12 new endpoints on the same 1,500-image test cohort and reuse six saved 8D
audits. Per seed, compare 12D AGF against 12D Vanilla, continued 8D AGF against
continued 8D Vanilla, 12D against equally continued 8D for each model type, and
continued 8D against its saved 8D source for each model type. There are 18 paired
comparisons. Each uses 200 whole-image bootstrap resamples for capture and AP.
Intervals condition on fitted models and omit seed-level uncertainty,
researcher-selection uncertainty and multiple-comparison adjustment.

## Effective-rank analysis

After training and scoring, analyze only TRAIN features for all 12 new endpoints.
Reuse, rather than recompute, the six completed 8D rank analyses from
`LOCAL_INPUTS/AGF_COCO_capacity_effective_rank_v1`, run
`7c974de9e38700416635ab01ad903186d0375c01bdd9fcc91e519f61cd6eef0c`.
Verify the rank protocol, exact model hashes, cohort identity and per-file checksums
before training begins. No readout or model is fitted by the rank analysis.

Use the unchanged validated rank implementation: centered covariance of each
block's coefficients conditional on nonzero gated activity; entropy effective
rank, participation ratio, covariance stable rank, coefficient-matrix stable rank,
and effective rank divided by allocated width. Compute contribution covariance
spectra through the decoder Gram matrix. Report all-estimable and at-least-32-active-
patch summaries, sample counts and undefined cases. The first 75 initialized
slots and 949 reserve slots remain separate; Vanilla's first 75 are a positional
control, not semantic initialization. These are empirical code-variation measures,
not ground-truth concept dimensions or MDL efficiency.

## T4 execution and recovery

Mount Drive and match the saved numerical package versions up front. A fresh
Python worker avoids restarting the notebook kernel after installation. No HF
token or interactive decision is needed later. Four disposable ten-update GPU
preflights check both widths and both supervision paths; they do not modify source
or production checkpoints and are excluded from the 36,000-update budget.

Local training checkpoints are saved every 50 updates; verified Drive checkpoints
every 250 and at endpoints. Graceful interruption flushes the latest complete
local checkpoint. Abrupt runtime loss can replay up to 249 training updates.
Completed fits and audits are skipped. Rank accumulation uses two checksummed
generations every 32 inference batches; completed rank analyses skip inference.
Run All unchanged after a reset, with only one notebook writer for this output.

Outputs are separate in `LOCAL_INPUTS/AGF_COCO_12D_followup_v1`. The final evidence ZIP
contains metrics, histories, category outcomes, bootstrap intervals, rank spectra,
figures, provenance and source. Model/optimizer checkpoints and rank accumulation
checkpoints remain in Drive and are excluded from the evidence ZIP.
