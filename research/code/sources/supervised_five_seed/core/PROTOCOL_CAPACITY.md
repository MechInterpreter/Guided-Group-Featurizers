# AGF warm-start capacity sweep: 6D, 8D and 5D/top-24

## New first stage: targeted versus random reserve additions

Before the longer plan below, run a short development-only screen from the
same completed ordinary 5D checkpoints. Use seeds 21, 22 and 23, the saved
0.40 controller cap, cached features, frozen TRAIN readouts and the original
selection-MSE limit. No test features are opened for this screen.

Four branches per seed each receive 500 additional updates:

| Branch | Structural edit | Purpose |
|---|---|---|
| Targeted AGF | Copy one TRAIN-selected reserve direction into each concept slot | Test residual-informed consolidation |
| Random AGF | Copy one random reserve direction from the same legal pool into each concept slot | Capacity- and budget-matched primary control |
| Ordinary AGF | No additions | Isolate the extra-training effect |
| Ordinary Vanilla BSF | No additions | Reconstruction-only budget reference |

The two addition branches have exactly the same usable capacity and Top-16
gate. The no-addition branches match the training budget, not addition
capacity. There are 12 screening endpoints, 6,000 new updates and a scheduled
total of 14,140 per endpoint. The original capacity stage below remains
24 endpoints and 72,000 updates. Combined: 36 endpoints / 78,000 updates,
plus disposable preflight steps that never become fitted endpoints.

Select 512 TRAIN images with the established mask-coverage allocator and
seed 107011. For each category, measure the initialized slot's remaining
object-contrast reconstruction error. Use each reserve direction's signed
raw coefficient and learned decoder vector to estimate how much adding its
contribution would reduce that error. Choose the greatest estimated gain
among directions with nonzero foreground raw activity. This score is an
additive approximation; it does not account for the new Top-16 gate. Keep
nonpositive best scores in the results rather than hiding hard categories.
The random control draws uniformly from the same legal per-category pool,
using seed 211000 + training seed. Neither development nor test outcomes
choose donors. No expensive single-direction removal/dependency graph runs.

Donors come only from groups 75 through 1,023. Copy the learned encoder
column, bias and decoder row into a new sixth slot of concept groups 0
through 74, in saved category order. Leave every original slot and donor
in place. Copying adds 75 directions; it does not preserve total capacity
or guarantee unchanged reconstruction. Record initial metrics before
fine-tuning so duplication and gate effects are visible. Existing and new
directions may train during the 500 updates; preserving the original five
means no removals or reassignments, not freezing their parameter values.
One donor can serve multiple categories, with that reuse recorded explicitly
in the donor lists; this is not proof it represents a shared semantic concept.

Use native rectangular six-coordinate groups for storage. The other 949
groups' sixth coordinates have zero encoder, bias, decoder and Adam moments.
They are zero-gradient fixed points under the unchanged losses and remain
zero, which is checked in saved endpoints and preflight. Thus effective
width is six for the 75 concept groups and five for reserves, with up to
96 active coordinates. No-addition controls have zero padding in all sixth
slots and retain effective width five / 80 active coordinates. Padding is
not represented as learned capacity. Allocated core storage is 9,443,328
parameters; usable addition capacity is 7,984,715, versus 7,869,440 in the
no-addition controls. Guided heads add the unchanged 150 parameters.

Carry old model Adam moments/steps, heads and the full bounded controller.
New-coordinate moments start at zero for both addition branches. All four
arms share 500-update image/noise streams seeded 111000 + training seed.
Use the same learning rate, clean-target/noisy-input reconstruction, contrast,
detection, clipping and decoder normalization as the capacity experiment.
Vanilla remains reconstruction-only. Profile progress commits every complete
25-image chunk; training commits locally every 50 and to Drive every 250
updates and at the endpoint. A graceful stop flushes the latest complete
local generation. An abrupt reset can replay the last incomplete unit.

Some saved capture or detection readouts differ from initialized category
slots. Keep them fixed for training and the primary metrics. Report separate
initialized-slot diagnostics using the same predeclared slot assignment for
every branch. Do not substitute those diagnostics for primary scores or
claim the two measure the same readout. Record each category's original
readout beside its donor selection. For Vanilla, the fixed slot numbers are
a diagnostic convention, not mask-initialized semantic groups. In particular, source capture readouts
match initialized slots for 73/75, 71/75 and 73/75 categories in seeds 21,
22 and 23. Effects on off-slot primary readouts may be indirect through gating.

Save initial and final selection/confirmation metrics, all category outcomes,
targeted-minus-random and targeted-minus-ordinary deltas, and the original
selection-MSE eligibility. Do not automatically adopt a winner. This is a
development-only diagnostic with one random donor draw per fitted seed,
not independent confirmation, a complete structural search, or evidence
that 500 updates suffice to reach a plateau.

First-stage output: `MyDrive/AGF_COCO_reserve_additions_v1`.
A separate `agf_reserve_additions_<run-prefix>_results.zip` is saved before
the longer phase begins. Its results are also copied under
`results/reserve_additions/` in the final capacity evidence ZIP.

The subsequent 6D, 8D and top-24 branches retain the original saved 5D roots,
budgets and randomness; they do not start from the screening winner. Exact
compatible progress from the preceding notebook release (source SHA256
`08f980a9465737be9c7df2b572f85d8104ea3802602894228b3ff00f9e5da9ae`)
is imported read-only when all configuration fields except packaging SHA
match. Newer progress is never overwritten. Completed fits/audits are reused.

Select Tesla T4 and click Run all. Authorize Google Drive at the beginning.
No subsequent HF token, source selection, package restart or upload is needed.
The notebook embeds and verifies its entire implementation.

## Fixed source

Use the completed ordinary 5D AGF-G and matched ordinary 5D Vanilla BSF
checkpoints for seeds 21, 22 and 23 from
`MyDrive/AGF_COCO_5D_continuation_v1`.
Run: `371d2d1d81ea6c9ce15fa17a6cdecc716f42304e385ac0cf9d2f3280572d6a0f`.
Source SHA256:
`4e2f00da8b218512d5a0d06241176819e14f884eeca1ac3269977a057594858b`.
Both model types have 13,640 scheduled updates. Their exact weights,
readouts, Adam moments, controller and completed audits are verified. Source
files are read-only. Missing, incomplete or corrupt sources stop execution;
there is no retraining or backbone-extraction fallback.

The source AGF test means are 41.04% capture, 75.01% patch AP and 66.04%
reconstruction VE. These are references, not results of this notebook.

## Eight new branches per seed

| Configuration | Groups | Width | Top-k | Maximum active coordinates | Core parameters | New updates per model |
|---|---:|---:|---:|---:|---:|---:|
| 6D/top-16 AGF and Vanilla | 1,024 | 6 | 16 | 96 | 9,443,328 | 3,000 |
| 8D/top-16 AGF and Vanilla | 1,024 | 8 | 16 | 128 | 12,591,104 | 3,000 |
| 5D/top-24 AGF and Vanilla | 1,024 | 5 | 24 | 120 | 7,869,440 | 3,000 |
| 5D/top-16 AGF and Vanilla budget controls | 1,024 | 5 | 16 | 80 | 7,869,440 | 3,000 |

Every new branch independently starts from its corresponding completed 5D
source. No branch starts from another new branch. AGF retains its learned
mask-informed representation; Vanilla retains its own reconstruction-trained
representation. Matching covers capacity, gate, source scheduled budget,
additional updates, image/noise schedule and data, not identical supervision
or wall-clock computation. Guided models have 150 additional trained head
parameters. Higher widths/top-k intentionally increase available active
coordinates; this is not an equal-active-coordinate experiment.

There are 18 requested trial/control endpoints and six additional 5D/top-16
budget-control endpoints. Total planned new training: 72,000 updates.
All endpoints have 16,640 scheduled updates. No warm-up is repeated.
The six saved source references and their existing audits are reused;
training the new budget controls is necessary to separate capacity effects
from the effect of 3,000 more updates.

Order: 6D AGF all seeds, 6D Vanilla all seeds, 8D AGF all seeds, 8D Vanilla
all seeds, 5D/top-24 AGF all seeds, its Vanilla controls, then both 5D/top-16
budget continuations. Development tables are saved after each seed and a
partial evidence ZIP after each completed three-seed arm. The unchanged plan
does not branch on test scores.

## Safe expansion of a learned dictionary

Increasing width must not shuffle flattened directions into the wrong groups.
The original five encoder columns, decoder rows and bias entries remain in
their original groups. For each group, seeded random decoder vectors are
projected perpendicular to that group's learned decoder span and normalized
to unit length. New encoder columns and biases start at zero. Thus their
initial coefficients are zero, preserving the original Top-16 predictions
up to floating-point roundoff. The sixth direction is identical in the 6D
and 8D starts for a given source/seed; 8D appends two more directions.
Both AGF and Vanilla use the same expansion rule and candidate RNG schedule.
Their projected vectors differ because their learned decoder spans differ.

These are learned-checkpoint capacity expansions, not fresh mask eigenvector
initializations. A negative outcome does not rule out training a wider
mask-initialized architecture from scratch. New decoder directions learn
after their encoder coordinates become active. Neither old nor new
directions are frozen during continuation.

Adam's original moments and scalar step counters carry forward. Old moments
move with their slots; added coordinates get zero first/second moments.
Head moments remain unchanged. Existing Adam step counters are shared by old
and new coordinates, rather than tracking separate coordinate ages.
Top-24 keeps the exact saved weights and Adam shapes but changes the gate
immediately, so it does not promise unchanged initial predictions.

## Training and adaptive weights

Use the established clean-target/noisy-input reconstruction loss and existing
category-balanced contrast and detection losses. AGF retains reconstruction
weight 1 and the shared-encoder GradNorm-inspired controller with contrast
bounds [0.0125, 0.40] and detection bounds [0.0025, 0.40]. Carry the saved
weights, weight optimizer, reference losses and adaptation count. No
second-order/meta-gradient or validation feedback is introduced. Ordinary
weighted backpropagation updates encoder, decoder and private logistic heads;
there is no PCGrad and no Exchanges. Vanilla optimizes reconstruction only.

Each branch samples eight TRAIN images with replacement per update, using
seed 97000 + training seed, source learning rate, 0.1 noise scaling, gradient
clipping and unit decoder normalization. The noise RNG is reset after model
construction so different widths cannot shift the paired noise stream.
All eight endpoints for each seed must have identical cumulative image and
noise hashes. Checkpoints include model/head/Adam/controller and image,
torch/CUDA and projection RNG state. No readouts or their original heat
normalization are refit. Detection heads are carried and continue training.

## Data, development, freeze and test

Reuse normalized cached DINOv3 patch features and aligned masks: 2,000 TRAIN,
400 selection-development, 400 confirmation-development and the same 1,500
previously inspected test images. All four ID sets must be disjoint and their
normalization/cohort identities must match the saved lineage. No images,
annotations or backbone weights are downloaded. Feature shards load into
local memory maps and GPU minibatches, not a full GPU dataset.

AGF development metrics are reported every 250 updates. Vanilla checkpoints
report reconstruction only during optimization, so capture/AP fields there
are intentionally None. Both models receive full capture/AP/reconstruction
audits on both development splits at the final endpoint. Report all outcomes,
including unchanged or failed reconstruction guards; final endpoints, not
development-best checkpoints, define this experiment.

Keep the original seed-native selection-MSE allowance of 1.02 times its
saved reference. Apply that absolute limit only to selection development,
not to test MSE or a different split's VE. Test tables also report matching
Vanilla reconstruction directly so the older reference cannot hide a
reconstruction tradeoff.

All 24 endpoints must finish, pass identity/budget/pairing checks and freeze
before new test feature inference. Evaluate every configuration and retain
all categories/seeds. Saved six source references reuse their existing test
reports and compact arrays without inference or readout refitting. Score the
24 new endpoints with the established category-fixed capture and raw group
norm patch-AP audit. Gating continues to use raw group norms; no width
normalization changes the baseline implementation.

Compare each AGF with its capacity/gate-matched Vanilla; each changed-capacity
arm with its same-budget 5D/top-16 counterpart; and the 5D/top-16 continuations
with their saved sources. Report per-seed and three-seed means, full category
outcomes and 200-draw paired whole-image bootstrap capture/AP intervals.
These intervals condition on fitted models; they do not quantify seed-level
or researcher-selection uncertainty or correct multiple comparisons. This
already inspected test cohort cannot establish independent confirmation.

## T4 preflight, resume and outputs

Before production training, run the bundled CPU synthetic checks and eight
disposable ten-update GPU checks. Exercise widened optimizer states, actual
supervised/controller backward passes, finite values and correct Top-k
shapes. Check preservation of starting Top-16 predictions and measure peak
GPU memory and seconds per update. These 80 disposable updates do not alter
source or trial RNGs. Show a gradient-only remaining-time estimate; data I/O,
development audits, Drive writes, test and intervals add time.

Verified two-slot local checkpoints save every 50 updates. Verified Drive
checkpoints save every 250 updates and at endpoints. A graceful cell stop
attempts to flush the newest complete local generation to Drive. An abrupt
runtime loss may replay up to 249 updates since the last verified Drive save.
Run All unchanged to restore all state and skip completed training, audits
and intervals. Completed saved checkpoints are never silently retrained.
Use one writer for this output; do not run copies concurrently.

Output: `MyDrive/AGF_COCO_capacity_sweep_v1`.
Results ZIP: `agf_capacity_<run-prefix>_results.zip`, downloaded automatically
after completion. It includes protocol, source, histories, checkpoints'
locations/hashes, frozen endpoint metadata, development/test CSVs, plots,
paired intervals and all category results. Large model/optimizer checkpoints
remain on Drive and are not copied into the evidence ZIP.
