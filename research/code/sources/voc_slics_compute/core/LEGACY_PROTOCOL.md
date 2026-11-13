# Prospective supervised comparison: GGF and SLiCS only

This replaces the historical 75x3 repair notebook. Select a T4 and Run All.
No Vanilla, Grassmannian, Group Lasso, detection-only BSF or synthetic benchmark
is trained or evaluated. Stop the other notebook's writer before running this
one. All prior stores remain read-only. Outputs use a new Drive folder.

## Question and scope

The primary question is whether GGF's object-guided learning improves single-
group object capture and patch detection relative to published supervised SLiCS
dictionary learning at the SAME number of concept groups and directions, on
the same patch task with the same annotations. It is not a comparison of SLiCS's
original image-level retrieval benchmark, and does not change GGF to global
image training. Published objectives and updates are applied to the common
patch vectors; this input/task adaptation is explicit for SLiCS.

The primary comparison is GGF-75 versus SLiCS-75: 75 category-associated groups,
12 directions per group, 900 decoder directions. Every category is retained.
Both receive identical normalized cached DINOv3 patch vectors, category masks,
instance annotations, initial SVD axes and polarity, and underlying image
presentation schedule. SLiCS uses category-presence supports; GGF uses its
original instance-contrast and category-detection losses. Equal annotation
access does not make their losses identical.

GGF retains its free encoder, Group Top-32, decoder, original detection heads,
reconstruction warm-up, and two adaptive object-loss weights with cap0.60.
The primary pair has no reserve groups, so GGF's group count differs from the
current larger model. This is an explicit capacity-controlled method comparison,
not an assertion that the small control is the strongest GGF configuration.
SLiCS retains its nonnegative supported coefficients and alternating atom fit;
no encoder, detection heads, background group, Top-k, ridge term, shrinkage or
extra objective is added. GGF has more total parameters because its encoder is
untied. Dictionary size is matched; total parameters, activity and FLOPs differ
and are reported. Do not label this an equal-parameter or equal-FLOP comparison.

The exact three completed 4,096-group /12D /Top-32 /cap0.60 GGF endpoints are
read-only capacity anchors. They are evaluated on the new cohort but excluded
from matched-capacity significance claims. No large-scale GGF endpoint is
retrained and no extra four large-scale starts are needed here. The ongoing
seven-seed BSF study remains separate.

## Training and development selection

1. Verify source identities, numerical versions, cached training/development
   arrays, normalization and three existing current-GGF weight/readout hashes.
2. Reserve a new 1,500-image holdout using annotation coverage only. Exclude all
   recorded prior cohorts and the other study's reserved final holdout. No test
   activations or model scores are opened here. Replay one TRAIN image through
   the cached backbone to verify extraction identity/authentication.
3. Fit SLiCS for seeds21,22,23 first. Each starts from the leading12 uncentered
   SVD directions of each category's training foreground patch vectors, signed
   by positive-versus-negative coefficient energy, as Algorithm1 specifies.
   Reuse saved SVD axes and recompute their polarity from TRAIN vectors only.
4. Fit the GGF-75 three-seed learning-rate candidates 0.0001,0.0003,0.001.
   Each seed shares one reconstruction warm-up across its three candidates.
   No fixed/adaptive cap sweep occurs. Cap stays0.60, initial weights and the
   bounded controller remain those of current GGF.
5. Select the GGF rate by mean three-seed DEVELOPMENT capture +0.2 AP, breaking
   exact ties by lower MSE and then declared grid order. Save every candidate,
   all three metrics and active-group counts. There is no GGF-derived MSE
   rejection cutoff for SLiCS, and no assumption either method must win.
6. Add seeds24--27 for both methods using the frozen settings. Freeze all
   selected models, fixed-budget final models and readouts before test features.

Each matched pair is fitted for10,640 eight-image updates on clean activations.
Both exclude annotation-unknown crowd patches from their reconstruction
training objective using the same crowd threshold. GGF's object/detection
losses already exclude those unknown patches. Test reconstruction is scored
on all patches, identically for both methods.
GGF uses7,640 reconstruction warm-up plus3,000 guided updates. SLiCS uses its
native supervised reconstruction objective throughout10,640 alternating
updates. This matches presentation counts, not optimizer computations. Within
each phase, images are shuffled into disjoint full minibatches each epoch,
following the paper's minibatch procedure. Both methods use the same exact
orders, verified by hashes. The order resets at7,640 to match GGF's phase
boundary. Both primary methods use clean vectors, avoiding a GGF-only noise
advantage; current large-GGF anchors retain their original noisy training.

GGF warm-up uses Adam0.0003. Guided learning uses the candidate/selected rate,
fresh Adam, fresh detection heads and the original bounded controller. Selection
considers the warm-up endpoint and each500 guided updates through3,000.
The SLiCS native SVD initializer and every500 updates, plus7,640 and10,640, are
eligible. Object score is capture +0.2 AP; earliest exact tie wins. SLiCS's
minimum-development-MSE selector is independently retained and reported as a
predeclared sensitivity. All models complete their fixed training budgets even
when an earlier checkpoint is selected. Selected training lengths remain visible.
Fixed10,640-update GGF/SLiCS endpoints are additionally evaluated, so selection
cannot hide an unfavorable final-budget comparison.

Both methods use their assigned category slot for the primary small-model
readout. No test-dependent remapping or best-test checkpoint selection occurs.
GGF logistic heat normalization uses only TRAIN patches after warm-up. The
current larger model retains its exact historical TRAIN-selected mappings.

The seven matched runs share the same deterministic SVD initializer. Seeds
vary the training image order; they are not seven different initial concept
geometries or datasets. Seed intervals describe that training-order variation.

## Faithful SLiCS inference and numerical checks

Algorithm1 uses supported NNLS, sequential atom updates only on currently
nonzero coefficient support, leading rank1 SVD, optimal sign and coefficient
clamping. The paper's practical unconstrained SVD update is retained; any
residual increase is logged. Empty support gives an exact zero reconstruction,
not a solver failure. Unsupported-background energy and support counts are
logged, rather than concealed with an added background dictionary.

SLiCS's primary object readout uses independent per-concept NNLS projection
(Eq.3), the paper's concept-filtered readout. Eq.2 joint decomposition is a
separate sensitivity on the SAME frozen dictionaries. Test masks never enter
either solver. The concept queried is fixed by its category slot, just as
GGF's category readout is fixed. Coefficients are never pruned to Top-32.

Reconstruction VE always uses the joint NNLS solution. Independent partial
concept projections are not additive and are NEVER summed to claim a full
reconstruction. Partial-readout object metrics and joint-reconstruction VE are
explicitly named as two native SLiCS evaluation operations. Object-readout
activity and joint-reconstruction activity are kept separately.

NNLS uses up to2,000 FISTA steps, a per-sample projected-gradient KKT tolerance
of1e-5 verified in float64 against the original objective, and a verified
float64 SciPy active-set fallback with20,000-iteration limit. Nonconvergence
raises; it cannot count as a successful result. Independent concept solves
are vectorized without changing their objectives. Training removes globally
forbidden dictionary columns before forming a Gram matrix. All fallbacks and
atom-update diagnostics remain in saved records. No regularization is added.

## Fresh evaluation and uncertainty

The new cohort has1,500 images and all75 predeclared categories. Candidate
selection uses masks/coverage only. A freeze gate checks every seven-seed
model/selector/readout and all saved anchor weights BEFORE extracting any
features. All methods then receive exactly the same images, patch activations,
normalization, capture definition and AP definition. Training-only readouts
are frozen; no classifier/readout is fitted on the holdout.

Co-primary contrasts are GGF-75 minus object-selected SLiCS-75 partial readouts
for capture and patch AP. Report paired seven-seed t intervals and two-sided
p-values, with Holm adjustment across these two tests. Reconstruction VE is
a reported tradeoff metric, not a hidden eligibility filter. Also report
fixed-budget-final, MSE-selected and joint-readout sensitivities without
selectively choosing the most favorable result. The first three starts and
four additional starts are summarized separately as well as pooled.

Provide1,000 paired whole-image bootstrap replicates per seed for the primary
and fixed-budget partial-readout contrasts. These intervals condition on the
fitted models. Seed intervals condition on the evaluation cohort; neither
captures all dataset or researcher-selection uncertainty. Three-seed larger-GGF
anchors are explicitly separated from seven-seed matched methods.

## Resume, source protection and outputs

Outputs: MyDrive/GGF_COCO_supervised_comparison_v1. Historical models, current
BSF tuning stores and their sources are read-only. No other study is restarted.
Run one writer at a time. The notebook defaults to completing the entire study,
without a wall-clock stopping limit. An optional development-only stage can
produce preliminary validation evidence while leaving test features closed;
switching back resumes the same run identity.

SGD model/Adam/controller/head/noise states save locally every50 updates and
to verified Drive generations every250, at phase boundaries and on graceful
Stop. Selected candidates save independently. Native SLiCS stages a full atom
sweep before committing; partial atom updates cannot corrupt committed weights.
Abrupt runtime loss can replay up to249 completed updates; completed fits and
audits are reused. Interrupted current audits restart their bounded operation.
Local duplicate checkpoint files are evicted ONLY after their hashes match
verified Drive generations. Drive checkpoints are never deleted.

The ZIP contains protocols, exact source, saved input pins, full development
histories, solver/support diagnostics, per-category scores, all model/selector
rows, uncertainty tables, plots and hashes. Model tensors and feature caches
remain in their stores. CPU tests cover numerical and integration logic; full
T4/Drive execution is still required. No notebook can guarantee an empirical
advantage or CVPR acceptance. Supported claims must match the observed results.
