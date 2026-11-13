# GGF: Manifold Zoo first, then fixed-budget baseline confirmation

Select **T4**, stop other writers to the source studies, and click **Run All**.
Grant Google Drive access at setup. COCO authentication, if needed, is checked
after the synthetic stage; `HF_TOKEN` comes from Colab Secrets.
There is no timer, automatic early stop, architecture sweep or additional
component training. This is the full confirmation study, not a short pilot.

## First: synthetic Manifold Zoo

Run the earlier reconstructed independent-factor Zoo before loading COCO
features or DINO weights: 128 input coordinates, 128 factors, 256 groups,
four directions per group and Top-4 selection. Use the same 300,000 training,
20,000 selection, 20,000 calibration and 100,000 test samples as the archived
BSFs. Read the original saved dataset from its checksum-verified Drive
artifact when available; otherwise regenerate with the unchanged sampler.
Every split must match the old array fingerprints exactly. If platform
rounding prevents that and the original dataset is absent, restore the
original artifact; the notebook will not silently compare different data.

Reuse the nine saved released-BSF results, with zero baseline refits. Train
GGF at seeds 0, 1 and 2 with the original two learning rates (0.0004, 0.003):
six small fits. Each has 200 epochs of floor batches, or 29,200 Adam updates,
matching the old total budget. The first 40 epochs are reconstruction warm-up;
160 epochs add guidance. OneCycleLR spans all 200 epochs. Initialization uses
the leading four eigenvectors of observed training-mixture second moments
conditional on each known factor's presence; it does not use isolated true
component vectors to initialize the dictionary. The first 128 groups have
fixed factor assignments and the remaining 128 are reserve groups.

This is a **factor-supervised synthetic analogue** of object guidance.
Known component contributions replace mask-derived object contrasts in the
supervised normalized contribution-error term, and balanced factor-presence
classification replaces patch detection. Component error is equally averaged
within each factor and then across factors represented in the batch. Factor
labels and component targets are a stronger supervision source than COCO
masks; this experiment is a recovery ceiling check, not an identical COCO
protocol. The existing first-order controller adjusts both supervision
weights every ten guided updates, with cap 0.60 and reconstruction weight one.
Training input is noiseless, as in the earlier Zoo experiment.

Select each seed's learning rate and checkpoint by selection reconstruction
MSE before opening synthetic test arrays for model evaluation. Report the
same trial's final endpoint separately as a diagnostic; a selected checkpoint
may precede completion of guidance, and its update count stays visible.
Compare calibration-locked one-group factor R2, paper-style test-mask-matched
R2, reconstruction VE, and recovered-factor counts. Additionally report
fixed-assignment factor R2, factor-presence AP and breakdowns by geometry
for GGF. Factor-presence AP is not COCO patch AP. Archived baselines have no
saved fixed-assignment/AP measurements, so those cells remain absent.

Perfect summation of the known true components checks the metric's arithmetic
ceiling. **It is not a learned result or a guarantee of 100% recovery.** A
100% factor count above R2 > 0.8 is distinct from mean factor R2 = 100%.
Historical baseline checkpoints were selected by the same development-MSE
policy; the supervised model uses additional labels, heads and computation.
The historical Torch CUDA build differs (cu128 versus the COCO environment's
cu130), and that limitation is recorded with the imported evidence.

Synthetic results and their source ZIP save separately under
`LOCAL_INPUTS/GGF_Manifold_Zoo_v1`; a comparison table prints before COCO begins.
They are also copied into the final confirmation ZIP. Re-running unchanged
reuses completed synthetic fits and audits. No COCO holdout opens in this stage.

## Fixed COCO configuration

Every final model has 4,096 groups, 12 directions per group, and a target
activity of 32 groups per patch. GGF uses the existing adaptive controller
with cap 0.60. Each receives **7,640 reconstruction warm-up + 3,000 additional
updates = 10,640 total**, matching the current paper. Group Lasso uses its
own activity mechanism; measured sparsity mismatches remain visible.

The current recipe is fixed before this run. It is a useful previously
explored configuration, not an established global optimum. There is no
1,024-versus-4,096 selection, new cap trial, longer continuation, Exchanges,
PCGrad, seven-seed component ablation, rank sweep or MDL computation.

## Run order

0. Complete the synthetic Zoo stage above and display its comparison.
1. Verify pinned source/numerical versions, cached features, all six saved
   pilot identities and the completed twelve-endpoint component catalog.
   Import the completed component results unchanged as historical evidence.
   Validate cached DINO weights by replaying one **training** image, reserve
   the annotation-only holdout plan, and run disposable full-batch T4 checks.
2. Fit the missing Grassmannian and Group Lasso families first, in released
   and paper-specified-adapter forms. Then finish paper/released Vanilla
   tuning. Each method has three additional-phase learning-rate candidates
   (0.0001, 0.0003, 0.001), evaluated on seeds 21, 22, 23. Warm-up always
   uses 0.0003 and is shared across that method/seed's candidates.
   The three paper-Lasso rates pair with penalties 0.0001, 0.001, 0.01;
   this is three declared joint candidates, not a nine-combination search.
3. Select each baseline by mean development MSE, requiring Lasso activity
   within 20% of 32 **in every seed** when feasible. If no candidate qualifies,
   keep the minimum-MSE result and flag its mismatch. Save the three-seed
   development comparison before starting the four additional seeds.
4. Complete seeds 21--27 for GGF and all six baseline implementations:
   **49 final endpoints**. Compatible seeds 21--23 and selected tuning
   endpoints are reused. Seeds 24--27 are additional training starts.
5. Freeze every final endpoint and TRAIN-selected readout. Only then extract
   features and score the unused 1,500-image coverage-enriched holdout.
   Report all categories, mean capture/AP/VE, seven-seed variation, paired
   seed intervals and 1,000 paired whole-image bootstrap draws per comparison.
6. Measure synchronized forward latency, peak allocated GPU memory and
   parameter counts on cached TRAIN features. Export the verified result ZIP.

There are **54 development candidate endpoints** (six baseline implementations
times three settings times three seeds), followed by the 49-endpoint final
catalog. Selected candidates are the final endpoints for their first three
seeds; they are not fitted again. Existing compatible candidates skip or
resume. If development selects a different Vanilla learning rate, the saved
0.0003 control remains in the tuning evidence, but the selected final Vanilla
endpoint must use the chosen setting. No incompatible result is substituted
merely to avoid training.

## What is reused and protected

Source checkpoints remain read-only in `AGF_COCO_scale_study_v1`; completed
component records remain read-only in `GGF_COCO_components_v1`. Cached data,
normalization, mask-derived initialization, category readouts, DINO weights,
and compatible warm-ups/continuations are reused. Different baseline families
still require their own warm-ups when no compatible fit exists.

New checkpoints, reports and any newly extracted holdout features live in
**`GGF_COCO_baseline_confirmation_v1`**. Original artifact keys such as `agf_g`
remain unchanged for compatibility; displayed method names use GGF. The six
saved pilot endpoints are checksum-pinned and cannot be overwritten. An
advanced checkpoint beyond the requested 3,000-update endpoint triggers a
restore error rather than silent retraining or a longer-budget substitution.

The pinned core's internal fit target remains 12,000 solely for checkpoint
identity compatibility. The worker always passes an explicit **3,000-update
stage limit** and verifies every final model has 10,640 total updates. It
never executes the old 12,000-update additional phase.

## Fairness and interpretation

All methods use the same cached normalized activations, fixed group width,
sampling/noise streams and update budgets. Parameters, supervision and FLOPs
differ. Baseline selection uses development reconstruction, not test object
scores. Group Lasso's actual activity is reported, including failures to
match the target. Original reconstruction-error eligibility is reported;
the existing GGF violations are not hidden or used to drop endpoints.

Released BSFs use the pinned unmodified public implementations. The separately
labelled paper-specified adapters retain their documented choices for decoder
normalization, Grassmannian scaling/projection, AuxK, and Lasso shrinkage.
These are adaptations under a common COCO budget, not exact replications of
the authors' original ImageNet schedule or unpublished checkpoints.

The first three seeds have already informed development; the four additional
seeds are reported separately as well as pooled. The untouched holdout is not
used to select settings, readouts or checkpoints. The completed component
ablation uses the earlier inspected test cohort and is kept separate from
the final holdout results. No recursive-generalization claim is tested here.

Seed intervals describe training-start variation conditional on the cohort;
image intervals describe paired image sampling conditional on fitted models.
Neither includes researcher-selection uncertainty. Six GGF-versus-baseline
contrasts are declared; intervals are reported without multiplicity correction.

## Resume and download

Model state includes Adam, controller, detection heads and sampling/noise RNG.
Local checkpoints save every 50 updates; verified Drive generations save
every 250 and at stage boundaries. On Colab's stop-cell button, wait for the
verified checkpoint/partial-ZIP message before disconnecting. A graceful
stop preserves the latest fully committed state; an abrupt runtime loss may
replay up to 249 updates. Stop during setup/mapping or bootstrap work may
repeat the unfinished bounded operation. Run All unchanged to resume.

After a completed model's development audit, local model/warm-up duplicates
are evicted only when their payload hashes match verified Drive generations.
This prevents retaining the entire seven-seed suite on the T4 runtime disk.
Drive checkpoints remain intact and are loaded again when needed.

The final download cell saves `ggf_confirmation_<run>_results.zip` to the
browser. The same file is already in the new Drive output directory. It
contains numerical reports, protocols, plots, source archives and SHA-256
hashes. Large checkpoint, image, backbone and activation tensors remain in
their stores and are not duplicated into the ZIP.

CPU tests validate logic and small-model execution. Full T4 performance,
Drive behavior and completion still require execution in Colab.
