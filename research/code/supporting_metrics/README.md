# Definitions and evidence for retained spectral diagnostics

This note supports Figure S6(b), labeled `tab:headroom`, and the entropy-rank
columns of Table S27, labeled `tab:voc-geometry`. Labels are the stable identifiers
if numbering changes. It restores the methods and provenance needed for those
items without restoring obsolete localization galleries or historical rank tables.
All paths in this note are relative to `code/`.

## Table S27: coefficient and contribution entropy rank

The VOC analysis uses the fixed training-image cohort and the first
training-selected Capture group for each of the 19 categories. The original
`geometry_consistency` profile's `analysis.py` contains `selected_indices`,
`rank_stats`, and `ranks`. Its source identity is recorded in
`frozen_sources/profiles.json`; its exported SHA-256 is
`4b154a2af16b36a99b0ecdebe616fc73932cf89a17da358a9c73d36748100bf6`.
The cohort consists of up to 256 training images selected by the smallest
SHA-256 hashes of `ggf-common-cohort-` followed by the integer image ID, with
the selected indices then sorted. The saved VOC records each contain 256 images.
This selection does not use test images or model scores.

For group coefficient vectors z_i retained by native gating, the analysis counts
an active patch when the coefficient norm exceeds 10^-6. For n active patches,
let z_bar be their mean and define the centered sample covariance

    C = sum_i (z_i - z_bar)(z_i - z_bar)^T / (n - 1).

Let lambda_j be the nonnegative covariance eigenvalues. Numerical roundoff below
zero is clipped to zero. With p_j = lambda_j / sum_h lambda_h, entropy rank is

    E(C) = exp(-sum_{j: p_j > 0} p_j log(p_j)).

This definition uses covariance eigenvalues, not their square roots. It is the
spectral-entropy construction of Roy and Vetterli (2007), “The effective rank:
A measure of effective dimensionality,” applied to the covariance matrix.
It uses natural logarithms. With zero covariance trace the rank is undefined;
fewer than two active samples cannot estimate the covariance. These states are
retained in the source records rather than assigned an invented rank.

For decoder D with directions in its columns, decoded contributions have
covariance D C D^T. Their entropy rank uses that covariance's nonzero eigenvalues.
The implementation evaluates the equivalent small matrix
C^(1/2) D^T D C^(1/2), avoiding a full activation-space eigendecomposition.
Coefficient and contribution rank can differ because the decoder directions
need not be orthogonal. The archived spectra additionally retain participation
ratio, sum(lambda)^2/sum(lambda^2); coefficient-matrix stable rank,
sum(lambda)/max(lambda); and covariance stable rank,
sum(lambda^2)/max(lambda)^2. These are distinct statistics.

`evidence/voc_heldout/selected_ranks.json` contains all 19 selected category rows
for each of three methods and seeds 21, 22 and 23. Both spectra, active counts,
group IDs and cohort IDs are preserved. Its original exported SHA-256 is
`c18ea8b390ca4cd8a74ab7163b260ec2088c648731f88a66a752f33f19404511`.
Average entropy ranks equally over the 19 category assignments within each
model, then report the mean and sample SD (n-1 denominator) of those three
run-level values. Every selected record in this table has an estimable,
nonzero-variance spectrum. Category assignments retain their multiplicity if
two categories select the same group.

These ranks describe variation among active responses. They do not measure
semantic coherence, count independent concepts, establish intrinsic manifold
dimension, or establish minimum-description-length efficiency. The off-target
column of Table S27 is a separate fixed-probe score-change measure, defined
in the Appendix's removal-control protocol.

## Figure S6(b): geometric headroom and training-basis transfer

For one category, X contains regional object-minus-comparison contrast vectors
as rows, using the original 2,000-image training and 400-image development
cohorts. The contrast construction is the one defined in the paper. Its
implementation is `contrasts` in `training/core/coco_ggf/wider_groups.py`.
No across-object mean subtraction or coordinate-wise whitening is performed.
For width b, the best orthogonal projection's retained energy on that same
finite sample is

    U_b(X) = sum_{j=1}^b lambda_j(X^T X) / trace(X^T X),

with eigenvalues in descending order. Compute this separately on the training
and development contrasts. These are the training-fitted/training-scored and
development-fitted/development-scored columns. The latter uses the evaluation
sample to fit the subspace, so it is an optimistic finite-sample bound.

For transfer, fit the orthonormal leading training eigenvectors Q_train,b, then
evaluate them on the category's development contrasts X_dev:

    V_b = ||X_dev Q_train,b||_F^2 / ||X_dev||_F^2.

The figure's training-fitted/development-scored column is this transfer value.
Each displayed column averages the category-wise fractions equally over all
75 categories, then multiplies by 100. It does not pool category energies.
Widths are 3, 4, 8 and 16. This is one saved geometric audit, not a multi-seed
method-comparison result; no training-seed SD is claimed for the panel.

`wider_geometric_headroom.json` is the byte-for-byte restored original
`geometric_headroom.json` export from the width/activity study
`367b64ceef5d`. It contains the 75 categories, 300 category-by-width rows,
support/energy totals, all macro values and the original interpretation.
Its SHA-256 is
`6eeb9c15a1772f9f326263d211a303c9858047a70e4972fbaea72a522a3a268b`.
The retained original calculation is `spectrum` and `headroom` in
`training/core/coco_ggf/wider_groups.py`, SHA-256
`a2236b19e2167396a71cb4e3b8861034ec10fc2c91c7d7eee891cf346e4aeef9`.

The bound ignores the learned gate, encoder, shared dictionary, joint
reconstruction and optimization. A development-fitted projection can retain
100% when the available sample rank is no larger than b. Neither that bound
nor the training-basis transfer score establishes population headroom,
semantic coherence, or a jointly attainable gain in Capture, AP and VE.

## Offline verification and limits

From the extracted bundle root, run:

```sh
python scripts/verify_bundle.py
```

This now also runs `code/supporting_metrics/verify_support.py`. Its default
recount uses only Python's standard library. It recomputes all 342 saved
coefficient/contribution spectral summaries, all Table S27 entropy-rank means
and sample SDs, and all 12 headroom macro values from the 300 category rows.
It verifies the source hashes and checks support, bounds and nested widths.
To exercise the original headroom function on a small known matrix pair,
without importing the training stack, install NumPy and SciPy and run:

```sh
python code/supporting_metrics/verify_support.py --check-source
```

The saved records support the numerical recount without GPUs or source-image
caches. Re-extracting rank spectra or headroom from original activations still
requires the recorded datasets, normalization, source activations, masks and
checkpoints as appropriate. This restoration does not rerun those experiments,
alter their results, or repeat any GPT/Gemini review.
