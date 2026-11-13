# Object-only development evaluation revision

This is an explicit scientific revision of the preceding supervised comparison.
Both methods retain the same training arithmetic, data, annotations, dictionary
capacity, seeds and update budgets. SLiCS is adapted to PATCH inputs; it is not
the original image-level evaluation. GGF has an untied encoder and detection
heads: dictionary capacity and image schedules are matched, not FLOPs or total
parameter counts. This notebook does not fit any unsupervised BSF controls.

Routine SLiCS validation computes only Eq. 3 partial-concept capture and patch
AP. It never calls the joint reconstruction solver. The shared selection score
is capture + 0.2 AP; exact checkpoint ties retain the earliest evaluated update;
learning-rate ties retain the declared grid order. The minimum-MSE selector and
secondary joint object readout have been removed. GGF's cheap reconstruction
diagnostics may still be printed but are not used for selection or tie-breaking.
SLiCS evaluates update 0, every 500, and the 7,640 / 10,640 boundaries. GGF
evaluates guided update 0, every 500, and 3,000 after its shared 7,640 warm-up.
Initialization and training paths differ and selected update counts are reported.
GGF screens three predefined Adam learning rates on training orders 21–23;
SLiCS has no Adam learning rate and follows its fixed alternating-update recipe.
No claim is made that tuning searches or end-to-end computation are identical.

The original journal remains read-only. Checksummed compatible training prefixes,
optimizer/controller/RNG states, initializers and targets can be copied into a
new run identity. Old rows, settings, MSE-selected models, freeze records, test
reports and final conclusions cannot be imported. Native SLiCS training identity
is explicitly checked against the pinned predecessor protocol before migration.
Previously computed object-only selection evidence remains valid; no old MSE
tie-break is carried into the learning-rate selection. A pending validation
resumes with the cheaper schedule; no training reset is required.

Run order: seed 21 SLiCS, all three GGF LRs; then 22, then 23; freeze the LR;
finish four additional training orders; freeze selected and fixed-budget models;
only then extract a fresh 1,500-image holdout and compute final metrics and CIs.
The original 1,500-image annotation-only reservation is retained, after pinned
identity, all-75 category quotas, prior-split disjointness and unopened-status
checks. A reserved cohort has not been consumed by testing. It remains excluded
from training and tuning, and excludes the separate BSF study's cohort. The
predecessor must have no all-model freeze, extracted test shards, audit files or
test events. Any such evidence stops execution. No testing is used to select models. Seven-order paired t
intervals and Holm correction for the two primary metrics are retained, as are
whole-image bootstrap intervals. Order variation is conditional on a common SVD
initializer; these are not seven independent initial geometries.

Final reconstruction uses strict joint NNLS with the unchanged tolerances and
verified CPU fallbacks. Partial projections are NEVER summed as reconstruction.
Final joint passes are resumable per image (local) and every four images (Drive),
with forced Drive save on graceful interruption. Duplicate weight/cohort pairs
reuse the exact completed reconstruction. Object inference is cached before
starting reconstruction, so an interrupted reconstruction does not restart it.
The final joint solve can still be slow. This revision reduces repeated work;
it is not a new NNLS solver or a promised end-to-end runtime improvement factor.

Historical protocol text is retained in LEGACY_* files for provenance only.
The operative machine-readable protocol is protocol.json in the new run.
