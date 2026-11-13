# AGF-X quick test from final ordinary 5D AGF-G

Select a Tesla T4 and click Run all. Authorize Google Drive at the beginning.
No HF token, dataset download, backbone extraction, optimization, warm-up,
guided continuation, or control retraining is called. The notebook embeds its
complete implementation and verifies the source archive and every file.

## Pinned start and controls
Use all seeds 21, 22, and 23 from the completed ordinary 5D continuation run:
371d2d1d81ea6c9ce15fa17a6cdecc716f42304e385ac0cf9d2f3280572d6a0f.
Source SHA256:
4e2f00da8b218512d5a0d06241176819e14f884eeca1ac3269977a057594858b.
Folder: LOCAL_INPUTS/AGF_COCO_5D_continuation_v1.
The source AGF-G test means are 41.04% capture, 75.01% patch AP, and 66.04% VE.
Original three-seed controls and cached feature identities are verified,
not chosen by latest folder or modification time. Missing/corrupt/incomplete
inputs stop; there is no retraining or re-extraction fallback.

All models retain 1,024 groups, five directions per group, Top-16 gating,
5,120 directions, 80 available coefficient slots, 7,869,440 core parameters,
150 trained detection-head parameters, and the original 13,640 scheduled
training updates. The saved adaptive cap remains 0.40; heads and controller
are frozen during Exchanges. No new gradient updates occur. Structural
snapshots are inference checkpoints with source optimizer/controller hashes;
this notebook does not claim to resume guided training from an exchanged
optimizer state.

## One bounded exchange pass per seed
Choose 512 TRAIN images with the existing mask-coverage allocator and fixed
seed 107011, requiring at least three usable images for every declared category.
No development/test outcomes determine those images. Profile all 5,120
directions on this screen. Width-aware additive contrast algebra supplies
at most 64 candidate pairs, using up to five incoming directions per target
and a coverage-broadening candidate-pool rule from the earlier implementation.

At the current starting model, measure fixed-gate single-direction removal
effects on capture, AP, and reconstruction for directions in that pool.
Use measured capture/AP dependencies to rank a shortlist of eight pairs.
Reconstruction removal importance is descriptive, since exchanged directions
stay in the dictionary. Fixed-gate removal and additive algebra are ONLY
shortlisting approximations; all candidate predictions use the actual new gate.

Score those eight individually on TRAIN, then freeze the order by actual
standalone mean capture gain, target gain, and proposal ID. Make at most
eight cumulative attempts and accept at most five nonconflicting exchanges.
No target category or donor/receiver group participates twice after an accepted
exchange. Rejections reserve no groups and leave the current model unchanged.
There is no search reranking based on development or test outcomes.

Exchange the encoder column, bias entry, and decoder row together for two
slots in different groups. Verify exact element-value preservation and the
slot-to-original-direction permutation after every accepted edit. Trained
heads, group readouts, capacity, and Top-16 rule stay fixed. Gate membership
may change, which is why all metrics must be recomputed for actual swaps.

## Acceptance checks
TRAIN must improve both target-category and mean capture by more than 1e-10.
Both existing 400-image development sets must improve mean capture and pass
the earlier preservation checks against both the current model and the
unchanged starting AGF-G: macro AP and VE may fall at most 0.001 (0.1 pp),
MSE may grow at most 2%, and every category's capture/AP may fall at most
0.01 (1 pp). These are protocol tolerances, not significance thresholds.

Additionally, selection-split MSE must remain <=1.02 times the SAME original
native-reference development MSE saved for that seed. Never apply that
selection-split threshold to a different split's MSE or to test VE.
The second development set uses relative preservation checks; its scores
are selection evidence because each attempted exchange is checked there.
All categories and all seeds remain in the results, including zero acceptances.

## Evaluation
Freeze and verify all three AGF-X endpoints BEFORE opening test feature shards.
Reuse saved AGF-G and continued 5D Vanilla test reports and compact arrays;
do not refit readouts or rescore unchanged controls. An unchanged AGF-X seed
also reuses its exact AGF-G report. Only genuinely edited models need new
1,500-image test inference. The test cohort has been inspected previously.
No post-test rollback or new search is allowed.

Save per-seed and three-seed mean capture, patch AP, VE and MSE, original
native eligibility, accepted counts, all decisions, full category outcomes,
parameter/readout identities, and paired whole-image bootstrap intervals
(200 draws). These intervals condition on fitted models and images; they do
not quantify training-seed or researcher-selection uncertainty or correct
multiple comparisons. This quick test makes no image-classifier or downstream
improvement claim. It tests one bounded search policy, not the optimality of
Exchanges or an independent generalization claim.
Unchanged checkpoints have exact zero difference intervals without bootstrap
draws; this identity shortcut is explicitly recorded, not reported as 200 draws.

## Resume and outputs
Every profile shard, graph shard, completed score and exchange decision is
saved with SHA256 read-back verification to Drive. An accepted snapshot is
committed before its decision row. Run all unchanged after a runtime reset:
completed shards, decisions, frozen models and audits are reused. An interrupted
unit may be repeated, but completed training is never repeated. Do not run two
copies of this notebook against the same output simultaneously.

Output: LOCAL_INPUTS/AGF_COCO_5D_exchanges_v1.
Partial ZIPs are saved after each completed seed; the final compact results ZIP
is verified and saved before automatic download. Numerical evidence/source
are in the ZIP; large model/activation files stay on Drive. Initial Drive
authorization may require interaction. Nothing later asks for an HF token.

CPU synthetic tests verify width indexing, additive/removal algebra, actual
gating, reversible value-preserving swaps, cumulative guards, partial-commit
resume, freeze barriers, reused-control evaluation, and export hashes. The
full-size cached-data T4 run is executed by Colab, not claimed as run locally.
