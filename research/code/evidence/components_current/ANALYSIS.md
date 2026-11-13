# Completed GGF component ablation

Verified on October 15, 2026. All results below are means across seeds 21, 22, and 23 on the same previously inspected 1,500-image test cohort.

## Configuration and verification

Each arm uses 4,096 groups, 12 directions per group, Top-32 selection, and 10,640 model updates (7,640 reconstruction warm-up plus 3,000 continuation updates). Adaptive GGF uses a 0.60 cap. Capacity, active-coordinate budget, update counts, and sampled image/noise streams are matched; computation in FLOPs is not matched.

The completed archive passed ZIP integrity checks and all 38 recorded artifact SHA-256 checks. The analysis independently recomputed the four arm means from 12 seed endpoints, checked 900 category rows, and verified nine paired image comparisons with 1,000 valid bootstrap replicates each and no invalid replicates. The saved warm-up and fixed training-selected readout identities agree between the mask-initialized arms within each seed. Six existing GGF/Vanilla endpoint results were reused; six new endpoints supplied the mask-initialization and fixed-guidance controls.

## Test means

| Arm | Capture (%) | Patch AP (%) | Reconstruction VE (%) |
| --- | ---: | ---: | ---: |
| Vanilla BSF | 21.37 | 51.66 | 80.35 |
| Mask initialization + reconstruction only | 23.56 | 68.55 | 79.56 |
| Mask initialization + fixed-weight guidance | 42.25 | 74.63 | 77.33 |
| Mask initialization + adaptive guidance (GGF) | 42.77 | 74.76 | 76.94 |

Higher is better for all three metrics. Differences below are percentage points, computed before rounding.

| Sequential change | Capture gain | Patch AP gain | VE change |
| --- | ---: | ---: | ---: |
| Add mask initialization | +2.19 | +16.89 | -0.79 |
| Add fixed-weight object guidance | +18.69 | +6.08 | -2.23 |
| Replace fixed with adaptive weights | +0.52 | +0.13 | -0.40 |
| Full GGF versus Vanilla | +21.40 | +23.10 | -3.41 |

## Interpretation

Mask initialization provides most of the patch-detection improvement along this ablation sequence. Object-guided learning provides most of the capture improvement: adding fixed guidance improves mean capture for all 75 categories. This supports the central explanation that reconstruction alone does not consolidate object evidence into the measured groups, and object supervision substantially improves that alignment.

Adaptive weighting is a smaller refinement. It improves capture in all three seeds and in 71 of 75 categories when category scores are averaged across seeds. Its paired three-seed 95% t interval is +0.29 to +0.75 capture points. Patch AP improves numerically in all three seeds, but its corresponding interval is -0.02 to +0.28 points, so this experiment does not establish a reliable AP advantage over the tested fixed weights. These seed intervals are distinct from the separately saved within-seed image-bootstrap intervals.

Fixed guidance already achieves approximately 97.6% of the full capture gain and 99.4% of the full AP gain over Vanilla along this sequence, while retaining 0.40 additional VE points. These percentages describe this particular ablation path, not independent causal shares. Adaptive GGF remains the numerical best for both object metrics; fixed guidance is a credible simpler alternative.

The fixed weights are derived only from training records: each seed uses the arithmetic mean of six logged adaptive-weight vectors at updates 500 through 3,000. They match the observed average scale of guidance; they are not an independently optimized fixed-weight baseline or an exact average over every training update.

## Reconstruction and scope

The gains have a consistent reconstruction cost. Vanilla and mask-initialized reconstruction-only controls pass the original development-MSE allowance in all three seeds; neither guided arm passes it in any seed. The result should therefore be presented as an object-alignment versus reconstruction tradeoff, not as an improvement satisfying the original reconstruction guardrail.

The experiment explains the existing headline GGF scores rather than raising them. It uses three seeds and the previously inspected test cohort, not a fresh final holdout. It does not supply the pending seven-seed comparison or current-configuration Grassmannian and Group Lasso baselines. Guidance jointly includes contrast and detection losses, so these arms do not isolate their individual contributions.

## Evidence

- `ANALYSIS.json`: recomputed metrics, per-seed gains, intervals, category counts, and scope.
- `analyze.py`: independent verification and aggregation script.
- `test_means.csv`, `test_models.csv`, `test_categories.csv`: saved endpoint summaries.
- `component_paired_images.csv`, `component_paired_seeds.csv`: paired uncertainty estimates.
- `components_frozen.json`, `protocol.json`, `fixed_weight_policy.json`: frozen lineage and comparison policy.

Source archive SHA-256: `2c2ec70992d9823acc6f079c20731d0e183a917f6cc7fcd8752648a5ba06307c`.
