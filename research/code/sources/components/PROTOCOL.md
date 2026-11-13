# GGF: three-seed component ablation and current Figure 1

Select a T4 and click Run All. Plan for roughly 6–8 T4 hours; Drive delays can extend this. There is no time limit or larger study after completion.

## What runs

The current plant/vase illustration is regenerated first from the completed seed-23, 4,096-group, 12D, Top-32, cap-0.60 GGF endpoint. This is inference only. Its PDF, PNG, measured arrays, projection audit and SHA manifest are saved separately in `ggf_figure1_current.zip` before training. The photograph is the previously selected COCO image 553000. The clouds use a fixed training cohort. Three principal components of decoded contributions are displayed; their retained variance is reported. The same PC-to-RGB function colors each group's cloud and image overlay. Only display fields are smoothly interpolated. No points are invented or moved to make the clouds tighter.

Then six new 3,000-update fits run, in seed order 21,22,23:

| Arm | Initialization | Additional learning | Work |
|---|---|---|---|
| Vanilla BSF | Saved random-initialized reconstruction warm-up | Reconstruction only | Reuse saved endpoint and test audit |
| Mask + reconstruction | Saved mask-initialized warm-up | Reconstruction only | 3 new fits |
| Fixed-weight guidance | The same saved mask-initialized warm-up | Reconstruction + fixed object losses | 3 new fits |
| GGF | The same saved mask-initialized warm-up | Reconstruction + adaptive object losses, cap0.60 | Reuse saved endpoint and test audit |

All models have 4,096 groups, 12 directions per group and Top-32 gating. Each has 7,640 warm-up updates plus 3,000 continuation updates, totaling 10,640. The additional phase uses fresh Adam and fresh zero-initialized detection heads where applicable, matching the saved pilot. Its learning rate is0.0003. The six new fits use exactly the saved pilot's image and noise streams, and their hashes must match. Capacity, update counts and inputs match; supervision and computation differ. The study does not claim equal FLOPs.

## Fixed weights use training records only

For each seed, freeze the arithmetic mean of the adaptive model's weights USED at the six already logged updates500,1000,1500,2000,2500,3000. Reconstruction weight remains1. Validation or test scores are never consulted. This is the mean of six logged training-weight vectors, **not** the exact time average over all3,000 updates. The source did not save every controller weight. This control compares the adaptive schedule with a constant weighting at its logged average scale. It does not compare independently tuned fixed weights or prove that adaptation beats every possible constant weighting. The policy and its source hashes are saved before any new fit.

## Evaluation and uncertainty

Freeze all12 endpoints before new test inference. Import the six completed Vanilla/GGF compact audits unchanged; score only the six new models on the same previously inspected1,500-image test cohort. Carry the saved TRAIN-selected readouts unchanged. No new readout fitting, backbone extraction, HF token, fresh holdout, Exchanges or seven-seed run occurs.

Report all three-seed means, seed SDs, per-category values and original native-warm development reconstruction-error eligibility, including failures. For Initialization, fixed object guidance and adaptive-vs-fixed scheduling, compute paired image bootstrap intervals with1,000 replicates per seed, plus separately identified paired seed t-intervals. The test cohort has already been inspected, so these are component diagnostics rather than a new independent final holdout.

## Resume and outputs

Sources are read-only. New checkpoints and results live in `MyDrive/GGF_COCO_components_v1`. Checkpoint identity includes sources, policies, training/dev identities and the numerical environment. Completed fits and audits are reused. Local state saves every100 updates; durable Drive state every500, and at evaluation/end boundaries. Use Colab's stop-cell button and wait for the verified save/partial-evidence message. A graceful stop saves the latest committed update, including Adam, detection heads and RNG. Abrupt runtime loss can lose up to499 updates since the last verified Drive checkpoint. Resume by Run All with unchanged settings. Do not delete source folders or alter settings between resumes.

The result ZIP contains summaries, protocols, figures, source archives and hashes. Large model/optimizer/feature tensors remain in their checkpoint stores rather than being duplicated in the ZIP. The source inventory and CPU checks run before any fitting. GPU execution still has to occur in your Colab runtime.
