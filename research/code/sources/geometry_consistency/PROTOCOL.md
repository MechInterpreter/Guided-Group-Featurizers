# GGF: supervised controls, VOC training and consolidation evidence

Select **T4**, mount the same Google Drive, then **Run All**. Three paired seeds: **21–23**. Live cell output and 30-second parent heartbeats; partial evidence saved between stages; **no time cutoff**.

This broader run includes new VOC training, interpretation, and six fully trained Concept-SAE models. It is substantially longer than the earlier comparison-only notebook and is not promised to finish today. Urgent matched comparison and analysis are saved before supporting stages. Colab execution remains necessary.

## Primary comparison

| Arm | Initialization/warm-up | Continuation | Heads | Weighting |
|---|---|---|---|---|
| Vanilla BSF | Random; reconstruction warm-up | Reconstruction | None | Reconstruction only |
| Supervised Vanilla | Same random warm-up | Reconstruction + contrast + detection | Same scalar heads as GGF | Fixed, development-selected |
| GGF | TRAIN-mask initialization; reconstruction warm-up | Reconstruction + contrast + detection | Same scalar heads | Existing adaptive controller |

All use 1,024 groups,12 directions/group, Top-16, Adam LR 0.0003,7,640 warm-up +3,000 continuation updates, eight images/update and matched within-seed image/noise streams. GGF retains cap 0.40. **Fixed supervision has no adaptive cap.** Heads contain two learned parameters/category:150 on COCO; twice the retained category count on VOC. The same TRAIN-only mapping policy searches all groups after warm-up; mappings stay frozen during continuation. No hard category ownership.

Fixed competitor candidates: contrast/detection coefficients **(0.05,0.01), (0.20,0.20), (0.40,0.40)**, reconstruction weight 1. Run all three at full budget for seed 21. Select maximum development capture+0.2AP within random-warm development MSE+2%; if none pass, select minimum development MSE and disclose failure; ties prefer earlier candidate. Reuse the winning endpoint and apply that same pair to seeds22–23. Report all candidates. This is three-point calibration, **not optimal fixed weights or search-budget matching to GGF's historical exploration**.

## Priority order and reuse

1. Verify immutable source manifests, code/numerical pins, train/dev identities, checkpoints.
2. Finish the COCO primary comparison, freeze all 9 endpoints, evaluate, and complete consolidation/rank/intervention analyses. Save an evidence ZIP. Load image metadata and produce direction atlases.
3. Train the same three primary arms **from scratch on VOC**, using COCO-chosen settings, then evaluate and analyze them.
4. Train **released Concept-SAE**, all three native stages for three seeds on both datasets.
5. Complete supporting **unsupervised** released/paper-equation Grassmannian and Group Lasso on COCO, with the same BSF interpretation audit.

Total:18 primary BSF endpoints,12 supporting BSF endpoints,6 Concept-SAE endpoints, plus4 nonwinning fixed-weight fits across the two datasets. Exact compatible saved BSF work reduces new training. The previous1,024 job's warm-ups/native endpoints can be imported after hash/recipe checks. Its adaptive supervised-Vanilla states **cannot** become the new fixed-weight control. All original sources and 4,096-group evidence remain read-only. Existing SLiCS and component ablations are not retrained.

## Dataset handling

COCO uses the already inspected1,500-image held-out cohort: label these historical audit results, **not fresh untouched confirmation**.

VOC uses the official checksum-verified2012 segmentation archive. Official train has 1,464 images; fixed annotation-blind SHA ordering reserves256 for development; the rest are FIT. Official validation has 1,449 images and is held out from fitting/tuning. Deduplication may reduce counts. Exclude detected COCO Flickr-source overlap and exact decoded/Flickr duplicates between VOC train/val. This is **not exhaustive perceptual near-duplicate certification**.

Freeze shared mapped category eligibility from annotations only, before any scores: at least 5 usable FIT/evaluation images and 3 development images for each positive/negative/contrast evidence type. Report excluded classes/counts. Ignore void consistently. Compute normalization, mask initialization, group maps and probes from **VOC FIT only**. Reuse backbone weights, not COCO featurizer weights or normalization. No VOC GGF hyperparameter sweep. This is **official validation evaluation, not official VOC test**. The old frozen COCO→VOC transfer does not substitute for this training experiment.

## Quantitative consolidation

Operational definition: a concept proxy is more consolidated when a single native group explains more object contrast and additional groups offer less necessary recovery at comparable full reconstruction.

Select greedy native contribution groups on TRAIN after fitting. Freeze them and evaluate prefixes1,2,4 on held-out images, without refitting coefficients/decoder. Stop before nonpositive marginal TRAIN gain and pad larger budgets with unchanged prediction. Report single-group capture, additional-group gain, all-group capture, **AP for the same group**, and image/seed uncertainty. Existing primary readouts remain unchanged; this is a separate post-fit diagnostic.

Contributions may overlap/cancel and include context. These are object proxies, **not a ground-truth feature-splitting rate, additive energy allocation, or proof of human interpretability**.

## Functional diagnostic, directions and rank

Fit one common TRAIN-only linear ridge predictor of patch mask fractions (fixed ridge0.001). Freeze it. Report probe AP on original and reconstructed activations. Remove a TRAIN-selected native group contribution from the original activation, x-z_gD_g, then measure target-class AP loss and off-target response changes. Compare4 random groups selected from 32 nearest TRAIN activity/contribution-RMS neighbors; retain matching statistics. Matching is approximate. This diagnoses a fitted readout, **not causal image editing or recursive generalization**.

For seed 21, create original/group presence/all 12 signed directions/decoded-contribution PCA overlays for person, plant, vase, fork, spoon and pizza where available. A fixed SHA rule selects an annotation-positive evaluation image per category, identically across models, **without filtering for wins**. TRAIN-only display scales and PCA; inactive patches transparent; patch-faithful overlays. Raw coordinates depend on basis; decoded-contribution PCA is complementary. 3D clouds contain actual learned contributions. These are representative seed 21 visualizations; numerical evidence covers all seeds.

On a common fixed256-image TRAIN cohort, collect centered active coefficient covariance spectra. Report entropy effective rank, participation ratio, coefficient-matrix stable rank=sum(lambda)/max(lambda), covariance stable rank=sum(lambda²)/max(lambda)², and utilization=entropy rank/allocated width, with activity/sample counts. Report initialized category slots separately from reserve slots; Vanilla slots are positional controls. Zero-variance normalized ranks are undefined; inactive/insufficient blocks separate. Soft Group Lasso activity means nonzero native coefficients. Rank does not establish intrinsic concept dimension or MDL efficiency.

## Published additional supervised comparator

Concept-SAE paper: https://arxiv.org/abs/2509.22015
Released source: https://github.com/RafaDD/Concept-SAE/tree/956d19b32baccd9c355b5ada9b281898ca998a16

Download unchanged pinned classes, verify SHA256, cache on Drive. Preserve concept tokenizer, concept aggregator, frozen-concept/free-residual stages and 36 free tokens. Released defaults:10 epochs/stage, effective batch 64 accumulated in microbatches8, Adam LR 1e-4/1e-3/1e-3, released StepLR schedules. Development-best stage checkpoints; completed training is evaluated.

Explicit adaptation: normalized frozen DINO patch sequences replace CLIP hook activations; human masks/presence replace proxy labels; mask grid14×14 replaces20×20; unknown pixels ignored. Preserve released linear mask/presence outputs and patch-merge L1, concept-aggregation KL/L1, and free-aggregation-weight L1. Released defaults differ from longer paper schedules/details.

Call it **released Concept-SAE adapted to the DINO/mask task**, not native CLIP/ImageNet replication, tuned optimum, or capacity/Top-k/update-matched BSF. Capture uses the category-assigned native additive decoded contribution including projector bias. AP uses native mask×presence; decoded-component-norm AP is also reported. Report actual parameters/native budget. Do not misrepresent its image-level token interface as a12D BSF. The shared1/2/4-group/rank/direction audit applies to the BSF family.

## Outputs and honest scope

CSV/JSON per seed/category; means±seed SD; Student-t95 df2; paired seed differences and whole-image bootstrap intervals. Consolidation uses1,000 paired whole-image draws. Image intervals condition on fitted models and do not replace seed uncertainty. No failed endpoint disappears.

Generate LaTeX result tables, PDF/PNG baseline plots and consolidation confidence bands, rank/intervention reports, direction atlas HTML/PNG/PDF, provenance and checklist_coverage.json. Measure stage wall time, peak allocated/reserved VRAM, hardware, library versions and parameter counts. Reused cost is distinguished from new invocation cost. Linear encoder/decoder forward FLOPs are a limited estimate, **not total method training FLOPs**.

New output: **MyDrive/GGF_checklist_evidence_v1**, with separate voc_training child store and compact COCO/VOC evidence ZIPs. Large models/features/optimizer states remain outside ZIPs.

Run All unchanged resumes verified checkpoints and completed audits. Unfinished analysis may repeat its current model-level pass. BSF saves local every 50 updates, Drive every 250 and at phase end/graceful stop. Abrupt loss may rewind. Concept-SAE saves at epoch boundaries and attempts a complete-batch save on graceful stop; abrupt loss may replay an epoch. Stop Cell and **wait for verified save/partial archive before disconnecting**. No concurrent writers. T4 is required because cross-device reuse is unvalidated.

A notebook is not completed evidence. Coverage is marked only after outputs exist. Paper rewriting, table/figure integration and renumbering, current venue checks, conference/arXiv preparation, mentor review and resource coordination remain separate. The notebook neither ticks Google Doc checkboxes nor shares/publishes anything. Optional supervised Grassmannian/Group Lasso are not included; supporting variants are correctly labeled unsupervised.

