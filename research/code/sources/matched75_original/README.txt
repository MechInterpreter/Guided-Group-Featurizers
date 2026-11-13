All-75 study preparation, 2026-10-27

Run ggf_all75_comparison_colab.ipynb on a Colab GPU with the original Drive caches mounted. Both regimes run in separate resumable journals under MyDrive/GGF_all75_20261027. The notebook embeds its source, reruns synthetic preflight and prints training progress. It reads manifests by exact run identity rather than assuming a full-SHA directory name.

Full-category: six genuine GGF75/SLiCS75 checkpoints for orders21-23 are already local and reverified. Train three Vanilla75 and three Supervised Vanilla75 endpoints. Existing GGF uses mask initialization and native adaptive supervision; Vanilla starts randomly and Supervised Vanilla shares its random reconstruction warm-up. Same 75x12 architecture and Top32 gate, 7640+3000 updates. Historical GGF LR tuning is disclosed; the new common LR is not an optimality claim.

Category split: train six GGF75, six Supervised Vanilla75 and three label-free Vanilla75 endpoints, sharing Vanilla across the reversed folds. Excluded annotations cannot enter initialization, crowd exclusions, targets, losses or development. Old1024 endpoints cannot be reused. Native SLiCS37/38 remains a separate reference as explicitly chosen by the user. Its existing seed21 folds must be payload reverified; no silent extra-dictionary variant or all75 label.

Total missing main endpoints:21. This count excludes supervised-weight search candidates. No wall-clock estimate is promised without actual GPU throughput. Training is prepared, not already completed.

Original metric dashboard is frozen in PROTOCOL.json. Specific-semantic coherence and visual-property consistency are separate. Concentration uses the exact signed top4 pooling, q=.95/.90, calibration-negative higher quantiles, strict thresholds, balanced-accuracy ranking, OR coverage/FPR at1/2/4/8 groups, overlap and unique/incremental coverage. Category-mask AP uses its separate AP selector, not reconstruction. Visual target alignment is judged independently. Capture, reconstruction, activity, parameters and compute are controls.

Source allocations were previously inspected. All640 allocated interpretability images are opened. These runs are controlled exploratory comparisons, not fresh-data confirmation. Independent presence/visual judgments and response collection are additional stages; TRAINING_COMPLETE.json is not a completed semantic evaluation. No results or labels are imputed.

Local development:
python scripts/ggf_all75/build_study.py
PYTHONPATH=<output/source>;<output/source/core> python -m pytest output/ggf_all75_20261027/source/test_all75.py -q

The source archive contains original training and concentration implementations, source hashes, explicit modifications, regression tests and the common evaluation protocol. Original result files are read only.
