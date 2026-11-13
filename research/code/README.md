# Guided Group Featurizers

This anonymous review snapshot accompanies the CVPR 2027 main-track submission. The supplementary PDF describes the retained experimental protocols and provides the key results.

## Check the seven-order coherence results offline

From this directory, run:

    python verify_coherence.py

The script uses only Python's standard library. It recounts the saved, de-identified GPT majority and Gemini judgments by case and bank, checks the confusion matrices and planned denominators, and recomputes the per-order scores and seven-order means and sample standard deviations. Each pair endpoint supplies three rows per group in fixed pair order. A whole-group endpoint supplies one row per group. Gemini rubric-invalid replies retain their missing status. No model inference or API access is required. The source score files in evidence/matched75 include the separate response concentration and false-positive measurements.

## Training and evaluation

Use `../notebooks/reproduce_paper.ipynb`. Its quick mode checks each pipeline
section with small inputs; full mode uses `bootstrap/full_pipeline.py` to rebuild
the study inputs and run the selected experiments. Public datasets, model access,
GPU resources and full-budget settings are documented in `../docs/REPRODUCING.md`.

## Additional analyses

The archived_analyses directory supplies the development screens, three-direction pilots and intermediate continuation studies referenced by the current Appendix. The evidence directory holds their numerical records and the main seven-order results. The bundle includes the source layers for the workflows listed in the notebook registry.

## Appendix evidence paths

Appendix paths are relative to this `code/` directory in both review bundles.
Run `python verify_numeric_evidence.py` here to recompute the matched 75-group
numeric and concentration summaries, VOC--SLiCS seed statistics and execution
profile arithmetic from the saved measurements. It requires only Python.

- `evidence/voc_slics_final/` contains per-seed scores, means, paired intervals,
  protocol and completion records, solver checks, and execution profiles.
  `FLOP_COVERAGE.json` describes coverage limits and `compute_profiles.json`
  lists the counted operators per phase. The A100 device metadata is also in
  `evidence/voc_final_compute/hardware.json`.
- `evidence/matched_baselines/`, `evidence/zoo_baselines/`,
  `evidence/zoo_guided/`, and `evidence/dictionary_size/` retain the named
  baseline, geometry and capacity records. Manifold Zoo source snapshots are
  unpacked in `source/` and `core_source/` under `evidence/zoo_guided/`, with
  generic mounted paths made portable. Original source hashes are retained.
- `evidence/terminology.csv` maps archived labels to the paper terminology.

Private storage locators and the original hosted VOC notebook are excluded.
The VOC implementation modules and original included-file hashes are supplied.
The maintained public-input workers download upstream data and generate features and checkpoints locally. Frozen historical entry points retain their recorded input-store requirements.
