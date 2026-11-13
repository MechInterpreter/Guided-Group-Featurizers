# Synthetic Manifold Zoo from scratch

`code/bootstrap/fresh_zoo.py` generates the configured synthetic data and calls
the original fitting and evaluation functions. It trains Vanilla, Grassmannian,
Group Lasso and GGF with the recorded configurations. Each method receives the
same newly generated arrays. Output is written to a new run directory.

The configuration is read from the included reference and compared with the
recorded GGF manifest. It retains 128 factors, 256 groups of width 4, four active
factors, seeds 0–2, both learning rates and the 200-epoch schedules. The baseline
matrix contains 18 trials and GGF adds six trials. Selection uses the original
functions. The selected GGF checkpoint and final-checkpoint diagnostic are
reported separately.

## Stages

Use Python 3.11+ and the numerical package versions recorded in
`code/evidence/zoo_guided/source/zoo_reference.json` for generation and baseline
stages. The GGF stage uses the versions in
`code/evidence/zoo_guided/results/manifest.json`. The recorded Torch builds differ:
baselines use `2.11.0+cu128`; GGF uses `2.11.0+cu130`. Use separate environments
with access to the same output directory, or switch the installed environment
between phase groups. The command checks versions before execution.

```sh
python code/workflows.py run fresh.zoo.generate --root /path/to/new_zoo --execute
python code/workflows.py run fresh.zoo.train-baselines --root /path/to/new_zoo --execute
python code/workflows.py run fresh.zoo.train-ggf --root /path/to/new_zoo --execute
python code/workflows.py run fresh.zoo.audit-baselines --root /path/to/new_zoo --execute
python code/workflows.py run fresh.zoo.audit-ggf --root /path/to/new_zoo --execute
```

The two audit commands use their corresponding training environments. All fitted
models must be complete before either audit stage. Training checkpoints support
resume within the same source and configuration. The output directory contains
its own generated data and newly trained checkpoints; these are created locally.

`DATA.json` records whether regenerated arrays match the original fingerprints.
Platform-dependent rounding can change those fingerprints. Results from this
run are stored separately from the published records, with the generated data
hash shared across all methods.

## Outputs and validation

`BASELINE_RESULTS.json` and `GUIDED_RESULTS.json` contain the original evaluation
endpoints and factor-level records. Reported metrics and sparsity-match flags
retain their original definitions. Generation, training and audit receipts record
the data and source identities.

CPU tests cover configuration, trial counts, audit gating, native-trainer resume
and a GGF warm-up/guided update sequence. The complete 200-epoch runs and their
GPU evaluations are awaiting execution.
