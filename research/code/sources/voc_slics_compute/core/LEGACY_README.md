# GGF versus SLiCS: supervised-only prospective comparison

Upload ggf_supervised_comparison_colab.ipynb to Colab, select T4, and Run All.
Stop the other baseline writer first. Mount the same Drive account. HF_TOKEN
may be read from Colab Secrets at setup for a training-image backbone replay
and eventual new-holdout extraction. Existing extraction versions are reused.

This notebook runs no BSF control or Zoo experiment. It fits seven-seed75x12
SLiCS and GGF, tunes only the small GGF learning rate on development data,
and evaluates frozen supervised methods on a new independent1,500-image cohort.
The three saved current4096x12 GGF endpoints are read-only capacity anchors.
No new large-scale endpoint is fitted. PROTOCOL.md defines the precise claims,
native SLiCS inference operations, selectors and capacity distinctions.

Default STAGES=['all']; Run All completes everything, with no timer. Optional
STAGES=['development'] completes the three-seed development study only, then
stops before new test features. Switching back to ['all'] resumes existing fits
and keeps the same run identity. Do not edit seeds, model settings or sources
between resumes. Results and checkpoints use a separate Drive directory.

This is the prospective supervised study, not the earlier historical3D repair.
The seven-seed full-scale BSF baseline notebook is a separate experiment and
is not invoked here. Full GPU execution has not been performed during delivery.
