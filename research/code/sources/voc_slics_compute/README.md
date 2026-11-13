# VOC SLiCS and GGF comparison, plus resource profiling

Run All on a T4 or A100 using the same Drive as your completed VOC run.
This notebook writes only to MyDrive/GGF_VOC_SLiCS_compute_v1. Previous
COCO/VOC checkpoints and results are read-only. It reuses the saved VOC
features, masks, normalization and split allocation; it never retrains
the existing 1024-group models or Concept-SAE.

## New comparison
Three paired training orders (21-23), one group per eligible VOC category:
19 groups, 12 directions/group, GGF Top-8 (the original 32/75 active fraction
rounded to 19 groups), cap 0.60, original COCO-selected guided LR 0.001,
7,640 reconstruction warm-up and 3,000 guided updates; SLiCS receives
10,640 native alternating updates on matching image batches. Both start
from the same VOC FIT-mask SVD directions, with native SLiCS polarity.
No VOC learning-rate sweep. Primary scores use completed training, not
the initial dictionary. Original development-selected checkpoints remain
separate secondary records. This is a new fit on the existing inspected
VOC allocation, not a fresh independent holdout.

GGF19 is a separate capacity-matched comparator, not a replacement for the
existing GGF1024 VOC result. Object Capture and patch AP use native per-category
outputs; full SLiCS reconstruction uses joint NNLS, never the sum of partial
projections. Strict float64 per-row KKT tolerance stays 1e-5. Joint evaluation
uses the verified accelerator and an independent SciPy objective check.
New full-model checkpoints and image-complete joint reconstruction prefixes
resume after reconnection. Stop the cell once and wait for its saved exit.

## Resource measurements
The notebook measures actual new fit-loop and stage times, training invocation
GPU peaks, and independent disposable inference/update probes. The probe compares
GGF19, SLiCS19, saved GGF1024, supervised Vanilla1024, Vanilla1024 and all three
Concept-SAE training stages. It separates GGF controller updates from ordinary
updates, and SLiCS partial readouts from joint reconstruction. Native training
batches remain eight BSF images or 64 Concept-SAE images with microbatches of
eight; inference uses eight images for every method. Both allocated and reserved
CUDA memory are retained. Cold first-update timings include optimizer allocation;
they are labelled separately from historical fit timers and repeat SDs.

FLOPs are counted over the executed scopes, including backward and controller
work when present, with PyTorch's registered formulas (multiply-add = two FLOPs).
The per-operator inventory explicitly lists missing formulas. SVD/eigendecomposition
and external SciPy active-set arithmetic do not have reliable complete counts.
Consequently these are covered-operation counts, not falsely labelled full totals.
Runtime and memory remain the primary cross-method cost comparison. No guessed
training-FLOP ratio is produced from these partial counts.

Training-model copies in profiling are disposable. No profiling optimizer step
is saved back to a trained endpoint. Concept-SAE probes retain its stage loss,
effective batch and freezing rules, with fresh Adam moments on the copied model.

## Outputs
Completed per-run/mean-SD VOC tables; paired training-order intervals; a comparison
figure; full category records and compact image audits; development histories;
hardware/software receipts; loop timers and GPU peaks; repeated inference/update
profiles; registered FLOPs and unsupported operator inventories; verified results ZIP.
This revision has CPU regression checks but has not yet completed a Colab GPU run.
