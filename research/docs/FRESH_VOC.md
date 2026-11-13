# VOC2012 comparison from public inputs

`code/bootstrap/fresh_voc.py` downloads the official VOC2012 archive and COCO
annotations, recreates the recorded duplicate checks and image allocation, and
extracts features from the pinned DINOv3 backbone. Normalization, initialization
and readouts use VOC FIT images. Official validation images remain the evaluation
split. The original selection functions retain 19 eligible categories.

The worker calls the original 19-group training and evaluation implementations
in `code/sources/voc_slics_compute/`. GGF uses width 12, Top-K 8, learning rate
0.001, cap 0.6, 7,640 warm-up updates and 3,000 guided updates. SLiCS uses the
same training masks and initial SVD directions for 10,640 updates. Both methods
run seeds 21, 22 and 23. GGF's completed endpoint is the primary result;
development-selected checkpoints are retained separately.

## Run

Use a CUDA runtime, the numerical versions in
`code/evidence/voc_slics_final/manifest.json`, and the extraction dependencies in
`requirements.txt`. Set `HF_TOKEN` with access to the specified DINOv3 model.
Choose an output directory outside the source tree.

```sh
python code/workflows.py run fresh.voc19.prepare --root /path/to/new_voc --execute
python code/workflows.py run fresh.voc19.train --root /path/to/new_voc --execute
python code/workflows.py run fresh.voc19.evaluate --root /path/to/new_voc --execute
```

The preparation stage records the source identity, public input hashes, split
plan and normalization. Training checkpoints support resume within that same
configuration. All six completed fits are required before evaluation. SLiCS
object scores use independent category NNLS; reconstruction uses joint NNLS.
Both retain the original strict numerical certificates and precision recovery.

`FITS_COMPLETE.json` records the six endpoints. `RESULTS.json` contains the new
evaluation reports; the journal also contains the original table exports.

## Validation

CPU tests check the output-directory and source guards, exact backbone pin,
six-endpoint requirement, original deterministic VOC split and FIT-only
normalization. The four-image extraction check passed in the supplied Colab log; full-budget
GPU fits remain unvalidated. The full planner also provides `controls.voc.*`,
`concept.voc.*` and `compute` for the 1,024-group, Concept-SAE and profiling
studies, sharing only newly extracted VOC inputs.
