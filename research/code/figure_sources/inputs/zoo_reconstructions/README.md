# Measured Manifold Zoo reconstruction panel

This panel uses selected seed-0 checkpoints. It is separate from the illustrative
true-family panel. No model was trained, continued, edited or selected by this
figure operation. The three-seed tables remain unchanged.

`projected_points.npz` contains the exact paired points drawn in the figure.
Run `python figure_sources/make_zoo_reconstructions.py` from the paper directory
to redraw it using NumPy and Matplotlib. Every row uses the same true embedding
frame, ground-truth center, scale, camera, limits and point colors for all models.
No model-specific PCA, rotation, affine fit or readout is used. Gray floor shadows
are display aids. The colors are point identities from ground-truth azimuth,
not RGB channels of model activations. Zero-gated responses are not discarded.

The seven rows are the first indexed factor of each nonlinear family, with
700 present-factor observations uniformly sampled using seed 48714 + factor ID.
Selection is independent of all model scores. Full-dimensional R2 labels use
all present-factor test observations, not only the displayed subset. Projection
can hide error orthogonal to the true factor frame. Seed 0 illustrates geometry;
all-factor, three-seed tables are the quantitative comparison.

`checkpoint_provenance.json` records Drive paths and checkpoint file SHA256s.
`inference_audit.json` records the original and regenerated split fingerprints,
full-dimensional scores and all 128 group assignments. Recomputed seed-0 recovery
and VE agree with the archive within 3e-8 (absolute fractions). Float32 platform
rounding prevents byte-identical split fingerprints. Benchmark functions, seeds,
geometry and split sizes are unchanged; this regeneration is for the figure.

`measure_source.py` is the read-only inference script. For a fresh run, extract
`core_source.zip` and `source.zip` from `evidence/zoo_guided/saved_results.zip`
into separate core and wrapper directories. Download the four trusted checkpoint
payloads from the recorded Drive paths as vanilla.pt, grass.pt, lasso.pt, ggf.pt,
and copy checkpoint_provenance.json to that directory as provenance.json.
Then run:

```
python measure_source.py --checkpoint-dir CHECKPOINTS --core-root CORE --wrapper-root WRAPPER --output OUTPUT
```

The large dataset payload is not needed; the script regenerates the fixed
benchmark and requires agreement with archived scores before exporting points.
Dependencies match the recorded notebook (NumPy 2.1.3, Torch 2.11, SciPy,
scikit-learn, einops and threadpoolctl). The baseline classes match all three
archived class-source hashes after newline normalization. CPU inference suffices.
