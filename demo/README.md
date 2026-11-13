# GGF Concept Atlas

## Recorded walkthrough

Use Node.js 22.13 or newer:

```sh
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The default walkthrough includes all 225 original seed-21 pairs for each of GGF, SLiCS, supervised Vanilla BSF and Vanilla BSF. It preserves marked regions, pixel-verified lossless crops, all three GPT judgments, disagreements and exceptions. Group numbers are method-specific. Research evidence separately shows seven-order aggregates and Gemini results.

Click a marked patch, a patch zoom or a bright geometry point to update the linked inspector. Native clouds use frozen group-specific PCA of actual decoded responses. The optional UMAP is a shared map of DINOv3 inputs. Coordinates are not rearranged by theme. Explore mode retains co-activation and original group evidence. Take a tour explains each control.

## Local live inference

Recorded exploration works immediately after installation. New-image inference additionally requires the trained four-method checkpoints, normalization, frozen projections and the original DINOv3 weights. These assets are deliberately excluded from Git. Train models through `../research/` or use your own trusted local copy of the frozen demo assets; a clone does not include them.

The model directory must contain `ggf.pt`, `slics.pt`, `supervised_vanilla.pt`, `vanilla.pt`, `normalization.pt`, `bases.npz`, `EXPORT_VERIFICATION.json`, `SHARED_UMAP_VERIFIED.json`, `shared_umap.joblib`, and `dinov3_hf/config.json` plus `dinov3_hf/model.safetensors`. Checkpoints and the UMAP joblib are loaded as trusted local artifacts, not accepted as uploads.

```sh
python -m venv .local/inference-venv
# Activate this environment, then install the server dependencies:
python -m pip install -r server/requirements.txt
npm run live
```

By default the service reads `.local/research/`. Set `GGF_MODEL_DIR` to use an existing local model directory and `GGF_PYTHON` to use an existing Python environment. These settings contain paths, not API credentials. Do not commit the files they reference. DINOv3 access is governed by its original provider's terms.

The service and frontend bind to localhost. A random backend token is generated locally under `.local/` and passed through ignored `.dev.vars`. No paid API is used for demo inference. Images and completed results stay in memory; the session tray allows switching without reuploading or rerunning inference. Refreshing the page clears the cache.

## Verify and test

```sh
npm run verify
npm run typecheck
npm run build
npm run test:coactivation
```

With the live server running, `npm run test:pairs` and `npm run test:uploads` exercise original evidence, actual point selection and inference/cache behavior. Browser tests use Edge on Windows; adapt the browser executable when testing another platform. `ASSETS.json` verifies every recorded public asset. Compressed group data is the canonical copy; raw duplicates, model caches and old deployment scaffolding are omitted.

The pretrained backbone, model checkpoints, images and third-party dependencies retain their original access conditions and licenses. COCO research examples are included as recorded evidence; the full dataset and backbone weights are not bundled. See `../research/LICENSES.md`.
