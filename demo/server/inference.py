"""Local inference service for the frozen concept-atlas models."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import time

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import transformers
from transformers import DINOv3ViTModel
from starlette.formparsers import MultiPartParser

MultiPartParser.spool_max_size = 9 * 1024 * 1024

from frozen_native import partial_nnls
from frozen_umap import FrozenUmap

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path(os.environ.get('GGF_MODEL_DIR', str(ROOT / '.local' / 'research')))
METHODS = ('ggf', 'slics', 'supervised_vanilla', 'vanilla')
EXPECTED_BACKBONE = 'c53a2e034dc9f2ac5c7e04e0564ef66c7ff99f21ec2a246e75a7b0be61fcede1'
Image.MAX_IMAGE_PIXELS = 24_000_000


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name.startswith('model.layer.'):
            name = name[len('model.'):]
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FrozenAtlas:
    def __init__(self):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone = DINOv3ViTModel.from_pretrained(
            PRIVATE / 'dinov3_hf', local_files_only=True,
        ).eval()
        if tensor_digest(self.backbone.state_dict()) != EXPECTED_BACKBONE:
            raise RuntimeError('Frozen backbone tensor verification failed')
        self.backbone.to(self.device).requires_grad_(False)
        norm = torch.load(PRIVATE / 'normalization.pt', map_location='cpu', weights_only=False)
        self.mean = torch.as_tensor(norm['mean'], device=self.device)
        self.scale = float(norm['scale'])
        self.bases = dict(np.load(PRIVATE / 'bases.npz', allow_pickle=False))
        verified = json.loads((PRIVATE / 'EXPORT_VERIFICATION.json').read_text())
        expected = {r['method']: r['checkpoint_sha256'] for r in verified['methods']}
        self.identity = {'backbone': 'DINOv3 ViT-B/16', 'backbone_tensor_sha256': EXPECTED_BACKBONE,
                         'seed': 21, 'groups': 75, 'directions': 12, 'checkpoint_sha256': expected,
                         'normalization_sha256': hashlib.sha256((PRIVATE / 'normalization.pt').read_bytes()).hexdigest(),
                         'projection_sha256': hashlib.sha256((PRIVATE / 'bases.npz').read_bytes()).hexdigest(),
                         'software': {'torch': str(torch.__version__), 'transformers': transformers.__version__}}
        self.umap = FrozenUmap(PRIVATE, ROOT / 'public/data/shared-umap.json')
        self.identity['shared_umap'] = self.umap.identity
        self.models = {}
        self.decoders = {}
        for method in METHODS:
            path = PRIVATE / f'{method}.pt'
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected[method]:
                raise RuntimeError('Frozen group checkpoint verification failed')
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.decoders[method] = (checkpoint['D'] if method == 'slics' else checkpoint['weights']['W_dec']).reshape(75, 12, 768).double().numpy()
            self.models[method] = (
                checkpoint['D'].reshape(75, 12, 768).to(self.device)
                if method == 'slics' else
                {k: v.to(self.device) for k, v in checkpoint['weights'].items()}
            )

    @torch.inference_mode()
    def infer(self, content: bytes):
        started = time.perf_counter()
        try:
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise ValueError('Choose a JPEG, PNG or WebP image.')
                if min(source.size)<32 or max(source.size)>8192 or max(source.size)/min(source.size)>12:
                    raise ValueError('Choose an image with sides between 32 and 8192 pixels and an aspect ratio below 12:1.')
                if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError('Choose an image with at most 24 million pixels.')
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError('This image could not be decoded.') from error
        width, height = image.size
        factor = 256 / min(width, height)
        nw, nh = int(width * factor), int(height * factor)
        left, top = int(round((nw - 224) / 2)), int(round((nh - 224) / 2))
        geometry = {'original': [width, height], 'resized': [nw, nh],
                    'box': [left, top, left + 224, top + 224], 'crop': 224}
        crop = image.resize((nw, nh), Image.Resampling.BICUBIC).crop(tuple(geometry['box']))
        pixels = np.asarray(crop, dtype=np.float32) / 255
        pixels = (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        batch = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(self.device)
        raw = self.backbone(pixel_values=batch).last_hidden_state[0, 5:]
        if raw.shape != (196, 768) or not torch.isfinite(raw).all():
            raise RuntimeError('Invalid frozen-backbone response')
        # Match the stored feature precision before the fitted normalization.
        features = (raw.half().float() - self.mean) * self.scale
        umap_result = self.umap.transform(features.cpu().numpy())
        results = {}
        for method in METHODS:
            if method == 'slics':
                codes, solver = partial_nnls(features, self.models[method])
            else:
                weights = self.models[method]
                codes = (features @ weights['W_enc'] + weights['b_enc']).reshape(196, 75, 12)
                norms = codes.norm(dim=-1)
                gate = torch.zeros_like(norms).scatter_(1, norms.topk(32, dim=-1).indices, 1.)
                codes = codes * gate[..., None]
                solver = None
            codes = codes.cpu().numpy()
            norms = np.linalg.norm(codes, axis=2)
            positions = np.einsum('pgw,gwk->gpk', codes.astype(np.float64), self.bases[method + '_native_transform'])
            positions -= self.bases[method + '_native_center'][:, None, :]
            overview = np.einsum('pgw,gwk->gpk', codes.astype(np.float64), self.decoders[method] @ self.bases['global_axes']) - self.bases['global_center']
            results[method] = {
                'overview': overview.round(6).tolist(),
                'norms': norms.T.round(6).tolist(),
                'positions': positions.round(6).tolist(),
                'scores': np.sort(norms, axis=0)[-4:].mean(0).round(6).tolist(),
                'solver': solver,
            }
        return {'geometry': geometry, 'methods': results, 'umap': umap_result,
                'request_id': secrets.token_hex(12), 'input_sha256': hashlib.sha256(content).hexdigest(),
                'device': self.device, 'model_identity': self.identity,
                'seconds': round(time.perf_counter() - started, 3),
                'backbone': 'DINOv3 ViT-B/16', 'seed': 21}


model = None
busy = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    global model
    model = await asyncio.to_thread(FrozenAtlas)
    yield


class AuthenticatedMemoryUploads:
    def __init__(self, app):
        self.app = app
        self.in_flight = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        token = os.environ.get('GGF_INFERENCE_TOKEN', '')
        supplied = dict(scope['headers']).get(b'authorization', b'')
        if len(token) < 40 or not secrets.compare_digest(supplied, ('Bearer ' + token).encode()):
            return await JSONResponse({'error': 'Authorization required'}, status_code=401)(scope, receive, send)
        if scope.get('method') == 'POST':
            if self.in_flight>=2:
                return await JSONResponse({'error':'The model service is busy.'},status_code=429)(scope,receive,send)
            self.in_flight+=1
            try:
                return await self.read_upload(scope,receive,send)
            finally:
                self.in_flight-=1
        return await self.app(scope, receive, send)

    async def read_upload(self,scope,receive,send):
        events = []
        total = 0
        while True:
            try:
                event = await asyncio.wait_for(receive(),timeout=15)
            except TimeoutError:
                return await JSONResponse({'error':'The upload timed out.'},status_code=408)(scope,receive,send)
            if event['type'] == 'http.disconnect':
                return
            total += len(event.get('body', b''))
            if total > 9 * 1024 * 1024:
                return await JSONResponse({'error': 'Choose an image smaller than 8 MB.'}, status_code=413)(scope, receive, send)
            events.append(event)
            if not event.get('more_body', False):
                break
        async def replay():
            return events.pop(0) if events else await receive()
        return await self.app(scope, replay, send)

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(AuthenticatedMemoryUploads)


def authorize(authorization: str | None = Header(default=None)):
    token = os.environ.get('GGF_INFERENCE_TOKEN', '')
    if len(token) < 40 or not secrets.compare_digest(authorization or '', 'Bearer ' + token):
        raise HTTPException(status_code=401, detail='Authorization required')


@app.get('/status', dependencies=[Depends(authorize)])
def status():
    return {'ready': model is not None,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}


@app.post('/infer', dependencies=[Depends(authorize)])
async def infer(image: UploadFile = File(...)):
    content = await image.read(8 * 1024 * 1024 + 1)
    await image.close()
    if len(content) > 8 * 1024 * 1024:
        return JSONResponse({'error': 'Choose an image smaller than 8 MB.'}, status_code=413)
    if busy.locked():
        return JSONResponse({'error': 'The model service is processing another image. Please try again shortly.'}, status_code=429)
    async with busy:
        try:
            result = await asyncio.to_thread(model.infer, content)
            return JSONResponse(result, headers={'Cache-Control': 'no-store'})
        except ValueError as error:
            return JSONResponse({'error': str(error)}, status_code=400)
        except Exception:
            return JSONResponse({'error': 'The frozen models could not process this image.'}, status_code=500)
