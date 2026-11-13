"""Official COCO assets, label-blind image splits, aligned masks, frozen DINO."""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from .storage import Store, array_sha, identity, sha_file

MODEL_ID = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
COCO_BUCKET = 'https://s3.amazonaws.com/images.cocodataset.org/'
ANNOTATIONS_URL = COCO_BUCKET + 'annotations/annotations_trainval2017.zip'


def download(url, dest):
    """Resume a local partial transfer if the official server supports Range."""
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): return dest
    partial = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            with requests.get(url, stream=True, timeout=(20, 120), headers={'Range': f'bytes={offset}-'} if offset else {}) as r:
                r.raise_for_status()
                append = offset > 0 and r.status_code == 206
                size = int(r.headers.get('content-length', 0))
                count = 0; start = time.monotonic()
                with partial.open('ab' if append else 'wb') as f:
                    for b in r.iter_content(4 * 1024**2):
                        f.write(b); count += len(b)
                        if count // (32 * 1024**2) != (count - len(b)) // (32 * 1024**2):
                            print(f'Downloaded {count / 1024**2:.0f} MiB; {time.monotonic()-start:.0f}s', flush=True)
                if size and count != size: raise OSError('Incomplete HTTP response')
            partial.replace(dest); return dest
        except (requests.RequestException, OSError):
            if attempt == 3: raise
            time.sleep(2 ** attempt)


def annotations(cache, local, annotation_path=''):
    cache, local = Path(cache), Path(local)
    if not cache.parent.exists(): raise OSError('Mount Drive before creating the cache')
    cache.mkdir(parents=True, exist_ok=True); local.mkdir(parents=True, exist_ok=True)
    target = cache / 'instances_val2017.json'
    if not target.exists():
        if annotation_path:
            raw = Path(annotation_path).read_bytes()
        else:
            archive = download(ANNOTATIONS_URL, local / 'annotations_trainval2017.zip')
            with zipfile.ZipFile(archive) as z: raw = z.read('annotations/instances_val2017.json')
        info = json.loads(raw)
        if len(info['images']) != 5000 or len(info['categories']) != 80:
            raise ValueError('Expected official COCO 2017 validation instance annotations (5000 images, 80 categories)')
        target.write_bytes(raw)
    info = json.loads(target.read_text())
    if len(info['images']) != 5000 or len(info['categories']) != 80: raise ValueError('COCO annotation cache is incomplete')
    return target, dict(source=ANNOTATIONS_URL, annotation_sha=sha_file(target), dataset='COCO val2017 split into pilot train/select/calibration/test')


def split_images(images, counts, seed):
    """Image metadata only. No class, object-count, area, or mask stratification."""
    ids = np.array(sorted(int(x['id']) for x in images))
    if sum(counts.values()) > len(ids): raise ValueError('Requested pilot exceeds 5000 COCO val images')
    order = np.random.default_rng(seed).permutation(ids); result = {}; lo = 0
    for name in ('train', 'select', 'calibration', 'test'):
        result[name] = order[lo:lo+counts[name]].tolist(); lo += counts[name]
    sets = [set(x) for x in result.values()]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i)): raise AssertionError('Image split overlap')
    return result


def geometry(width, height, resize=256, crop=224):
    scale = resize / min(width, height)
    nw, nh = int(width * scale), int(height * scale)
    left, top = int(round((nw-crop)/2)), int(round((nh-crop)/2))
    return dict(original=[width, height], resized=[nw, nh], box=[left, top, left+crop, top+crop], crop=crop)


def transform_image(image, geom, mask=False):
    interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC
    return image.resize(tuple(geom['resized']), interpolation).crop(tuple(geom['box']))


def patch_fractions(mask, geom, patch=16):
    a = np.asarray(transform_image(Image.fromarray(mask.astype(np.uint8)), geom, mask=True), dtype=np.float32)
    n = geom['crop'] // patch
    return a.reshape(n, patch, n, patch).mean((1, 3)).reshape(-1)


def masks_for_image(coco, image_id, geom, categories):
    """Separate audit artifact. Return fractions; crowd pixels are unknown."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[int(image_id)]))
    h, w = geom['original'][1], geom['original'][0]
    unions = {}; instances = []; crowd = np.zeros((h, w), np.uint8)
    for ann in anns:
        mask = coco.annToMask(ann).astype(np.uint8)
        if ann.get('iscrowd', 0): crowd |= mask; continue
        c = categories.index(ann['category_id'])
        unions[c] = unions.get(c, np.zeros_like(mask)) | mask
        instances.append(dict(annotation_id=int(ann['id']), category=c, fraction=patch_fractions(mask, geom)))
    frac = np.zeros((196, len(categories)), np.float32)
    for c, m in unions.items(): frac[:, c] = patch_fractions(m, geom)
    return dict(fraction=np.rint(frac * 255).astype(np.uint8), crowd=patch_fractions(crowd, geom), instances=instances)


def extract(cfg, cache_root, local_root, source_sha, token=None, annotation_path='', require_drive=True, token_callback=None):
    """One extraction per distinct image. Same cached patches for every method."""
    from pycocotools.coco import COCO
    import transformers
    from transformers import AutoImageProcessor, AutoModel
    from huggingface_hub import model_info

    cache_root, local_root = Path(cache_root), Path(local_root)
    if require_drive and not Path('/content/drive/MyDrive').is_dir(): raise OSError('Mount Drive first')
    ann_path, meta = annotations(cache_root, local_root, annotation_path)
    coco = COCO(str(ann_path)); categories = sorted(coco.cats)
    counts = cfg['images']; splits = split_images(list(coco.imgs.values()), counts, cfg['split_seed'])
    pin = cache_root / 'dinov3_pin.json'
    if pin.exists(): backbone = json.loads(pin.read_text())
    else:
        if token is None and token_callback is not None: token = token_callback()
        if token is None: raise ValueError('Supply your authorized HF read token for the first DINOv3 extraction')
        backbone = dict(model=MODEL_ID, revision=model_info(MODEL_ID, token=token).sha)
        pin.write_text(json.dumps(backbone, indent=2))
    if backbone['model'] != MODEL_ID: raise ValueError('Backbone cache identity mismatch')
    config = dict(version=1, source_sha=source_sha, **meta, backbone=backbone, images=counts, split_seed=cfg['split_seed'],
                  image_ids=splits, resize=256, crop=224, patch=16, dtype='float16', transformers=transformers.__version__)
    # Tensor hash is recorded separately and does not change cache namespace mid-extraction.
    config['backbone'] = {k: backbone[k] for k in ('model', 'revision')}
    store = Store(cache_root / 'activations', config, local_root / 'activation_rescue', require_drive)
    store.write('image_splits.json', splits); store.write('categories.json', [coco.cats[c] for c in categories])
    model = processor = None
    for split, ids in splits.items():
        for lo in range(0, len(ids), cfg['extract_chunk']):
            key = f'features_{split}_{lo:05d}'
            existing = store.load(key)
            if existing is not None:
                if existing['ids'] != ids[lo:lo+cfg['extract_chunk']]: raise ValueError('Cached image order mismatch')
                if split in ('calibration', 'test') and store.load(f'masks_{split}_{lo:05d}') is None:
                    raise OSError('A feature shard lacks its aligned masks; restore the shard pair')
                print(f'Reused {split} {min(lo+cfg["extract_chunk"], len(ids))}/{len(ids)}', flush=True); continue
            if model is None:
                if token is None and token_callback is not None: token = token_callback()
                store.event('backbone_load', model=MODEL_ID, revision=backbone['revision'])
                processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=backbone['revision'], token=token)
                model = AutoModel.from_pretrained(MODEL_ID, revision=backbone['revision'], token=token, torch_dtype=torch.float32).to(cfg['device']).eval()
                h = hashlib.sha256()
                for name, t in sorted(model.state_dict().items()):
                    a = t.detach().cpu().contiguous().numpy(); h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
                store.write('backbone_weights.json', dict(**backbone, tensor_sha=h.hexdigest(), torch=str(torch.__version__)))
            chosen = ids[lo:lo+cfg['extract_chunk']]; rows = []; hashes = []; mask_rows = []; geometries = []
            for b in range(0, len(chosen), cfg['extract_batch']):
                crops = []
                for image_id in chosen[b:b+cfg['extract_batch']]:
                    im = coco.imgs[image_id]
                    url = COCO_BUCKET + 'val2017/' + im['file_name']
                    local_image = download(url, local_root / 'images' / im['file_name'])
                    hashes.append(dict(image_id=image_id, sha256=sha_file(local_image)))
                    with Image.open(local_image) as image:
                        image = image.convert('RGB')
                        if image.size != (im['width'], im['height']): raise ValueError('Image dimensions differ from annotations')
                        g = geometry(*image.size); crops.append(transform_image(image, g)); geometries.append(g)
                    if split in ('calibration', 'test'): mask_rows.append(masks_for_image(coco, image_id, g, categories))
                inputs = processor(images=crops, do_resize=False, do_center_crop=False, return_tensors='pt').to(cfg['device'])
                if tuple(inputs['pixel_values'].shape[-2:]) != (224, 224): raise ValueError('Processor changed the mask-aligned crop')
                with torch.inference_mode(): x = model(**inputs).last_hidden_state[:, 5:].float().cpu()
                if tuple(x.shape[1:]) != (196, 768): raise ValueError(f'Expected 196x768 DINOv3 patch tokens, got {x.shape}')
                if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite DINO features')
                rows.append(x.half())
            # Publish masks first: a committed feature shard always has its mask counterpart.
            if mask_rows: store.save(f'masks_{split}_{lo:05d}', dict(ids=chosen, masks=mask_rows, geometry=geometries))
            value = dict(ids=chosen, x=torch.cat(rows), image_hashes=hashes, geometry=geometries)
            store.save(key, value)
            store.event('extraction_chunk', split=split, completed=lo+len(chosen), total=len(ids), activation_sha=array_sha(value['x'].numpy()))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    weights_file = store.results / 'backbone_weights.json'
    recorded_weights = json.loads(weights_file.read_text()) if weights_file.exists() else None
    return store, dict(**meta, cache=str(store.root), splits=splits, categories=[coco.cats[c] for c in categories], backbone=backbone, backbone_weights=recorded_weights)


def load_split(cache, cfg, split, with_masks=False):
    xs = []; ids = []; masks = []
    for lo in range(0, cfg['images'][split], cfg['extract_chunk']):
        p = cache.load(f'features_{split}_{lo:05d}')
        if p is None: raise FileNotFoundError(f'Missing activation shard {split}/{lo}')
        xs.append(p['x'].float().numpy()); ids += p['ids']
        if with_masks:
            q = cache.load(f'masks_{split}_{lo:05d}')
            if q is None or p['ids'] != q['ids']: raise ValueError('Image/mask alignment failed')
            masks += q['masks']
    return dict(x=np.concatenate(xs), ids=ids, masks=masks if with_masks else None)


def normalization(train):
    mean = train.mean(0, dtype=np.float64).astype(np.float32)
    centered = train.astype(np.float32) - mean
    scale = float(np.sqrt(train.shape[-1] / np.mean(np.sum(centered.astype(np.float64)**2, axis=-1))))
    return dict(mean=mean, scale=scale, train_sha=array_sha(train), kind='training-only per-position mean and one global RMS scale')


def normalize(raw, norm): return ((raw.astype(np.float32) - norm['mean']) * norm['scale']).astype(np.float32)

