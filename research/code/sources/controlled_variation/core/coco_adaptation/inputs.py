"""Read verified parent artifacts; masks for supervision; genuinely new test IDs."""
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coco_baselines import data, training
from coco_baselines.storage import Store, identity, sha_file


class ReadStore(Store):
    """Read-only opening: never rewrites a parent manifest/checkpoint."""
    def __init__(self, root, local):
        self.root = Path(root)
        self.output = self.root.parent.parent
        manifest = json.loads((self.output/'results'/self.root.name/'manifest.json').read_text())
        self.config, self.run_id = manifest['config'], manifest['run_id']
        if identity(self.config) != self.run_id or self.root.name != self.run_id[:20]:
            raise ValueError('Parent manifest identity mismatch')
        self.rescue = Path(local)/self.run_id[:20]
        self.results = self.output/'results'/self.root.name
        self.require_drive = False


class JournalStore(Store):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.rescue/'events.jsonl').exists() and (self.results/'events.jsonl').exists():
            shutil.copyfile(self.results/'events.jsonl', self.rescue/'events.jsonl')


class CompatibleStore(JournalStore):
    """Import only unchanged, unanchored computations; prior files stay read-only."""
    def attach_previous(self, root):
        self.previous=None
        if root is None or not Path(root).exists():
            self.write('reuse_preflight.json',dict(compatible=False,reason='Previous run not found; original native models still reused'))
            return
        old=ReadStore(root,self.rescue/'previous')
        # Only coverage/version fields differ; every old numerical and algorithm
        # setting must match. New settings cannot excuse changed old settings.
        a=old.config;b=self.config
        changed=[k for k,v in a['config'].items() if k!='version' and b['config'].get(k)!=v]
        if a['parent_run_id']!=b['parent_run_id']:changed.append('parent_run_id')
        if a['numerical_versions']!=b['numerical_versions']:changed.append('numerical_versions')
        if a['config'].get('version')!=2:changed.append('unsupported_previous_version')
        if a['source_sha']!='3cee2081cb60279ca7bcadfb2971a2fc8dae6240f5cd70f1effdbe1742372bfb':changed.append('unsupported_previous_source')
        report=dict(compatible=not changed,changed_fields=changed,previous_run_id=old.run_id,
                    previous_source_sha=a['source_sha'],policy='No anchored fits, anchors, frozen-model catalogs or final reports imported.')
        self.write('reuse_preflight.json',report)
        if not changed:
            self.previous=old
            for section in ('histories','decisions','proposals','targets'):
                for p in (old.results/section).glob('*.json'):
                    if self.reusable(p.stem):self.write(section+'/'+p.name,json.loads(p.read_text()))

    @staticmethod
    def reusable(key):
        return bool(re.fullmatch(r'supervision_masks_(train|select)_\d+',key) or
            re.fullmatch(r'(vanilla|grassmannian|group_lasso)_s\d+_(training_audit|targets|native|continued_reconstruction_seg\d+|supervised_fixed_seg\d+|supervised_fixed_equal_search_compute_seg\d+|unanchored_round\d+(_(control|candidate\d+|proposal_audit|proposals))?)',key))

    def load(self,key):
        value=super().load(key)
        old=getattr(self,'previous',None)
        if value is not None or old is None or not self.reusable(key):return value
        value=old.load(key)
        if value is None:return None
        self.save(key,value)
        for section in ('histories','decisions','proposals','targets'):
            p=old.results/section/(key+'.json')
            if p.exists():self.write(section+'/'+key+'.json',json.loads(p.read_text()))
        self.event('compatible_checkpoint_imported',key=key,previous_run_id=old.run_id,
                   updates=value.get('updates') if isinstance(value,dict) else None)
        print('Reused compatible previous run:',key,flush=True)
        return value


def parent(output, local, expected_run, methods):
    output = Path(output)
    root = output/'checkpoints'/expected_run[:20]
    s = ReadStore(root, Path(local)/'parent_rescue')
    if s.run_id != expected_run: raise ValueError('Wrong COCO baseline run')
    frozen = json.loads((s.results/'models_frozen.json').read_text())
    if not frozen['complete']: raise ValueError('Parent models did not finish')
    choices = [c for c in frozen['choices'] if c['method'] in methods]
    if {c['method'] for c in choices} != set(methods): raise ValueError('Missing selected native method')
    for c in choices:
        state = s.load(c['key'])
        if state is None: raise FileNotFoundError('Missing parent checkpoint '+c['key'])
        w = state['best'] if c['sparsity_matched'] else state['fallback']
        if training.weight_sha(w) != c['weights_sha']: raise ValueError('Parent model hash mismatch')
    cache = ReadStore(s.config['activation_cache'], Path(local)/'parent_activation_rescue')
    provenance = json.loads((s.results/'provenance.json').read_text())
    norm = s.load('normalization')
    if norm is None: raise FileNotFoundError('Training normalization is missing')
    return s, cache, provenance, norm, choices


def add_masks(cache, base_cfg, split, ann_path, categories, store):
    """Never run DINO again for existing patches; align new masks using saved geometry."""
    from pycocotools.coco import COCO
    coco = COCO(str(ann_path)); out = []
    cat_ids = [c['id'] for c in categories]
    for lo in range(0, base_cfg['images'][split], base_cfg['extract_chunk']):
        feat = cache.load(f'features_{split}_{lo:05d}')
        if feat is None: raise FileNotFoundError('Missing feature shard')
        key = f'supervision_masks_{split}_{lo:05d}'
        pack = store.load(key)
        if pack is None:
            masks = [data.masks_for_image(coco, i, geom, cat_ids) for i, geom in zip(feat['ids'], feat['geometry'])]
            pack = dict(ids=feat['ids'], masks=masks)
            store.save(key, pack)
            store.event('supervision_masks', split=split, completed=lo+len(feat['ids']), total=base_cfg['images'][split])
        if pack['ids'] != feat['ids']: raise ValueError('Supervision masks/features disagree')
        out.extend(pack['masks'])
    return out


def fresh_ids(all_ids, used_splits, count, seed):
    used = set(i for ids in used_splits.values() for i in ids)
    available = np.array(sorted(set(all_ids)-used))
    if len(available) < count: raise ValueError('Too few unused COCO images')
    ids = np.random.default_rng(seed).permutation(available)[:count].tolist()
    if set(ids) & used: raise AssertionError('Fresh test overlaps prior run')
    return ids


def split_pack(pack, indices):
    return dict(x=pack['x'][indices], ids=[pack['ids'][i] for i in indices], masks=[pack['masks'][i] for i in indices])


def prepare(store, source, cache, provenance, norm, cfg, ann_path):
    base_cfg = source.config['config']
    if sha_file(ann_path) != provenance['annotation_sha']: raise ValueError('COCO annotations changed')
    packs = {}
    for split in ('train', 'select'):
        p = data.load_split(cache, base_cfg, split)
        if p['ids']!=provenance['splits'][split]:raise ValueError('Parent activation order changed')
        if split=='train' and data.array_sha(p['x'])!=norm['train_sha']:raise ValueError('Training activation bytes differ from stored normalization')
        p['masks'] = add_masks(cache, base_cfg, split, ann_path, provenance['categories'], store)
        p['x'] = data.normalize(p['x'], norm); packs[split] = p
    raw = data.load_split(cache, base_cfg, 'calibration', True)
    raw['x'] = data.normalize(raw['x'], norm)
    n = len(raw['ids']); order = np.random.default_rng(cfg['partition_seed']).permutation(n)
    packs['confirm'] = split_pack(raw, order[:n//2])
    packs['mapping'] = split_pack(raw, order[n//2:])
    all_ids = [im['id'] for im in json.loads(Path(ann_path).read_text())['images']]
    test_ids = fresh_ids(all_ids, provenance['splits'], cfg['fresh_test_images'], cfg['fresh_test_seed'])
    ids = {k: v['ids'] for k,v in packs.items()}; ids['fresh_test'] = test_ids
    sets = [set(v) for v in ids.values()]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)): raise AssertionError('Adaptation split overlap')
    store.write('adaptation_splits.json', dict(image_ids=ids, roles=dict(train='Weights and annotation-based proposals',select='Checkpoint/candidate ranking',confirm='Edit acceptance only',mapping='Final group/threshold calibration only',fresh_test='New images excluded from original baseline and all adaptation decisions'),
        historical_test_status='Original 1000 inspected test images are historical development data, available ONLY to supplement preservation checks. Excluded from final claims.'))
    historical=data.load_split(cache,base_cfg,'test',True)
    if historical['ids']!=provenance['splits']['test']:raise ValueError('Historical development order changed')
    historical['x']=data.normalize(historical['x'],norm)
    packs['historical_development']=historical
    return packs, test_ids


def extract_fresh(store, provenance, ids, ann_path, norm, cfg, local, token_callback):
    """Called only after all adaptation weights and choices have frozen."""
    if store.load('all_models_frozen') is None: raise RuntimeError('Freeze adaptations before opening fresh-test masks')
    from pycocotools.coco import COCO
    from transformers import AutoImageProcessor, AutoModel
    coco = COCO(str(ann_path)); cat_ids = [c['id'] for c in provenance['categories']]
    model = processor = None; xs=[]; masks=[]
    for lo in range(0,len(ids),cfg['extract_chunk']):
        key=f'fresh_test_{lo:05d}'; pack=store.load(key); chosen=ids[lo:lo+cfg['extract_chunk']]
        if pack is None:
            if model is None:
                token=token_callback(); pin=provenance['backbone']
                processor=AutoImageProcessor.from_pretrained(pin['model'],revision=pin['revision'],token=token)
                model=AutoModel.from_pretrained(pin['model'],revision=pin['revision'],token=token,torch_dtype=torch.float32).to(cfg['device']).eval()
                h=hashlib.sha256()
                for name,t in sorted(model.state_dict().items()):
                    a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
                expected=provenance.get('backbone_weights') or {}
                if expected.get('tensor_sha') and h.hexdigest()!=expected['tensor_sha']:
                    raise RuntimeError('DINO tensor hash differs from the parent; stop rather than mix backbones')
                store.write('fresh_backbone.json',dict(**pin,tensor_sha=h.hexdigest()))
            crops=[]; mm=[]; geometries=[]; hashes=[]; rows=[]
            for b in range(0,len(chosen),cfg['extract_batch']):
                crops=[]
                for image_id in chosen[b:b+cfg['extract_batch']]:
                    im=coco.imgs[image_id]; path=data.download(data.COCO_BUCKET+'val2017/'+im['file_name'],Path(local)/'images'/im['file_name'])
                    with Image.open(path) as image:
                        image=image.convert('RGB')
                        if image.size!=(im['width'],im['height']):raise ValueError('Image dimensions disagree')
                        geom=data.geometry(*image.size);crops.append(data.transform_image(image,geom))
                    geometries.append(geom);hashes.append(sha_file(path));mm.append(data.masks_for_image(coco,image_id,geom,cat_ids))
                inputs=processor(images=crops,do_resize=False,do_center_crop=False,return_tensors='pt').to(cfg['device'])
                with torch.inference_mode(): xx=model(**inputs).last_hidden_state[:,5:].float().cpu()
                if tuple(xx.shape[1:])!=(196,768) or not torch.isfinite(xx).all():raise ValueError('Unexpected DINO patch output')
                rows.append(xx.half())
            pack=dict(ids=chosen,x=torch.cat(rows),masks=mm,geometry=geometries,image_hashes=hashes)
            store.save(key,pack)
        if pack['ids']!=chosen:raise ValueError('Fresh test order changed')
        xs.append(pack['x'].float().numpy());masks.extend(pack['masks'])
        store.event('fresh_test_extraction',completed=lo+len(chosen),total=len(ids))
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(x=data.normalize(np.concatenate(xs),norm),masks=masks,ids=ids)
