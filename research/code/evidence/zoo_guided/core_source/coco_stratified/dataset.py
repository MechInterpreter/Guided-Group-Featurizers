"""Annotation-only stratification using exact cropped masks before model fitting."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coco_baselines import data
from coco_baselines.storage import sha_file,identity
from coco_adaptation import coverage


def annotations(cache,local):
    from pycocotools.coco import COCO
    path=Path(cache)/'instances_train2017.json'
    if not path.exists():
        archive=data.download(data.ANNOTATIONS_URL,Path(local)/'annotations_trainval2017.zip')
        with zipfile.ZipFile(archive) as z:raw=z.read('annotations/instances_train2017.json')
        info=json.loads(raw)
        if len(info['categories'])!=80 or len(info['images'])<100000:raise ValueError('Expected official COCO train2017')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    coco=COCO(str(path))
    return coco,path,sha_file(path)


def candidate_ids(coco,seed,per_class):
    rng=np.random.default_rng(seed);chosen=set()
    for c in sorted(coco.cats):
        ids=sorted(set(a['image_id'] for a in coco.loadAnns(coco.getAnnIds(catIds=[c],iscrowd=False))))
        chosen.update(rng.permutation(ids)[:per_class].tolist())
    return sorted(chosen)


def scan(store,coco,cfg):
    ids=candidate_ids(coco,cfg['split_seed'],cfg['candidate_images_per_category'])
    cats=sorted(coco.cats);rows=[]
    for lo in range(0,len(ids),cfg['scan_chunk']):
        key=f'coverage_scan_{lo:06d}';part=store.load(key);current=ids[lo:lo+cfg['scan_chunk']]
        if part is None:
            evidence=[]
            for image_id in current:
                im=coco.imgs[image_id];mask=data.masks_for_image(coco,image_id,data.geometry(im['width'],im['height']),cats)
                evidence.append(coverage.evidence(dict(ids=[image_id],masks=[mask]),cfg)[0])
            part=dict(ids=current,evidence=np.stack(evidence));store.save(key,part)
        if part['ids']!=current:raise ValueError('Coverage candidate order changed')
        rows.append(part['evidence']);store.event('coverage_scan',images_done=lo+len(current),total=len(ids))
    return ids,np.concatenate(rows)


def allocate(ids,evidence,counts,minimums,seed,min_categories=1):
    """Joint rare-first allocation, then prevalence-targeted filling; no model scores."""
    names=list(counts);ids=np.asarray(ids);n=len(ids);C=evidence.shape[1]
    # Joint usability requires sampled positive AND negative patches, plus an eligible contrast.
    usable=evidence.all(axis=2).astype(np.int32)
    needed=sum(minimums.values());available=usable.sum(0)
    eligible=np.flatnonzero(available>=int(np.ceil(needed*1.2)))
    if len(eligible)<min_categories:raise ValueError(f'Only {len(eligible)} categories have sufficient cropped coverage; increase candidate pool or revise the declared scope before training.')
    if sum(counts.values())>n:raise ValueError('Candidate pool smaller than requested splits')
    rng=np.random.default_rng(seed);priority=np.empty(n,int);priority[rng.permutation(n)]=np.arange(n)
    free=np.ones(n,bool);take={s:[] for s in names};have={s:np.zeros(C,int) for s in names};supply=available.copy()
    while True:
        deficits=[(s,int(c),minimums[s]-have[s][c]) for s in names for c in eligible if have[s][c]<minimums[s]]
        if not deficits:break
        total_deficit=np.zeros(C,int)
        for _,c,d in deficits:total_deficit[c]+=d
        s,c,_=min(deficits,key=lambda q:(supply[q[1]]/total_deficit[q[1]],-q[2],names.index(q[0]),q[1]))
        candidates=np.flatnonzero(free & (usable[:,c]>0))
        if not len(candidates) or len(take[s])>=counts[s]:
            raise ValueError(f'Cannot meet {s} category-index {c} quota within the declared split sizes. No training has started; inspect coverage and enlarge sizes/pool.')
        weight=np.maximum(minimums[s]-have[s][eligible],0)/np.maximum(supply[eligible],1)
        gain=usable[candidates][:,eligible]@weight
        order=np.lexsort((priority[candidates],-gain));i=int(candidates[order[0]])
        take[s].append(i);free[i]=False;have[s]+=usable[i];supply-=usable[i]
    # Fill toward one shared category prevalence target, not five unrelated distributions.
    target=usable.mean(0);target[eligible]=np.maximum(target[eligible],max(minimums[s]/counts[s] for s in names))
    for s in sorted(names,key=lambda x:counts[x]):
        desired=target*counts[s]
        while len(take[s])<counts[s]:
            candidates=np.flatnonzero(free)
            # Bounded deterministic pool to avoid an O(N^2 C) fill over common classes.
            candidates=candidates[np.argsort(priority[candidates])[:1024]]
            deficit=desired[eligible]-have[s][eligible]
            weights=1/np.maximum(desired[eligible],1)
            bits=usable[candidates][:,eligible]
            cost=np.sum(((deficit-bits)**2-deficit**2)*weights,axis=1)
            order=np.lexsort((priority[candidates],cost));i=int(candidates[order[0]])
            take[s].append(i);free[i]=False;have[s]+=usable[i]
    splits={s:ids[rng.permutation(take[s])].tolist() for s in names}
    assert sum(len(v) for v in splits.values())==len(set(i for v in splits.values() for i in v))
    for s in names:assert (have[s][eligible]>=minimums[s]).all()
    report=dict(category_indices=eligible.tolist(),candidate_usable_counts=available.tolist(),
        excluded_categories=[dict(category_index=c,usable_images=int(available[c]),required_with_reserve=int(np.ceil(needed*1.2))) for c in range(C) if c not in eligible],
        split_counts={s:have[s].tolist() for s in names},minimums=minimums,sizes=counts,
        shared_target_prevalence=target.tolist(),
        note='All splits use the same supported categories and cropped-mask quotas. Multi-label overlap prevents exact equal category proportions; actual prevalence is reported. This is a coverage-enriched benchmark, not natural COCO frequencies.')
    return splits,report


def plan(store,coco,cfg):
    old=store.load('split_plan')
    if old is not None:return old
    ids,evidence=scan(store,coco,cfg)
    splits,report=allocate(ids,evidence,cfg['images'],cfg['minimum_images_per_category'],cfg['split_seed'],cfg['minimum_categories'])
    categories=[coco.cats[c] for c in sorted(coco.cats)]
    for r in report['excluded_categories']:r['category']=categories[r['category_index']]['name']
    report['categories']=[dict(index=c,name=categories[c]['name'],**{s:report['split_counts'][s][c] for s in splits}) for c in range(len(categories))]
    out=dict(splits=splits,coverage=report,categories=categories)
    out['fingerprint']=identity(out);store.save('split_plan',out);store.write('coverage_plan.json',out)
    return out


def tensor_digest(state):
    """The original experiment's names + shapes + exact tensor bytes."""
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def verify_backbone_state(state,expected):
    direct=tensor_digest(state)
    if direct==expected:
        return dict(expected=expected,observed=direct,verified=direct,renamed_keys=0,rule='original names')
    # HF DINOv3ViTModel now wraps its layer ModuleList in an encoder called model.
    # Rename only that documented prefix; never discard, reshape or alter a tensor.
    legacy={};changed=0
    for name,t in state.items():
        new=name[len('model.'):] if name.startswith('model.layer.') else name
        if new in legacy:raise ValueError('DINOv3 compatibility mapping has duplicate tensor names')
        legacy[new]=t;changed+=new!=name
    compatible=tensor_digest(legacy)
    if changed and compatible==expected:
        return dict(expected=expected,observed=direct,verified=compatible,renamed_keys=changed,
            rule='model.layer.* -> layer.*; names only, all tensor values unchanged')
    raise ValueError('DINOv3 tensor verification failed: expected='+expected+'; observed='+direct+
        '; documented-name-mapping='+compatible+'. No weight changes or hash override permitted.')


def backbone(provenance,device,token_callback):
    from transformers import AutoImageProcessor,AutoModel,__version__
    token=token_callback();pin=provenance['backbone']
    processor=AutoImageProcessor.from_pretrained(pin['model'],revision=pin['revision'],token=token)
    model=AutoModel.from_pretrained(pin['model'],revision=pin['revision'],token=token,dtype=torch.float32).eval()
    verification=verify_backbone_state(model.state_dict(),provenance['backbone_weights']['tensor_sha'])
    verification.update(model=pin['model'],revision=pin['revision'],transformers=__version__)
    model._agf_backbone_verification=verification
    print('Verified original DINOv3 tensor values:',verification,flush=True)
    return processor,model.to(device)


def extract(store,coco,plan,split,cfg,provenance,local,token_callback):
    if split=='test' and store.load('all_models_frozen') is None:raise RuntimeError('Freeze all requested models before test extraction')
    ids=plan['splits'][split];cats=sorted(coco.cats);processor=model=None;xs=[];masks=[]
    try:
        for lo in range(0,len(ids),cfg['extract_chunk']):
            chosen=ids[lo:lo+cfg['extract_chunk']];key=f'features_{split}_{lo:06d}';pack=store.load(key)
            if pack is None:
                if model is None:processor,model=backbone(provenance,cfg['device'],token_callback)
                tensors=[];mm=[];geometry=[];hashes=[]
                for b in range(0,len(chosen),cfg['extract_batch']):
                    images=[]
                    for i in chosen[b:b+cfg['extract_batch']]:
                        im=coco.imgs[i];path=data.download(data.COCO_BUCKET+'train2017/'+im['file_name'],Path(local)/'images'/im['file_name'])
                        g=data.geometry(im['width'],im['height'])
                        with Image.open(path) as image:
                            if image.size!=(im['width'],im['height']):raise ValueError('Image dimensions disagree with masks')
                            images.append(data.transform_image(image.convert('RGB'),g))
                        geometry.append(g);hashes.append(sha_file(path));mm.append(data.masks_for_image(coco,i,g,cats))
                    xx=processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt').to(cfg['device'])
                    with torch.inference_mode():features=model(**xx).last_hidden_state[:,5:].float().cpu()
                    if tuple(features.shape[1:])!=(196,768) or not torch.isfinite(features).all():raise ValueError('Unexpected DINO patch features')
                    tensors.append(features.half())
                pack=dict(ids=chosen,x=torch.cat(tensors),masks=mm,geometry=geometry,image_hashes=hashes);store.save(key,pack)
            if pack['ids']!=chosen:raise ValueError('Shard IDs changed')
            xs.append(pack['x'].float().numpy());masks.extend(pack['masks']);store.event('extraction',split=split,done=lo+len(chosen),total=len(ids))
        result=dict(x=np.concatenate(xs),ids=ids,masks=masks)
        observed=coverage.evidence(result,cfg).all(2).sum(0)
        expected=np.asarray(plan['coverage']['split_counts'][split])
        if not np.array_equal(observed,expected):raise ValueError('Extracted mask coverage differs from planned coverage')
        return result
    finally:
        del model,processor
        if torch.cuda.is_available():torch.cuda.empty_cache()


def load_development(store,coco,plan,cfg,provenance,local,token_callback):
    packs={};norm=store.load('normalization')
    for split in ('train','select','confirm','mapping'):
        p=extract(store,coco,plan,split,cfg,provenance,local,token_callback)
        if split=='train' and norm is None:
            norm=data.normalization(p['x']);store.save('normalization',norm)
        elif split=='train' and data.array_sha(p['x'])!=norm['train_sha']:
            raise ValueError('Training activations changed relative to saved normalization')
        p['x']=data.normalize(p['x'],norm);packs[split]=p
    return packs,norm
