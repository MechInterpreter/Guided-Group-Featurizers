"""Official VOC2012 val, aligned semantic/instance masks, frozen COCO normalization."""
import gc,hashlib,json,os,re,tarfile,time
from pathlib import Path,PurePosixPath
from xml.etree import ElementTree as ET
import numpy as np
import torch
from PIL import Image
from coco_baselines import data
from coco_baselines.storage import identity,sha_file
from coco_adaptation.coverage import evidence
from coco_coordinated import run as memory

URL='https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
MD5='6cd6e144f989b92b3379bac3b3de84fd'
NAMES=['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining table','dog','horse','motorcycle','person','potted plant','sheep','couch','train','tv']

def mapping(categories):
    byname={c['name']:i for i,c in enumerate(categories)}
    if len(byname)!=len(categories) or any(n not in byname for n in NAMES):raise ValueError('COCO taxonomy does not contain the declared VOC20 mapping')
    return {j+1:byname[n] for j,n in enumerate(NAMES)}

def mask_pair(classes,instances,geom,category_map,annotation_base):
    classes=np.asarray(classes);instances=np.asarray(instances)
    if classes.shape!=instances.shape or classes.ndim!=2:raise ValueError('Class and instance masks differ')
    if not set(np.unique(classes))<=set(range(21))|{255}:raise ValueError('Unexpected VOC class palette')
    unknown=(classes==255)|(instances==255)
    unions=np.zeros((196,80),np.float32);objects=[]
    for c,index in category_map.items():unions[:,index]=data.patch_fractions(((classes==c)&~unknown).astype(np.uint8),geom)
    for object_id in np.unique(instances):
        if object_id in (0,255):continue
        pixels=(instances==object_id)&~unknown;labels=classes[pixels];labels=labels[(labels>0)&(labels<21)]
        if not len(labels):continue
        count=np.bincount(labels,minlength=21);c=int(count.argmax())
        if count[c]/len(labels)<.90:raise ValueError('Ambiguous VOC instance-to-class assignment')
        fraction=data.patch_fractions((pixels&(classes==c)).astype(np.uint8),geom)
        objects.append(dict(annotation_id=int(annotation_base*256+int(object_id)),category=category_map[c],fraction=fraction))
    return dict(fraction=np.rint(unions*255).astype(np.uint8),crowd=data.patch_fractions(unknown.astype(np.uint8),geom),instances=objects)

def safe_member(name):
    p=PurePosixPath(name)
    return not(p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name)

def assets(ctx):
    asset=Path(ctx['options']['output'])/'input_assets/voc2012';asset.mkdir(parents=True,exist_ok=True)
    archive=data.download(URL,asset/'VOCtrainval_11-May-2012.tar')
    digest=hashlib.md5()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
    if digest.hexdigest()!=MD5:raise ValueError('VOC archive MD5 differs from torchvision official metadata. Restore/remove corrupt archive, then resume.')
    root=ctx['local']/'voc_assets';root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        split='VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt';member=tar.getmember(split)
        ids=tar.extractfile(member).read().decode().split()
        if len(ids)!=1449 or len(set(ids))!=1449 or any(not re.fullmatch(r'\d{4}_\d{6}',x) for x in ids):raise ValueError('Expected official1449-image VOC2012 segmentation validation list')
        wanted={split}
        for image_id in ids:
            for folder,suffix in [('JPEGImages','.jpg'),('Annotations','.xml'),('SegmentationClass','.png'),('SegmentationObject','.png')]:
                wanted.add(f'VOCdevkit/VOC2012/{folder}/{image_id}{suffix}')
        # No tar.extractall: accept only explicitly requested regular files.
        seen=set()
        for member in tar:
            if member.name not in wanted:continue
            if not safe_member(member.name) or not member.isfile() or member.size>30*1024**2 or member.name in seen:raise ValueError('Unsafe or duplicate archive member')
            seen.add(member.name);dest=root/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
            raw=tar.extractfile(member).read()
            if not dest.exists() or dest.read_bytes()!=raw:dest.write_bytes(raw)
        if seen!=wanted:raise ValueError('VOC archive misses required validation files')
    return root/'VOCdevkit/VOC2012',ids

def flickr_id(url):
    if not url:return None
    if re.fullmatch(r'\d+',url.strip()):return url.strip()
    # COCO farm/static Flickr URLs contain the original photo identifier before '_'.
    match=re.search(r'/(\d+)_[^/]+\.(?:jpg|jpeg)(?:\?|$)',url,re.I)
    return match.group(1) if match else None

def rgb_hash(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert('RGB'))
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

def prepare_dataset(ctx,coco):
    root,all_ids=assets(ctx);catmap=mapping(ctx['provenance']['categories'])
    # Exclude metadata source-photo matches conservatively against all COCO annotation images,
    # a superset of the training/inspected cohorts in this source lineage.
    coco_photos={p for im in coco.imgs.values() if (p:=flickr_id(im.get('flickr_url','')))}
    # An exact decoded-image check supplements metadata for locally available COCO images.
    cached={}
    roots=[ctx['local']/'images',Path(ctx['options']['cache_local'])/'images']
    used={i for ids in ctx['provenance']['splits'].values() for i in ids}|set(ctx['test_ids'])
    for folder in roots:
        for p in folder.glob('*.jpg'):
            if p.stem.isdigit() and int(p.stem) in used:cached[rgb_hash(p)]=int(p.stem)
    ids=[];masks=[];records=[];excluded=[];metadata_available=0
    for ordinal,image_id in enumerate(all_ids):
        p=root/'JPEGImages'/(image_id+'.jpg');xml=ET.parse(root/'Annotations'/(image_id+'.xml'))
        photo=flickr_id(xml.findtext('source/flickrid',''));metadata_available+=photo is not None
        decoded=rgb_hash(p)
        reason='source Flickr photo matches COCO metadata' if photo in coco_photos else 'exact decoded image matches cached COCO' if decoded in cached else None
        if reason:excluded.append(dict(image=image_id,reason=reason));continue
        with Image.open(p) as im:geom=data.geometry(*im.size)
        with Image.open(root/'SegmentationClass'/(image_id+'.png')) as im:classes=np.array(im)
        with Image.open(root/'SegmentationObject'/(image_id+'.png')) as im:objects=np.array(im)
        # Dataset-specific deterministic ID namespace controls patch sampling, not learned labels.
        numeric=2000000000+ordinal
        masks.append(mask_pair(classes,objects,geom,catmap,numeric));ids.append(numeric)
        records.append(dict(image=image_id,id=numeric,image_sha=sha_file(p),class_sha=hashlib.sha256(classes.tobytes()).hexdigest(),
            object_sha=hashlib.sha256(objects.tobytes()).hexdigest(),geometry=geom))
        if (ordinal+1)%100==0:ctx['store'].event('VOC_annotation_preflight',done=ordinal+1,total=1449)
    if not ids:raise ValueError('No non-overlapping VOC validation images remain')
    ev=evidence(dict(ids=ids,masks=masks),ctx['cfg']);counts=ev.sum(0);minimum=max(5,ctx['cfg']['decision_min_images'])
    common=sorted(set(catmap.values())&set(ctx['cfg']['category_indices']))
    eligible=[c for c in common if (counts[c]>=minimum).all()]
    if not eligible:raise ValueError('No shared VOC classes have sufficient annotation-only coverage')
    plan=dict(dataset='VOC2012 segmentation',split='val',official_test=False,archive_url=URL,archive_md5=MD5,
        records=records,category_indices=eligible,mapping=catmap,minimum_usable_images=minimum,
        category_coverage=[dict(category_index=c,category=ctx['provenance']['categories'][c]['name'],
            positive_images=int(counts[c,0]),negative_images=int(counts[c,1]),contrast_images=int(counts[c,2]),included=c in eligible) for c in common],
        excluded_duplicates=excluded,duplicate_audit=dict(method='source Flickr photo IDs against all loaded COCO annotations plus exact decoded hashes against cached COCO images',
            VOC_metadata_available=metadata_available,VOC_images=1449,COCO_photos_available=len(coco_photos),COCO_decoded_hashes=len(cached),
            exhaustive_perceptual_near_duplicate_search=False,
            limitation='This is not a certified exhaustively deduplicated holdout; unrecognized/re-encoded/cropped near duplicates may remain.'),
        category_selection='annotations only, before model scoring; all models use exactly the same fixed category denominator',
        normalization_sha=identity(ctx['norm']),backbone=ctx['provenance']['backbone'])
    saved=ctx['store'].load('voc_plan')
    # Local exact-hash availability may grow after a resumed backbone replay.
    # Freeze included IDs and all feature/mask identities; audit cache coverage alone is not a cohort change.
    if saved is not None:
        if identity({k:v for k,v in saved.items() if k!='duplicate_audit'})!=identity({k:v for k,v in plan.items() if k!='duplicate_audit'}):raise ValueError('Frozen VOC cohort or annotations changed')
        plan=saved
    else:ctx['store'].save('voc_plan',plan)
    ctx['store'].write('voc_plan.json',plan)
    print('VOC PLAN FROZEN',len(ids),'images;',len(eligible),'of',len(common),'shared categories; no VOC fitting.',flush=True)
    return dict(root=root,ids=ids,masks=masks,records=records,plan=plan)

def extract_features(ctx,voc):
    if ctx['store'].load('primary_frozen') is None:raise ValueError('Freeze models before VOC feature extraction')
    from transformers import AutoImageProcessor,AutoModel
    from coco_stratified.dataset import verify_backbone_state
    asset=ctx['fresh_provenance']['backbone']['model']
    stamp=identity(voc['plan']);folder=ctx['local']/'normalized';folder.mkdir(exist_ok=True)
    path=folder/('voc_'+stamp+'.npy');meta=path.with_suffix('.json');n=len(voc['ids']);s=ctx['store']
    valid=path.exists() and meta.exists() and json.loads(meta.read_text()).get('file_sha')==sha_file(path)
    if valid:
        previous=json.loads(meta.read_text());x=np.load(path,mmap_mode='r')
        if previous['plan_sha']!=stamp or x.shape!=(n,196,768) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Cached normalized VOC features differ')
        print('Reused complete local VOC feature cache',n,flush=True)
        return dict(x=x,ids=voc['ids'],masks=voc['masks'],identity=stamp)
    target=None
    if not valid:target=np.lib.format.open_memmap(path.with_suffix('.pending.npy'),mode='w+',dtype=np.float32,shape=(n,196,768))
    processor=backbone=None
    try:
        for lo in range(0,n,16):
            key=f'voc_features_{lo:05d}';chosen=voc['ids'][lo:lo+16];shard=s.load(key)
            if shard is None:
                if backbone is None:
                    processor=AutoImageProcessor.from_pretrained(asset);backbone=AutoModel.from_pretrained(asset,dtype=torch.float32).eval().to(ctx['device'])
                    verify_backbone_state(backbone.state_dict(),ctx['provenance']['backbone_weights']['tensor_sha'])
                rows=[]
                for j in range(lo,min(lo+16,n),4):
                    crops=[]
                    for record in voc['records'][j:min(j+4,n)]:
                        with Image.open(voc['root']/'JPEGImages'/(record['image']+'.jpg')) as im:crops.append(data.transform_image(im.convert('RGB'),record['geometry']))
                    values=processor(images=crops,do_resize=False,do_center_crop=False,return_tensors='pt').to(ctx['device'])
                    with torch.inference_mode():xx=backbone(**values).last_hidden_state[:,5:].float().cpu().half()
                    if tuple(xx.shape[1:])!=(196,768) or not torch.isfinite(xx).all():raise ValueError('VOC features violate source backbone shape/finite contract')
                    rows.append(xx)
                shard=dict(ids=chosen,plan_sha=stamp,x=torch.cat(rows),normalization_sha=identity(ctx['norm']))
                s.save(key,shard)
            if shard['ids']!=chosen or shard['plan_sha']!=stamp or shard['normalization_sha']!=identity(ctx['norm']):raise ValueError('VOC feature identity differs')
            if target is not None:target[lo:lo+len(chosen)]=data.normalize(shard['x'].float().numpy(),ctx['norm'])
            del shard
            s.event('VOC_feature_cache',completed=min(lo+16,n),total=n)
        if target is not None:target.flush();del target;os.replace(path.with_suffix('.pending.npy'),path)
    finally:
        del backbone,processor;gc.collect();torch.cuda.empty_cache()
    x=np.load(path,mmap_mode='r')
    if x.shape!=(n,196,768) or not np.isfinite(x).all():raise ValueError('Invalid normalized VOC cache')
    meta.write_text(json.dumps(dict(file_sha=sha_file(path),plan_sha=stamp)))
    return dict(x=x,ids=voc['ids'],masks=voc['masks'],identity=stamp)
