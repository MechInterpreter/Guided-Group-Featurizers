"""Actual VOC2012 training: official train -> fit/dev; official val -> evaluation.

COCO fixes capacity, cap, LR, update budgets. Only the competitor's fixed
loss pair is selected on VOC development. Normalization/init/mapping/probe
are recomputed from VOC FIT images, never COCO or VOC validation images.
"""
import copy,gc,hashlib,json,os,re,tarfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import torch
from PIL import Image
from coco_baselines import data
from coco_baselines.storage import identity,sha_file
from coco_adaptation.coverage import evidence
from coco_adaptation.inputs import JournalStore
from coco_ggf.drive_recovery import install_drive_recovery
from coco_coordinated import run as memory
from voc import URL,MD5,NAMES,safe_member,mapping,mask_pair,flickr_id,rgb_hash

def partition(records,dev_images=256):
    train=[r for r in records if r['official_split']=='train']
    ordered=sorted(train,key=lambda r:hashlib.sha256(('ggf-voc-dev-20261019-'+r['image']).encode()).hexdigest())
    chosen={r['image'] for r in ordered[:dev_images]}
    return {r['image']:('select' if r['image'] in chosen else 'train') if r['official_split']=='train' else 'eval' for r in records}

def streaming_norm(raw,indices):
    indices=list(indices);n=len(indices)
    if not n:raise ValueError('No training images for normalization')
    total=np.zeros(raw.shape[1:],np.float64);sq=0.
    h=hashlib.sha256()
    for i in indices:
        a=np.asarray(raw[i],np.float32);h.update(a.tobytes());total+=a;sq+=float(np.square(a,dtype=np.float64).sum())
    mean=(total/n).astype(np.float32)
    # Match the mean stored as float32 in the existing normalizer.
    centered_sum=sq-2*float((total*mean).sum())+n*float(np.square(mean,dtype=np.float64).sum())
    scale=np.sqrt(raw.shape[-1]/(centered_sum/(n*raw.shape[1])))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate training normalization')
    return dict(mean=mean,scale=float(scale),train_sha=h.hexdigest(),kind='VOC FIT-only per-position mean and one global RMS scale')

def assets(ctx):
    previous=Path(ctx['options']['source_output']).parent/'GGF_COCO_1024_supervised_VOC_v1/input_assets/voc2012/VOCtrainval_11-May-2012.tar'
    if previous.exists():archive=previous
    else:archive=data.download(URL,ctx['store'].output/'input_assets/voc2012/VOCtrainval_11-May-2012.tar')
    h=hashlib.md5()
    with archive.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    if h.hexdigest()!=MD5:raise ValueError('Official VOC archive checksum mismatch')
    root=ctx['local']/'voc_full_assets';root.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        splits={};wanted=set()
        for split,count in [('train',1464),('val',1449)]:
            name=f'VOCdevkit/VOC2012/ImageSets/Segmentation/{split}.txt';m=tar.getmember(name)
            if not m.isfile():raise ValueError('Invalid split archive member')
            ids=tar.extractfile(m).read().decode().split()
            if len(ids)!=count or len(set(ids))!=count or any(not re.fullmatch(r'\d{4}_\d{6}',i) for i in ids):raise ValueError('VOC split inventory changed')
            splits[split]=ids;wanted.add(name)
            for i in ids:
                for folder,suffix in [('JPEGImages','.jpg'),('Annotations','.xml'),('SegmentationClass','.png'),('SegmentationObject','.png')]:wanted.add(f'VOCdevkit/VOC2012/{folder}/{i}{suffix}')
        if set(splits['train'])&set(splits['val']):raise ValueError('Official VOC split overlap')
        seen=set()
        for m in tar:
            if m.name not in wanted:continue
            if not safe_member(m.name) or not m.isfile() or m.size>30*1024**2 or m.name in seen:raise ValueError('Unsafe VOC archive')
            seen.add(m.name);p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True);raw=tar.extractfile(m).read()
            if not p.exists() or p.read_bytes()!=raw:p.write_bytes(raw)
        if seen!=wanted:raise ValueError('Incomplete VOC archive')
    return root/'VOCdevkit/VOC2012',splits

def prepare(parent,coco,require_drive=True):
    from transformers import AutoModel,AutoImageProcessor
    from coco_stratified.dataset import verify_backbone_state
    root,splits=assets(parent);catmap=mapping(parent['provenance']['categories'])
    coco_photos={p for im in coco.imgs.values() if (p:=flickr_id(im.get('flickr_url','')))}
    records=[];masks=[];excluded=[];train_hashes=set();train_photos=set()
    for split in ['train','val']:
        for image in splits[split]:
            path=root/'JPEGImages'/(image+'.jpg');xml=ET.parse(root/'Annotations'/(image+'.xml'))
            photo=flickr_id(xml.findtext('source/flickrid',''));decoded=rgb_hash(path)
            duplicate=decoded in train_hashes or photo is not None and photo in train_photos
            reason='Flickr metadata overlap with COCO' if photo in coco_photos else ('duplicate within VOC training inventory' if split=='train' else 'exact/Flickr VOC train-validation duplicate') if duplicate else None
            if reason:excluded.append(dict(image=image,reason=reason));continue
            if split=='train':train_hashes.add(decoded);train_photos.update([photo] if photo else [])
            with Image.open(path) as im:geom=data.geometry(*im.size)
            with Image.open(root/'SegmentationClass'/(image+'.png')) as im:cl=np.array(im)
            with Image.open(root/'SegmentationObject'/(image+'.png')) as im:ob=np.array(im)
            numeric=2100000000+len(records)
            masks.append(mask_pair(cl,ob,geom,catmap,numeric));records.append(dict(image=image,id=numeric,official_split=split,
                image_sha=sha_file(path),class_sha=hashlib.sha256(cl.tobytes()).hexdigest(),object_sha=hashlib.sha256(ob.tobytes()).hexdigest(),geometry=geom))
    partitions=partition(records);by={key:[i for i,r in enumerate(records) if partitions[r['image']]==key] for key in ['train','select','eval']}
    cover={key:evidence(dict(ids=[records[i]['id'] for i in ids],masks=[masks[i] for i in ids]),parent['cfg']).sum(0) for key,ids in by.items()}
    # Every retained class must have training positives, negatives, object contrasts,
    # plus independently usable development/evaluation annotations. No model scores.
    common=sorted(set(catmap.values())&set(parent['cfg']['category_indices']))
    categories=[c for c in common if all((cover[k][c]>= (3 if k=='select' else 5)).all() for k in by)]
    if not categories:raise ValueError('No jointly eligible VOC categories')
    plan=dict(dataset='VOC2012 segmentation',records=records,partitions=partitions,category_indices=categories,
        coverage={k:{str(c):cover[k][c].tolist() for c in common} for k in by},excluded=excluded,
        selection='annotation-only fixed denominator; min3 usable dev images and min5 fit/eval images per evidence type',
        duplicate_audit='Flickr IDs against COCO metadata; exact decoded/Flickr between VOC official train and val; no exhaustive perceptual near-duplicate certification',
        official_test=False,evaluation='Official validation held out from all fitting and tuning',
        COCO_settings_frozen=True,normalization='VOC fit-only',archive_md5=MD5)
    options=copy.deepcopy(parent['options']);options['output']=str(Path(options['output'])/'voc_training');options['local']=str(parent['local']/'voc_training')
    local=Path(options['local']);local.mkdir(exist_ok=True)
    s=JournalStore(options['output'],dict(study='ggf_VOC_training_v1',plan=plan,core_sha=options['source_sha'],wrapper_sha=options['wrapper_sha'],
        rule=parent['rule'],versions=parent['store'].config['versions']),local/'rescue',require_drive)
    install_drive_recovery(s);s.write('dataset_plan.json',plan)
    rawpath=local/'raw.npy';raw=np.lib.format.open_memmap(rawpath,mode='w+',dtype=np.float16,shape=(len(records),196,768))
    processor=backbone=None;stamp=identity(plan)
    try:
        for lo in range(0,len(records),16):
            key=f'voc_raw_{lo:05d}';part=s.load(key);rr=records[lo:lo+16]
            if part is None:
                if backbone is None:
                    asset=parent['fresh_provenance']['backbone']['model'];processor=AutoImageProcessor.from_pretrained(asset)
                    backbone=AutoModel.from_pretrained(asset,dtype=torch.float32).eval().to(parent['device'])
                    verify_backbone_state(backbone.state_dict(),parent['provenance']['backbone_weights']['tensor_sha'])
                rows=[]
                for start in range(0,len(rr),4):
                    crops=[]
                    for r in rr[start:start+4]:
                        with Image.open(root/'JPEGImages'/(r['image']+'.jpg')) as im:crops.append(data.transform_image(im.convert('RGB'),r['geometry']))
                    inp=processor(images=crops,do_resize=False,do_center_crop=False,return_tensors='pt').to(parent['device'])
                    with torch.inference_mode():rows.append(backbone(**inp).last_hidden_state[:,5:].float().cpu().half())
                part=dict(plan_sha=stamp,ids=[r['id'] for r in rr],x=torch.cat(rows));s.save(key,part)
            if part['plan_sha']!=stamp or part['ids']!=[r['id'] for r in rr] or tuple(part['x'].shape)!=(len(rr),196,768) or not torch.isfinite(part['x']).all():raise ValueError('VOC raw feature identity changed')
            raw[lo:lo+len(rr)]=part['x'].numpy();s.event('VOC_feature_cache',done=lo+len(rr),total=len(records));del part
    finally:
        del backbone,processor;gc.collect();torch.cuda.empty_cache()
    raw.flush();norm=s.load('normalization')
    if norm is None:norm=streaming_norm(raw,by['train']);s.save('normalization',norm)
    ctx=dict(parent,store=s,options=options,local=local,norm=norm,cfg=dict(parent['cfg'],category_indices=categories),packs={},_pairing_proofs={})
    for split,idx in by.items():
        p=local/(split+'.npy');target=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(len(idx),196,768))
        for lo in range(0,len(idx),16):target[lo:lo+16]=data.normalize(np.asarray(raw[idx[lo:lo+16]],np.float32),norm)
        target.flush();del target
        ctx['packs'][split]=dict(x=np.load(p,mmap_mode='r'),ids=[records[i]['id'] for i in idx],masks=[masks[i] for i in idx],
            identity=identity(dict(plan=stamp,split=split,norm=identity(norm))))
    del raw;ctx['voc_assets']=dict(root=root,records={r['id']:r for r in records});ctx['voc_plan']=plan
    s.write('training_protocol.json',dict(groups=1024,width=12,top_k=16,cap=.4,warm_updates=7640,guided_updates=3000,seeds=[21,22,23],
        reserved_category_slots=len(categories),heads=2*len(categories),train_images=len(by['train']),development_images=len(by['select']),evaluation_images=len(by['eval']),
        backbone_weights_reused=True,featurizers_trained_from_scratch=True,COCO_model_weights_not_used=True))
    return ctx
