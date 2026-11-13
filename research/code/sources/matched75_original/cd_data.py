"""Read cached features only; construct category-restricted training views."""
import copy,gc,json,os,shutil
from pathlib import Path
import numpy as np
import torch
from coco_baselines import data
from coco_baselines.storage import identity,sha_file
from coco_coordinated import run as memory
from coco_ggf import twelve_capacity as loader
from coco_ggf import matched_baselines as matched
from cd_common import SPEC,category_split,safe_output,make_store,write_json

def prepare(options,require_drive=True):
    safe_output(options['output'],options['paths'].values())
    # Read feature lineage, not historical all-category model checkpoints.
    # The latter are neither inputs to these new fits nor needed to resume them.
    local=Path(options['local']);paths={k:Path(v) for k,v in options['paths'].items()}
    parent=matched.discover(paths['capacity'],local/'read_capacity',run=loader.PARENT_RUN)
    if parent.config['source_sha']!=loader.PARENT_SOURCE_SHA or parent.config['versions']!=matched.versions():
        raise ValueError('Cached numerical lineage differs from the recorded environment')
    original=matched.discover(paths['original'],local/'read_original',run=parent.config['activation_run'])
    strong=matched.discover(paths['strong'],local/'read_strong',run=parent.config['strong_run'])
    if original.run_id!=memory.EXPECTED_PARENT:raise ValueError('Unexpected activation lineage')
    provenance=json.loads((original.results/'provenance.json').read_text());norm=memory.required(original,'normalization')
    plan=memory.required(strong,'fresh_plan');stamp=memory.required(strong,'activation_identity_test');ids=plan['splits']['test']
    if stamp!=parent.config['test_identity'] or stamp['ids']!=ids or len(set(ids))!=1500 or identity(norm)!=parent.config['normalization_sha'] or stamp['normalization_sha']!=identity(norm):
        raise ValueError('Cached evaluation or normalization identity differs')
    splits=[set(provenance['splits'][k]) for k in ['train','select','confirm']]+[set(ids)]
    if any(splits[i]&splits[j] for i in range(len(splits)) for j in range(i)):raise ValueError('Image split overlap')
    checks=make_store(Path(options['output'])/'input_verification',local/'inputs',dict(source_sha=options['core_sha'],activation_run=original.run_id,strong_run=strong.run_id),require_drive)
    ctx=dict(options=options,store=checks,sources=dict(original=original,strong=strong,capacity_parent=parent),
        cfg=copy.deepcopy(parent.config['fit_config']),device='cuda',norm=norm,provenance=provenance,packs={},
        test_ids=ids,test_stamp=stamp,test_chunk=strong.config['config']['extract_chunk'],
        stamps=memory.source_snapshot([original,strong,parent]),checkpoint_stamps=[])
    for source,keys in [(original,['normalization']),(strong,['fresh_plan','activation_identity_test'])]:
        ctx['checkpoint_stamps'] += [matched.source_key_stamp(source,k) for k in keys]
    ctx['stamps'].append(dict(path=str(original.results/'provenance.json'),sha256=sha_file(original.results/'provenance.json')))
    memory.memory_safe_metrics()
    for split in ['train','select','confirm']:ctx['packs'][split]=loader.cache.cached_split(ctx,split,options['cache_local'])
    matched.assert_sources(ctx)
    eligible=list(ctx['cfg']['category_indices'])
    if len(eligible)!=75:raise ValueError('Expected the original 75 eligible categories')
    ctx.update(options=options,versions=matched.versions(),folds=category_split(ctx['provenance']['categories'],eligible),eligible=eligible)
    if any(len(ctx['packs'][s]['ids'])!=n for s,n in [('train',2000),('select',400),('confirm',400)]):raise ValueError('Original split sizes changed')
    cfg=dict(protocol=SPEC,code_sha256=options['experiment_sha'],core_sha=options['core_sha'],folds=ctx['folds'],
        train=ctx['packs']['train']['identity'],development=ctx['packs']['select']['identity'],
        calibration=ctx['packs']['confirm']['identity'],evaluation=ctx['test_stamp'],normalization=identity(ctx['norm']),versions=ctx['versions'])
    ctx['store']=make_store(options['output'],Path(options['local'])/'journal',cfg,require_drive)
    ctx['store'].write('PROTOCOL.json',cfg)
    ctx['store'].write('category_split.json',dict(folds=[dict(f,guide_names=[ctx['provenance']['categories'][c]['name'] for c in f['guide']],excluded_names=[ctx['provenance']['categories'][c]['name'] for c in f['excluded']]) for f in ctx['folds']],
        train_ids=ctx['packs']['train']['ids'],development_ids=ctx['packs']['select']['ids'],calibration_ids=ctx['packs']['confirm']['ids'],evaluation_ids=ctx['test_ids']))
    return ctx

def restricted_annotation_subset(annotation_records,allowed_ids,image_ids):
    allowed=set(allowed_ids);images=set(image_ids)
    return [copy.deepcopy(a) for a in annotation_records if a['category_id'] in allowed and a['image_id'] in images]

def annotation_file(ctx):
    explicit=ctx['options'].get('annotation_path','')
    candidates=[Path(explicit)] if explicit else []
    for root in ctx['options']['paths'].values():
        root=Path(root)
        for relative in ['input_assets/instances_train2017.json','cache/instances_train2017.json','instances_train2017.json']:
            candidates.append(root/relative)
        candidates.extend(root.glob('**/instances_train2017.json'))
    for p in candidates:
        if p.is_file() and sha_file(p)==ctx['provenance']['annotation_sha']:return p
    from coco_stratified.dataset import annotations
    _,p,h=annotations(Path(ctx['options']['output'])/'input_assets',Path(ctx['options']['local']))
    if h!=ctx['provenance']['annotation_sha']:raise ValueError('COCO annotation identity changed')
    return p

def guidance_packs(ctx,fold):
    """Rebuild crowd masks too. Never derive an ignore mask from excluded labels."""
    if fold in ctx.setdefault('_guidance_cache',{}):return ctx['_guidance_cache'][fold]
    key=f'guidance_masks_fold{fold}';s=ctx['store'];saved=s.load(key)
    guide=ctx['folds'][fold]['guide'];allowed=[ctx['provenance']['categories'][c]['id'] for c in guide]
    if saved is None:
        from pycocotools.coco import COCO
        full=COCO(str(annotation_file(ctx)));all_cats=sorted(full.cats)
        wanted=ctx['packs']['train']['ids']+ctx['packs']['select']['ids']
        anns=restricted_annotation_subset(full.loadAnns(full.getAnnIds(imgIds=wanted)),allowed,wanted)
        limited=COCO();limited.dataset=dict(images=[full.imgs[i] for i in wanted],categories=[full.cats[c] for c in all_cats],annotations=anns);limited.createIndex()
        del full;gc.collect();saved=dict(guide=guide,allowed_annotation_ids=sorted(a['id'] for a in anns),packs={})
        for split in ['train','select']:
            masks=[]
            for j,i in enumerate(ctx['packs'][split]['ids']):
                im=limited.imgs[i];g=data.geometry(im['width'],im['height'])
                m=data.masks_for_image(limited,i,g,all_cats)
                assert not np.asarray(m['fraction'])[:,[c for c in range(80) if c not in guide]].any()
                assert all(o['category'] in guide for o in m['instances'])
                masks.append(m)
                if j%200==0:s.event('rebuild_guidance_masks',fold=fold,split=split,done=j,total=len(ctx['packs'][split]['ids']))
            saved['packs'][split]=masks
        s.save(key,saved);del limited;gc.collect()
    if saved['guide']!=guide:raise ValueError('Guidance cache category identity differs')
    result={split:dict(x=ctx['packs'][split]['x'],ids=ctx['packs'][split]['ids'],masks=saved['packs'][split],
        identity=identity(dict(features=ctx['packs'][split]['identity'],guide=guide,masks=saved['packs'][split]))) for split in ['train','select']}
    ctx['_guidance_cache'][fold]=result
    return result

def training_context(ctx,fold,seed,kind,store):
    from cd_common import SPEC
    from coco_ggf.width_balance import settings
    packs=guidance_packs(ctx,fold)
    cfg=dict(ctx['cfg'],category_indices=ctx['folds'][fold]['guide'],decision_min_images=3,target_min_images=3,batch=512,preview_images=0)
    rule=dict(SPEC,final_guided_updates=SPEC['guided_updates'],warm_lr=SPEC['lr'],local_save_every=50)
    balance=copy.deepcopy(settings()['balance']);balance['bounds']['capture'][1]=balance['bounds']['detection'][1]=SPEC['cap']
    return dict(store=store,packs=packs,cfg=cfg,rule=rule,balance_recipe=balance,device=ctx['device'],norm=ctx['norm'],
        provenance=dict(categories=ctx['provenance']['categories']),local=Path(ctx['options']['local'])/f'fold{fold}_s{seed}_{kind}',
        options=dict(ctx['options'],concept_source=ctx['options']['concept_source']),active_key=None)

def load_evaluation(ctx,block_frozen):
    if len(block_frozen)!=5 or {r['kind'] for r in block_frozen}!={'ggf','supervised_vanilla','vanilla','slics','concept_sae'}:raise ValueError('Freeze the five-method block before excluded-category calibration/evaluation')
    if 'test' in ctx['packs']:return ctx['packs']['test']
    source=ctx['sources']['strong'];ids=ctx['test_ids'];stamp=ctx['test_stamp']
    folder=Path(ctx['options']['local'])/'evaluation';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'features.npy';target=None
    if not path.exists():target=np.lib.format.open_memmap(path.with_suffix('.pending.npy'),mode='w+',dtype=np.float32,shape=(1500,196,768))
    masks=[]
    for lo in range(0,len(ids),ctx['test_chunk']):
        shard=memory.required(source,f'features_test_{lo:06d}');want=ids[lo:lo+ctx['test_chunk']]
        if shard['ids']!=want or shard['normalization_sha']!=identity(ctx['norm']):raise ValueError('Evaluation shard lineage differs')
        if target is not None:target[lo:lo+len(want)]=data.normalize(np.asarray(shard['x'],np.float32),ctx['norm'])
        masks.extend(shard['masks'])
        if lo%200==0:ctx['store'].event('load_cached_evaluation',done=lo+len(want),total=1500)
    if target is not None:target.flush();del target;os.replace(path.with_suffix('.pending.npy'),path)
    x=np.load(path,mmap_mode='r')
    if memory.streaming_array_sha(x)!=stamp['normalized_sha']:raise ValueError('Evaluation activation hash differs')
    ctx['packs']['test']=dict(x=x,ids=ids,masks=masks,identity=identity(stamp))
    matched.assert_sources(ctx);return ctx['packs']['test']
