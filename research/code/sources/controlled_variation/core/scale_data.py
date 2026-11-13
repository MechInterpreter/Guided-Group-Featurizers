"""Read-only prior inputs, new TRAIN initialization and a genuinely new holdout."""
import copy,gc,json,os
from pathlib import Path
import numpy as np
import torch
from coco_baselines import data,training
from coco_baselines.storage import identity,sha_file
from coco_adaptation import inputs,coverage
from coco_coordinated import run as memory
from coco_stratified import dataset
from coco_ggf import twelve_capacity as old,matched_baselines as matched,width_balance
from coco_ggf.drive_recovery import install_drive_recovery
from scale_plan import settings

PARENT_RUN='385b9aeacddfa4d3db6b82f7054ffc513ce25dd42dde915a0d4ae483b02ed581'
PARENT_SOURCE='6c7443f7a5527e3c344b5f109db61d5fbce10567ec275ec14c44bf52bd4f0d39'


def prepare(options,require_drive=True):
    paths=options['paths'];output=Path(options['output']);local=Path(options['local'])
    old.cache.safe_destination(output,paths.values())
    # Existing input verification and feature loader only; never call old.run.
    ctx=old.prepare(paths,output/'verified_inputs',local/'inputs',options['cache_local'],options['source_sha'],require_drive)
    parent=matched.discover(paths['twelve'],local/'read_twelve',run=PARENT_RUN)
    status=json.loads((parent.results/'completion_status.json').read_text())
    if parent.config['source_sha']!=PARENT_SOURCE or not all(status[k] for k in ['training_complete','test_complete','rank_complete']):
        raise ValueError('Complete the exact 12D follow-up first')
    frozen=memory.required(parent,'capacity_frozen');result=memory.required(parent,'capacity_result')
    if not frozen['complete'] or len(frozen['models'])!=12 or not result['complete'] or len(result['models'])!=18:
        raise ValueError('Completed 12D catalog is incomplete')
    ctx['sources']['twelve']=parent;ctx['stamps']+=memory.source_snapshot([parent])
    ctx['checkpoint_stamps'] += [matched.source_key_stamp(parent,k) for k in ['capacity_frozen','capacity_result']]
    rule=settings(options.get('baselines','both'),options.get('fresh',True))
    import importlib.metadata
    extraction_versions={p:importlib.metadata.version(p) for p in ['transformers','huggingface-hub','pycocotools','pillow']}
    payload=dict(recipe=rule,source_sha=options['source_sha'],versions=matched.versions(),extraction_versions=extraction_versions,parent_run=PARENT_RUN,
        parent_source_sha=PARENT_SOURCE,normalization_sha=identity(ctx['norm']),activation_run=ctx['sources']['original'].run_id,
        training_identity=ctx['packs']['train']['identity'],selection_identity=ctx['packs']['select']['identity'],confirmation_identity=ctx['packs']['confirm']['identity'])
    ctx.update(rule=rule,local=local,active_key=None,balance_recipe=copy.deepcopy(width_balance.settings()['balance']))
    ctx['store']=inputs.JournalStore(output,payload,local/'rescue',require_drive);install_drive_recovery(ctx['store'])
    ctx['cfg']=dict(ctx['cfg'],batch=512,audit_image_batch=1,bootstrap=rule['bootstrap'],fresh_test_seed=rule['fresh_seed'],preview_images=0)
    ctx['base']=dict(ctx['base'],block_rank=12)
    ctx['reference_reports']=result['models']
    ctx['store'].write('prior_results_context.json',dict(models=result['models'],scope='Historical references only; different initialization/budget schedule from this new study.'))
    ctx['store'].write('protocol.json',payload)
    ctx['store'].write('source_checkpoint_hashes.json',ctx['checkpoint_stamps'])
    return ctx


def mask_dictionary(ctx):
    s=ctx['store'];old=s.load('mask_dictionary12')
    if old is not None:
        if training.weight_sha({'D':old['D']})!=old['sha']:raise ValueError('Initializer hash changed')
        return old['D']
    b=ctx['rule']['width'];d=ctx['packs']['train']['x'].shape[-1];blocks=[];records=[]
    for c in ctx['cfg']['category_indices']:
        key=f'initializer_cat{c}';part=s.load(key)
        if part is None:
            gram=np.zeros((d,d),np.float64);count=0
            for x,mm in zip(ctx['packs']['train']['x'],ctx['packs']['train']['masks']):
                good=(mm['crowd']<=.1)&(mm['fraction'][:,c].astype(float)/255>=.5)
                xx=np.asarray(x[good],np.float64);gram+=xx.T@xx;count+=len(xx)
            if count<b:raise ValueError(f'Too few training foreground patches for {c}: {count}')
            eigenvectors=np.linalg.eigh(gram/count)[1][:,-b:][:,::-1].T.copy()
            for row in eigenvectors:
                if row[np.argmax(np.abs(row))]<0:row*=-1
            part=dict(D=torch.from_numpy(eigenvectors).float(),patches=count,category=c)
            s.save(key,part)
        blocks.append(part['D']);records.append(dict(category=c,patches=part['patches']))
        s.event('mask_initialization',done=len(blocks),total=len(ctx['cfg']['category_indices']))
    D=torch.cat(blocks);s.save('mask_dictionary12',dict(D=D,sha=training.weight_sha({'D':D}),records=records))
    s.write('initialization/foreground_patches.json',records);return D


def annotations(ctx):
    coco,path,h=dataset.annotations(ctx['store'].output/'input_assets',ctx['local'])
    if h!=ctx['provenance']['annotation_sha']:raise ValueError('Original COCO annotations differ')
    return coco


def holdout_plan(ctx,coco):
    s=ctx['store'];oldplan=s.load('fresh_holdout_plan')
    used=set(i for ids in ctx['provenance']['splits'].values() for i in ids)
    for src in [ctx['sources']['strong']]:
        previous=memory.required(src,'fresh_plan')
        used.update(i for ids in previous['splits'].values() for i in ids)
    if oldplan is not None:
        if set(oldplan['splits']['test'])&used or len(oldplan['splits']['test'])!=ctx['rule']['final_holdout_images']:
            raise ValueError('Fresh holdout overlaps a prior split')
        return oldplan
    # Metadata and masks only, before any new model metrics; all declared
    # categories need usable contrasts plus sampled positive/negative patches.
    ids=dataset.candidate_ids(coco,ctx['rule']['fresh_seed'],300)
    ids=[i for i in ids if i not in used];ev=[]
    for lo in range(0,len(ids),100):
        key=f'holdout_coverage_{lo:06d}';part=s.load(key);chosen=ids[lo:lo+100]
        if part is None:
            rows=[]
            for image_id in chosen:
                im=coco.imgs[image_id];mm=data.masks_for_image(coco,image_id,data.geometry(im['width'],im['height']),sorted(coco.cats))
                rows.append(coverage.evidence(dict(ids=[image_id],masks=[mm]),ctx['cfg'])[0])
            part=dict(ids=chosen,evidence=np.stack(rows));s.save(key,part)
        if part['ids']!=chosen:raise ValueError('Holdout scan order changed')
        ev.extend(part['evidence']);s.event('holdout_coverage',done=min(lo+100,len(ids)),total=len(ids))
    ev=np.stack(ev);cohort=ctx['cfg']['category_indices']
    # Restrict allocator to the declared 75 categories. No category is selected
    # or removed because a newly fitted model does badly on it.
    local_ev=ev[:,cohort];split,report=dataset.allocate(ids,local_ev,{'test':ctx['rule']['final_holdout_images']},{'test':10},ctx['rule']['fresh_seed'],len(cohort))
    if report['category_indices']!=list(range(len(cohort))):raise ValueError('Fresh coverage cannot support all declared categories')
    chosen=set(split['test']);full=ev[[i for i,x in enumerate(ids) if x in chosen]].all(2).sum(0)
    out=dict(splits=split,coverage={'split_counts':{'test':full.tolist()}},categories=ctx['provenance']['categories'],
             excluded_prior_ids=sorted(used),category_indices=cohort,selection='annotation coverage only; no features or model scores')
    s.save('fresh_holdout_plan',out);s.write('fresh_holdout_plan.json',out);return out


def backbone_preflight(ctx,coco):
    """Download/authenticate now, rather than prompt after days of fitting."""
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor,AutoModel
    from PIL import Image
    asset=ctx['store'].output/'input_assets'/'dinov3';asset.mkdir(parents=True,exist_ok=True)
    pin=ctx['provenance']['backbone'];token=os.environ.get('HF_TOKEN') or None
    record=asset/'verified.json'
    # Materialize weights to Drive so resumed runtimes do not redownload them.
    if not record.exists():
        snapshot_download(pin['model'],revision=pin['revision'],token=token,local_dir=str(asset),
                          allow_patterns=['*.json','*.safetensors','*.bin'])
    else:
        previous=json.loads(record.read_text())
        if previous['model']!=pin['model'] or previous['revision']!=pin['revision']:
            raise ValueError('Cached backbone pin differs')
    processor=AutoImageProcessor.from_pretrained(str(asset));m=AutoModel.from_pretrained(str(asset),dtype=torch.float32).eval().to(ctx['device'])
    verified=dataset.verify_backbone_state(m.state_dict(),ctx['provenance']['backbone_weights']['tensor_sha'])
    image_id=ctx['packs']['train']['ids'][0];im=coco.imgs[image_id]
    path=data.download(data.COCO_BUCKET+'train2017/'+im['file_name'],ctx['local']/'images'/im['file_name'])
    with Image.open(path) as image:
        crop=data.transform_image(image.convert('RGB'),data.geometry(im['width'],im['height']))
    with torch.inference_mode():
        xx=processor(images=[crop],do_resize=False,do_center_crop=False,return_tensors='pt').to(ctx['device'])
        fresh=m(**xx).last_hidden_state[:,5:].float().cpu().numpy()[0]
    old=ctx['packs']['train']['x'][0]/ctx['norm']['scale']+ctx['norm']['mean']
    error=fresh-old
    if np.max(np.abs(error))>.005 or np.sqrt(np.mean(error**2))>.0005:
        raise ValueError('Fresh DINO pipeline fails cached TRAIN-image replay; do not mix extraction pipelines')
    verified.update(model=pin['model'],revision=pin['revision'],train_image_id=image_id,max_abs_error=float(np.max(np.abs(error))),rms_error=float(np.sqrt(np.mean(error**2))),
                    replay_tolerance='float16-cache rounding: max .005, RMS .0005',asset_path=str(asset))
    ctx['store'].write('backbone_preflight.json',verified);record.write_text(json.dumps(verified,indent=2));ctx['fresh_provenance']=copy.deepcopy(ctx['provenance'])
    ctx['fresh_provenance']['backbone']=dict(model=str(asset),revision=None)
    del m,processor;gc.collect();torch.cuda.empty_cache();return verified


def load_holdout(ctx,coco):
    if ctx['store'].load('all_models_frozen') is None:raise ValueError('Freeze every final model before test extraction')
    if ctx['rule']['fresh']:
        plan=memory.required(ctx['store'],'fresh_holdout_plan')
        p=dataset.extract(ctx['store'],coco,plan,'test',ctx['cfg'],ctx['fresh_provenance'],ctx['local'],lambda:None)
        p['x']=data.normalize(p['x'],ctx['norm']);p['identity']=identity(dict(ids=p['ids'],normalization=identity(ctx['norm'])))
        return p
    # Optional explicitly labeled continuity evaluation, never called fresh.
    old.load_test(ctx);return ctx['packs'].pop('test')
