import copy,gc,json
from pathlib import Path
import numpy as np
import torch
from coco_baselines import data
from coco_ggf import matched_baselines as matched


def annotations(ctx,source_output):
    from coco_stratified import dataset
    root=Path(source_output)/'input_assets'
    if not (root/'instances_train2017.json').exists():root=ctx['store'].output/'input_assets'
    coco,path,digest=dataset.annotations(root,ctx['local'])
    if digest!=ctx['provenance']['annotation_sha']:raise ValueError('COCO annotation identity changed')
    return coco

def backbone_preflight(ctx,coco,source_output):
    """Reuse cached pinned DINO weights read-only and replay one TRAIN image."""
    from transformers import AutoImageProcessor,AutoModel
    from PIL import Image
    from coco_stratified import dataset
    asset=Path(source_output)/'input_assets/dinov3';record=asset/'verified.json'
    if not record.exists():
        from scale_data import backbone_preflight as fresh
        return fresh(ctx,coco)
    pin=ctx['provenance']['backbone'];old=json.loads(record.read_text())
    if old['model']!=pin['model'] or old['revision']!=pin['revision']:raise ValueError('Cached DINO pin differs')
    processor=AutoImageProcessor.from_pretrained(str(asset));model=AutoModel.from_pretrained(str(asset),dtype=torch.float32).eval().to(ctx['device'])
    verified=dataset.verify_backbone_state(model.state_dict(),ctx['provenance']['backbone_weights']['tensor_sha'])
    image_id=ctx['packs']['train']['ids'][0];im=coco.imgs[image_id]
    path=data.download(data.COCO_BUCKET+'train2017/'+im['file_name'],ctx['local']/'images'/im['file_name'])
    with Image.open(path) as image:crop=data.transform_image(image.convert('RGB'),data.geometry(im['width'],im['height']))
    with torch.inference_mode():
        values=processor(images=[crop],do_resize=False,do_center_crop=False,return_tensors='pt').to(ctx['device'])
        fresh=model(**values).last_hidden_state[:,5:].float().cpu().numpy()[0]
    expected=ctx['packs']['train']['x'][0]/ctx['norm']['scale']+ctx['norm']['mean'];err=fresh-expected
    if np.max(np.abs(err))>.005 or np.sqrt(np.mean(err**2))>.0005:raise ValueError('DINO TRAIN-image replay failed')
    verified.update(asset_path=str(asset),train_image_id=image_id,max_abs_error=float(np.max(np.abs(err))),
        rms_error=float(np.sqrt(np.mean(err**2))),reused_read_only=True)
    ctx['store'].write('backbone_preflight.json',verified)
    ctx['fresh_provenance']=copy.deepcopy(ctx['provenance']);ctx['fresh_provenance']['backbone']=dict(model=str(asset),revision=None)
    del model,processor;gc.collect();torch.cuda.empty_cache();return verified
