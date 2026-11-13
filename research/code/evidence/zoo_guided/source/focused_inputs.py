"""Read-through verified source states; every new save uses a separate store."""
import copy,gc,json,re
from pathlib import Path
import numpy as np
import torch
from coco_baselines import data,training
from coco_baselines.storage import identity,sha_file
from coco_adaptation import inputs
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched
from focused_protocol import *

class ReuseStore:
    def __init__(self,job,source,pinned):
        self.job=job;self.source=source;self.pinned={r['key']:r for r in pinned}
    def __getattr__(self,name):return getattr(self.job,name)
    @staticmethod
    def reusable(key):
        return bool(re.fullmatch(r'(?:dev_)?g4096_k32_s(?:21|22|23|24|25|26|27)_[A-Za-z0-9_]+',key)
            or re.fullmatch(r'initializer_cat[0-9]+',key) or key=='mask_dictionary12')
    def load(self,key):
        saved=self.job.load(key)
        if saved is not None:return saved
        if not self.reusable(key):return None
        saved=self.source.load(key)
        if saved is not None and isinstance(saved,dict) and 'updates' in saved:
            cap=WARM if key.endswith('_warm') else ADDITIONAL
            if saved['updates']>cap:
                raise ValueError('Saved state is beyond the required endpoint; cannot silently rewind or substitute: '+key)
        if saved is not None and key in self.pinned:
            r=self.pinned[key]
            if saved['updates']!=ADDITIONAL or training.weight_sha(saved['weights'])!=r['weights_sha']:
                raise ValueError('Pinned pilot weights changed: '+key)
        return saved
    def save(self,key,value,*args,**kwargs):
        if key in self.pinned:raise ValueError('Cannot rewrite a completed pilot endpoint: '+key)
        return self.job.save(key,value,*args,**kwargs)
    @property
    def _width_drive_recovery_original_save(self):return self.force_save
    def force_save(self,key,value,*args,**kwargs):
        if key in self.pinned:raise ValueError('Cannot rewrite a completed pilot endpoint: '+key)
        save=getattr(self.job,'_width_drive_recovery_original_save',self.job.save)
        return save(key,value,*args,**kwargs)

def completed_components(ctx,source,component,folder):
    frozen=json.loads((component.results/'components_frozen.json').read_text())
    summary=json.loads((component.results/'component_summary.json').read_text())
    done=json.loads((component.results/'completion_status.json').read_text())
    if not frozen['complete'] or not summary['complete'] or not done['complete'] or len(frozen['models'])!=12:
        raise ValueError('The exact completed component study is required; no retraining fallback')
    if summary['fresh_holdout_opened'] or summary['seven_seed_study']:
        raise ValueError('Unexpected component study scope')
    pins=json.loads((Path(__file__).parent/'pilot_pins.json').read_text())
    rows=[copy.deepcopy(r) for r in frozen['models'] if r['kind'] in ['agf_g','released_vanilla']]
    if {(r['seed'],r['kind']) for r in rows}!={(s,k) for s in SCREEN for k in ['agf_g','released_vanilla']} or len(rows)!=6:
        raise ValueError('Need six exact current pilot endpoints')
    for row in rows:
        validate_row(row);pin=pins[row['key']]
        if any(row[k]!=pin[k] for k in ['weights_sha','targets_sha','seed','kind']):raise ValueError('Completed result pin changed')
        state=memory.required(source,row['key'])
        if state['updates']!=ADDITIONAL or training.weight_sha(state['weights'])!=row['weights_sha']:
            raise ValueError('Restore the exact saved3000-update pilot; never restart it')
        del state;gc.collect()
    out=Path(folder)/'completed_components';out.mkdir(parents=True,exist_ok=True)
    for name in ['component_summary.json','components_frozen.json','fixed_weight_policy.json','test_means.csv','test_models.csv','component_paired_images.csv','component_paired_seeds.csv','completion_status.json']:
        raw=(component.results/name).read_bytes();(out/name).write_bytes(raw)
    return rows,dict(complete=True,models=12,new_training_updates=0,source_run=component.run_id,
        frozen_catalog_sha=sha_file(component.results/'components_frozen.json'),
        scope='Previously inspected1500-image test diagnostics, imported unchanged; not new final-holdout evidence.')

def reject_opened_holdout(source):
    # A previous extraction or score would invalidate an "untouched" claim.
    if source.load('test_complete') is not None or any(source.root.glob('features_test_*')):
        raise ValueError('The reserved holdout was already extracted/scored in the source study; do not call it untouched')

def annotations(ctx,source_output):
    from coco_stratified import dataset
    root=Path(source_output)/'input_assets'
    if not (root/'instances_train2017.json').exists():root=ctx['store'].output/'input_assets'
    coco,path,digest=dataset.annotations(root,ctx['local'])
    if digest!=ctx['provenance']['annotation_sha']:raise ValueError('COCO annotation identity changed')
    return coco

def trim_verified_local(ctx,row):
    """Evict only checksummed local duplicates, never a Drive/source state."""
    store=ctx.get('store')
    if not isinstance(store,ReuseStore):return
    job=store.job;base=Path(job.rescue).resolve();local=Path(ctx['local']).resolve()
    remote=Path(job.root).resolve()
    if not base.is_relative_to(local) or base==local or remote.is_relative_to(local) or base.is_relative_to(remote):
        raise ValueError('Local checkpoint cleanup target is outside the dedicated local cache')
    keys=[row['key'],row['key'].split('_lr',1)[0]+'_warm']
    for key in keys:
        folder=(base/key).resolve()
        if folder.parent!=base:raise ValueError('Invalid checkpoint cache key')
        if not folder.exists():continue
        local_records=job._records(base,key)
        if not local_records:continue
        local_meta=json.loads(local_records[0][1].read_text())
        verified=job._records(remote,key)
        if not any(json.loads(meta.read_text())['sha256']==local_meta['sha256'] for _,meta,_ in verified):continue
        for p in list(folder.glob('slot*.pt'))+list(folder.glob('slot*.json')):
            if p.resolve().parent!=folder:raise ValueError('Cache path escaped its checkpoint directory')
            p.unlink()
        if not any(folder.iterdir()):folder.rmdir()
        print('Local cache released; verified Drive state retained:',key,flush=True)

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
