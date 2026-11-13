"""Exact existing metrics; full TRAIN mappings on disk, bounded readout audits."""
import copy,gc,json
from pathlib import Path
import numpy as np
import torch
from coco_baselines import metrics,training
from coco_baselines.storage import identity
from coco_adaptation import core
from coco_ggf import study,matched_baselines as matched


class ReadoutView:
    def __init__(self, parent, groups):
        self.parent=parent;self.ids=list(groups)
        self.n_groups=len(groups);self.group_size=parent.group_size;self.d=parent.d
    def eval(self):self.parent.eval();return self
    def encode(self,x):return self.parent.encode(x)[:,self.ids]
    def atoms(self):return self.parent.atoms()[self.ids]


def checked(report,categories):
    if [r['category_index'] for r in report['categories']]!=categories:
        raise ValueError('Category cohort changed')
    if any(r['capture'] is None or r['ap'] is None or not np.isfinite([r['capture'],r['ap']]).all() for r in report['categories']):
        raise ValueError('Undefined category result; denominator must not shrink')
    return report


@torch.no_grad()
def evaluate(m,pack,targets,cfg,device,compact=False):
    ids=sorted({t[k] for t in targets for k in ['contrast_group','detector_group']})
    index={g:i for i,g in enumerate(ids)}
    mapped=[dict(t,contrast_group=index[t['contrast_group']],detector_group=index[t['detector_group']]) for t in targets]
    view=ReadoutView(m,ids)
    a=metrics.audit_arrays(view,pack['x'],pack['masks'],pack['ids'],cfg,device)
    r=core.category_scores(a,mapped,cfg)
    for row,t in zip(r['categories'],targets):
        row.update(contrast_group=t['contrast_group'],detector_group=t['detector_group'])
    r['reconstruction']=training.measure(m,pack['x'].reshape(-1,m.d),device,min(cfg['batch'],512))
    checked(r,cfg['category_indices'])
    if compact:r['_compact']=study.compact_audit(a,matched.mapping(mapped),cfg)
    del a;gc.collect();return r


@torch.no_grad()
def train_targets(ctx,m,key):
    """Unchanged TRAIN-only all-group mapping policy with disk-backed arrays.

    This avoids simultaneous multi-gigabyte concatenations for 4096 groups.
    Files are temporary and never form part of the results ZIP.
    """
    s=ctx['store'];old=s.load(key)
    if old is not None:return old
    p=ctx['packs']['train'];cfg=ctx['cfg'];G=m.n_groups;b=m.group_size;d=m.d
    folder=Path(ctx['local'])/'mapping'/key;folder.mkdir(parents=True,exist_ok=True)
    # Exact usable-object count using the same foreground/background rules.
    objects=[]
    for i,mm in enumerate(p['masks']):
        known=mm['crowd']<=cfg['negative_fraction']
        for o in mm['instances']:
            fg=np.asarray(o['fraction'],float)*known
            bg=(mm['fraction'][:,o['category']].astype(float)/255<=cfg['negative_fraction'])&known
            if fg.sum()>=cfg['min_object_patch_mass'] and bg.sum()>=cfg['min_background_patches']:
                objects.append(dict(image_id=int(p['ids'][i]),image_index=i,annotation_id=o['annotation_id'],category=o['category'],weights=fg/fg.sum()-bg/bg.sum()))
    if not objects:raise ValueError('No usable training objects')
    H=min(196,cfg['detection_patches']);n=len(p['ids']);O=len(objects)
    heat=np.lib.format.open_memmap(folder/'heat.npy',mode='w+',dtype=np.float32,shape=(n*H,G))
    dz=np.lib.format.open_memmap(folder/'dz.npy',mode='w+',dtype=np.float32,shape=(O,G,b))
    dx=np.zeros((O,d),np.float32);frac=[];crowd=[];image_index=[]
    byimage={}
    for j,o in enumerate(objects):byimage.setdefault(o['image_index'],[]).append(j)
    m.eval()
    for i in range(n):
        x=np.array(p['x'][i],copy=True);z=m.encode(torch.as_tensor(x,device=ctx['device'])).cpu().numpy()
        pick=np.random.default_rng(cfg['patch_sample_seed']+int(p['ids'][i])).choice(196,H,replace=False)
        heat[i*H:(i+1)*H]=np.linalg.norm(z,axis=-1)[pick]
        frac.append(p['masks'][i]['fraction'][pick]);crowd.append(p['masks'][i]['crowd'][pick]);image_index.extend([i]*H)
        for j in byimage.get(i,[]):
            w=objects[j]['weights'];dx[j]=w@x;dz[j]=np.einsum('p,pgr->gr',w,z)
        if (i+1)%100==0:s.event('training_mapping',key=key,done=i+1,total=n,groups=G)
    heat.flush();dz.flush()
    a=dict(heat=heat,dz=dz,dx=dx,atoms=m.atoms().cpu().numpy(),fraction=np.concatenate(frac),crowd=np.concatenate(crowd),
           image_index=np.asarray(image_index),objects=[{k:v for k,v in o.items() if k!='weights'} for o in objects])
    targets=core.choose_targets(a,cfg,ctx['provenance']['categories'])
    if [t['category_index'] for t in targets]!=cfg['category_indices']:raise ValueError('TRAIN mapping lacks a declared category')
    s.save(key,targets);s.write('targets/'+key+'.json',targets)
    del a,heat,dz;gc.collect()
    for name in ['heat.npy','dz.npy']:(folder/name).unlink(missing_ok=True)
    return targets
