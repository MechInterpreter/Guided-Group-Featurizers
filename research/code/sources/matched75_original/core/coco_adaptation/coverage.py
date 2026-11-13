"""Annotation-only coverage planning, before any candidate weights are evaluated."""
import numpy as np
from .inputs import split_pack


def evidence(pack, cfg):
    """Mirror audit_arrays' crop, crowd, patch sampling and contrast eligibility."""
    C=pack['masks'][0]['fraction'].shape[1]
    out=np.zeros((len(pack['ids']),C,3),dtype=bool)
    for i,(image_id,mm) in enumerate(zip(pack['ids'],pack['masks'])):
        pick=np.random.default_rng(cfg['patch_sample_seed']+int(image_id)).choice(196,min(cfg['detection_patches'],196),replace=False)
        f=mm['fraction'][pick].astype(float)/255
        known=mm['crowd'][pick]<=cfg['negative_fraction']
        out[i,:,0]=((f>=cfg['positive_fraction']) & known[:,None]).any(0)
        out[i,:,1]=((f<=cfg['negative_fraction']) & known[:,None]).any(0)
        known=mm['crowd']<=cfg['negative_fraction']
        for obj in mm['instances']:
            c=obj['category'];fg=np.asarray(obj['fraction'],float)*known
            bg=(mm['fraction'][:,c].astype(float)/255<=cfg['negative_fraction']) & known
            if fg.sum()>=cfg['min_object_patch_mass'] and bg.sum()>=cfg['min_background_patches']:out[i,c,2]=True
    return out


def combine(a,b):
    return dict(x=np.concatenate([a['x'],b['x']]),ids=a['ids']+b['ids'],masks=a['masks']+b['masks'])


def requirements(target_lists,cfg):
    needed=set();names={}
    for targets in target_lists:
        for t in targets:
            c=int(t['category_index']);names[c]=t['category']
            if t['training_capture'] is not None and t['training_capture']>=cfg['protect_capture_threshold']:needed.add((c,2))
            if t['training_ap'] is not None and t['training_ap']>=cfg['protect_ap_threshold']:needed.update(((c,0),(c,1)))
    return sorted(needed),names


def plan(selection,confirmation,pool,target_lists,cfg):
    """Use disjoint historical development images, never final mapping/test images.

    Requirements depend on training-selected protections, never edit outcomes.
    Joint greedy coverage keeps scarce examples available to both checks. If this
    bounded planner cannot cover a class, stop rather than disable its protection.
    """
    req,names=requirements(target_lists,cfg);minimum=cfg['decision_min_images']
    ids=[set(p['ids']) for p in (selection,confirmation,pool)]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Preservation development splits overlap')
    matrices=[evidence(p,cfg) for p in (selection,confirmation,pool)]
    counts=[m.sum(0).astype(int) for m in matrices[:2]];before=[c.copy() for c in counts]
    available=set(range(len(pool['ids'])));take=[[],[]]
    rng=np.random.default_rng(cfg['coverage_seed']);order=rng.permutation(len(pool['ids']))
    priority={int(i):j for j,i in enumerate(order)}
    while True:
        deficits=[(s,c,k,minimum-counts[s][c,k]) for s in range(2) for c,k in req if counts[s][c,k]<minimum]
        if not deficits:break
        def scarcity(d):
            s,c,k,missing=d
            return (sum(bool(matrices[2][i,c,k]) for i in available)/missing,-missing,s,c,k)
        s,c,k,_=min(deficits,key=scarcity)
        eligible=[i for i in available if matrices[2][i,c,k]]
        if not eligible or len(take[s])>=cfg['coverage_max_extra_images']:
            detail=[dict(split=('selection','confirmation')[ss],category=names[cc],evidence=('positive_images','negative_images','contrast_images')[kk],have=int(counts[ss][cc,kk]),need=minimum) for ss,cc,kk,_ in deficits]
            raise ValueError('Insufficient preservation coverage before training: '+str(detail))
        def gain(i):return sum(min(1,minimum-counts[s][cc,kk]) for cc,kk in req if counts[s][cc,kk]<minimum and matrices[2][i,cc,kk])
        i=max(eligible,key=lambda i:(gain(i),-priority[i]));available.remove(i);take[s].append(i);counts[s]+=matrices[2][i]
    packs=[combine(p,split_pack(pool,np.asarray(ix,dtype=int))) for p,ix in zip((selection,confirmation),take)]
    rows=[dict(split=('selection','confirmation')[s],category=names[c],evidence=('positive_images','negative_images','contrast_images')[k],before=int(before[s][c,k]),after=int(counts[s][c,k]),minimum=minimum) for s in range(2) for c,k in req]
    report=dict(passed=True,requirements=rows,supplement_ids={('selection','confirmation')[s]:[pool['ids'][i] for i in take[s]] for s in range(2)},
                source='Previously inspected original test images, now explicitly historical development data',
                policy='Original ranking sets unchanged. Supplements only check native-group preservation. Final mapping and fresh test excluded.',
                cross_split_overlap=bool(set(packs[0]['ids'])&set(packs[1]['ids'])))
    return packs[0],packs[1],report
