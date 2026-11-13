"""Frozen-unit geometry. No featurizer fitting and no display-based metrics."""
import numpy as np

def background_composites(foreground,mask,background_a,background_b):
    if foreground.shape!=background_a.shape or foreground.shape!=background_b.shape or mask.shape!=foreground.shape[:2]:raise ValueError('Composite geometry mismatch')
    a=np.where(mask[:,:,None],foreground,background_a)
    b=np.where(mask[:,:,None],foreground,background_b)
    if not np.array_equal(a[mask],b[mask]):raise AssertionError('Foreground pixels changed')
    return a,b

def panel(pack,categories,per_image=32):
    ids=[];patch=[];labels=[];valid=[];images=[]
    for i,(image_id,m) in enumerate(zip(pack['ids'],pack['masks'])):
        f=np.asarray(m['fraction'],float)[:,categories]/255
        known=np.asarray(m['crowd'])<=.1
        rng=np.random.default_rng(int(image_id)+61023)
        chosen=[]
        for c in rng.permutation(len(categories)):
            p=np.flatnonzero((f[:,c]>=.5)&known)
            if len(p):chosen.append(int(rng.choice(p)))
        chosen=list(dict.fromkeys(chosen))
        for p in rng.permutation(196):
            if len(chosen)>=per_image:break
            if known[p] and int(p) not in chosen:chosen.append(int(p))
        chosen=np.array(chosen,dtype=int)
        ids.extend([image_id]*len(chosen));images.extend([i]*len(chosen));patch.extend(chosen)
        labels.extend(f[chosen]>=.5);valid.extend(((f[chosen]>=.5)|(f[chosen]<=.1))&known[chosen,None])
    return dict(ids=np.array(ids),images=np.array(images),patch=np.array(patch),y=np.array(labels),valid=np.array(valid))

def balanced_indices(y,valid,limit,seed):
    rng=np.random.default_rng(seed)
    a=np.flatnonzero(y&valid);b=np.flatnonzero(~y&valid)
    n=min(len(a),len(b),limit//2)
    if n<3:return np.array([],int)
    out=np.r_[rng.choice(a,n,replace=False),rng.choice(b,n,replace=False)]
    return rng.permutation(out)

def _balanced_subset(y,eligible,seed=61023):
    y=np.asarray(y,bool);eligible=np.asarray(eligible,bool)
    a=np.flatnonzero(y&eligible);b=np.flatnonzero(~y&eligible)
    n=min(len(a),len(b))
    rng=np.random.default_rng(seed)
    return np.r_[rng.choice(a,n,False),rng.choice(b,n,False)] if n else np.array([],int)

def _neighbor_accuracy(r,q,ry,qy,k):
    if not len(r) or not np.any(qy) or not np.any(~qy):return None
    k=min(int(k),len(r))
    d=np.maximum((q*q).sum(1)[:,None]+(r*r).sum(1)[None]-2*q@r.T,0)
    cutoff=np.partition(d,k-1,axis=1)[:,k-1,None]
    tol=1e-9+1e-10*np.abs(cutoff)
    low=d<cutoff-tol;equal=np.abs(d-cutoff)<=tol
    remaining=k-low.sum(1)
    w=low.astype(float)+equal*np.divide(remaining,equal.sum(1),out=np.zeros(len(q)),where=equal.sum(1)>0)[:,None]
    prediction=w@ry/k
    # Soft balanced accuracy includes BOTH query classes. A constant response
    # has exactly 0.5 accuracy, independently of reference-label prevalence.
    return float(.5*(prediction[qy].mean()+1-prediction[~qy].mean()))

def geometry(ref,query,ry,qy,normalized=False,k=10):
    """V2: total category organization and conditional active geometry.

    The all-patch diagnostic retains zeros and balances both classes. The
    conditional diagnostic balances each class AFTER filtering active patches,
    and is unavailable if either active class lacks support. Neither is a
    validated universal semantic-coherence score. No legacy score is reused.
    """
    r=np.asarray(ref,float);q=np.asarray(query,float)
    ry=np.asarray(ry,bool);qy=np.asarray(qy,bool)
    rn=np.linalg.norm(r,axis=1);qn=np.linalg.norm(q,axis=1)
    active=rn>1e-10;qa=qn>1e-10
    if normalized:
        r=np.divide(r,rn[:,None],out=np.zeros_like(r),where=active[:,None])
        q=np.divide(q,qn[:,None],out=np.zeros_like(q),where=qa[:,None])
    ri=_balanced_subset(ry,np.ones(len(ry),bool))
    qi=_balanced_subset(qy,np.ones(len(qy),bool),71023)
    ari=_balanced_subset(ry,active)
    aqi=_balanced_subset(qy,qa,71023)
    supported=len(ari)>=6 and len(aqi)>=6
    return dict(balanced_neighbor_accuracy=_neighbor_accuracy(r[ri],q[qi],ry[ri],qy[qi],k),
        active_balanced_neighbor_accuracy=_neighbor_accuracy(r[ari],q[aqi],ry[ari],qy[aqi],k) if supported else None,
        chance_baseline=.5,active_geometry_status='supported' if supported else 'insufficient_active_class_support',
        target_nonzero_rate=float(qa[qy].mean()) if qy.any() else None,
        nontarget_nonzero_rate=float(qa[~qy].mean()) if (~qy).any() else None,
        active_reference_target_count=int(np.sum(active&ry)),active_reference_nontarget_count=int(np.sum(active&~ry)),
        active_query_target_count=int(np.sum(qa&qy)),active_query_nontarget_count=int(np.sum(qa&~qy)),
        conditional_reference_per_class=len(ari)//2,conditional_query_per_class=len(aqi)//2)

def pca_fit(x,limit=3):
    x=np.asarray(x,float);active=np.linalg.norm(x,axis=1)>1e-10;x=x[active]
    if len(x)<2:return None
    mean=x.mean(0);_,s,v=np.linalg.svd(x-mean,full_matrices=False)
    v=v[:limit]
    for a in v:
        if a[np.argmax(np.abs(a))]<0:a*=-1
    u=(x-mean)@v.T;lo,hi=np.quantile(u,[.02,.98],axis=0)
    return dict(mean=mean,axes=v,lo=lo,hi=hi,variance_fraction=float((s[:limit]**2).sum()/max((s*s).sum(),1e-30)))

def pca_rgb(x,pc):
    u=(np.asarray(x)-pc['mean'])@pc['axes'].T
    rgb=np.clip((u-pc['lo'])/np.maximum(pc['hi']-pc['lo'],1e-10),0,1)
    return np.pad(rgb,((0,0),(0,max(0,3-rgb.shape[1]))))[:,:3]
