"""Category-mask AP selector. No training, image sampling or model code."""
import numpy as np

def masks_to_labels(masks,categories):
    fractions=np.stack([np.asarray(m['fraction'])[:,categories] for m in masks])/255.
    crowd=np.stack([np.asarray(m['crowd']) for m in masks])
    if fractions.shape[1]!=196 or crowd.shape!=fractions.shape[:2]:raise ValueError('Expected full196 saved masks')
    y=fractions>=.5
    valid=((fractions>=.5)|(fractions<=.1))&(crowd[:,:,None]<=.1)
    return y.reshape(-1,len(categories)),valid.reshape(-1,len(categories))

def nonnegative_ap(scores,y,valid,weights=None):
    """Exact noninterpolated AP, including the entire tied-zero group.

    Accept one score vector and one/multiple binary targets. Scanning only the
    nonzero scores makes Top-k group norms cheap without treating zeros as missing.
    """
    s=np.asarray(scores);y=np.asarray(y,dtype=bool);v=np.asarray(valid,dtype=bool)
    scalar=y.ndim==1
    if scalar:y=y[:,None];v=v[:,None]
    if s.ndim!=1 or y.shape!=v.shape or len(s)!=len(y) or not np.isfinite(s).all() or (s<0).any():raise ValueError('Invalid AP input')
    w=np.ones(len(s)) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=s.shape or not np.isfinite(w).all() or (w<0).any():raise ValueError('Invalid weights')
    n=(v*w[:,None]).sum(0);p=(y*v*w[:,None]).sum(0)
    idx=np.flatnonzero(s>0);idx=idx[np.argsort(-s[idx],kind='stable')]
    result=np.zeros(y.shape[1]);tp=np.zeros_like(result)
    if len(idx):
        ends=np.r_[np.flatnonzero(s[idx][1:]!=s[idx][:-1]),len(idx)-1]
        tp=np.cumsum(y[idx]*v[idx]*w[idx,None],axis=0)[ends]
        seen=np.cumsum(v[idx]*w[idx,None],axis=0)[ends]
        delta=np.diff(np.vstack([np.zeros(y.shape[1]),tp]),axis=0)
        result=np.sum(delta*np.divide(tp,seen,out=np.zeros_like(tp),where=seen>0),axis=0)
        tp=tp[-1]
    # At the last threshold all valid samples, including zeros, enter together.
    result+=(p-tp)*np.divide(p,n,out=np.zeros_like(p),where=n>0)
    result=np.divide(result,p,out=np.full_like(p,np.nan),where=(p>0)&(n>p))
    return float(result[0]) if scalar else result

def select_by_calibration_ap(unit_scores,y,valid,progress=lambda *_:None):
    if len(unit_scores.shape)!=2 or unit_scores.shape[1]!=len(y):raise ValueError('Expected units x patches')
    aps=np.empty((len(unit_scores),y.shape[1]))
    for unit,score in enumerate(unit_scores):
        aps[unit]=nonnegative_ap(score,y,valid)
        if unit%32==0:progress(unit,len(unit_scores))
    if not np.isfinite(aps).all():raise ValueError('Unsupported calibration category; denominator is preserved')
    # np.argmax explicitly breaks exact ties by smallest native unit ID.
    chosen=np.argmax(aps,axis=0)
    return chosen,aps

def strong_views(scores,image_ids,k=6):
    a=np.asarray(scores).reshape(len(image_ids),196)
    strength=np.partition(a,-4,axis=1)[:,-4:].mean(1)
    order=sorted(range(len(image_ids)),key=lambda i:(-float(strength[i]),int(image_ids[i])))
    out=[]
    for i in order:
        if strength[i]<=0:continue
        patches=sorted(range(196),key=lambda p:(-float(a[i,p]),p))[:4]
        out.append(dict(image_id=int(image_ids[i]),mean_top4=float(strength[i]),positive_top4=[dict(patch=p,score=float(a[i,p])) for p in patches if a[i,p]>0]))
        if len(out)==k:break
    return out

def image_weighted_ap(scores,y,valid,n_images):
    """Prepare exact AP for many image-block resamples without dense repeats."""
    s=np.asarray(scores);y=np.asarray(y,bool);v=np.asarray(valid,bool)
    known=v.reshape(n_images,196).sum(1);positive=(v&y).reshape(n_images,196).sum(1)
    idx=np.flatnonzero((s>0)&v);idx=idx[np.argsort(-s[idx],kind='stable')]
    ends=np.r_[np.flatnonzero(s[idx][1:]!=s[idx][:-1]),len(idx)-1] if len(idx) else np.array([],int)
    images=idx//196;yy=y[idx]
    def evaluate(counts):
        n=float(counts@known);p=float(counts@positive)
        if p<=0 or p>=n:return float('nan'),float('nan')
        result=0.;last=0.
        if len(idx):
            w=counts[images];tp=np.cumsum(w*yy,dtype=float)[ends];seen=np.cumsum(w,dtype=float)[ends]
            delta=np.diff(np.r_[0,tp]);result=float(np.sum(delta*np.divide(tp,seen,out=np.zeros_like(tp),where=seen>0)));last=tp[-1]
        return (result+(p-last)*p/n)/p,p/n
    return evaluate

def evaluate_selected(score,y,valid,n_images,seed=126811,bootstrap=500):
    if len(score)!=n_images*196:raise ValueError('Image block identity differs')
    y=np.asarray(y,dtype=bool);valid=np.asarray(valid,dtype=bool)
    prevalence=float((y&valid).sum()/valid.sum());ap=nonnegative_ap(score,y,valid)
    positive=(y&valid).reshape(n_images,196).any(1)
    local_valid=valid&np.repeat(positive,196)
    conditional=nonnegative_ap(score,y,local_valid)
    # Conditional on the calibration-frozen selector; resample whole images.
    rng=np.random.default_rng(seed);draws=[];lifts=[];weighted=image_weighted_ap(score,y,valid,n_images)
    for _ in range(bootstrap):
        counts=rng.multinomial(n_images,np.full(n_images,1/n_images));val,p=weighted(counts)
        if np.isfinite(val):draws.append(val);lifts.append(val-p)
    if len(draws)<bootstrap*.95:raise ValueError('Too many unsupported image bootstrap draws')
    return dict(patch_AP=ap,positive_patch_prevalence=prevalence,AP_minus_prevalence=ap-prevalence,normalized_AP_lift=(ap-prevalence)/(1-prevalence),
        AP_image_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),AP_lift_image_bootstrap_95=np.quantile(lifts,[.025,.975]).tolist(),
        within_target_images_AP=conditional,positive_images=int(positive.sum()),valid_patches=int(valid.sum()),positive_patches=int((y&valid).sum()),
        bootstrap_valid_draws=len(draws),uncertainty='Conditional on fixed calibration-selected unit; image-block bootstrap, not model-run or selector uncertainty.')
