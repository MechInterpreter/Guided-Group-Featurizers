"""Frozen, operational coherence diagnostics. No fitting or model selection."""
import numpy as np
from scipy.stats import t as student

FOREGROUND = ['rotation', 'scale', 'brightness']

def average_precision(y, score):
    """Threshold-grouped AP: ties cannot gain credit from input order."""
    y=np.asarray(y,dtype=bool);score=np.asarray(score,dtype=float)
    if y.ndim!=1 or score.shape!=y.shape or not np.isfinite(score).all():raise ValueError('Invalid AP arrays')
    if not y.any() or y.all():raise ValueError('AP requires both classes')
    order=np.argsort(-score,kind='stable');y=y[order];score=score[order]
    ends=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(y)-1]
    tp=np.cumsum(y)[ends];precision=tp/(ends+1);recall=tp/y.sum()
    return float(np.sum(np.diff(np.r_[0,recall])*precision))

def trajectory_consistency(delta, decoder, ids=None, weights=None):
    """Mean pairwise cosine of decoded changes; zero changes contribute zero.

    delta is objects x nonidentity conditions x block width. Only distinct
    base objects are paired. Gram products avoid materializing DINO vectors.
    """
    z=np.asarray(delta,dtype=float);D=np.asarray(decoder,dtype=float)
    if z.ndim!=3 or D.ndim!=2 or z.shape[-1]!=D.shape[0]:raise ValueError('Trajectory shape')
    n=z.shape[0];ids=np.arange(n) if ids is None else np.asarray(ids)
    w=np.ones(n) if weights is None else np.asarray(weights,dtype=float)
    if len(ids)!=n or w.shape!=(n,) or np.any(w<0):raise ValueError('Object weights')
    gram=D@D.T;norm2=np.einsum('nkr,rs,nks->nk',z,gram,z).clip(0)
    norm=np.sqrt(norm2);scale=max(float(norm.max(initial=0)),1.)
    valid=norm>1e-10*scale
    unit=np.divide(z,norm[:,:,None],out=np.zeros_like(z),where=valid[:,:,None])
    cos=np.einsum('ikr,rs,jks->ijk',unit,gram,unit).clip(-1,1)
    pairs=(ids[:,None]!=ids[None,:]) & (np.arange(n)[:,None]<np.arange(n)[None,:])
    pairw=w[:,None]*w[None,:]*pairs;den=pairw.sum()*z.shape[1]
    value=float((cos*pairw[:,:,None]).sum()/den) if den>0 else None
    return dict(consistency=value,zero_change_fraction=float((~valid).mean()),decoded_change_RMS=float(np.sqrt(norm2.mean())),distinct_objects=int(len(set(ids.tolist()))))

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid p-values')
    order=np.argsort(p);adjusted=np.empty(len(p));running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]);adjusted[index]=min(1.,running)
    return adjusted

def paired_summary(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2 or not np.isfinite(a-b).all():raise ValueError('Paired observations required')
    d=a-b;mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/np.sqrt(len(d))
    half=float(student.ppf(.975,len(d)-1)*se)
    p=float(student.sf(mean/se,len(d)-1)) if se else (0. if mean>0 else 1. if mean<0 else .5)
    return dict(difference=mean,sample_SD=sd,paired_t95_low=mean-half,paired_t95_high=mean+half,one_sided_p=p,n_pairs=len(d))

def quadratic_kappa(a,b,levels=5):
    a=np.asarray(a,int)-1;b=np.asarray(b,int)-1
    if a.shape!=b.shape or not len(a) or np.any((a<0)|(a>=levels)|(b<0)|(b>=levels)):raise ValueError('Rating range')
    obs=np.zeros((levels,levels));np.add.at(obs,(a,b),1);obs/=len(a)
    expected=obs.sum(1)[:,None]*obs.sum(0)[None,:]
    weight=(np.arange(levels)[:,None]-np.arange(levels)[None,:])**2/(levels-1)**2
    den=float((expected*weight).sum())
    return None if den==0 else float(1-(obs*weight).sum()/den)
