"""Same tied-score whole-image bootstrap, bounded vectorized torch reductions."""
import numpy as np
import torch
from coco_baselines import metrics
from coco_ggf import study

@torch.no_grad()
def weighted_many(prepared,counts,device,batch=32):
    idx=torch.as_tensor(prepared['images'],device=device,dtype=torch.long)
    ends=torch.as_tensor(prepared['ends'],device=device,dtype=torch.long)
    y=torch.as_tensor(prepared['y'],device=device,dtype=torch.float64)
    valid=torch.as_tensor(prepared['valid'],device=device,dtype=torch.float64)
    result=[]
    for lo in range(0,len(counts),batch):
        weights=torch.as_tensor(counts[lo:lo+batch],device=device,dtype=torch.float64)[:,idx]
        tp=torch.cumsum(weights*y,1);known=torch.cumsum(weights*valid,1)
        te=tp[:,ends];ke=known[:,ends]
        precision=torch.where(ke>0,te/ke,torch.zeros_like(te))
        increments=torch.diff(te,dim=1,prepend=torch.zeros((len(te),1),device=device,dtype=torch.float64))
        score=(increments*precision).sum(1)/tp[:,-1]
        score[(tp[:,-1]==0)|(tp[:,-1]==known[:,-1])]=float('nan')
        result.append(score.cpu().numpy())
    return np.concatenate(result)

def paired(a,b,cfg,device='cpu'):
    if (a['image_ids'],a['category_indices'],a['objects'])!=(b['image_ids'],b['category_indices'],b['objects']):raise ValueError('Paired audit identities differ')
    for k in ['positive','valid','patch_image_index','denominator']:
        if not np.array_equal(a[k],b[k]):raise ValueError('Paired audit data differ: '+k)
    n=len(a['image_ids']);C=len(a['category_indices']);rng=np.random.default_rng(cfg['fresh_test_seed']+91)
    counts=np.asarray([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(cfg['bootstrap'])],np.float64)
    den=counts@a['denominator'];num=counts@(b['error']-a['error'])
    caps=np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-12);aps=[]
    for j in range(C):
        pa=study.prepared_ap(a['detection_scores'][:,j],a['positive'][:,j],a['valid'][:,j],a['patch_image_index'])
        pb=study.prepared_ap(b['detection_scores'][:,j],b['positive'][:,j],b['valid'][:,j],b['patch_image_index'])
        aps.append(weighted_many(pa,counts,device)-weighted_many(pb,counts,device))
    aps=np.asarray(aps).T;ok=np.isfinite(caps).all(1)&np.isfinite(aps).all(1)
    if not ok.any():raise ValueError('No full-cohort paired bootstrap replicate')
    ca=caps[ok];da=aps[ok]
    return dict(capture_gain_ci95=metrics.interval(ca.mean(1)),ap_gain_ci95=metrics.interval(da.mean(1)),
        per_category=[dict(category_index=c,capture_gain_ci95=metrics.interval(ca[:,j]),ap_gain_ci95=metrics.interval(da[:,j])) for j,c in enumerate(a['category_indices'])],
        valid_bootstrap_replicates=int(ok.sum()),invalid_full_cohort_replicates=int((~ok).sum()),
        backend='Bounded float64 torch; stable sort and tie bins identical to reference CPU method; same image draws',
        note='Paired whole-image bootstrap; full fixed category denominator, conditional on fitted weights and cohort; seed variation reported separately.')

