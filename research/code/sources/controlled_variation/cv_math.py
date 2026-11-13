"""CPU analysis functions. Every fitted transformation takes TRAIN arrays only."""
import hashlib, json, os, io
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance

LEVELS={'rotation':[-30.,-15.,0.,15.,30.], 'scale':[.75,.875,1.,1.125,1.25], 'brightness':[.6,.8,1.,1.2,1.4], 'background':[.6,.8,1.,1.2,1.4]}
VARIANTS=[(a,v) for a,vs in LEVELS.items() for v in vs]

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()

def safe_output(output,sources):
    p=Path(output).resolve()
    for source in sources:
        q=Path(source).resolve()
        if p==q or p in q.parents or q in p.parents:raise ValueError('Study output overlaps a read-only source')
    return p

def render(rgb,mask,attribute,value):
    """Same crop and neutral background for every method. Geometry acts on RGBA."""
    rgb=np.asarray(rgb,np.uint8);mask=np.asarray(mask,np.uint8)
    if rgb.shape!=(224,224,3) or mask.shape!=(224,224):raise ValueError('Expected pinned 224-pixel crop')
    alpha=mask*255 if mask.max()<=1 else mask
    rgba=Image.fromarray(np.dstack([rgb,alpha]))
    box=rgba.getchannel('A').getbbox()
    if box is None:raise ValueError('Empty object mask')
    piece=rgba.crop(box);factor=144/max(piece.size)
    piece=piece.resize(tuple(max(1,round(v*factor)) for v in piece.size),Image.Resampling.BICUBIC)
    rgba=Image.new('RGBA',(224,224));rgba.paste(piece,((224-piece.width)//2,(224-piece.height)//2))
    if attribute=='rotation':rgba=rgba.rotate(value,resample=Image.Resampling.BICUBIC,expand=False)
    elif attribute=='scale':
        n=round(224*value);layer=rgba.resize((n,n),Image.Resampling.BICUBIC)
        rgba=Image.new('RGBA',(224,224));rgba.paste(layer,((224-n)//2,(224-n)//2))
    elif attribute=='brightness':
        a=rgba.getchannel('A');rgba=ImageEnhance.Brightness(rgba.convert('RGB')).enhance(value).convert('RGBA');rgba.putalpha(a)
    elif attribute!='background':raise ValueError('Unknown transformation')
    background=round(np.clip(114*(value if attribute=='background' else 1.),0,255))
    image=Image.alpha_composite(Image.new('RGBA',(224,224),(background,)*3+(255,)),rgba).convert('RGB')
    m=np.asarray(rgba.getchannel('A'),np.float32)/255
    fraction=m.reshape(14,16,14,16).mean((1,3)).reshape(196)
    return image,fraction

def target(attribute,value):
    return np.log(value) if attribute=='scale' else value

def pca_from_stats(n,total,cross,D):
    """Exact PCA of centered decoded contributions zD, computed in block space."""
    b,d=D.shape
    if n<2:return dict(valid=False,mean=np.zeros(b),V=np.zeros((3,d)),variance=np.zeros(3),limits=np.ones(3),reason='fewer than two active TRAIN patches')
    mean=total/n;cov=(cross-n*np.outer(mean,mean))/(n-1);ev,U=np.linalg.eigh((cov+cov.T)/2)
    root=(U*np.sqrt(np.maximum(ev,0)))@U.T
    _,sv,V=np.linalg.svd(root@D,full_matrices=False);v=V[:3].copy()
    for row in v:
        if row[np.argmax(abs(row))]<0:row*=-1
    var=np.zeros(3);var[:min(3,len(sv))]=sv[:3]**2
    return dict(valid=bool(var[0]>1e-12),mean=mean,V=v,variance=var,
                variance_fraction=float(var.sum()/max(np.sum(sv**2),1e-30)),
                limits=np.maximum(2*np.sqrt(var),1e-6),samples=int(n))

def project(z,D,pca):
    return (z-pca['mean'])@D@pca['V'].T

def fit_ridge(X,Y,alpha=.001):
    """Fixed regularization, TRAIN-only centering/scaling; intercept unpenalized."""
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    mu=X.mean(0);sd=np.maximum(X.std(0),1e-6);Z=np.c_[(X-mu)/sd,np.ones(len(X))]
    reg=np.eye(Z.shape[1])*alpha;reg[-1,-1]=0
    if X.shape[1]>len(X):
        A=Z[:,:-1];ym=Y.mean(0)
        coef=A.T@np.linalg.solve(A@A.T+len(X)*alpha*np.eye(len(X)),Y-ym)
        W=np.concatenate([coef,np.asarray(ym)[None]],axis=0)
    else:W=np.linalg.solve(Z.T@Z/len(Z)+reg,Z.T@Y/len(Z))
    return dict(mean=mu,scale=sd,W=W)

def predict(probe,X):
    return np.c_[(np.asarray(X)-probe['mean'])/probe['scale'],np.ones(len(X))]@probe['W']

def r2(y,p):
    den=np.square(y-np.mean(y)).sum()
    return None if den<=1e-12 else float(1-np.square(y-p).sum()/den)

def capture_terms(d,r):
    return float(np.square(d).sum()),float(np.square(d-r).sum())

def transplant(x,source,target_part):
    return x-source+target_part

def energy_match(control,delta):
    a=np.linalg.norm(control);b=np.linalg.norm(delta)
    return None if a<=1e-12 else control*(b/a)

class Cache:
    """Transactional local/Drive shards. A sidecar commits exact content bytes."""
    def __init__(self,local,drive,protocol):
        self.local=Path(local);self.drive=Path(drive);self.protocol=protocol
        self.local.mkdir(parents=True,exist_ok=True);self.drive.mkdir(parents=True,exist_ok=True)
    def get(self,key):
        for root in [self.local,self.drive]:
            p=root/(key+'.npz');m=p.with_suffix('.json')
            if not p.exists() or not m.exists():continue
            meta=json.loads(m.read_text());raw=p.read_bytes()
            if meta['protocol']!=self.protocol:raise ValueError('Cache protocol changed')
            if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('Committed cache checksum mismatch: '+str(p))
            with np.load(io.BytesIO(raw),allow_pickle=False) as f:out={k:f[k] for k in f.files}
            if root!=self.local:self._write(self.local,key,raw,meta)
            return out
        return None
    def _write(self,root,key,raw,meta):
        p=root/(key+'.npz');tmp=p.with_suffix('.pending');tmp.write_bytes(raw)
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=meta['sha256']:raise OSError('Cache write verification failed')
        os.replace(tmp,p);m=p.with_suffix('.json');tmp=m.with_suffix('.pending');tmp.write_text(json.dumps(meta));os.replace(tmp,m)
    def put(self,key,**arrays):
        buf=io.BytesIO();np.savez_compressed(buf,**arrays);raw=buf.getvalue()
        meta=dict(protocol=self.protocol,sha256=hashlib.sha256(raw).hexdigest())
        for root in [self.local,self.drive]:self._write(root,key,raw,meta)
        return arrays

def image_bootstrap(records,value,draws=1000):
    """Average variants WITHIN base image, then resample base images."""
    by={}
    for row in records:by.setdefault(row['image_id'],[]).append(row[value])
    v=np.asarray([np.mean(a) for a in by.values()]);rng=np.random.default_rng(42010)
    if len(v)<2:return [None,None]
    means=np.asarray([np.mean(v[rng.integers(len(v),size=len(v))]) for _ in range(draws)])
    return np.quantile(means,[.025,.975]).tolist()
