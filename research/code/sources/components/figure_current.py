"""Measured 12D GGF plant/vase visualization. Inference only; no model updates."""
import gc,json,hashlib,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter,map_coordinates
from coco_baselines import data,training
from coco_baselines.storage import identity,sha_file
from coco_ggf import matched_baselines as matched
from scale_models import model

def field(a,res=1024):
    a=np.asarray(a,float).reshape(14,14);s=gaussian_filter(a,.35,mode='nearest')
    t=(np.arange(res)+.5)*14/res-.5;y,x=np.meshgrid(t,t,indexing='ij')
    return np.clip(map_coordinates(s,[y,x],order=3,mode='nearest'),a.min(),a.max())

def projection(c,decoder):
    # PCA on actual decoded contributions. QR reduces the 768D computation
    # exactly to the decoder span, without treating raw coefficients as metric.
    q,r=np.linalg.qr(np.asarray(decoder,float).T,mode='reduced')
    m=np.asarray(c,float)@r.T;mean=m.mean(0)
    _,s,vt=np.linalg.svd(m-mean,full_matrices=False);v=vt[:3]
    for a in v:
        if a[np.argmax(np.abs(a))]<0:a*=-1
    transform=r.T@v.T;center=mean@v.T;u=np.asarray(c,float)@transform-center
    return transform,center,u,(s[:3]**2/max(float(np.sum(s**2)),1e-30)).tolist()

def rgb(u,lo,hi):return .08+.87*np.clip((u-lo)/np.maximum(hi-lo,1e-12),0,1)

def cached_export_valid(out,archive,signature):
    """Accept the first delivered export as well as newer per-file audits."""
    out,archive=Path(out),Path(archive);record=out/'figure1_current.json'
    if not record.is_file() or not archive.is_file():return False
    try:
        cached=json.loads(record.read_text())
        if cached.get('signature')!=signature:return False
        hashes=cached.get('artifact_hashes')
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:return False
            if not hashes:hashes=json.loads(z.read('SHA256.json'))
            names=['figure1_current.pdf','figure1_current.png','figure1_current.npz']
            return all(n in hashes and (out/n).is_file() and sha_file(out/n)==hashes[n] and hashlib.sha256(z.read(n)).hexdigest()==hashes[n] for n in names)
    except (OSError,ValueError,KeyError,zipfile.BadZipFile):return False

def make(ctx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out=ctx['store'].results/'figure1_current';out.mkdir(exist_ok=True)
    row=next(r for r in ctx['base_context']['rows'] if r['seed']==23 and r['kind']=='agf_g')
    src=Path(__file__).resolve().parent;meta=json.loads((src/'figure_input.json').read_text())
    if identity(ctx['norm'])!=meta['normalization_sha256']:raise ValueError('Figure normalization differs from latest GGF')
    fp=src/'figure_input.npz';photo_path=src/'plant_vase.jpg'
    if sha_file(fp)!=meta['features_sha256'] or sha_file(photo_path)!=meta['image_sha256']:raise ValueError('Figure input hash mismatch')
    train=ctx['packs']['train'];ids=train['ids'][:256]
    signature=identity(dict(weights=row['weights_sha'],targets=row['targets_sha'],train_identity=train['identity'],ids=ids,input=meta,version=1))
    record_path=out/'figure1_current.json';arrays=out/'figure1_current.npz'
    if cached_export_valid(out,ctx['store'].output/'ggf_figure1_current.zip',signature):
        print('FIGURE_READY '+str(out/'figure1_current.png'),flush=True);return out
    targets=[]
    for name in ['potted plant','vase']:
        matches=[t for t in row['targets'] if t['category'].lower()==name]
        if len(matches)!=1:raise ValueError('Missing unique TRAIN-selected '+name+' readout')
        targets.append(matches[0])
    groups=[t['detector_group'] for t in targets]
    weights=torch.load(ctx['base_context']['local']/'verified_weights/s23_agf_g.pt',map_location='cpu',weights_only=True)
    if training.weight_sha(weights)!=row['weights_sha']:raise ValueError('Current figure checkpoint changed')
    m=model('released_vanilla',4096,12,32,768).to(ctx['device']);m.load_state_dict(weights);m.eval()
    raw=np.load(fp)['raw_features'];image_x=data.normalize(raw[None],ctx['norm'])[0];all_codes=[]
    x=np.asarray(train['x'][:len(ids)]).reshape(-1,768)
    with torch.inference_mode():
        for lo in range(0,len(x),512):
            z=m.encode(torch.as_tensor(np.array(x[lo:lo+512],copy=True),device=ctx['device']))
            all_codes.append(z[:,groups].cpu().numpy());del z
        image_code=m.encode(torch.as_tensor(image_x,device=ctx['device']))[:,groups].cpu().numpy()
    all_codes=np.concatenate(all_codes);dec=weights['W_dec'].reshape(4096,12,768)[groups].numpy()
    del m,weights;gc.collect();torch.cuda.empty_cache()
    geom=meta['geometry'];im=Image.open(photo_path).convert('RGB');ow,oh=geom['original'];rw,rh=geom['resized'];l,t,r,b=geom['box']
    photo=np.asarray(im.transform((1024,1024),Image.Transform.EXTENT,(l*ow/rw,t*oh/rh,r*ow/rw,b*oh/rh),resample=Image.Resampling.BICUBIC),float)/255
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':14.6})
    fig=plt.figure(figsize=(12.5,4.4),facecolor='white');records=[];saved={};panels=[]
    # All labels sit outside images. No black plates, dashed divider or fabricated points.
    def img(a,box):
        ax=fig.add_axes(box);ax.imshow(a,interpolation='lanczos');ax.axis('off')
    maxnorm=max(np.linalg.norm(image_code[:,j],axis=-1).max() for j in range(2))
    for j,(name,target) in enumerate(zip(['Potted plant','Vase'],targets)):
        code=all_codes[:,j];active=np.linalg.norm(code,axis=1)>0;active_code=code[active]
        if len(active_code)<4:raise ValueError(name+' has insufficient active TRAIN patches for a 3D projection')
        transform,center,u,fractions=projection(active_code,dec[j]);lo,hi=u.min(0),u.max(0)
        iu=image_code[:,j]@transform-center;cloud_colors=rgb(u,lo,hi);image_colors=rgb(iu,lo,hi)
        # Display all points up to 3000; larger clouds use a fixed uniform subset.
        rng=np.random.default_rng(20261015+j);ix=np.sort(rng.choice(len(u),min(3000,len(u)),replace=False))
        y=.55 if j==0 else .08
        ax=fig.add_axes([.01,y,.21,.36],projection='3d');p=u[ix]
        ax.scatter(*p.T,c=cloud_colors[ix],s=2.0,alpha=.83,depthshade=False,linewidths=0,rasterized=True)
        extent=np.maximum(np.ptp(p,axis=0),1e-8);ax.set_box_aspect(extent);ax.view_init(27,-57);ax.set_axis_off()
        floor=p[:,2].min()-.07*extent[2];bounds=[(p[:,a].min(),p[:,a].max()) for a in [0,1]]
        ax.add_collection3d(Poly3DCollection([[(bounds[0][0],bounds[1][0],floor),(bounds[0][1],bounds[1][0],floor),(bounds[0][1],bounds[1][1],floor),(bounds[0][0],bounds[1][1],floor)]],facecolor='#f1f3f4',edgecolor='#d6dade',linewidth=.3,alpha=.5))
        fig.text(.115,y+.36,name+' responses',ha='center',fontsize=15)
        fig.text(.115,y-.005,f'3 PCs: {100*sum(fractions):.1f}% variance',ha='center',fontsize=14.6,color='#52606a')
        norm=np.linalg.norm(image_code[:,j],axis=-1);strength=field(norm);alpha=.8*np.clip(strength/max(maxnorm,1e-12),0,1)**.65
        hue=np.stack([field(image_colors[:,k]*norm) for k in range(3)],axis=-1)/np.maximum(strength[...,None],1e-12)
        overlay=photo*(1-alpha[...,None])+np.clip(hue,0,1)*alpha[...,None];panels.append(overlay)
        img(overlay,[.235,y-.005,.15,.38]);fig.text(.31,y+.375,'Group '+str(groups[j]),ha='center',fontsize=14.6)
        scalar=np.abs(image_code[:,j,0]);v=field(scalar)/max(float(np.abs(image_code[:,:,0]).max()),1e-12)
        colors=plt.get_cmap('Reds')(.1+.9*v)[...,:3];a=.88*v**.7;background=.23*photo+.77
        img(background*(1-a[...,None])+colors*a[...,None],[.40,y-.005,.15,.38]);fig.text(.475,y+.375,'Direction 1',ha='center',fontsize=14.6,color='#314955')
        fig.add_artist(FancyArrowPatch((.38,y+.18),(.402,y+.18),transform=fig.transFigure,arrowstyle='->',mutation_scale=10,color='#355f73',linewidth=1.1))
        saved.update({f'{j}_all_active_code':active_code,f'{j}_active_flat_indices':np.flatnonzero(active),f'{j}_plot_indices':ix,
            f'{j}_decoder':dec[j],f'{j}_coefficient_to_pc':transform,f'{j}_pc_center':center,f'{j}_cloud_pc':u,
            f'{j}_image_code':image_code[:,j],f'{j}_image_pc':iu,f'{j}_cloud_rgb':cloud_colors,f'{j}_image_rgb':image_colors})
        records.append(dict(category=name,group=groups[j],category_index=target['category_index'],active_training_patches=int(active.sum()),
            displayed_points=len(ix),pca_variance_fractions=fractions,color_min=lo.tolist(),color_max=hi.tolist(),decoder_width=12))
    norms=np.stack([field(np.linalg.norm(image_code[:,j],axis=-1)) for j in range(2)],axis=-1)
    colors=np.array([[.03,.90,.75],[.86,.05,.91]])
    joint=(norms@colors)/np.maximum(norms.sum(-1,keepdims=True),1e-12)
    alpha=.82*np.clip(norms.max(-1)/max(maxnorm,1e-12),0,1)**.62
    combined=photo*(1-alpha[...,None])+joint*alpha[...,None]
    img(combined,[.575,.1,.29,.81]);fig.text(.72,.925,'Joint group activity',ha='center',fontsize=15)
    fig.text(.72,.08,'Plant / cyan; vase / magenta',ha='center',fontsize=14.6,color='#46535e')
    img(panels[0],[.87,.54,.125,.36]);img(panels[1],[.87,.10,.125,.36])
    fig.text(.932,.91,'Plant / RGB',ha='center',fontsize=14.6);fig.text(.932,.47,'Vase / RGB',ha='center',fontsize=14.6)
    fig.text(.5,.015,'GGF / 4,096 groups / 12D / Top-32 / seed 23. Clouds: 3D projections of active training contributions.',ha='center',fontsize=14.6,color='#46535e')
    for ext in ['pdf','png']:fig.savefig(out/('figure1_current.'+ext),dpi=450)
    plt.close(fig);np.savez_compressed(arrays,**saved,training_image_ids=ids,image_id=meta['image_id'])
    record=dict(signature=signature,seed=23,weights_sha=row['weights_sha'],pilot_run=ctx['parent'].run_id,groups=4096,width=12,k=32,cap=.6,
        training_cohort_ids=ids,image_input=meta,groups_displayed=records,projection='Centered PCA of decoded contributions on active patches from the first256 fixed TRAIN images; 3D projection, not intrinsic dimension.',
        colors='The exact same TRAIN-fitted PC-to-RGB function colors points and individual-group image patches; full 12D norm controls overlay opacity. The large joint-presence panel instead uses categorical cyan/magenta.',
        interpolation='Gaussian sigma0.35 patch units then clipped cubic scalar interpolation at patch centers. Display only.',
        sampling='Fixed uniform subset of up to3000 active training patches per group, without mask or geometry selection.',retraining=False,test_selection=False,
        artifact_hashes={p.name:sha_file(p) for p in [arrays,out/'figure1_current.pdf',out/'figure1_current.png']})
    record_path.write_text(json.dumps(record,indent=2))
    archive=ctx['store'].output/'ggf_figure1_current.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file():z.write(p,p.name)
        for name in ['plant_vase.jpg','figure_input.json','figure_current.py']:z.write(src/name,'source/'+name)
        z.writestr('SHA256.json',json.dumps({p.name:sha_file(p) for p in out.iterdir() if p.is_file()},indent=2))
    matched.assert_sources(ctx);print('FIGURE_READY '+str(out/'figure1_current.png'),flush=True)
    print('New measured Figure1 export:',archive,flush=True);return out
