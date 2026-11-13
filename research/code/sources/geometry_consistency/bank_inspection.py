"""Controls first, all-unit inspection second. Reuse V1 inputs, version metrics."""
import gc,html,json,os
from pathlib import Path
import numpy as np
import torch
from cv_math import Cache,digest
from study_math import pca_fit,pca_rgb

REVISION='whole_bank_controls_first_v2'

def release(*items):
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def prepare_bank(ctx,row,source,stores,rp,tp):
    """Cheap bank labeling only. No neighbor calculations or PCA required."""
    import study as s
    m,D,A,units=s.load(ctx,row,source,stores)
    ref=s.encode_panel(ctx,row,ctx['packs']['select'],rp,m,D,A)
    test=s.encode_panel(ctx,row,ctx['packs']['confirm'],tp,m,D,A)
    scale_path=ctx['out']/'response_scales'/f"{row['kind']}_{row['seed']}.npy"
    scales=np.maximum(np.sqrt(np.mean(ref['score'].astype(float)**2,axis=0)),1e-8)
    if scale_path.exists():
        if not np.array_equal(np.load(scale_path),scales):raise ValueError('Existing response scale differs')
    else:
        temp=scale_path.with_suffix('.pending.npy');np.save(temp,scales);os.replace(temp,scale_path)
    path=ctx['out']/'discovery'/f"{row['kind']}_{row['seed']}.json"
    discovery=s.read_done(path)
    if discovery is None:
        ap,prev=s.ap_profiles(ref['score'],rp);bp,bprev=s.ap_profiles(test['score'],tp)
        uplift=np.maximum((ap-prev)/(1-prev),0);testup=np.maximum((bp-bprev)/(1-bprev),0)
        discovery=[]
        for g in range(len(ap)):
            a=np.nan_to_num(uplift[g]);c=int(a.argmax());assigned=bool(a.max()>0 and np.any(ref['score'][:,g]>1e-10))
            total=float(np.nansum(testup[g]));cohort=('directly_guided' if g<75 else 'not_directly_guided') if row['kind']=='agf_g' else ('guided_token' if g<75 else 'free_token') if row['kind']=='concept_sae' else ('category_dictionary' if row['kind']=='slics' else 'other_groups' if g not in units else 'category_readout')
            val=float(bp[g,c]);pre=float(bprev[c])
            if not np.isfinite(val):val=0.
            discovery.append(dict(method=row['kind'],seed=row['seed'],unit=g,scope=cohort,category=int(ctx['cats'][c]) if assigned else None,assigned=assigned,test_AP=val if assigned else 0.,test_uplift=(val-pre)/(1-pre) if assigned and np.isfinite(pre) else 0.,test_focus=float(np.nan_to_num(testup[g,c])/total) if assigned and total>0 else 0.,stable_category=bool(assigned and c==np.nan_to_num(testup[g]).argmax()),calibration_activity=float(np.mean(ref['score'][:,g]>1e-10))))
        s.done_json(path,discovery);s.summarize(ctx,'discovery')
    if len(discovery)!=ref['score'].shape[1]:raise ValueError('Incomplete native-bank inventory')
    return m,D,A,units,ref,test,discovery

def unit_pca(ctx,row,g,ref,test,rp,tp,D,A):
    """Reuse compatible old PCA, or fit once to all active calibration labels."""
    import study as s
    sig=digest(dict(revision=REVISION,source=ctx['signature'],weights=row['weights_sha'],unit=g))
    cache=Cache(Path(ctx['options']['local'])/'atlas_pca_v2'/sig,ctx['out']/'atlas_pca_v2'/sig,sig)
    saved=cache.get('pca')
    if saved is not None:
        return None if not bool(saved['valid']) else {k:v for k,v in saved.items() if k!='valid'}
    active=np.flatnonzero(ref['score'][:,g]>1e-10)
    if len(active)>1024:active=np.random.default_rng(818+g).choice(active,1024,replace=False)
    x=s.embed(ref,rp,g,active,D,A)
    legacy=ctx['out']/'pca'/row['kind']/str(row['seed'])/f'unit_{g}.npz'
    pc=None
    if legacy.exists() and len(x)>=2:
        with np.load(legacy,allow_pickle=False) as old:
            if (np.array_equal(old['reference_ids'],rp['ids'][active]) and np.array_equal(old['reference_patch'],rp['patch'][active])
                and old['axes'].shape[1]==x.shape[1] and np.allclose(old['mean'],x.mean(0))):
                candidate={k:old[k].copy() for k in ['mean','axes','lo','hi','variance_fraction']}
                if all(np.isfinite(v).all() for v in candidate.values()) and np.allclose(candidate['axes']@candidate['axes'].T,np.eye(len(candidate['axes'])),atol=1e-6):pc=candidate
    if pc is None:pc=pca_fit(x)
    if pc is None or np.square(x-x.mean(0)).sum()<=1e-20:
        cache.put('pca',valid=np.array(False));return None
    cache.put('pca',valid=np.array(True),**pc)
    return pc

def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(text,encoding='utf8');os.replace(tmp,path)

def _color(v):return '#'+''.join(f'{int(round(float(a)*255)):02x}' for a in np.clip(v,0,1))

def patch_overlay(url,patch,values,colors=False,scale=1.):
    parts=[f'<svg viewBox="0 0 224 224" role="img"><image href="{html.escape(url,quote=True)}" width="224" height="224"/>']
    for p,value in zip(patch,values):
        if colors:
            color=_color(value);opacity=.55
        else:
            strength=float(np.clip(value/max(scale,1e-10),0,1));color='#009c95';opacity=.72*strength
        if opacity>0:parts.append(f'<rect x="{int(p)%14*16}" y="{int(p)//14*16}" width="16" height="16" fill="{color}" opacity="{opacity:.3f}"/>')
    parts.append('</svg>');return ''.join(parts)

STYLE='''<style>body{font:15px Arial,sans-serif;margin:24px;color:#20262b;background:#fafafa}a{color:#007b78}.card{background:white;border:1px solid #ddd;padding:16px;margin:16px 0;border-radius:8px}.views{display:flex;gap:12px;flex-wrap:wrap}.pair{display:flex;gap:4px}.image{width:160px}svg{width:100%}.scatter{width:260px}small{color:#4c555b}nav{position:sticky;top:0;background:#fafafa;padding:10px}table{border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #ddd}</style>'''

def gallery(ctx,row,m,D,A,units,ref,test,rp,tp,discovery):
    """Every native unit, including inactive and category-unassigned units.

    Uses existing 32-patch panel encodings. Heatmaps explicitly show sampled
    locations, not a new dense 196-patch forward pass. No favorable-case filter.
    """
    import study as s
    from PIL import Image
    from coco_baselines import data
    root=ctx['out']/'whole_bank_atlas_v2';folder=root/row['kind']/str(row['seed']);folder.mkdir(parents=True,exist_ok=True)
    images=root/'images';images.mkdir(exist_ok=True)
    G=ref['score'].shape[1];cards=[];inventory=[]
    image_order=np.unique(tp['images'])
    # At most one download/resize per source image, shared by all models/units.
    def image_url(i):
        image_id=int(ctx['packs']['confirm']['ids'][i]);target=images/f'{image_id}.jpg'
        if not target.exists():
            p=data.download(data.COCO_BUCKET+f'train2017/{image_id:012d}.jpg',Path(ctx['options']['local'])/'images'/f'{image_id:012d}.jpg')
            with Image.open(p) as im:
                transformed=data.transform_image(im.convert('RGB'),data.geometry(*im.size))
                pending=target.with_suffix('.pending');transformed.save(pending,format='JPEG',quality=92);os.replace(pending,target)
        return f'../../images/{image_id}.jpg'
    for entry in discovery:
        g=entry['unit'];committed=folder/'cards'/f'{g}.json';card=s.read_done(committed)
        if card is None:
            pc=unit_pca(ctx,row,g,ref,test,rp,tp,D,A)
            strength=np.array([test['score'][tp['images']==i,g].max(initial=0) for i in image_order])
            chosen=image_order[np.argsort(-strength,kind='stable')[:2]]
            nonzero=bool(np.any(strength>1e-10))
            info=dict(unit=g,category=entry['category'],scope=entry['scope'],calibration_activity=entry['calibration_activity'],evaluation_activity=float(np.mean(test['score'][:,g]>1e-10)),pca_status='available' if pc is not None else 'inactive_or_constant',evaluation_image_ids=[int(ctx['packs']['confirm']['ids'][i]) for i in chosen] if nonzero else [])
            views=[];scale=max(float(np.quantile(ref['score'][:,g],.99)),1e-8)
            if nonzero:
                for i in chosen:
                    ix=np.flatnonzero(tp['images']==i);url=image_url(i);v=s.embed(test,tp,g,ix,D,A)
                    norm=patch_overlay(url,tp['patch'][ix],test['score'][ix,g],scale=scale)
                    on=test['score'][ix,g]>1e-10
                    color=patch_overlay(url,tp['patch'][ix][on],pca_rgb(v,pc)[on],colors=True) if pc is not None else '<p>No nonconstant PCA</p>'
                    views.append(f'<div><small>Image {int(ctx["packs"]["confirm"]["ids"][i])}</small><div class="pair"><div class="image">Response{norm}</div><div class="image">PCA coordinates{color}</div></div></div>')
            scatter=''
            if pc is not None:
                ix=np.flatnonzero(test['score'][:,g]>1e-10)
                if len(ix)>128:ix=np.random.default_rng(91023+g).choice(ix,128,False)
                v=s.embed(test,tp,g,ix,D,A);u=(v-pc['mean'])@pc['axes'].T
                if len(u):
                    xy=np.pad(u[:,:2],((0,0),(0,max(0,2-u.shape[1]))))
                    lower=np.minimum(xy.min(0),np.pad(pc['lo'][:2],(0,max(0,2-len(pc['lo'])))))
                    upper=np.maximum(xy.max(0),np.pad(pc['hi'][:2],(0,max(0,2-len(pc['hi'])))))
                    xy=(xy-lower)/np.maximum(upper-lower,1e-8)*200+20
                    dots=''.join(f'<circle cx="{x:.2f}" cy="{240-y:.2f}" r="2" fill="{_color(c)}"/>' for (x,y),c in zip(xy,pca_rgb(v,pc)))
                    scatter=f'<div class="scatter"><small>Evaluation contributions, PC1 / PC2</small><svg viewBox="0 0 240 240">{dots}</svg></div>'
            categories=ctx.get('provenance',{}).get('categories',[])
            category=entry['category']
            name=categories[category]['name'] if category is not None and len(categories)>category else f'index {category}'
            label='No positive annotated-category association' if category is None else f'Calibration-associated category: {html.escape(name)}'
            legend=f'<div style="width:180px;height:12px;background:linear-gradient(90deg,white,#009c95)"></div><small>Native response 0 to {scale:.4g}, fixed on calibration images</small>'
            text=f'<section class="card" id="unit-{g}"><h2>Unit {g} · {html.escape(entry["scope"])}</h2><p>{label}. Active calibration patches {entry["calibration_activity"]:.1%}. Active evaluation patches {info["evaluation_activity"]:.1%}.</p>'+legend+'<div class="views">'+''.join(views)+scatter+'</div>'
            if not nonzero:text+='<p>No nonzero response on the evaluation panel. Retained in the inventory.</p>'
            text+='<small>Same frozen unit across images. Images are its two strongest sampled evaluation responses, selected independently per method, so this is discovery rather than a matched-image superiority comparison. Only sampled patch locations are colored. Teal strength uses a fixed calibration 99th percentile. RGB encodes PCA location and is not aligned across groups or methods.</small></section>'
            card=dict(info=info,html=text);s.done_json(committed,card)
        cards.append(card['html']);inventory.append(card['info'])
        if (g+1)%64==0 or g+1==G:s.log('WHOLE BANK INSPECTION',method=row['kind'],seed=row['seed'],units=g+1,total=G)
    pages=(G+31)//32
    for page in range(pages):
        nav=' '.join(f'<a href="page_{j+1}.html">{j+1}</a>' for j in range(pages))
        atomic_text(folder/f'page_{page+1}.html',f'<!doctype html><meta charset="utf-8"><title>{row["kind"]} bank</title>{STYLE}<nav><a href="../../index.html">All methods</a> | {nav}</nav><h1>{row["kind"]}, run {row["seed"]}</h1><p>Every native unit is retained. Geometry and images are descriptive evidence, not a semantic correctness score.</p>'+''.join(cards[page*32:(page+1)*32]))
    s.csv_write(folder/'inventory.csv',inventory)
    s.done_json(folder/'COMPLETE.json',dict(complete=True,units=G,revision=REVISION))
    rebuild_index(root)

def rebuild_index(root):
    links=[]
    for path in sorted(root.glob('*/*/COMPLETE.json')):
        method=path.parent.parent.name;seed=path.parent.name
        inventory=path.parent/'inventory.csv'
        links.append(f'<li><a href="{method}/{seed}/page_1.html">{method}, run {seed}</a> · <a href="{method}/{seed}/inventory.csv">Every-unit inventory</a></li>')
    atomic_text(root/'index.html',f'<!doctype html><meta charset="utf-8"><title>Whole-bank inspection</title>{STYLE}<h1>Whole-bank response and contribution-PCA inspection</h1><p>All five methods and all saved runs. Every native unit is included, including units without an annotated-category association. These are 32 sampled patch locations per image. The full uncolored image remains visible. Teal denotes response strength and RGB denotes PCA coordinates. Each unit chooses its own strongest evaluation images. Inspect these as discovery examples, not as matched comparisons or evidence that the colors identify a semantic manifold.</p><ul>'+''.join(links)+'</ul>')

def run_bank_v2(ctx,row,source,stores,rp,tp):
    import study as s
    complete=ctx['out']/'whole_bank_v2_complete'/f'{row["kind"]}_{row["seed"]}.json'
    if s.read_done(complete) is not None:return
    m,D,A,units,ref,test,discovery=prepare_bank(ctx,row,source,stores,rp,tp)
    # Finish the browsable atlas BEFORE optional slower neighbor summaries.
    gallery(ctx,row,m,D,A,units,ref,test,rp,tp,discovery)
    for scope,entries in [('guided_geometry_v2',[dict(unit=g,category=ctx['cats'][c],scope='fixed_category_readout') for c,g in enumerate(units)]),('dictionary_geometry_v2',discovery)]:
        result=[]
        for entry in entries:
            g=entry['unit'];category=entry['category']
            key=f'{g}_{category}'
            path=ctx['out']/'geometry_units_v2'/row['kind']/str(row['seed'])/f'{key}.json'
            q=s.read_done(path)
            if q is None:
                q=dict(entry,method=row['kind'],seed=row['seed'],metric_revision=REVISION)
                if category is not None:
                    q.update(s.unit_geometry(ctx,row,g,ctx['cats'].index(category),ref,test,rp,tp,D,A,entry['scope']))
                else:q['status']='no_positive_object_association'
                s.done_json(path,q)
            result.append(q)
        s.done_json(ctx['out']/scope/f'{row["kind"]}_{row["seed"]}.json',result);s.summarize(ctx,scope)
    s.done_json(complete,dict(complete=True,revision=REVISION,units=len(discovery)))
    del m,ref,test;release()

def run_stages(ctx,rows,source,stores,rp,tp,check_sources=lambda:None):
    import study as s
    for name in ['guided_geometry_v2','dictionary_geometry_v2','whole_bank_v2_complete']:(ctx['out']/name).mkdir(exist_ok=True)
    s.atomic_json(ctx['out']/'REVISION_V2.json',dict(revision=REVISION,input_cache_identity=ctx['signature'],legacy_results='preserved, not used as corrected geometry',execution_order=['controlled_consistency','whole_bank_inspection','balanced_geometry_v2'],all_patch_geometry='Both classes retained including zero vectors, soft balanced neighbor accuracy, chance 0.5',conditional_geometry='Rebalance active reference and query classes separately. Missing if either class has fewer than three active patches.',scope='Object-category-associated organization and controlled response consistency, not proof of a unique semantic manifold or excluded-category transfer'))
    atomic_text(ctx['out']/'READ_ME_FIRST_V2.txt','Open whole_bank_atlas_v2/index.html for the every-unit inspection. Use guided_geometry_v2 and dictionary_geometry_v2 CSVs for corrected geometry, and read their metric_support CSVs. Earlier guided_geometry and dictionary_geometry purity files are preserved legacy diagnostics with unequal sparse/dense active-reference baselines. Do not use those raw purity scores to rank semantic coherence. Controlled calculations and input cache identities are unchanged. Updated controlled and discovery summary tables can be regenerated from the committed per-model records. No source checkpoints or legacy geometry records are changed. Earlier fingerprint and rotation studies remain separate evidence and are not silently substituted for a different image/patch cohort.\n')
    ordered=[next(r for r in rows if r['seed']==seed and r['kind']==kind) for seed in s.RUNS for kind in s.METHODS]
    if any(s.read_done(ctx['out']/'controlled_consistency'/f'{r["kind"]}_{r["seed"]}.json') is None for r in ordered):
        root,pairs=s.syn_pairs(ctx);features=s.extract_pairs(ctx,root,pairs)
        for row in ordered:
            if s.read_done(ctx['out']/'controlled_consistency'/f'{row["kind"]}_{row["seed"]}.json') is not None:continue
            m,D,A,units,ref,test,_=prepare_bank(ctx,row,source,stores,rp,tp)
            del m,ref,test;release()
            s.run_controls(ctx,row,source,stores,pairs,features);check_sources()
    s.summarize(ctx,'controlled_consistency')
    s.atomic_json(ctx['out']/'controlled_first_complete_v2.json',dict(complete=True,models=len(ordered)))
    s.log('CONTROLLED TESTS COMPLETE',models=len(ordered),next='Every native group/token response and PCA inspection')
    for row in ordered:
        run_bank_v2(ctx,row,source,stores,rp,tp);check_sources()
    s.done_json(ctx['out']/'COMPLETE_V2.json',dict(complete=True,models=len(ordered),revision=REVISION,no_training=True))
