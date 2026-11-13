"""Exact-crop assets, method-blinded review cards, and independently locked labels."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from eval_common import *
from coco_baselines import data
import original_concentration as original

SALT='all75-pair-v1'
key=lambda *xs:hashlib.sha256((SALT+'|'+'|'.join(map(str,xs))).encode()).hexdigest()
def font(size=18):
    for p in ['C:/Windows/Fonts/segoeui.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def restore_images(root):
    root=Path(root);inputs=read(root/'INPUTS.json');rows=[]
    for i,r in inputs['records'].items():
        p=root/'images'/f'{i}.png';side=p.with_suffix('.json')
        if side.exists():
            row=read(side)
            if row['original_sha256']!=r['original_sha256'] or row['geometry']!=r['geometry']:raise ValueError('Image provenance differs')
            if row['status']=='ready' and sha(p)!=row['crop_sha256']:raise ValueError('Crop changed')
            rows.append(row);continue
        original_path=root/'original_images'/f'{int(i):012d}.jpg'
        try:data.download(r['source_url'],original_path)
        except (OSError,Exception) as e:
            # Missing stays explicit; it is never replaced with another image.
            row=dict(r,status='missing',error=type(e).__name__);save(side,row);rows.append(row);continue
        if sha(original_path)!=r['original_sha256']:raise ValueError('Downloaded source JPEG differs from frozen features: '+str(i))
        with Image.open(original_path) as im:
            if list(im.size)!=r['geometry']['original']:raise ValueError('Source geometry differs')
            crop=data.transform_image(im.convert('RGB'),r['geometry'])
        p.parent.mkdir(parents=True,exist_ok=True);crop.save(p,format='PNG')
        row=dict(r,status='ready',crop_path=str(p.relative_to(root)),crop_sha256=sha(p),crop_pixel_sha256=hashlib.sha256(np.asarray(crop).tobytes()).hexdigest(),
            pixel_replay_status='Original JPEG SHA and recorded crop geometry verified; crop pixels hashed now. Original extraction did not supply an independent crop-pixel hash.')
        save(side,row);rows.append(row)
        if len(rows)%50==0:print('EXACT CROPS',len(rows),'/',len(inputs['records']),flush=True)
    save(root/'IMAGE_ASSETS.json',dict(images=rows,source_inputs_sha256=sha(root/'INPUTS.json'),missing=sum(r['status']!='ready' for r in rows)))

def assets(root):return {r['image_id']:r for r in read(Path(root)/'IMAGE_ASSETS.json')['images']}
def verified_views(root,bank_id):
    p=Path(root)/'portable'/bank_id/'VIEWS.json';master=Path(root)/'PORTABLE_LOCK.json'
    if master.exists():
        expected=next(r for r in read(master)['banks'] if r['bank_id']==bank_id)
        if sha(p)!=expected['views_sha256']:raise ValueError('Portable view coordinates changed')
    return read(p)
def picture(root,view,records):
    row=records.get(view.get('image_id'));canvas=Image.new('RGB',(224,306),'#e9edf2');draw=ImageDraw.Draw(canvas)
    valid=bool(row and row['status']=='ready' and view.get('patches'))
    if row and row['status']=='ready':
        p=resolve(root,row['crop_path'])
        if sha(p)!=row['crop_sha256']:raise ValueError('Review crop changed')
        with Image.open(p) as im:im=im.convert('RGB')
        marked=im.copy();d=ImageDraw.Draw(marked)
        for j,p in enumerate(view.get('patches',[])):
            x=p%14*16;y=p//14*16;d.rectangle((x,y,x+15,y+15),outline='#f31567',width=2)
            canvas.paste(im.crop((x,y,x+16,y+16)).resize((56,56),Image.Resampling.NEAREST),(j*56,232))
        canvas.paste(marked,(0,0))
        if not view.get('patches'):draw.text((4,287),'No positive response',fill='black',font=font(14))
    else:draw.text((16,90),'Missing planned image',fill='black',font=font(16))
    return canvas,not valid

def card(root,path,title,views,records):
    canvas=Image.new('RGB',(1344,364),'white');d=ImageDraw.Draw(canvas);d.text((8,5),title,fill='black',font=font(19));invalid=[]
    for j in range(6):
        d.text((j*224+6,31),str(j+1),fill='black',font=font(16))
        view=views[j] if j<len(views) else {};im,bad=picture(root,view,records);canvas.paste(im,(j*224,56));invalid.append(bad)
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,format='PNG');return invalid

def lock_pack(folder,public,private):
    save(folder/'PUBLIC.json',public);save(folder/'PRIVATE.json',private)
    hashes={str(p.relative_to(folder)):sha(p) for p in sorted((folder/'cards').glob('*.png'))}
    save(folder/'REVIEW_LOCK.json',dict(public_sha256=sha(folder/'PUBLIC.json'),private_sha256=sha(folder/'PRIVATE.json'),cards=hashes))

def verify_pack(folder):
    folder=Path(folder);lock=read(folder/'REVIEW_LOCK.json')
    if sha(folder/'PUBLIC.json')!=lock['public_sha256'] or sha(folder/'PRIVATE.json')!=lock['private_sha256']:raise ValueError('Public/private review mapping changed')
    for p,h in lock['cards'].items():
        if sha(folder/p)!=h:raise ValueError('Blinded card changed')
    return lock

def generic_reviews(root):
    root=Path(root);collection=read(root/'COLLECTION_COMPLETE.json');records=assets(root);cases=[]
    for bank in collection['banks']:
        view=verified_views(root,bank['bank_id'])
        if view['signature']!=bank['signature']:raise ValueError('Native view cache differs')
        for unit in range(bank['groups']):
            opaque=key('group',bank['bank_id'],bank['weights_sha'],unit)
            slots=sorted(range(6),key=lambda j:key(opaque,j));selected=view['roles']['audit'][str(unit)]
            views=[selected[j] if j<len(selected) else {} for j in slots]
            cases.append(dict(opaque=opaque,bank_id=bank['bank_id'],unit=unit,views=views,slots=slots))
    cases.sort(key=lambda r:key('display',r['opaque']));folder=root/'reviews/coherence';public=[];private=[]
    for j,r in enumerate(cases,1):
        cid=f'G{j:04d}';path=folder/'cards'/f'{cid}.png';bad=card(root,path,f'{cid} | pairs: views1-2, 3-4, 5-6 | judge marked regions',r['views'],records)
        public.append(dict(case_id=cid,card=f'cards/{cid}.png',pair_ids=[0,1,2],invalid_pairs=[any(bad[2*k:2*k+2]) for k in range(3)]))
        private.append(dict(case_id=cid,bank_id=r['bank_id'],unit=r['unit'],source_slots=r['slots'],image_ids=[v.get('image_id') for v in r['views']],invalid_pairs=public[-1]['invalid_pairs']))
    pub=dict(schema='all75-blinded-pair-review-v1',cases=public,
        semantic_rubric='Specific shared object, identifiable part/action/relation or named material/region at BOTH marked patches. Generic color/edge/orientation or broad animal/furniture umbrella is not enough. State the shared meaning.',
        visual_rubric='Separately judge shared specific texture/form/color/material/geometric structure. Parts and cross-category geometry are valid visual concepts; do not use category labels as unit names.',
        codes=['yes','no','unclear'],unavailable='Missing or zero-positive pairs must be unclear',
        annotation='Three fresh contexts, one independent file each; every case and three pairs retained. Use public cards only. No method/native-unit/category identities in this pack.')
    lock_pack(folder,pub,dict(cases=private,collection_sha256=sha(root/'COLLECTION_COMPLETE.json')))
    for name in ['context1','context2','context3']:
        save(folder/'templates'/f'{name}.json',dict(public_sha256=sha(folder/'PUBLIC.json'),method_blind=True,reviewer='',cases=[dict(case_id=r['case_id'],pairs=[dict(pair=k,semantic='',visual='',shared_meaning='',exceptions='') for k in range(3)]) for r in public]))
        for start in range(0,len(public),12):
            save(folder/'batches'/name/f'B{start//12:03d}.json',dict(public_sha256=sha(folder/'PUBLIC.json'),method_blind=True,reviewer='',
                cases=[dict(case_id=r['case_id'],pairs=[dict(pair=k,semantic='',visual='',shared_meaning='',exceptions='') for k in range(3)]) for r in public[start:start+12]]))
    # Fixed validation sample before any AI outcomes, <=30 groups. These are
    # validation cases, not new independent groups to add to the primary count.
    strata={}
    banks={b['bank_id']:b for b in collection['banks']}
    for r in private:
        b=banks[r['bank_id']];s=(b['regime'],b['kind'],b['capacity_matched']);strata.setdefault(s,[]).append(r)
    chosen=[]
    for s,rr in sorted(strata.items()):chosen+=sorted(rr,key=lambda r:key('human-validation',r['case_id']))[:(2 if not s[2] else 4)]
    save(folder/'HUMAN_VALIDATION_SAMPLE.json',dict(case_ids=[r['case_id'] for r in chosen],rule='Fixed hash sample before AI outcomes;4 per main regime/method family and2 native-reference cases; not pooled as independent extra groups',maximum_expected_minutes=60))
    print('BLINDED GROUPS',len(cases),'pairs',len(cases)*3,flush=True)

def presence_pack(root):
    root=Path(root);inputs=read(root/'INPUTS.json');records=assets(root);folder=root/'reviews/presence';items=[]
    allids=sorted(sum(inputs['ids'].values(),[]),key=lambda i:key('presence',i))
    for j,i in enumerate(allids,1):
        cid=f'I{j:04d}';r=records[i];rel=f'images/{cid}.png'
        if r['status']=='ready':
            p=folder/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(resolve(root,r['crop_path']),p)
        items.append(dict(case_id=cid,image=rel if r['status']=='ready' else None,status=r['status']))
    save(folder/'PUBLIC.json',dict(cases=items,definitions=read(root/'TARGET_DEFINITIONS.json'),
        instruction='Judge each crop against EVERY fixed category definition. No response overlays or method identities. Explicit yes/no/unclear for all75; missing image stays missing, never absent.'))
    save(folder/'PRIVATE.json',dict(cases=[dict(case_id=f'I{j:04d}',image_id=i,role=records[i]['role'],crop_sha256=records[i].get('crop_sha256'),status=records[i]['status']) for j,i in enumerate(allids,1)]))
    save(folder/'REVIEW_LOCK.json',dict(public_sha256=sha(folder/'PUBLIC.json'),private_sha256=sha(folder/'PRIVATE.json'),
        image_sha256={r['image']:sha(folder/r['image']) for r in items if r['image']}))
    for start in range(0,len(items),12):
        batch=items[start:start+12];canvas=Image.new('RGB',(896,780),'white');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(batch):
            x=j%4*224;y=j//4*260;d.text((x+5,y+3),r['case_id'],font=font(16),fill='black')
            if r['image']:
                with Image.open(folder/r['image']) as im:canvas.paste(im,(x,y+26))
            else:d.text((x+5,y+80),'Missing',font=font(16),fill='black')
        p=folder/'sheets'/f'P{start//12:03d}.png';p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
        save(folder/'templates'/f'P{start//12:03d}.json',dict(public_sha256=sha(folder/'PUBLIC.json'),reviewer='',method_blind=True,cases=[dict(case_id=r['case_id'],yes=[],no=[],unclear=[],all_definitions_considered=False,rationale='') for r in batch]))
    # Fixed, response-blind second pass before canonical presence labels exist.
    private=read(folder/'PRIVATE.json')['cases'];qa=[]
    for role in ['calibration','audit']:
        selected=sorted((r for r in private if r['role']==role),key=lambda r:key('presence-QA',r['case_id']))[:20]
        qa.extend(next(r for r in items if r['case_id']==s['case_id']) for s in selected)
    qfolder=root/'reviews/presence_qa'
    save(qfolder/'PUBLIC.json',dict(cases=[dict(r,image='../presence/'+r['image'] if r['image'] else None) for r in qa],definitions=read(root/'TARGET_DEFINITIONS.json'),
        instruction='Independent crop-only second pass. Do not read canonical presence labels or model responses. Explicit yes/no/unclear partition of all75 categories.'))
    save(qfolder/'TEMPLATE.json',dict(public_sha256=sha(qfolder/'PUBLIC.json'),reviewer='',method_blind=True,cases=[dict(case_id=r['case_id'],yes=[],no=[],unclear=[],all_definitions_considered=False,rationale='') for r in qa]))

def lock_presence(root,role):
    root=Path(root);folder=root/'reviews/presence';lock=read(folder/'REVIEW_LOCK.json')
    if sha(folder/'PUBLIC.json')!=lock['public_sha256'] or sha(folder/'PRIVATE.json')!=lock['private_sha256']:raise ValueError('Presence review mapping changed')
    private=read(folder/'PRIVATE.json');inputs=read(root/'INPUTS.json');cats=inputs['categories'];mapping={r['case_id']:r for r in private['cases']};received={};sources=[]
    for p in sorted((folder/'labels').glob('*.json')):
        d=read(p)
        if d['public_sha256']!=lock['public_sha256'] or not d.get('method_blind') or not d.get('reviewer'):raise ValueError('Presence provenance/blinding missing')
        sources.append(dict(path=str(p.relative_to(root)),sha256=sha(p),reviewer=d['reviewer']))
        for r in d['cases']:
            if r['case_id'] not in mapping or r['case_id'] in received:raise ValueError('Unknown/duplicate presence case')
            received[r['case_id']]=r
    labels={str(c):{} for c in cats};count=0
    for cid,m in mapping.items():
        if m['role']!=role:continue
        if m['status']!='ready':
            for c in cats:labels[str(c)][str(m['image_id'])]='missing'
        else:
            if cid not in received:raise ValueError('Presence case pending: '+cid)
            r=received[cid]
            if not r.get('all_definitions_considered'):raise ValueError('All category definitions must be considered')
            sets=[set(r[k]) for k in ['yes','no','unclear']]
            if set.union(*sets)!=set(cats) or sum(len(v) for v in sets)!=len(cats) or any(len(r[k])!=len(sets[j]) for j,k in enumerate(['yes','no','unclear'])):
                raise ValueError('Explicit disjoint75-category presence partition required')
            for c in cats:labels[str(c)][str(m['image_id'])]='yes' if c in sets[0] else ('no' if c in sets[1] else 'unclear')
        count+=1
    save(root/'presence'/f'{role}_LOCK.json',dict(complete=True,role=role,ids=inputs['ids'][role],labels=labels,sources=sources,
        definitions_sha256=sha(root/'TARGET_DEFINITIONS.json'),public_sha256=lock['public_sha256'],cases=count,reviewer_scope='Independent crop-only judgments; AI is not human ground truth'))

def alignment_pack(root,role):
    root=Path(root);selection=read(root/'CONCENTRATION_SELECTION_COMPLETE.json');records=assets(root);defs={r['category_index']:r for r in read(root/'TARGET_DEFINITIONS.json')['categories']};cases=[]
    if role=='audit':
        cal=read(root/'ALIGNMENT_CALIBRATION_LOCK.json')
        if not cal.get('complete'):raise ValueError('Lock calibration alignment before audit alignment review')
    for ref in selection['locks']:
        p=resolve(root,ref['path'])
        if sha(p)!=ref['sha256']:raise ValueError('Presence selector changed')
        lock=read(p);view=verified_views(root,ref['bank_id'])
        for row in lock['rows']:
            q=next(r for r in row['selection']['quantiles'] if r['negative_quantile']==.95)
            if q['status']!='complete':continue
            u=q['selections'][0]['groups'][0];c=row['category_index'];cases.append(dict(bank_id=ref['bank_id'],unit=u,category_index=c,views=view['roles'][role][str(u)]))
    cases.sort(key=lambda r:key('target-alignment',r['bank_id'],r['category_index']));folder=root/'reviews'/('alignment_'+role);public=[];private=[]
    for j,r in enumerate(cases,1):
        cid=f'T{j:04d}';definition=defs[r['category_index']];bad=card(root,folder/'cards'/f'{cid}.png',cid+' | target: '+definition['category_name']+' | marked regions only',r['views'],records)
        public.append(dict(case_id=cid,card=f'cards/{cid}.png',target=definition['definition'],codes=['target','related_context','other','mixed','unclear'],invalid_views=bad))
        private.append(dict(case_id=cid,bank_id=r['bank_id'],unit=r['unit'],category_index=r['category_index'],invalid_views=bad))
    lock_pack(folder,dict(cases=public,role=role,instruction='Target must dominate the marked region; target elsewhere in the crop is not sufficient. A coherent different concept is other, not incoherent.'),dict(cases=private,selection_sha256=sha(root/'CONCENTRATION_SELECTION_COMPLETE.json')))
    save(folder/'TEMPLATE.json',dict(public_sha256=sha(folder/'PUBLIC.json'),method_blind=True,reviewer='',cases=[dict(case_id=r['case_id'],codes=['']*6,rationale='') for r in public]))

def presence_quality(root):
    root=Path(root);folder=root/'reviews/presence_qa';public=read(folder/'PUBLIC.json');p=folder/'labels.json';labels=read(p)
    if labels['public_sha256']!=sha(folder/'PUBLIC.json') or not labels.get('reviewer') or not labels.get('method_blind'):raise ValueError('Independent QA provenance required')
    primary={role:read(root/'presence'/f'{role}_LOCK.json') for role in ['calibration','audit']}
    prior_reviewers={s['reviewer'] for r in primary.values() for s in r['sources']}
    if labels['reviewer'] in prior_reviewers:raise ValueError('QA reviewer must be independent of canonical reviewer contexts')
    private={r['case_id']:r for r in read(root/'reviews/presence/PRIVATE.json')['cases']};received={r['case_id']:r for r in labels['cases']}
    if len(received)!=len(labels['cases']) or set(received)!={r['case_id'] for r in public['cases']}:raise ValueError('All fixed QA cases required')
    cats=read(root/'INPUTS.json')['categories'];cells=[]
    for case in public['cases']:
        q=received[case['case_id']];m=private[case['case_id']]
        if m['status']!='ready':continue
        sets=[set(q[k]) for k in ['yes','no','unclear']]
        if not q.get('all_definitions_considered') or set.union(*sets)!=set(cats) or sum(map(len,sets))!=len(cats):raise ValueError('Complete QA category partition required')
        for c in cats:
            original=primary[m['role']]['labels'][str(c)][str(m['image_id'])];second='yes' if c in sets[0] else ('no' if c in sets[1] else 'unclear')
            cells.append(dict(case_id=case['case_id'],category_index=c,canonical=original,second=second))
    yes=[r for r in cells if r['canonical']=='yes'];second_yes=[r for r in cells if r['second']=='yes']
    save(root/'PRESENCE_QA.json',dict(complete=True,planned_cases=len(public['cases']),judged_cells=len(cells),
        raw_agreement=sum(r['canonical']==r['second'] for r in cells)/len(cells) if cells else None,
        canonical_positive_agreement=sum(r['second']=='yes' for r in yes)/len(yes) if yes else None,
        second_positive_agreement=sum(r['canonical']=='yes' for r in second_yes)/len(second_yes) if second_yes else None,
        cells=cells,canonical_unchanged=True,labels_sha256=sha(p),public_sha256=sha(folder/'PUBLIC.json'),
        canonical_lock_sha256={r:sha(root/'presence'/f'{r}_LOCK.json') for r in primary},
        scope='Fixed40-crop second pass; AI agreement is not human ground truth. Report positive agreement and individual disagreements, not only absence-dominated raw agreement.'))

def assemble_context(root,context):
    root=Path(root);folder=root/'reviews/coherence';verify_pack(folder);cases=[];sources=[]
    if context not in ['context1','context2','context3']:raise ValueError('Unknown context stream')
    for p in sorted((folder/'batch_labels'/context).glob('*.json')):
        d=read(p)
        if d['public_sha256']!=sha(folder/'PUBLIC.json') or not d.get('reviewer') or not d.get('method_blind'):raise ValueError('Batch reviewer provenance absent')
        cases.extend(d['cases']);sources.append(dict(path=str(p.relative_to(root)),sha256=sha(p),reviewer=d['reviewer']))
    expected=[r['case_id'] for r in read(folder/'PUBLIC.json')['cases']]
    if len({r['case_id'] for r in cases})!=len(cases) or {r['case_id'] for r in cases}!=set(expected):raise ValueError('Complete disjoint blinded batches required')
    lookup={r['case_id']:r for r in cases}
    save(folder/'labels'/f'{context}.json',dict(public_sha256=sha(folder/'PUBLIC.json'),method_blind=True,reviewer='Blinded '+context+'; raw contexts preserved',cases=[lookup[c] for c in expected],sources=sources))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['images','coherence','presence','presence-qa','lock-calibration-presence','lock-audit-presence','alignment-calibration','alignment-audit','assemble-context1','assemble-context2','assemble-context3'],required=True);a=p.parse_args()
    if a.phase=='presence-qa':presence_quality(a.root)
    elif a.phase.startswith('lock-'):lock_presence(a.root,a.phase.split('-')[1])
    elif a.phase.startswith('alignment-'):alignment_pack(a.root,a.phase.split('-')[1])
    elif a.phase.startswith('assemble-'):assemble_context(a.root,a.phase.split('-')[1])
    else:{'images':restore_images,'coherence':generic_reviews,'presence':presence_pack}[a.phase](a.root)
