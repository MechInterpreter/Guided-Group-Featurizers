"""Portable stage driver; no training calls, fabricated labels or silent fallback."""
import argparse,json,zipfile
from pathlib import Path
from eval_common import *

def exports(root):
    root=Path(root);out=root/'exports';out.mkdir(parents=True,exist_ok=True)
    exclude={'responses','original_images','images','exports','input_verification'};metadata=[];coherence=[];presence=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(root);parts=rel.parts
        if parts[0] in exclude:continue
        if 'cards' in parts and parts[0]=='reviews':coherence.append((p,str(rel)))
        elif parts[:2]==('reviews','presence') and ('images' in parts or 'sheets' in parts):presence.append((p,str(rel)))
        elif p.suffix in ['.json','.html','.txt','.npz','.csv']:metadata.append((p,str(rel)))
    # Exact PNG input crops are shared by all methods and retained once here.
    presence.extend((p,str(p.relative_to(root))) for p in sorted((root/'images').glob('*.png')))
    index=[]
    for name,files in [('all75_evaluation_metadata.zip',metadata),('all75_blinded_cards.zip',coherence),('all75_presence_images.zip',presence)]:
        path=out/name;tmp=path.with_suffix('.pending.zip')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,6) as z:
            for p,rel in files:z.write(p,rel)
        os.replace(tmp,path);index.append(dict(name=name,sha256=sha(path),bytes=path.stat().st_size,files=len(files)))
    save(out/'EXPORT_INDEX.json',dict(files=index,raw_native_maps_remain_on_Drive=True,portable_pooling_and_views_in_metadata=True),immutable=False)
    return index

def package(root,download_images=True):
    from score_evaluation import numeric
    from prepare_evaluation_reviews import restore_images,generic_reviews,presence_pack
    from report_evaluation import report
    root=Path(root);collection=read(root/'COLLECTION_COMPLETE.json')
    if not collection.get('complete'):raise ValueError('Native collection incomplete')
    numeric(root)
    report(root)
    portable=[]
    for bank in collection['banks']:portable.append(cache_views(root,bank))
    save(root/'PORTABLE_LOCK.json',dict(complete=True,banks=portable,collection_sha256=sha(root/'COLLECTION_COMPLETE.json')))
    if download_images:
        restore_images(root);generic_reviews(root);presence_pack(root)
    report(root);index=exports(root)
    print('EVALUATION RETURN BUNDLES',json.dumps(index),flush=True)
    print('Numerical scores are ready. Semantic coherence, presence-based concentration and visual target alignment are pending their independent labels.',flush=True)

def run(root,phase):
    from score_evaluation import coherence,select_presence,concentration,alignment
    from prepare_evaluation_reviews import lock_presence,alignment_pack,presence_quality
    from report_evaluation import report
    root=Path(root)
    if phase=='package':package(root)
    elif phase=='lock-presence':lock_presence(root,'calibration');lock_presence(root,'audit');presence_quality(root)
    elif phase=='target-calibration':select_presence(root);alignment_pack(root,'calibration')
    elif phase=='target-audit':alignment(root,'calibration');alignment_pack(root,'audit')
    elif phase=='score':
        coherence(root);concentration(root);alignment(root,'audit');report(root);exports(root)
    elif phase=='report':report(root)
    else:raise ValueError(phase)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',required=True,choices=['package','lock-presence','target-calibration','target-audit','score','report']);a=p.parse_args();run(a.root,a.phase)
