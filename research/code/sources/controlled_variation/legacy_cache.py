"""Read verified v2 analysis shards without changing their source output."""
import json
from pathlib import Path
from cv_math import Cache
OLD_SHA='0936b16fc632bebdc76a15252992bf6287b3c9a84b886b2ba16edfc6bf6bbc7c'

class ReadThroughCache:
    def __init__(self,new,old=None):self.new,self.old=new,old
    def get(self,key):
        a=self.new.get(key)
        return a if a is not None or self.old is None else self.old.get(key)
    def put(self,*a,**kw):return self.new.put(*a,**kw)

def find(ctx,options,records):
    from coco_baselines.storage import identity
    root=Path(options['legacy_output']);hits=[]
    for p in (root/'analysis/results').glob('*/protocol.json'):
        q=json.loads(p.read_text());cohort=p.parent/'cohort.json'
        if q.get('source_sha')==OLD_SHA and q.get('normalization')==identity(ctx['norm']) and cohort.exists() and json.loads(cohort.read_text())==records:
            sig=identity(dict(protocol=q,cohort=records,backbone=ctx['provenance']['backbone_weights']))
            if (root/'features'/sig).is_dir():hits.append(dict(root=root,feature_id=sig))
    if len(hits)>1:raise ValueError('Multiple legacy feature sources match; select one explicitly')
    return hits[0] if hits else None

def cache(new,legacy,section,sig,local):
    old=None
    if legacy and (legacy['root']/section/sig).is_dir():old=Cache(Path(local)/'legacy_read'/section/sig,legacy['root']/section/sig,sig)
    return ReadThroughCache(new,old)
