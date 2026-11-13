import hashlib
from pathlib import Path
import pytest
import torch
from endpoint_restore import restore

def digest(weights):
    return hashlib.sha256(weights['D'].numpy().tobytes()).hexdigest()

class Frozen:
    def __init__(self, values):self.values=values;self.reads=[]
    def load(self,key):self.reads.append(key);return self.values.get(key)
    def save(self,*a,**kw):raise AssertionError('Source write')

def example(tmp_path):
    w={'D':torch.ones(2,3)}
    row=dict(key='reused_endpoint',weights_sha=digest(w),additional_updates=3000)
    return dict(local=tmp_path,store=Frozen({})),row,dict(weights=w,updates=3000)

def test_restore_reused_endpoint_without_source_writes(tmp_path):
    ctx,row,state=example(tmp_path);upstream=Frozen({row['key']:state})
    assert restore(ctx,[row],None,digest,[ctx['store'],upstream])==[row['key']]
    assert upstream.reads==[row['key']]
    assert restore(ctx,[row],None,digest,[ctx['store'],upstream])==[]
    assert upstream.reads==[row['key']]

@pytest.mark.parametrize('damage',['weights','updates','missing'])
def test_no_silent_substitution(tmp_path,damage):
    ctx,row,state=example(tmp_path)
    if damage=='weights':state['weights']={'D':torch.zeros(2,3)}
    if damage=='updates':state['updates']=2999
    src=Frozen({} if damage=='missing' else {row['key']:state})
    with pytest.raises((ValueError,FileNotFoundError)):
        restore(ctx,[row],None,digest,[ctx['store'],src])
    assert not list((Path(ctx['local'])/'frozen_weights').glob('*.pt'))

def test_actual_catalog_with_nine_upstream_endpoints(tmp_path,monkeypatch):
    """Reproduce the reported journal/endpoint split using real slot stores."""
    import study,cv_study,native_sources
    from coco_adaptation.inputs import JournalStore
    from coco_coordinated.run import FrozenStore
    from coco_baselines import training
    upstream=JournalStore(tmp_path/'scale',dict(study='verified_scale'),tmp_path/'writer',False)
    catalog=JournalStore(tmp_path/'catalog',dict(study='primary_catalog'),tmp_path/'catalog_writer',False)
    rows=[]
    for i,(seed,kind) in enumerate((s,k) for s in [21,22,23] for k in study.METHODS[:3]):
        w={'D':torch.full((2,3),float(i+1))};key=f'g1024_k16_s{seed}_{kind}_endpoint'
        rows.append(dict(key=key,kind=kind,seed=seed,additional_updates=3000,weights_sha=training.weight_sha(w)))
        upstream.save(key,dict(updates=3000,weights=w))
    a=FrozenStore(catalog.root,tmp_path/'read_catalog');b=FrozenStore(upstream.root,tmp_path/'read_scale')
    assert all(a.load(r['key']) is None for r in rows)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in [catalog.output,upstream.output] for p in base.rglob('*') if p.is_file()}
    monkeypatch.setattr(cv_study,'discover_primary',lambda *args:(a,rows))
    monkeypatch.setattr(native_sources,'discover',lambda *args:({},[]))
    ctx=dict(local=tmp_path/'analysis',store=Frozen({}),sources={'scale':b})
    study.source_models(ctx,dict(study_source=str(tmp_path/'catalog'),local=str(ctx['local'])))
    paths=list((ctx['local']/'frozen_weights').glob('*.pt'))
    assert len(paths)==9
    for row in rows:
        value=torch.load(ctx['local']/'frozen_weights'/(row['weights_sha']+'.pt'),weights_only=True)
        assert training.weight_sha(value)==row['weights_sha']
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in [catalog.output,upstream.output] for p in base.rglob('*') if p.is_file()}
    assert before==after
