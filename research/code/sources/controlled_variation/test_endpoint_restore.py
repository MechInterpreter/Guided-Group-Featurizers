import hashlib
from pathlib import Path
import pytest
import torch
from resume_weights import restore

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
