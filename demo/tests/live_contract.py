"""Exercise authenticated live inference with a fresh, transformed image."""
from pathlib import Path
import hashlib
import io
import json
import urllib.error
import urllib.request
import numpy as np
from PIL import Image, ImageOps

ROOT=Path.cwd()
TOKEN=(ROOT/'.local/inference-token').read_text().strip()
ORIGIN='http://127.0.0.1:8765'

def request(path, content=None, authenticated=True):
    headers={'Authorization':'Bearer '+TOKEN} if authenticated else {}
    if content is not None:
        boundary='ggf-contract-test'
        headers['Content-Type']='multipart/form-data; boundary='+boundary
        content=(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="input.png"\r\nContent-Type: image/png\r\n\r\n'.encode()+content+f'\r\n--{boundary}--\r\n'.encode())
    return urllib.request.urlopen(urllib.request.Request(ORIGIN+path,data=content,headers=headers),timeout=90)

for path in ['/status','/infer']:
    try:request(path,None if path=='/status' else b'image',False)
    except urllib.error.HTTPError as error:assert error.code==401
    else:raise AssertionError('Unauthenticated request accepted')
try:request('/infer',b'not an image')
except urllib.error.HTTPError as error:assert error.code==400
else:raise AssertionError('Malformed image accepted')
manifest=json.loads((ROOT/'public/data/manifest.json').read_text())
content=(ROOT/f'public/images/{manifest["imageIds"][0]}.jpg').read_bytes()
recorded=json.load(request('/infer',content))
shared=json.loads((ROOT/'public/data/shared-umap.json').read_text(encoding='utf-8'))
assert recorded['umap']['model_sha256']==shared['model_sha256']
assert np.asarray(recorded['umap']['positions']).shape==(196,3)
assert np.isfinite(recorded['umap']['positions']).all()
checks=[]
for method in manifest['methods']:
    name=method['id']
    errors=[]
    for group in range(75):
        source=json.loads((ROOT/f'data/groups/{name}-{group}.json').read_text())
        expected=np.array(source['norms'][:196])
        actual=np.array(recorded['methods'][name]['norms'][group])
        np.testing.assert_allclose(actual,expected,atol=.015,rtol=.002)
        errors.append(float(abs(actual-expected).max()))
    checks.append({'method':name,'groups':75,'max_norm_difference':max(errors)})
    print(name,'all 75 live group maps agree with recorded values; max difference',max(errors),flush=True)
with Image.open(io.BytesIO(content)) as source:
    fresh=ImageOps.mirror(source.convert('RGB')).rotate(17, resample=Image.Resampling.BICUBIC).resize((321,257))
    buffer=io.BytesIO();fresh.save(buffer,format='PNG');fresh_bytes=buffer.getvalue()
assert hashlib.sha256(content).digest()!=hashlib.sha256(fresh_bytes).digest()
result=json.load(request('/infer',fresh_bytes))
assert result['input_sha256']==hashlib.sha256(fresh_bytes).hexdigest()
assert result['request_id']!=recorded['request_id']
assert result['geometry']['original']==[321,257]
assert result['umap']['model_sha256']==shared['model_sha256']
assert np.asarray(result['umap']['positions']).shape==(196,3)
assert np.isfinite(result['umap']['positions']).all()
assert not np.allclose(result['umap']['positions'],recorded['umap']['positions'])
identity=result['model_identity']
assert identity['seed']==21 and identity['groups']==75 and identity['directions']==12
assert identity['software']['transformers']=='4.57.3'
for method in manifest['methods']:
    name=method['id'];response=result['methods'][name]
    norms=np.asarray(response['norms']);positions=np.asarray(response['positions']);overview=np.asarray(response['overview'])
    assert norms.shape==(75,196) and positions.shape==(75,196,3) and overview.shape==(75,196,3)
    assert np.isfinite(norms).all() and (norms>=0).all() and np.isfinite(positions).all() and np.isfinite(overview).all()
    np.testing.assert_allclose(np.sort(norms,axis=1)[:,-4:].mean(1),response['scores'],atol=.000002)
    assert not np.allclose(norms,recorded['methods'][name]['norms'])
    assert identity['checkpoint_sha256'][name]==hashlib.sha256((ROOT/f'.local/research/{name}.pt').read_bytes()).hexdigest()
qa=ROOT/'.local/qa';qa.mkdir(exist_ok=True)
(qa/'live-upload.png').write_bytes(fresh_bytes)
(qa/'live-contract.json').write_text(json.dumps({'passed':True,'recorded':checks,'fresh_input_sha256':result['input_sha256'],'request_id':result['request_id'],'device':result['device'],'seconds':result['seconds'],'model_identity':identity},indent=2))
print('Fresh upload, changed responses, all four methods, authentication, malformed-input rejection and model hashes passed. Latency:',result['seconds'],'seconds.',flush=True)
