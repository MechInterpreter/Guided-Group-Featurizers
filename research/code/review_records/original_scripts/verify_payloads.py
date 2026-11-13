from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import json,sys,time
sys.path.insert(0,'scripts/ggf_all75')
import openrouter_coherence_full as f
data=f.catalog()
ind=f.read(f.OUT/'INDEPENDENT_RECOUNT.json')
def verify_group(group):
    for family in f.pilot.FAMILIES:
        file=f.response_path(group,family)
        raw=file.read_bytes()
        assert f.sha(raw)==ind['response_sha256'][f.rel(file)]
        rec=json.loads(raw)
        _,_,body=f.inputs(group,family,data['rubrics'])
        assert f.sha(body)==rec['request_sha256'],file
    return 2
start=time.perf_counter()
count=0
with ThreadPoolExecutor(max_workers=12) as executor:
    for n in executor.map(verify_group,data['groups']):
        count+=n
        if count % 2400 == 0:
            print(f'{count} payloads verified in {time.perf_counter()-start:.1f}s',flush=True)
assert count==9450
record=f.encode(dict(complete=True,manifest_sha256=f.sha(f.MANIFEST.read_bytes()),
                     independent_recount_sha256=f.sha((f.OUT/'INDEPENDENT_RECOUNT.json').read_bytes()),
                     request_payload_hashes_verified=count))
path=f.OUT/'REQUEST_VERIFICATION.json'
if path.exists():
    assert path.read_bytes()==record
else:
    f.save(path,record)
print(f'All {count} payload hashes verified in {time.perf_counter()-start:.1f}s',flush=True)
