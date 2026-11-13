"""Check Git-selected files and nested archives without displaying secret values."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
 'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
 'openai_openrouter_key':rb'\bsk-(?:proj-|or-v1-|ant-api[0-9]+-)?[A-Za-z0-9_-]{24,}',
 'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
 'google_api_key':rb'\bAIza[0-9A-Za-z_-]{30,}',
 'huggingface_token':rb'\bhf_[A-Za-z0-9]{28,}',
 'aws_access_key':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'slack_token':rb'\bxox[baprs]-[A-Za-z0-9-]{20,}',
 'credential_url':rb'https?://[^\s/<>"\x27:@]{2,}:[^\s/<>"\x27@]{4,}@',
 'personal_machine_path':rb'(?:[A-Za-z]:[/\\]+Users[/\\]+[A-Za-z0-9_.-]+|/(?:Users|home)/[A-Za-z0-9_.-]+)',
 'private_hosting_identity':rb'appgprj_[a-f0-9]{12,}|https://[^/\s]+\.chatgpt\.site',
}
COMPILED={name:re.compile(pattern,re.I if name in ['personal_machine_path','private_hosting_identity'] else 0) for name,pattern in PATTERNS.items()}
BLOCKED_PARTS={'.local','.env','.dev.vars','.openai','.wrangler','.sites-runtime','node_modules','__pycache__','private_research','uploads','checkpoints'}


def git(*args):
 return subprocess.check_output(['git','-C',str(ROOT),*args])


def audit(files,extra_secrets=()):
 findings=[];seen=set();containers={};counts={'files':0,'nested_members':0,'bytes_scanned':0,'reconstructed_archives':0}
 def inspect(name,raw,depth=0):
  inner=name.split('!')[-1]
  if Path(inner).suffix.lower() in {'.pt','.pth','.safetensors','.joblib','.pem','.p12','.pfx'} or set(Path(inner).parts)&BLOCKED_PARTS or (Path(inner).name.startswith(('.env.','.dev.vars','inference-token')) and not inner.endswith('.env.example')):
   findings.append({'path':name,'kind':'private_file_path','line':None})
  digest=hashlib.sha256(raw).digest()
  if digest in seen:return
  seen.add(digest);counts['bytes_scanned']+=len(raw)
  for kind,pattern in COMPILED.items():
   match=pattern.search(raw)
   if match:findings.append({'path':name,'kind':kind,'line':raw[:match.start()].count(b'\n')+1})
  for secret in extra_secrets:
   if secret and secret in raw:findings.append({'path':name,'kind':'known_local_secret','line':None})
  if raw.startswith(b'PK\x03\x04'):
   if depth>=8:raise ValueError('Archive nesting exceeds review limit: '+name)
   if not zipfile.is_zipfile(io.BytesIO(raw)):
    if 'original_containers/chunks.zip!' in name:return
    raise ValueError('Incomplete archive: '+name)
   with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for item in archive.infolist():
     if item.is_dir():continue
     if item.file_size>256*1024*1024:raise ValueError('Archive member exceeds review limit: '+name+'!'+item.filename)
     counts['nested_members']+=1
     inspect(name+'!'+item.filename,archive.read(item),depth+1)
  elif raw.startswith(b'\x1f\x8b'):
   if depth>=8:raise ValueError('Compression nesting exceeds review limit: '+name)
   counts['nested_members']+=1;inspect(name+'!gunzip',gzip.decompress(raw),depth+1)
  if counts['bytes_scanned']>4*1024**3:raise ValueError('Scan budget exceeded; no clean result issued.')
 for name,raw in files:
  counts['files']+=1
  parts=Path(name).parts
  if set(parts)&BLOCKED_PARTS or Path(name).name.startswith(('.env.','.dev.vars','inference-token')):
   if not name.endswith('.env.example'):findings.append({'path':name,'kind':'private_file_path','line':None})
  if len(raw)>100*1024**2:findings.append({'path':name,'kind':'oversized_git_blob','line':None})
  if name.endswith(('original_containers/chunks.zip','original_containers/index.json')):containers[name]=raw
  inspect(name,raw)
 for name,index_raw in containers.items():
  if not name.endswith('/index.json'):continue
  chunk_name=name[:-len('index.json')]+'chunks.zip'
  if chunk_name not in containers:raise ValueError('Missing archive fragments: '+name)
  index=json.loads(index_raw);chunk_raw=containers[chunk_name]
  if hashlib.sha256(chunk_raw).hexdigest()!=index['chunk_archive_sha256']:raise ValueError('Changed archive fragments: '+chunk_name)
  with zipfile.ZipFile(io.BytesIO(chunk_raw)) as chunks:
   for sha,record in index['archives'].items():
    parts=[]
    for item in record['chunks']:
     part=chunks.read(item['sha256'])
     if len(part)!=item['bytes'] or hashlib.sha256(part).hexdigest()!=item['sha256']:raise ValueError('Changed archive fragment')
     parts.append(part)
    raw=b''.join(parts)
    if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Changed reconstructed archive')
    counts['reconstructed_archives']+=1;inspect(name+'!reconstructed-'+sha+'.zip',raw)
 return {'passed':not findings,**counts,'findings':findings}


def blobs(records):
 process=subprocess.Popen(['git','-C',str(ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 try:
  for name,sha in records:
   process.stdin.write(sha.encode()+b'\n');process.stdin.flush()
   header=process.stdout.readline().split()
   if len(header)!=3:raise ValueError('Unreadable Git object: '+name)
   size=int(header[2]);raw=process.stdout.read(size)
   if len(raw)!=size or process.stdout.read(1)!=b'\n':raise ValueError('Incomplete Git object: '+name)
   if header[1]==b'blob':yield name,raw
 finally:
  process.stdin.close();process.stdout.close();process.wait()


def staged_files():
 records=[]
 for entry in git('ls-files','--stage','-z').split(b'\0'):
  if not entry:continue
  metadata,name=entry.split(b'\t',1);mode,sha,stage=metadata.split()
  path=name.decode('utf-8')
  if mode not in [b'100644',b'100755']:raise ValueError('Links or submodules require explicit review: '+path)
  records.append((path,sha.decode()))
 yield from blobs(records)


def history_files():
 records=[]
 for row in git('rev-list','--objects','--all').splitlines():
  fields=row.split(b' ',1)
  if len(fields)==2:records.append(('history/'+fields[1].decode('utf-8','replace'),fields[0].decode()))
 yield from blobs(records)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--history',action='store_true');parser.add_argument('--report',type=Path)
 args=parser.parse_args()
 result=audit(staged_files())
 if args.history:
  history=audit(history_files());result['history']=history;result['passed'] &= history['passed']
 if args.report:
  report=args.report.resolve()
  if not report.is_relative_to(ROOT/'.local'):raise ValueError('Reports must stay under ignored .local/.')
  report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2));return 0 if result['passed'] else 1


if __name__=='__main__':sys.exit(main())
