import fs from 'node:fs/promises';
import {readFileSync as nativeReadFileSync} from 'node:fs';
import path from 'node:path';
import {gunzipSync} from 'node:zlib';
export default {...fs,async readFile(file,options){
 try{return await fs.readFile(file,options)}catch(error){
  const name=String(file).replaceAll('\\','/');
  if(error.code!=='ENOENT'||!name.startsWith('data/groups/'))throw error;
  const raw=gunzipSync(await fs.readFile(path.join('public',name+'.gz')));
  const encoding=typeof options==='string'?options:options?.encoding;
  return encoding?raw.toString(encoding):raw;
 }
}};

export function readFileSync(file,options){
 const name=String(file).replaceAll('\\','/');
 if(name.startsWith('data/groups/')){const raw=gunzipSync(nativeReadFileSync(path.join('public',name+'.gz')));return typeof options==='string'?raw.toString(options):options?.encoding?raw.toString(options.encoding):raw;}
 return nativeReadFileSync(file,options);
}
