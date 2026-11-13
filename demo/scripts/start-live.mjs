import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import net from 'node:net';
const root=path.resolve(import.meta.dirname,'..');
const python=process.env.GGF_PYTHON??path.join(root,'.local','inference-venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
try{await fs.access(python);}catch{throw Error('Create .local/inference-venv and install server/requirements.txt first. See README.md.');}
for(const port of [5173,8765]) await new Promise((resolve,reject)=>{
 const server=net.createServer();server.once('error',()=>reject(Error(`Port ${port} is already in use. Close the existing demo before running npm run live.`)));server.listen(port,'127.0.0.1',()=>server.close(resolve));
});
const children=[];
let stopping=false;
function stop(code=0){if(stopping)return;stopping=true;for(const child of children)child.kill('SIGTERM');setTimeout(()=>process.exit(code),750).unref();}
process.on('SIGINT',()=>stop());process.on('SIGTERM',()=>stop());
function start(command,args){const child=spawn(command,args,{cwd:root,stdio:'inherit',windowsHide:true});children.push(child);child.once('error',e=>{console.error(e.message);stop(1)});child.once('exit',code=>{if(!stopping)stop(code??1)});return child;}
start(python,['scripts/start-inference.py']);
const deadline=Date.now()+120000;
let ready=false;
while(Date.now()<deadline&&!stopping){
 try{
  const token=(await fs.readFile(path.join(root,'.local/inference-token'),'utf8')).trim();
  const response=await fetch('http://127.0.0.1:8765/status',{headers:{Authorization:'Bearer '+token},signal:AbortSignal.timeout(2000)});
  if(response.ok&&(await response.json()).ready){
   const varsPath=path.join(root,'.dev.vars');let vars=await fs.readFile(varsPath,'utf8').catch(()=> '');
   for(const [key,value] of Object.entries({INFERENCE_ENDPOINT:'http://127.0.0.1:8765',INFERENCE_TOKEN:token})){
    const line=key+'='+JSON.stringify(value),pattern=new RegExp('^'+key+'=.*$','m');vars=pattern.test(vars)?vars.replace(pattern,line):vars.trimEnd()+'\n'+line+'\n';
   }
   await fs.writeFile(varsPath,vars,{mode:0o600});ready=true;break;
  }
 }catch{}
 await new Promise(resolve=>setTimeout(resolve,500));
}
if(!ready){console.error('The frozen-model service did not become ready. Check its startup error above.');stop(1);}
else{console.log('Verified frozen models ready. Opening the local dashboard service.');start(process.execPath,['scripts/run-framework.mjs','dev','--host','127.0.0.1']);}
