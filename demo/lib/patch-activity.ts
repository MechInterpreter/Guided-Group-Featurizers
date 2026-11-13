import {coactiveRowIndices, measureCoactivation} from './coactivation.ts';

export type PatchReference = {referenceIds:number[]; values:Float64Array; groups:number; patches:number};
export type PatchFrame = Record<number, ArrayLike<number>>;
export function referenceFrame(data:PatchReference,index:number):PatchFrame {
  if(index<0||index>=data.referenceIds.length) return {};
  return Object.fromEntries(Array.from({length:data.groups},(_,group)=>{
    const offset=(group*data.referenceIds.length+index)*data.patches;
    return [group,data.values.subarray(offset,offset+data.patches)];
  }));
}
export function sharedPatchIndices(a:ArrayLike<number>|undefined,b:ArrayLike<number>|undefined,scaleA:number,scaleB:number,cutoff:number) {
  if(!a||!b||a.length!==196||b.length!==196||![scaleA,scaleB,cutoff].every(n=>Number.isFinite(n)&&n>0)) return [];
  return Array.from({length:196},(_,p)=>p).filter(p=>Number.isFinite(a[p])&&Number.isFinite(b[p])&&a[p]>=0&&b[p]>=0&&a[p]/scaleA>=cutoff&&b[p]/scaleB>=cutoff);
}
export function sharedPatchRows(data:PatchReference,rows:number[][],scales:number[],cutoff:number,source:number,target:number) {
  return coactiveRowIndices(rows,scales,cutoff,source,target).filter(index=>{
    const offsetA=(source*data.referenceIds.length+index)*196;
    const offsetB=(target*data.referenceIds.length+index)*196;
    for(let p=0;p<196;p++) if(data.values[offsetA+p]/scales[source]>=cutoff&&data.values[offsetB+p]/scales[target]>=cutoff) return true;
    return false;
  });
}
export function measurePatchCoactivation(data:PatchReference,rows:number[][],scales:number[],cutoff:number) {
  if(data.referenceIds.length!==rows.length||data.groups!==scales.length) throw Error('Patch reference does not match image reference');
  return measureCoactivation(rows,scales,cutoff).map(link=>{
    const joint=sharedPatchRows(data,rows,scales,cutoff,link.source,link.target).length;
    return {...link,joint,weight:joint/link.either};
  }).filter(link=>link.joint>0).sort((a,b)=>b.weight-a.weight||b.joint-a.joint||a.source-b.source||a.target-b.target);
}
async function unpack(response:Response) {
  if(!response.ok) throw Error('Patch responses could not be loaded.');
  const bytes=await response.arrayBuffer();const header=new Uint8Array(bytes);
  return header[0]===31&&header[1]===139
    ?await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():bytes;
}
type Index={groups:number;patches:number;referenceIds:number[];methods:Record<string,{file:string;sha256:string;bytes:number;extraIds:number[]}>};
let indexPromise:Promise<Index>|undefined;
const referenceCache=new Map<string,Promise<PatchReference>>();
const frameCache=new Map<string,Promise<PatchFrame>>();
const index=()=>indexPromise??=fetch('/data/patch-activity/index.json').then(async r=>{
  if(!r.ok) throw Error('Patch reference index unavailable');return r.json() as Promise<Index>;
}).catch(error=>{indexPromise=undefined;throw error;});
export function loadPatchReference(method:string):Promise<PatchReference> {
  if(!referenceCache.has(method)) referenceCache.set(method,(async()=>{
    const metadata=await index(),entry=metadata.methods[method];
    if(!entry) throw Error('Unknown patch response method');
    const bytes=await unpack(await fetch('/data/patch-activity/'+entry.file));
    if(bytes.byteLength!==entry.bytes||bytes.byteLength!==metadata.groups*metadata.referenceIds.length*metadata.patches*8) throw Error('Incomplete patch response file');
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
    if(digest!==entry.sha256) throw Error('Patch response verification failed');
    const view=new DataView(bytes),values=new Float64Array(bytes.byteLength/8);
    for(let i=0;i<values.length;i++) {values[i]=view.getFloat64(i*8,true);if(!Number.isFinite(values[i])||values[i]<0) throw Error('Invalid patch response');}
    return {referenceIds:metadata.referenceIds,values,groups:metadata.groups,patches:metadata.patches};
  })().catch(error=>{referenceCache.delete(method);throw error;}));
  return referenceCache.get(method)!;
}
export async function loadPatchFrame(method:string,imageId:number):Promise<PatchFrame> {
  const metadata=await index();const referenceIndex=metadata.referenceIds.indexOf(imageId);
  if(referenceIndex>=0) return referenceFrame(await loadPatchReference(method),referenceIndex);
  if(!metadata.methods[method]?.extraIds.includes(imageId)) return {};
  const key=method+'-'+imageId;
  if(!frameCache.has(key)) frameCache.set(key,unpack(await fetch('/data/patch-activity/'+key+'.json.gz')).then(bytes=>{
    const frame=JSON.parse(new TextDecoder().decode(bytes)) as Record<number,number[]>;
    if(Object.entries(frame).some(([g,n])=>!Number.isInteger(Number(g))||Number(g)<0||Number(g)>=75||n.length!==196||n.some(v=>!Number.isFinite(v)||v<0))) throw Error('Invalid image patch map');
    return frame;
  }).catch(error=>{frameCache.delete(key);throw error;}));
  return frameCache.get(key)!;
}
