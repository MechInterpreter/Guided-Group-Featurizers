import assert from 'node:assert/strict';
import {readFileSync} from '../scripts/asset-fs.mjs';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {coactivationReference,measureCoactivation} from '../lib/coactivation.ts';
import {sharedPatchIndices,measurePatchCoactivation,sharedPatchRows} from '../lib/patch-activity.ts';
const a=Array(196).fill(0), b=[...a];a[0]=1;b[1]=1;
assert.deepEqual(sharedPatchIndices(a,b,1,1,.5),[],'Different patches must not create a same-patch connection');
b[0]=.5;assert.deepEqual(sharedPatchIndices(a,b,1,1,.5),[0]);
assert.deepEqual(sharedPatchIndices(undefined,b,1,1,.5),[]);
assert.deepEqual(sharedPatchIndices(a,b,0,1,.5),[]);
const manifest=JSON.parse(readFileSync('public/data/manifest.json','utf8'));
const metadata=JSON.parse(readFileSync('public/data/patch-activity/index.json','utf8'));
const reference=coactivationReference(manifest);
assert.deepEqual(metadata.referenceIds,reference.imageIds);
for(const method of manifest.methods){
 const entry=metadata.methods[method.id],bytes=gunzipSync(readFileSync('public/data/patch-activity/'+entry.file));
 assert.equal(createHash('sha256').update(bytes).digest('hex'),entry.sha256);
 const values=Float64Array.from({length:bytes.length/8},(_,i)=>bytes.readDoubleLE(i*8));
 const data={referenceIds:reference.imageIds,values,groups:75,patches:196};
 const scales=method.groups.map(g=>g.q99),rows=reference.rows[method.id];
 const links=measurePatchCoactivation(data,rows,scales,.45),imageLinks=measureCoactivation(rows,scales,.45);
 const raw=Array.from({length:75},(_,g)=>JSON.parse(readFileSync(`data/groups/${method.id}-${g}.json`,'utf8')).norms);
 const expected=[];
 for(let source=0;source<75;source++)for(let target=source+1;target<75;target++){
  let joint=0,either=0;
  for(let i=0;i<rows.length;i++){
   const activeA=rows[i][source]/scales[source]>=.45,activeB=rows[i][target]/scales[target]>=.45;
   if(activeA||activeB) either++;
   const index=manifest.imageIds.indexOf(reference.imageIds[i]);
   if(activeA&&activeB&&Array.from({length:196},(_,p)=>p).some(p=>raw[source][index*196+p]/scales[source]>=.45&&raw[target][index*196+p]/scales[target]>=.45))joint++;
  }
  if(joint)expected.push({source,target,joint,either,weight:joint/either,samples:rows.length});
 }
 expected.sort((a,b)=>b.weight-a.weight||b.joint-a.joint||a.source-b.source||a.target-b.target);
 assert.deepEqual(links,expected);
 for(const link of links){assert(link.joint<=imageLinks.find(l=>l.source===link.source&&l.target===link.target).joint);assert.equal(sharedPatchRows(data,rows,scales,.45,link.source,link.target).length,link.joint);}
 console.log(method.name,links.length,'same-patch pairs independently recounted from original group JSON');
}
console.log('Same-patch scope, denominators, exact patch coordinates and reference hashes verified.');
