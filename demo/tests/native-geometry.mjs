import assert from 'node:assert/strict';
import fs from '../scripts/asset-fs.mjs';
import {nativeGeometryPoints,nativeBounds,nativeObservation} from '../lib/native-geometry.ts';

const manifest=JSON.parse(await fs.readFile('public/data/manifest.json','utf8'));
for(const method of manifest.methods){
 for(let group=0;group<75;group++){
  const data=JSON.parse(await fs.readFile(`data/groups/${method.id}-${group}.json`,'utf8'));
  const bounds=nativeBounds(data);
  for(const evidence of ['review','reference']){
   const points=nativeGeometryPoints(data,evidence,0,false,.45,method.groups[group].q99);
   const expected=[...new Set(evidence==='review'?data.reviewedIndices:[...data.cloudIndices,...data.reviewedIndices])];
   assert.deepEqual(points.map(point=>point.id),expected);
   for(const point of points){
    assert.deepEqual(point.xyz,nativeObservation(data,point.id).xyz);
    assert.equal(point.group,group);
    for(let k=0;k<3;k++)assert(point.xyz[k]>=bounds.min[k]&&point.xyz[k]<=bounds.max[k]);
   }
  }
  assert.equal(nativeGeometryPoints(data,'review',0,false,.45,1).length,24);
 }
 console.log(method.name,'75 actual native group clouds, frozen bounds, source coordinates and exact 24 review patches verified.');
}
const data=JSON.parse(await fs.readFile('data/groups/ggf-32.json','utf8'));
const bounds=nativeBounds(data),live={positions:Array.from({length:196},(_,p)=>[1e6+p,-1e6,p/10]),norms:Array(196).fill(2)};
const points=nativeGeometryPoints(data,'reference',0,true,.45,1,live);
for(let p=0;p<196;p++)assert.deepEqual(points.find(point=>point.id===-p-1).xyz,live.positions[p]);
assert.deepEqual(nativeBounds(data),bounds,'Uploads must not resize or refit the frozen reference frame');
const repeated={...data,cloudIndices:[0,1],reviewedIndices:[],norms:[1,1],positions:[[1,2,3],[1,2,3]]};
assert.deepEqual(nativeGeometryPoints(repeated,'reference',0,false,.45,1).map(point=>point.xyz),[[1,2,3],[1,2,3]]);
console.log('Native live outliers and overlapping observations remain unaltered.');
