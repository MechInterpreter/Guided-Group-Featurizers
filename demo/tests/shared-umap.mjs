import assert from 'node:assert/strict';
import fs from '../scripts/asset-fs.mjs';
import {gunzipSync} from 'node:zlib';
import {sharedUmapPoints,assertSharedUmapLive} from '../lib/shared-umap.ts';

const map=JSON.parse(await fs.readFile('public/data/shared-umap.json','utf8'));
const manifest=JSON.parse(await fs.readFile('public/data/manifest.json','utf8'));
assert.deepEqual(map.image_ids,manifest.imageIds);
assert.equal(map.positions.length,797);
assert.equal(map.fit_patch_count,64*196);
assert.equal(map.parameters.n_components,3);
assert.equal(map.parameters.metric,'cosine');
assert.deepEqual(JSON.parse(gunzipSync(await fs.readFile('public/data/shared-umap.json.gz'))),map);
for(const frame of map.positions){
 assert.equal(frame.length,196);
 for(const xyz of frame){assert.equal(xyz.length,3);assert(xyz.every(Number.isFinite));for(let k=0;k<3;k++)assert(xyz[k]>=map.bounds.min[k]&&xyz[k]<=map.bounds.max[k]);}
}
const reviewed=[...new Set(Object.values(manifest.reviewedOverview).flatMap(rows=>rows.map(row=>row[1])))].sort((a,b)=>a-b);
assert.deepEqual(map.reviewed_indices,reviewed);
let common;
for(const method of manifest.methods){
 for(let group=0;group<75;group++){
  const source=JSON.parse(await fs.readFile(`data/groups/${method.id}-${group}.json`,'utf8'));
  const points=sharedUmapPoints(map,'review',source,group,method.groups[group].q99,0,false,.45);
  const geometry=points.map(point=>[point.id,...point.xyz]);
  if(common)assert.deepEqual(geometry,common);else common=geometry;
  assert.deepEqual(points.filter(point=>point.pickable).map(point=>point.id).sort((a,b)=>a-b),[...source.reviewedIndices].sort((a,b)=>a-b));
  for(const point of points)assert.deepEqual(point.xyz,map.positions[Math.floor(point.id/196)][point.id%196]);
 }
 console.log(method.name,'all 75 groups: identical UMAP coordinates and exact original review membership');
}
let referenceGeometry;
for(const method of manifest.methods){
 const data=JSON.parse(await fs.readFile(`data/groups/${method.id}-32.json`,'utf8'));
 const geometry=sharedUmapPoints(map,'reference',data,32,method.groups[32].q99,0,false,.45).map(point=>[point.id,...point.xyz]);
 if(referenceGeometry)assert.deepEqual(geometry,referenceGeometry);else referenceGeometry=geometry;
}
const data=JSON.parse(await fs.readFile('data/groups/ggf-32.json','utf8'));
const live={positions:Array.from({length:196},(_,p)=>[1e6+p,-1e6,p/10]),norms:Array(196).fill(2)};
const points=sharedUmapPoints(map,'review',data,32,1,0,true,.45,live);
for(let p=0;p<196;p++)assert.deepEqual(points.find(point=>point.id===-p-1).xyz,live.positions[p]);
assertSharedUmapLive({umap:{positions:live.positions,model_sha256:map.model_sha256}},map);
assert.throws(()=>assertSharedUmapLive({umap:{positions:live.positions,model_sha256:'different-fit'}},map));
console.log('156,212 real patch coordinates, lossless export, immutable highlights, unchanged live outliers and projection-hash rejection verified.');
