import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {coactivationReference, measureCoactivation, visibleCoactivation} from '../lib/coactivation.ts';

const rows = [[1,1,0], [1,1,0], [1,1,0], [1,0,1], [0,0,1], [null,1,1]];
const links = measureCoactivation(rows, [1,1,1], .5);
assert.deepEqual(links[0], {source:0,target:1,joint:3,either:4,weight:.75,samples:5});
assert.equal(visibleCoactivation(links,.5,null).length,1);
assert.equal(visibleCoactivation(links,.5,[true,false,true]).length,0);
assert.equal(visibleCoactivation(links,.5,[true,true,false]).length,1);
assert.deepEqual(measureCoactivation([[0,0]], [1,1], .5), []);
assert.throws(() => measureCoactivation(rows,[1,0,1],.5));

const manifest = JSON.parse(readFileSync('public/data/manifest.json','utf8'));
const reference = coactivationReference(manifest);
assert.equal(reference.imageIds.length,87);
for (const method of manifest.methods) {
  const scales = method.groups.map(group => group.q99);
  const data = reference.rows[method.id];
  const edges = measureCoactivation(data,scales,.45);
  assert.equal(new Set(edges.map(edge => edge.source+':'+edge.target)).size,edges.length);
  for (const edge of edges) {
    const source = new Set(data.flatMap((row,i) => row[edge.source]/scales[edge.source]>=.45 ? [i] : []));
    const target = new Set(data.flatMap((row,i) => row[edge.target]/scales[edge.target]>=.45 ? [i] : []));
    const joint = [...source].filter(i=>target.has(i)).length;
    const either = new Set([...source,...target]).size;
    assert.equal(edge.joint,joint);
    assert.equal(edge.either,either);
    assert.equal(edge.weight,joint/either);
    assert.equal(edge.samples,87);
  }
  console.log(method.name, edges.length, 'measured pairs independently recounted');
}
console.log('Co-activation counts, missing-data handling and current-image filter passed.');

