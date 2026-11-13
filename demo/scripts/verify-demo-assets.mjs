import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
const root=path.resolve(import.meta.dirname,'..');
const hashes=JSON.parse(await fs.readFile(path.join(root,'ASSETS.json'),'utf8'));
for(const [name,expected] of Object.entries(hashes)){
 const file=path.resolve(root,name);
 if(!file.startsWith(root+path.sep)||createHash('sha256').update(await fs.readFile(file)).digest('hex')!==expected)throw Error('Recorded demo asset differs: '+name);
}
await fs.mkdir(path.join(root,'.local/qa'),{recursive:true});
console.log(`Verified ${Object.keys(hashes).length} recorded demo assets.`);
