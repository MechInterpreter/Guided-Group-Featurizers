import {chromium} from 'playwright-core';
import assert from 'node:assert/strict';
import fs from '../scripts/asset-fs.mjs';
import * as THREE from 'three';
import {nativeGeometryPoints,nativeBounds} from '../lib/native-geometry.ts';
import {focusPair} from '../lib/pair-geometry.ts';

const data=JSON.parse(await fs.readFile('public/data/pair-reviews.json','utf8'));
const manifest=JSON.parse(await fs.readFile('public/data/manifest.json','utf8'));
let checked=0;
for(const method of manifest.methods)for(const entry of data.methods[method.id]){
 const source=JSON.parse(await fs.readFile(`data/groups/${method.id}-${entry.group}.json`,'utf8'));
 const points=nativeGeometryPoints(source,'reference',0,false,.45,method.groups[entry.group].q99);
 assert.deepEqual(entry.pairs.flatMap(p=>p.views),source.reviewedViews);
 for(const pair of entry.pairs){
  const highlighted=focusPair(points,pair),ids=new Set(pair.views.flatMap(v=>v.patches.map(p=>v.imageIndex*196+p)));
  assert.deepEqual(highlighted.map(p=>[p.id,p.xyz]),points.map(p=>[p.id,p.xyz]));
  assert.deepEqual(new Set(highlighted.filter(p=>p.pickable).map(p=>p.id)),ids);
  assert.equal(ids.size,8);
  for(const endpoint of ['semantic','visual']){
   const votes=pair.reviews.map(r=>r[endpoint]);
   assert.equal(pair.majority[endpoint],votes.filter(v=>v==='yes').length>=2?'yes':votes.filter(v=>v==='no').length>=2?'no':'unclear');
  }
  checked++;
 }
}
assert.equal(checked,900);console.log('All 900 pairs preserve exact reviewed views, independent endpoint votes and every stored geometry coordinate.');

const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--use-angle=swiftshader','--enable-webgl']});
const context=await browser.newContext({viewport:{width:1600,height:1100}});
await context.addInitScript(()=>localStorage.setItem('ggf-native-geometry-tour-v2','seen'));
const page=await context.newPage(),errors=[];let posts=0;
page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/api/infer'))posts++});
async function ready(group){await page.waitForFunction(g=>document.querySelectorAll(`.pair-method[data-highlighted-group="${g}"] canvas`).length===4,group);}
try{
 await page.goto('http://127.0.0.1:5173',{waitUntil:'networkidle',timeout:120000});await ready(0);
 assert.equal(await page.getByRole('button',{name:'Reviewed pairs',exact:true}).getAttribute('aria-pressed'),'true');
 assert(await page.getByRole('button',{name:'Previous reviewed pair'}).isDisabled());
 for(const group of [0,32,69,74])for(const pairIndex of [0,1,2]){
  await page.getByLabel('Walkthrough group',{exact:true}).selectOption(String(group));
  await page.getByLabel('Fixed review pair',{exact:true}).selectOption(String(pairIndex));await ready(group);
  for(const method of manifest.methods){
   const card=page.locator(`.method-atlas[data-method="${method.id}"]`),entry=data.methods[method.id][group],pair=entry.pairs[pairIndex];
   assert.equal(await card.locator('.pair-evidence').getAttribute('data-case-id'),entry.case_id);
   assert.equal(await card.locator('.pair-evidence').getAttribute('data-pair'),String(pairIndex));
   assert.deepEqual(await card.locator('.pair-crop image').evaluateAll(images=>images.map(image=>image.getAttribute('href'))),pair.views.map(v=>data.crops[String(v.imageId)].src));
   assert.deepEqual(await card.locator('.pair-view').evaluateAll(views=>views.map(v=>[Number(v.dataset.imageId),v.dataset.patches.split(',').map(Number)])),pair.views.map(v=>[v.imageId,v.patches]));
   for(const endpoint of ['semantic','visual'])assert.equal(await card.locator(`[data-endpoint="${endpoint}"]`).getAttribute('data-verdict'),pair.majority[endpoint]);
   await card.locator('.pair-original-reviews summary').click();
   assert.deepEqual(await card.locator('.original-meaning').allTextContents(),pair.reviews.map(r=>r.shared_meaning||'No shared meaning recorded.'));
   assert.deepEqual(await card.locator('.original-exceptions').allTextContents(),pair.reviews.map(r=>'Exceptions: '+(r.exceptions||'None recorded.')));
   await card.locator('.pair-original-reviews summary').click();
   const selected=pair.views[1],patch=selected.patches[2];
   await card.getByRole('button',{name:`Inspect original view ${pairIndex*2+2} patch ${patch}`,exact:true}).click();
   await page.waitForFunction(({image,patch})=>document.querySelector('.atlas-source-caption')?.textContent===`Image ${image} · patch ${patch}`,{image:selected.imageId,patch});
   assert.equal(await page.locator('.atlas-selected-method>span').textContent(),'GROUP '+String(group).padStart(2,'0'));
   assert.equal(await card.locator('.pair-zooms button[aria-pressed=true]').textContent(),String(patch));
  }
 }
 assert(await page.getByRole('button',{name:'Next reviewed pair'}).isDisabled());
 await page.getByRole('button',{name:'Previous reviewed pair'}).click();
 assert.equal(await page.getByLabel('Fixed review pair',{exact:true}).inputValue(),'1');
 assert.equal(posts,0);
 console.log('First, middle and last groups, all fixed pairs, all four methods: exact notes, votes, views, zoom clicks and end boundaries passed.');

 await page.getByLabel('Walkthrough group',{exact:true}).selectOption('32');await page.getByLabel('Fixed review pair',{exact:true}).selectOption('0');await ready(32);
 const card=page.locator('.method-atlas[data-method="ggf"]'),entry=data.methods.ggf[32],pair=entry.pairs[0];
 const source=JSON.parse(await fs.readFile('data/groups/ggf-32.json','utf8'));
 const points=focusPair(nativeGeometryPoints(source,'reference',0,false,.45,manifest.methods[0].groups[32].q99),pair).filter(p=>p.pickable);
 await page.getByRole('button',{name:'Reset atlas views',exact:true}).click();await page.waitForTimeout(750);
 await card.locator('canvas').scrollIntoViewIfNeeded();
 const rect=await card.locator('canvas').boundingBox(),bounds=nativeBounds(source);
 const center=new THREE.Vector3(...bounds.min).add(new THREE.Vector3(...bounds.max)).multiplyScalar(.5),span=new THREE.Vector3(...bounds.max).sub(new THREE.Vector3(...bounds.min)),scale=2.25/Math.max(span.x,span.y,span.z,1e-5);
 const camera=new THREE.PerspectiveCamera(40,rect.width/rect.height,.01,100);camera.position.set(2.6,1.75,3.4).normalize().multiplyScalar(1.2/Math.tan(40*Math.PI/360)*1.25);camera.lookAt(0,0,0);camera.updateMatrixWorld();
 const projected=points.map(point=>{const v=new THREE.Vector3(...point.xyz).sub(center).multiplyScalar(scale).project(camera);return {...point,x:rect.x+(v.x*.5+.5)*rect.width,y:rect.y+(-v.y*.5+.5)*rect.height,z:v.z}});
 const point=projected.find(p=>p.z>-1&&p.z<1&&p.x>rect.x+20&&p.x<rect.x+rect.width-20&&p.y>rect.y+20&&p.y<rect.y+rect.height-20);assert(point);
 let nearest=null,distance=150;for(const p of projected){const d=(p.x-point.x)**2+(p.y-point.y)**2;if(d<distance){nearest=p;distance=d}}
 await page.mouse.click(point.x,point.y);
 await page.waitForFunction(({image,patch})=>document.querySelector('.atlas-source-caption')?.textContent===`Image ${image} · patch ${patch}`,{image:manifest.imageIds[Math.floor(nearest.id/196)],patch:nearest.id%196});
 assert.equal(await page.getByRole('button',{name:'Reviewed pairs',exact:true}).getAttribute('aria-pressed'),'true');
 assert.deepEqual(JSON.parse(await card.locator('.scene').getAttribute('data-pca-bounds')),bounds);
 await page.getByLabel('Geometry space').selectOption('umap');await page.waitForTimeout(300);assert.equal(await page.locator('.pair-method[data-geometry="umap"]').count(),4);
 await page.getByLabel('Geometry space').selectOption('native');
 await page.screenshot({path:'.local/qa/pair-walkthrough-desktop.png',fullPage:true});

 // A real inference must leave review mode and return to this exact cursor from cache.
 const reply=page.waitForResponse(r=>r.url().endsWith('/api/infer')&&r.request().method()==='POST',{timeout:120000});
 await page.locator('input[type=file]').setInputFiles('public/review-crops/498645.webp');const response=await reply;assert.equal(response.status(),200);const live=await response.json();
 const cached=page.getByRole('button',{name:/^Open cached upload 1:/});await cached.waitFor();
 await page.waitForFunction(()=>document.querySelector('.atlas-source')?.textContent==='LIVE UPLOAD');
 assert.equal(await page.locator('.pair-evidence').count(),0);assert.equal(posts,1);
 await page.getByRole('button',{name:'Reviewed pairs',exact:true}).click();await ready(32);
 assert.equal(await page.getByLabel('Fixed review pair',{exact:true}).inputValue(),'0');
 await cached.click();await page.waitForFunction(id=>document.querySelector('.atlas-receipt code:last-child')?.textContent===id,live.request_id);
 assert.equal(posts,1);assert.equal(await page.locator('.pair-evidence').count(),0);
 await page.getByRole('button',{name:'Recorded examples',exact:true}).click();await ready(32);
 assert.equal(await page.getByLabel('Fixed review pair',{exact:true}).inputValue(),'0');assert.equal(posts,1);
 console.log('Actual geometry click and UMAP switch stay linked; real uploads enter Explore and return to the last reviewed pair with no rerun.');
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(350);
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.getByRole('button',{name:'Next reviewed pair'}).click();assert.equal(await page.getByLabel('Fixed review pair',{exact:true}).inputValue(),'1');
 await page.screenshot({path:'.local/qa/pair-walkthrough-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);
 await fs.writeFile('.local/qa/pair-walkthrough-browser.json',JSON.stringify({passed:true,pairs_verified:checked,sampled_navigation_steps:12,methods:4,exact_notes:true,actual_geometry_click:true,real_live_upload:true,session_cache_preserved:true,mobile:true,errors},null,2));
}finally{await browser.close()}
