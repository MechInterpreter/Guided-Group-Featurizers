import {chromium} from 'playwright-core';
import assert from 'node:assert/strict';
import fs from '../scripts/asset-fs.mjs';
import * as THREE from 'three';
import {nativeGeometryPoints,nativeBounds} from '../lib/native-geometry.ts';

const origin='http://127.0.0.1:5173';
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--use-angle=swiftshader','--enable-webgl']});
const context=await browser.newContext({viewport:{width:1600,height:1100}});
await context.addInitScript(()=>localStorage.setItem('ggf-native-geometry-tour-v2','seen'));
const page=await context.newPage(),errors=[];page.on('pageerror',error=>errors.push(error.message));
const manifest=JSON.parse(await fs.readFile('public/data/manifest.json','utf8'));
const source=async(method,group)=>JSON.parse(await fs.readFile(`data/groups/${method}-${group}.json`,'utf8'));

async function selectProjected(card,points,bounds,predicate){
 await page.getByRole('button',{name:'Reset atlas views',exact:true}).click();await page.waitForTimeout(750);
 await card.locator('canvas').scrollIntoViewIfNeeded();
 const rect=await card.locator('canvas').boundingBox();
 const center=new THREE.Vector3(...bounds.min).add(new THREE.Vector3(...bounds.max)).multiplyScalar(.5);
 const span=new THREE.Vector3(...bounds.max).sub(new THREE.Vector3(...bounds.min)),scale=2.25/Math.max(span.x,span.y,span.z,1e-5);
 const camera=new THREE.PerspectiveCamera(40,rect.width/rect.height,.01,100);camera.position.set(2.6,1.75,3.4).normalize().multiplyScalar(1.2/Math.tan(40*Math.PI/360)*1.25);camera.lookAt(0,0,0);camera.updateMatrixWorld();
 const projected=points.filter(point=>point.pickable!==false).map(point=>{const v=new THREE.Vector3(...point.xyz).sub(center).multiplyScalar(scale).project(camera);return {...point,x:rect.x+(v.x*.5+.5)*rect.width,y:rect.y+(-v.y*.5+.5)*rect.height,z:v.z}}).filter(point=>point.z>-1&&point.z<1);
 const candidate=projected.find(point=>predicate(point)&&point.x>rect.x+25&&point.x<rect.x+rect.width-25&&point.y>rect.y+25&&point.y<rect.y+rect.height-25);
 assert(candidate,'A visible measured point must be selectable');
 let nearest=null,distance=150;for(const point of projected){const d=(point.x-candidate.x)**2+(point.y-candidate.y)**2;if(d<distance){nearest=point;distance=d}}
 await page.mouse.click(candidate.x,candidate.y);return nearest;
}

try{
 await page.goto(origin,{waitUntil:'networkidle',timeout:120000});
 await page.waitForFunction(()=>document.querySelectorAll('.method-atlas canvas').length===4);
 await page.getByRole('button',{name:'Explore & uploads',exact:true}).click();
 assert.equal(await page.getByLabel('Geometry space').inputValue(),'native');
 assert.equal(await page.getByLabel('Map evidence set').inputValue(),'reference');
 for(const method of manifest.methods){
  const card=page.locator('.method-atlas[data-method="'+method.id+'"]');
  const group=Number(await card.getAttribute('data-highlighted-group')),data=await source(method.id,group);
  assert.equal(await card.getAttribute('data-geometry'),'native');
  const groupColor=await card.getAttribute('data-group-color');assert.match(groupColor,/^#[0-9a-f]{6}$/);
  assert.deepEqual(await card.locator('.axis').allTextContents(),['PC 1','PC 2','PC 3']);
  assert.deepEqual(JSON.parse(await card.locator('.scene').getAttribute('data-pca-bounds')),nativeBounds(data));
  assert((await card.locator('.umap-card-caption').textContent()).includes((data.variance*100).toFixed(1)+'% variance'));
  await card.locator('header button').click();await page.waitForTimeout(250);
  const points=nativeGeometryPoints(data,'reference',0,false,.45,method.groups[group].q99),reviewed=new Set(data.reviewedIndices);
  const point=await selectProjected(card,points,nativeBounds(data),point=>reviewed.has(point.id));
  const image=manifest.imageIds[Math.floor(point.id/196)],patch=point.id%196;
  await page.waitForFunction(({image,patch})=>document.querySelector('.atlas-source-caption')?.textContent==='Image '+image+' · patch '+patch,{image,patch});
  assert((await page.locator('.atlas-selected-method').textContent()).includes(method.name));
  assert.equal(await page.locator('.atlas-inspector').getAttribute('data-group-color'),groupColor);
  assert.equal(await page.locator('.atlas-photo>rect').last().getAttribute('stroke'),groupColor);
  assert.equal(await page.locator('.atlas-patch-row button svg rect').first().getAttribute('stroke'),groupColor);
  const norm=point.id<data.norms.length?data.norms[point.id]:data.additionalImages[Math.floor(point.id/196)].norms[patch];
  assert.equal(await page.locator('.atlas-patch-measure strong').textContent(),norm.toFixed(3));
  const gallery=page.getByRole('region',{name:'Original blinded-review evidence'});assert.equal(await gallery.locator('button').count(),6);
  await page.getByRole('button',{name:'Inspect measured PCA',exact:true}).click();await page.getByRole('dialog',{name:'Measured group PCA'}).locator('canvas').waitFor();await page.keyboard.press('Escape');
  console.log(method.name,'native main cloud, original PCA bounds/variance and actual point-to-patch selection passed.');
 }
 await page.getByRole('button',{name:'Gloves',exact:true}).click();await page.waitForTimeout(600);
 await page.screenshot({path:'.local/qa/native-geometry-desktop.png',fullPage:true});
 const responsePromise=page.waitForResponse(response=>response.url().endsWith('/api/infer')&&response.request().method()==='POST',{timeout:120000});
 await page.locator('input[type=file]').setInputFiles('public/review-crops/498645.webp');
 const response=await responsePromise;assert.equal(response.status(),200);const live=await response.json();
 await page.waitForFunction(()=>document.querySelector('.atlas-source')?.textContent==='LIVE UPLOAD');
 for(const method of manifest.methods){
  const ranked=live.methods[method.id].scores.map((raw,group)=>({group,relative:raw/method.groups[group].q99})).sort((a,b)=>b.relative-a.relative||a.group-b.group);
  const group=ranked[0].group,card=page.locator('.method-atlas[data-method="'+method.id+'"]'),data=await source(method.id,group);
  await card.getByRole('button',{name:'Inspect '+method.name+' group '+group,exact:true}).click();await page.waitForTimeout(400);
  const frame={positions:live.methods[method.id].positions[group],norms:live.methods[method.id].norms[group]};
  const points=nativeGeometryPoints(data,'review',0,true,.45,method.groups[group].q99,frame);
  await page.getByLabel('Map evidence set').selectOption('review');await page.waitForTimeout(350);
  const point=await selectProjected(card,points,nativeBounds(data),point=>point.id<0);
  assert(point.id<0,'The selected visible live point must stay in the upload');const patch=-point.id-1;
  await page.waitForFunction(patch=>document.querySelector('.atlas-source-caption')?.textContent==='Your uploaded image · patch '+patch,patch);
  assert.equal(await page.locator('.atlas-patch-measure strong').textContent(),frame.norms[patch].toFixed(3));
  assert.deepEqual(JSON.parse(await card.locator('.scene').getAttribute('data-pca-bounds')),nativeBounds(data));
  console.log(method.name,'real upload uses its own decoded-response PCA and exact patch identity; frozen reference frame unchanged.');
 }
 await page.getByLabel('Geometry space').selectOption('umap');await page.waitForTimeout(500);
 for(const card of await page.locator('.method-atlas').all())assert.equal(await card.getAttribute('data-geometry'),'umap');
 assert.equal(await page.locator('.atlas-source').textContent(),'LIVE UPLOAD');
 await page.getByLabel('Geometry space').selectOption('native');await page.waitForTimeout(500);
 assert.equal(await page.locator('.atlas-source').textContent(),'LIVE UPLOAD');
 await page.screenshot({path:'.local/qa/native-geometry-live.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(350);
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.screenshot({path:'.local/qa/native-geometry-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);
 await fs.writeFile('.local/qa/native-geometry-browser.json',JSON.stringify({passed:true,methods:4,native_default:true,real_native_live_points:true,optional_input_umap:true,mobile:true,errors},null,2));
}finally{await browser.close()}
