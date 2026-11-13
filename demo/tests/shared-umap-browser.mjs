import {chromium} from 'playwright-core';
import assert from 'node:assert/strict';
import fs from '../scripts/asset-fs.mjs';
import * as THREE from 'three';
import {sharedUmapPoints} from '../lib/shared-umap.ts';

const origin='http://127.0.0.1:5173';
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--use-angle=swiftshader','--enable-webgl']});
const context=await browser.newContext({viewport:{width:1600,height:1100}});
const page=await context.newPage(),errors=[];
page.on('pageerror',error=>errors.push(error.message));
const tour=()=>page.locator('.atlas-tour');
const waitStep=async n=>{await page.waitForFunction(n=>document.querySelector('.tour-progress')?.textContent.includes(n+' of 16'),n);await page.waitForTimeout(420)};
try{
 await fs.mkdir('.local/qa',{recursive:true});
 await page.goto(origin,{waitUntil:'networkidle',timeout:120000});
 await page.waitForFunction(()=>document.querySelectorAll('.method-atlas canvas').length===4);
 await tour().waitFor();await waitStep(1);
 await page.screenshot({path:'.local/qa/umap-tour-welcome.png',fullPage:true});
 await page.waitForTimeout(700);assert((await tour().textContent()).includes('1 of 16'));
 for(let n=1;n<=16;n++){
  await waitStep(n);
  const bounds=await page.locator('.tour-bubble').boundingBox();
  assert(bounds.x>=0&&bounds.y>=0&&bounds.x+bounds.width<=1601&&bounds.y+bounds.height<=1101,'Tour bubble outside viewport at step '+n+': '+JSON.stringify(bounds));
  assert(await page.locator('.tour-spotlight').count(),'Missing tutorial target at step '+n);
  if(n===11)await page.getByRole('dialog',{name:'Measured group PCA'}).locator('canvas').waitFor();
  if(n===15)await page.getByRole('dialog',{name:'Research benchmark evidence'}).waitFor();
  await tour().getByRole('button',{name:n===16?'Finish':'Next',exact:true}).click();
 }
 assert.equal(await tour().count(),0);assert.equal(await page.locator('.simple-atlas').getAttribute('inert'),null);
 await page.getByRole('button',{name:'Take a tour',exact:true}).click();await waitStep(1);
 await tour().getByRole('button',{name:'Next',exact:true}).click();await waitStep(2);
 await tour().getByRole('button',{name:'Back',exact:true}).click();await waitStep(1);
 await page.keyboard.press('Escape');assert.equal(await tour().count(),0);
 await page.reload({waitUntil:'networkidle'});assert.equal(await tour().count(),0);
 console.log('All 16 manual tutorial steps, targets, Back, Finish, Escape, replay and remembered dismissal passed.');
 await page.getByRole('button',{name:'Explore & uploads',exact:true}).click();
 const map=JSON.parse(await fs.readFile('public/data/shared-umap.json','utf8'));
 const manifest=JSON.parse(await fs.readFile('public/data/manifest.json','utf8'));
 const model=map.model_sha256;
 await page.getByLabel('Geometry space').selectOption('umap');await page.waitForTimeout(400);
 for(const card of await page.locator('.method-atlas').all()){
  assert.equal(await card.getAttribute('data-coordinate-sha'),model);
  assert.deepEqual(JSON.parse(await card.locator('.scene').getAttribute('data-pca-bounds')),map.bounds);
  assert((await card.locator('.axis').allTextContents()).every(text=>text.startsWith('UMAP')));
 }
 await page.screenshot({path:'.local/qa/umap-atlas-desktop.png',fullPage:true});
 for(const method of manifest.methods){
  const card=page.locator('.method-atlas[data-method="'+method.id+'"]');
  await card.locator('header button').click();
  const group=Number((await page.locator('.atlas-selected-method>span').textContent()).match(/\d+/)[0]);
  const data=JSON.parse(await fs.readFile(`data/groups/${method.id}-${group}.json`,'utf8'));
  const gallery=page.getByRole('region',{name:'Original blinded-review evidence'});await gallery.waitFor();
  assert.equal(await gallery.locator('button').count(),6);
  const view=data.reviewedViews[1];await gallery.getByRole('button',{name:'View original review image '+view.imageId,exact:true}).click();
  await page.waitForFunction(id=>document.querySelector('.atlas-source-caption')?.textContent.startsWith('Image '+id+' · patch '),view.imageId);
  assert.deepEqual(await page.locator('.atlas-patch-row button').evaluateAll(buttons=>buttons.map(button=>Number(button.getAttribute('aria-label').match(/patch (\d+)/)[1]))),view.patches);
  const p=view.patches[2];await page.getByRole('button',{name:'Select patch '+p,exact:true}).click();
  assert.equal(await page.locator('.atlas-source-caption').textContent(),'Image '+view.imageId+' · patch '+p);
  await page.getByRole('button',{name:'Inspect measured PCA',exact:true}).click();
  await page.getByRole('dialog',{name:'Measured group PCA'}).locator('canvas').waitFor();await page.keyboard.press('Escape');
  console.log(method.name,'original six images, exact four patches, patch selection and separate group PCA passed.');
 }
 // Return to a fixed example and select an actual projected review point.
 await page.getByRole('button',{name:'Gloves',exact:true}).click();await page.waitForTimeout(900);
 const card=page.locator('.method-atlas[data-method="ggf"]'),canvas=card.locator('canvas'),rect=await canvas.boundingBox();
 const data=JSON.parse(await fs.readFile('data/groups/ggf-32.json','utf8'));
 const points=sharedUmapPoints(map,'review',data,32,manifest.methods[0].groups[32].q99,manifest.imageIds.indexOf(502857),false,.45).filter(point=>point.pickable);
 const center=new THREE.Vector3(...map.bounds.min).add(new THREE.Vector3(...map.bounds.max)).multiplyScalar(.5),span=new THREE.Vector3(...map.bounds.max).sub(new THREE.Vector3(...map.bounds.min)),scale=2.25/Math.max(span.x,span.y,span.z);
 const camera=new THREE.PerspectiveCamera(40,rect.width/rect.height,.01,100);camera.position.set(2.6,1.75,3.4).normalize().multiplyScalar(1.2/Math.tan(40*Math.PI/360)*1.25);camera.lookAt(0,0,0);camera.updateMatrixWorld();
 const projected=points.map(point=>{const v=new THREE.Vector3(...point.xyz).sub(center).multiplyScalar(scale).project(camera);return {...point,x:rect.x+(v.x*.5+.5)*rect.width,y:rect.y+(-v.y*.5+.5)*rect.height,z:v.z}});
 const candidate=projected.find(point=>point.z>-1&&point.z<1&&point.x>rect.x+25&&point.x<rect.x+rect.width-25&&point.y>rect.y+25&&point.y<rect.y+rect.height-25);
 assert(candidate);let nearest=null,distance=150;for(const point of projected){const d=(point.x-candidate.x)**2+(point.y-candidate.y)**2;if(d<distance){distance=d;nearest=point;}}
 await page.mouse.click(candidate.x,candidate.y);
 await page.waitForFunction(({image,patch})=>document.querySelector('.atlas-source-caption')?.textContent==='Image '+image+' · patch '+patch,{image:manifest.imageIds[Math.floor(nearest.id/196)],patch:nearest.id%196});
 console.log('Actual 3D UMAP point click retained original image and patch identity.');
 const responsePromise=page.waitForResponse(response=>response.url().endsWith('/api/infer')&&response.request().method()==='POST',{timeout:120000});
 await page.locator('input[type=file]').setInputFiles('public/review-crops/498645.webp');
 const response=await responsePromise;assert.equal(response.status(),200);const live=await response.json();
 assert.equal(live.umap.model_sha256,model);assert.equal(live.umap.positions.length,196);
 await page.waitForFunction(()=>document.querySelector('.atlas-source')?.textContent==='LIVE UPLOAD');
 for(const method of manifest.methods){
  const methodCard=page.locator('.method-atlas[data-method="'+method.id+'"]');
  const ranked=live.methods[method.id].scores.map((raw,group)=>({group,relative:raw/method.groups[group].q99})).sort((a,b)=>b.relative-a.relative||a.group-b.group);
  assert.equal(await methodCard.locator('header>span>b').textContent(),String(ranked.filter(row=>row.relative>=.45).length));
  const group=ranked[0].group;await methodCard.getByRole('button',{name:'Inspect '+method.name+' group '+group,exact:true}).click();
  await page.waitForTimeout(250);
  assert((await page.locator('.atlas-source-caption').textContent()).startsWith('Your uploaded image'));
  const p=Number((await page.locator('.atlas-source-caption').textContent()).match(/patch (\d+)/)[1]);
  assert.equal(await page.locator('.atlas-patch-measure strong').textContent(),live.methods[method.id].norms[group][p].toFixed(3));
  await page.getByRole('button',{name:'Inspect measured PCA',exact:true}).click();await page.getByRole('dialog',{name:'Measured group PCA'}).locator('canvas').waitFor();await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'Browse all',exact:true}).click();assert.equal(await page.locator('.atlas-ranking>button').count(),75);await page.getByRole('button',{name:'Show strongest',exact:true}).click();
 }
 await page.screenshot({path:'.local/qa/umap-atlas-live.png',fullPage:true});
 console.log('Real upload: one 196-point UMAP transform, four frozen methods, live response values, group rankings and native PCA passed.');
 // Original evidence remains linked after a live upload.
 await page.getByRole('region',{name:'Original blinded-review evidence'}).locator('button').first().click();
 assert.equal(await page.locator('.atlas-source').textContent(),'RECORDED IMAGE');
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(400);
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.getByRole('button',{name:'Take a tour',exact:true}).click();
 for(let n=1;n<=16;n++){
  await waitStep(n);const b=await page.locator('.tour-bubble').boundingBox();
  assert(b.x>=0&&b.y>=0&&b.x+b.width<=391&&b.y+b.height<=845,'Mobile tutorial bubble outside viewport at '+n+': '+JSON.stringify(b));
  if(n===2)await page.screenshot({path:'.local/qa/umap-tour-mobile.png',fullPage:true});
  await tour().getByRole('button',{name:n===16?'Finish':'Next',exact:true}).click();
 }
 await page.screenshot({path:'.local/qa/umap-atlas-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);
 console.log('Mobile layout and all 16 mobile walkthrough steps passed without browser errors.');
 await fs.writeFile('.local/qa/shared-umap-browser.json',JSON.stringify({passed:true,model_sha256:model,tour_steps:16,desktop:true,mobile:true,live_request:live.request_id,seconds:live.seconds,errors},null,2));
}finally{await browser.close()}
