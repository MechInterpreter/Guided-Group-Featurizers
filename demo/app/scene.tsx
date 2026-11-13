'use client';

import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

export type ScenePoint = {id:number; xyz:number[]; group?:number; color?:number[]; strength?:number; size?:number; pickable?:boolean};
export type SceneConnection = {source:number; target:number; weight:number};
export type SceneBounds={min:number[];max:number[]};
export type SceneAnnotation={group:number;xyz:number[];label:string};
type Props={axisLabels?:[string,string,string];fixedBounds?:SceneBounds;annotations?:SceneAnnotation[];labelGroups?:number[];onConnectionPick?:(source:number,target:number)=>void; selectedConnection?:{source:number;target:number}; connections?:SceneConnection[]; points:ScenePoint[]; selected?:number; selectedGroup?:number; activity?:number[]; focusGroup?:number|null; resetKey?:number; onPick?:(id:number,group?:number)=>void; onProject?:(x:number,y:number)=>void; large?:boolean; label:string};

export default function Scene({axisLabels,fixedBounds,annotations,labelGroups,points,connections,selectedConnection,onConnectionPick,selected,selectedGroup,activity,focusGroup,resetKey,onPick,onProject,large,label}:Props){
 const host=useRef<HTMLDivElement>(null);const latest=useRef({labelGroups,selected,selectedGroup,activity,connections,selectedConnection,onConnectionPick,onPick,onProject});latest.current={labelGroups,selected,selectedGroup,activity,connections,selectedConnection,onConnectionPick,onPick,onProject};
 const savedView=useRef<{position:THREE.Vector3;target:THREE.Vector3}|null>(null);
 const engine=useRef<any>(null);const marker=useRef<HTMLDivElement>(null);const [error,setError]=useState(false);
 useEffect(()=>{
  if(!host.current||!points.length)return;
  const element=host.current;let renderer:THREE.WebGLRenderer;
  try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});}catch{setError(true);return;}
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x12232b,0);element.insertBefore(renderer.domElement,element.firstChild);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.01,100);
  const bounds=fixedBounds?new THREE.Box3(new THREE.Vector3(...fixedBounds.min as [number,number,number]),new THREE.Vector3(...fixedBounds.max as [number,number,number])):new THREE.Box3().setFromPoints(points.map(p=>new THREE.Vector3(...p.xyz as [number,number,number])));
  const center=bounds.getCenter(new THREE.Vector3()),span=bounds.getSize(new THREE.Vector3());const scale=2.25/Math.max(span.x,span.y,span.z,1e-5);
  const vectors=points.map(p=>new THREE.Vector3(...p.xyz as [number,number,number]).sub(center).multiplyScalar(scale));
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vectors.flatMap(v=>v.toArray()),3));
  const rgb=points.flatMap(p=>p.color??[.20,.78,.70]);geometry.setAttribute('color',new THREE.Float32BufferAttribute(rgb,3));
  geometry.setAttribute('pointSize',new THREE.Float32BufferAttribute(points.map(point=>point.size??1),1));
  const intensities=new Float32Array(points.length).fill(.75);geometry.setAttribute('strength',new THREE.BufferAttribute(intensities,1));
  const material=new THREE.ShaderMaterial({transparent:true,depthWrite:false,vertexColors:true,uniforms:{uSize:{value:points.length<=100?.15:large?.065:.043}},
   vertexShader:`attribute float pointSize; attribute float strength; varying vec3 vColor; varying float vAlpha; uniform float uSize; void main(){vColor=vec3(.22)+color*.78;vAlpha=strength;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_Position=projectionMatrix*mv;gl_PointSize=clamp(pointSize*uSize*(210.0/-mv.z),1.4,11.0);}`,
   fragmentShader:`varying vec3 vColor; varying float vAlpha; void main(){vec2 q=gl_PointCoord-0.5;float d=dot(q,q);if(d>.25)discard;float a=exp(-d*12.0)*vAlpha;gl_FragColor=vec4(vColor,a);}`});
  const cloud=new THREE.Points(geometry,material);scene.add(cloud);

  const groupCenters = new Map<number, THREE.Vector3>();
  const groupCounts = new Map<number, number>();
  points.forEach((point, index) => {
    if (point.group === undefined || point.id < 0) return;
    const centroid = groupCenters.get(point.group) ?? new THREE.Vector3();
    centroid.add(vectors[index]);
    groupCenters.set(point.group, centroid);
    groupCounts.set(point.group, (groupCounts.get(point.group) ?? 0) + 1);
  });
  for (const [group, centroid] of groupCenters) centroid.divideScalar(groupCounts.get(group)!);
  const linkGeometry = new THREE.BufferGeometry();
  const linkMaterial = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: .7, depthWrite: false
  });
  const linkLines = new THREE.LineSegments(linkGeometry, linkMaterial);
  linkLines.frustumCulled = false;
  scene.add(linkLines);
  let previousConnections: SceneConnection[] | undefined;
  let previousConnectionGroup: number | undefined;
  let previousSelectedConnection: typeof selectedConnection;

  const labelRoot=document.createElement('div');labelRoot.className='pca-label-layer';element.appendChild(labelRoot);
  const labels=(annotations??[]).map(annotation=>{
   const node=document.createElement('div');node.className='pca-label';node.dataset.group=String(annotation.group);
   const number=document.createElement('span');number.textContent='G'+String(annotation.group).padStart(2,'0');
   const description=document.createElement('strong');description.textContent=annotation.label;node.appendChild(number);node.appendChild(description);labelRoot.appendChild(node);
   return {node,group:annotation.group,position:new THREE.Vector3(...annotation.xyz as [number,number,number]).sub(center).multiplyScalar(scale)};
  });
  let hoveredGroup:number|undefined;
  const hoverPoint=(event:PointerEvent)=>{
   if(event.buttons||!annotations?.length)return;
   const rect=renderer.domElement.getBoundingClientRect();let distance=64;hoveredGroup=undefined;
   const projected=new THREE.Vector3();
   for(let i=0;i<vectors.length;i++){
    if(points[i].pickable===false)continue;projected.copy(vectors[i]).project(camera);if(projected.z<-1||projected.z>1)continue;
    const d=((projected.x*.5+.5)*rect.width+rect.left-event.clientX)**2+((-projected.y*.5+.5)*rect.height+rect.top-event.clientY)**2;
    if(d<distance){distance=d;hoveredGroup=points[i].group;}
   }
   renderer.domElement.style.cursor=hoveredGroup===undefined?'grab':'pointer';
   renderer.domElement.title=hoveredGroup===undefined?'':annotations.find(annotation=>annotation.group===hoveredGroup)?.label??'';
  };
  renderer.domElement.addEventListener('pointermove',hoverPoint);
  const grid=new THREE.GridHelper(3.6,12,0x35515c,0x253e48);grid.position.y=-1.22;(grid.material as THREE.Material).transparent=true;(grid.material as THREE.Material).opacity=.42;scene.add(grid);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.085;controls.minDistance=.25;controls.maxDistance=15;controls.enablePan=true;
  if(savedView.current){camera.position.copy(savedView.current.position);controls.target.copy(savedView.current.target)}else{camera.position.set(2.6,1.75,3.4);if(fixedBounds)camera.position.normalize().multiplyScalar(1.2/Math.tan(camera.fov*Math.PI/360)*1.25);controls.target.set(0,0,0)}controls.update();
  const resize=()=>{const w=element.clientWidth,h=element.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();};
  const observer=new ResizeObserver(resize);observer.observe(element);resize();

  const connectionAt = (clientX: number, clientY: number) => {
    const rect = renderer.domElement.getBoundingClientRect();
    let found: SceneConnection | null = null;
    let distance = 36;
    for (const connection of latest.current.connections ?? []) {
      const a = groupCenters.get(connection.source)?.clone().project(camera);
      const b = groupCenters.get(connection.target)?.clone().project(camera);
      if (!a || !b || a.z < -1 || a.z > 1 || b.z < -1 || b.z > 1) continue;
      const ax = (a.x * .5 + .5) * rect.width + rect.left;
      const ay = (-a.y * .5 + .5) * rect.height + rect.top;
      const bx = (b.x * .5 + .5) * rect.width + rect.left;
      const by = (-b.y * .5 + .5) * rect.height + rect.top;
      const dx = bx - ax, dy = by - ay;
      const length = dx * dx + dy * dy;
      if (length < 1) continue;
      const t = ((clientX - ax) * dx + (clientY - ay) * dy) / length;
      if (t < .04 || t > .96) continue;
      const candidate = (clientX - ax - t * dx) ** 2 + (clientY - ay - t * dy) ** 2;
      if (candidate < distance) {distance = candidate; found = connection;}
    }
    return {connection: found, distance};
  };
  const hoverConnection = (event: PointerEvent) => {
    if(annotations?.length)return;
    if (event.buttons) return;
    const hit = connectionAt(event.clientX, event.clientY).connection;
    renderer.domElement.style.cursor = hit ? 'pointer' : 'grab';
    renderer.domElement.title = hit ? 'Groups ' + hit.source + ' and ' + hit.target + ' · view shared images' : '';
  };

  let down=[0,0];const onDown=(e:PointerEvent)=>{down=[e.clientX,e.clientY]};
  const choose=(e:PointerEvent)=>{
   if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;
   const rect=renderer.domElement.getBoundingClientRect();let best=-1,distance=150;const v=new THREE.Vector3();
   vectors.forEach((p,i)=>{if(points[i].pickable===false)return;v.copy(p).project(camera);if(v.z<-1||v.z>1)return;const x=(v.x*.5+.5)*rect.width+rect.left,y=(-v.y*.5+.5)*rect.height+rect.top;const d=(x-e.clientX)**2+(y-e.clientY)**2;if(d<distance){best=i;distance=d;}});
   const hit = connectionAt(e.clientX, e.clientY);
   if (latest.current.onConnectionPick && hit.connection && distance > 4 && hit.distance < distance) {
     latest.current.onConnectionPick(hit.connection.source, hit.connection.target);
   } else if(best>=0) latest.current.onPick?.(points[best].id,points[best].group);
  };
  renderer.domElement.addEventListener('pointerdown',onDown);renderer.domElement.addEventListener('pointerup',choose);
  renderer.domElement.addEventListener('pointermove',hoverConnection);
  let frame=0,lastSelected=-1,lastGroup=-1,lastActivity:undefined|number[];let tween:any=null;
  const idMap=new Map(points.map((p,i)=>[p.id,i]));
  const render=()=>{
   frame=requestAnimationFrame(render);const now=performance.now();
   if(tween){const t=Math.min(1,(now-tween.start)/600),s=t*t*(3-2*t);camera.position.lerpVectors(tween.from,tween.to,s);controls.target.lerpVectors(tween.targetFrom,tween.targetTo,s);if(t===1)tween=null;}
   controls.update();const state=latest.current;
   if(state.activity!==lastActivity||state.selectedGroup!==lastGroup){
    for(let i=0;i<points.length;i++){const group=points[i].group;const value=group!==undefined&&state.activity?Math.max(.04,Math.min(1,state.activity[group])):.80;intensities[i]=value*(points[i].strength??1);}
    geometry.attributes.strength.needsUpdate=true;lastActivity=state.activity;lastGroup=state.selectedGroup??-1;
   }

   if (state.connections !== previousConnections || state.selectedGroup !== previousConnectionGroup || state.selectedConnection !== previousSelectedConnection) {
     const positions: number[] = [];
     const colors: number[] = [];
     for (const connection of state.connections ?? []) {
       const from = groupCenters.get(connection.source);
       const to = groupCenters.get(connection.target);
       if (!from || !to) continue;
       positions.push(...from.toArray(), ...to.toArray());
       const selected = connection.source === state.selectedGroup || connection.target === state.selectedGroup;
       const chosen = state.selectedConnection?.source === connection.source &&
         state.selectedConnection?.target === connection.target;
       const color = new THREE.Color(chosen ? '#ffe6a0' : selected ? '#75efc3' : '#72aab9');
       color.multiplyScalar(.3 + .7 * connection.weight);
       colors.push(...color.toArray(), ...color.toArray());
     }
     linkGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
     linkGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
     linkGeometry.setDrawRange(0, positions.length / 3);
     linkLines.visible = positions.length > 0;
     element.dataset.renderedConnections = String(positions.length / 6);
     previousConnections = state.connections;
     previousConnectionGroup = state.selectedGroup;
     previousSelectedConnection = state.selectedConnection;
   }

   const fullLabelGroup=hoveredGroup??state.selectedGroup??state.labelGroups?.[0];
   for(const annotation of labels){
    const show=annotation.group===fullLabelGroup;
    if(!show){annotation.node.style.display='none';continue;}
    const position=annotation.position.clone().project(camera);
    annotation.node.style.display=position.z>-1&&position.z<1?'':'none';
    annotation.node.style.transform=`translate(${(position.x*.5+.5)*element.clientWidth}px,${(-position.y*.5+.5)*element.clientHeight}px)`;
    annotation.node.className='pca-label expanded'+(annotation.group===state.selectedGroup?' chosen':'');
   }
   const which=state.selected===undefined?undefined:idMap.get(state.selected);
   if(which!==undefined&&marker.current){const v=vectors[which].clone().project(camera),rect=element.getBoundingClientRect();const x=(v.x*.5+.5)*rect.width,y=(-v.y*.5+.5)*rect.height;marker.current.style.transform=`translate(${x}px,${y}px)`;marker.current.style.opacity=v.z>-1&&v.z<1?'1':'0';state.onProject?.(x+rect.left,y+rect.top);}
   else if(marker.current)marker.current.style.opacity='0';
   renderer.render(scene,camera);
  };render();
  engine.current={camera,controls,vectors,points,focus:(group:number|null)=>{
   let target=new THREE.Vector3(),radius=1.2;
   if(group!==null){const vs=vectors.filter((_,i)=>points[i].group===group);if(vs.length){const box=new THREE.Box3().setFromPoints(vs);box.getCenter(target);radius=Math.max(.24,box.getSize(new THREE.Vector3()).length()*.5);}}
   const direction=camera.position.clone().sub(controls.target).normalize();const position=target.clone().addScaledVector(direction,radius/Math.tan(camera.fov*Math.PI/360)*1.25);
   tween={start:performance.now(),from:camera.position.clone(),to:position,targetFrom:controls.target.clone(),targetTo:target};
  }};
  return()=>{renderer.domElement.removeEventListener('pointermove',hoverPoint);labelRoot.remove();savedView.current={position:camera.position.clone(),target:controls.target.clone()};cancelAnimationFrame(frame);observer.disconnect();controls.dispose();renderer.domElement.removeEventListener('pointermove',hoverConnection);linkGeometry.dispose();linkMaterial.dispose();geometry.dispose();material.dispose();grid.geometry.dispose();(grid.material as THREE.Material).dispose();renderer.dispose();renderer.domElement.remove();engine.current=null;};
 },[points,large,fixedBounds,annotations]);
 useEffect(()=>{if(focusGroup!==undefined)engine.current?.focus(focusGroup);},[focusGroup,resetKey]);
 return <div className={'scene '+(large?'scene-large':'')} ref={host} data-pca-bounds={fixedBounds?JSON.stringify(fixedBounds):undefined} data-selected-connection={selectedConnection ? selectedConnection.source+'-'+selectedConnection.target : undefined} role="img" aria-label={label}>
  <div className="scene-vignette"/><div className="selected-point" ref={marker}><span/></div>
  <span className="axis axis-one">{axisLabels?.[0]??"PC 1"}</span><span className="axis axis-two">{axisLabels?.[1]??"PC 2"}</span><span className="axis axis-three">{axisLabels?.[2]??"PC 3"}</span>
  {error&&<div className="scene-error">This browser could not start WebGL. Try a browser with hardware acceleration enabled.</div>}
 </div>;
}

