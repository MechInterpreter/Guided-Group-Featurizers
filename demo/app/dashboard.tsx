'use client';

import {useEffect,useMemo,useRef,useState,useCallback} from 'react';

import {Activity,ArrowDownLeft,ArrowUpRight,Check,ChevronLeft,ChevronRight,Expand,Focus,Grid2X2,HelpCircle,Layers3,LockKeyhole,Maximize2,Pause,Play,RotateCcw,Search,Settings2,Upload,X,Zap,Image as ImageIcon,Waypoints} from 'lucide-react';

import Scene,{ScenePoint} from './scene';
import {loadPatchReference,loadPatchFrame,measurePatchCoactivation,sharedPatchRows,sharedPatchIndices,type PatchReference,type PatchFrame} from '../lib/patch-activity';
import {readInferenceResponse} from '../lib/inference-response';
import BenchmarkEvidence from '../components/benchmark-evidence';
import CoactivationPanel from '../components/coactivation-panel';
import ConnectionEvidence from '../components/connection-evidence';
import {coactivationReference, coactiveRowIndices, measureCoactivation, visibleCoactivation} from '../lib/coactivation';

type Group={id:number;variance:number;q99:number;count:number;theme:string;outcome:string;specificity:string;themeMatches:number[]};

type Method={id:string;name:string;color:string;paperGroup:number;paperTitle:string;specificPairs:number;themeMatches:number[];paperImages:number[];groups:Group[];theme:string};

type ImageRecord={id:number;src:string;geometry:{original:number[];resized:number[];box:number[];crop:number}};

type Manifest={imageIds:number[];images:ImageRecord[];methods:Method[];activity:Record<string,number[][]>;overview:Record<string,number[][]>;sharedVariance:number;recordedImages:number;seed:number;playbackIds?:number[];reviewedImageCount?:number;reviewedOverview?:Record<string,number[][]>;activityExtra?:Record<string,Record<string,Record<string,number>>>};

type GroupData={group:number;method:string;variance:number;referenceCount:number;cloudIndices:number[];markedIndices:number[];positions:number[][];norms:number[];topPatches:number[][];q99:number;colorLow:number[];colorHigh:number[];additionalImages?:Record<string,{norms:number[];positions:number[][];topPatches:number[]}>;reviewedViews?:{imageId:number;imageIndex:number;patches:number[];pooled:number}[];reviewedIndices?:number[]};

async function readGroup(response:Response):Promise<GroupData>{
 if(!response.ok)throw new Error('Group data unavailable');
 const bytes=await response.arrayBuffer();
 const header=new Uint8Array(bytes);
 const plain=header[0]===31&&header[1]===139
  ?await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text()
  :new TextDecoder().decode(bytes);
 return JSON.parse(plain) as GroupData;
}
const cache=new Map<string,Promise<GroupData>>();

function loadGroup(method:string,group:number) {
 const key=method+'-'+group;
 if(!cache.has(key)) cache.set(key,fetch('/data/groups/'+key+'.json.gz').then(readGroup));
 return cache.get(key)!;
}

function useGroup(method:string,group:number){

 const [data,setData]=useState<GroupData|null>(null);

 useEffect(()=>{let alive=true;setData(null);loadGroup(method,group).then(d=>{if(alive)setData(d)});return()=>{alive=false}},[method,group]);return data?.method===method&&data.group===group?data:null;

}

function rgb(pc:number[],data:GroupData){return pc.map((v,k)=>.04+.92*Math.max(0,Math.min(1,(v-data.colorLow[k])/Math.max(data.colorHigh[k]-data.colorLow[k],1e-9))))}

function shortTheme(theme:string){return theme.replace(/\.$/,'')}

const ATLAS_STRIDE=1000000;

function pointIndex(imageIndex:number,patch:number){return imageIndex*196+patch}

function frame(data:GroupData|null,imageIndex:number){

 if(!data||imageIndex<0)return null;

 if(imageIndex*196<data.norms.length)return {norms:data.norms.slice(imageIndex*196,(imageIndex+1)*196),positions:data.positions.slice(imageIndex*196,(imageIndex+1)*196),topPatches:data.topPatches[imageIndex]};

 return data.additionalImages?.[imageIndex]??null;

}

function recordedPoint(data:GroupData,id:number){const row=frame(data,Math.floor(id/196));return row?{xyz:row.positions[id%196],norm:row.norms[id%196]}:null;}

function GroupCloud({method,group,imageIndex,patch,selected,onPick,onSelect,onProject,large,live,reviewedOnly}:any){

 const data=useGroup(method.id,group);const current=live?.[method.id];

 const points=useMemo<ScenePoint[]>(()=>{

  if(!data)return [];

  const points=new Map<number,ScenePoint>();

  if(!reviewedOnly)for(const id of data.cloudIndices){const point=recordedPoint(data,id);if(point)points.set(id,{id,xyz:point.xyz,color:reviewedOnly?[.38,.52,.56]:rgb(point.xyz,data),strength:reviewedOnly?.10:1,pickable:!reviewedOnly});}

  const foreground=reviewedOnly?[...(data.reviewedIndices??data.markedIndices)]:[];

  const row=frame(data,imageIndex);

  if(!reviewedOnly&&row)for(let p=0;p<196;p++)if(row.norms[p]>0)foreground.push(imageIndex*196+p);

  if(selected&&row&&row.norms[patch]>0&&(!reviewedOnly||foreground.includes(imageIndex*196+patch)))foreground.push(imageIndex*196+patch);

  for(const id of foreground){const point=recordedPoint(data,id);if(point)points.set(id,{id,xyz:point.xyz,color:rgb(point.xyz,data),strength:1,pickable:true});}

  if(current)for(let p=0;p<196;p++)if(current.norms[group][p]>0)points.set(-p-1,{id:-p-1,xyz:current.positions[group][p],color:[.97,.79,.40],strength:1,pickable:true});

  return [...points.values()];

 },[data,imageIndex,current,group,patch,selected,reviewedOnly]);

 const selectedPoint=current?-patch-1:pointIndex(imageIndex,patch);

 return <div className={'cloud-card '+(selected?'cloud-card-selected':'')} style={{'--method':method.color} as any}>

  <div className="cloud-card-title"><button className="method-title" onClick={onSelect}><i style={{background:method.color}}/>{method.name}</button><span>GROUP {String(group).padStart(2,'0')}</span></div>

  {data?<Scene points={points} selected={selected?selectedPoint:undefined} onPick={(i:number)=>onPick(i)} onProject={selected?onProject:undefined} large={large} label={method.name+' group '+group+' measured contribution PCA'}/>:<div className="loading-cloud"><span className="spinner"/>Loading measured responses</div>}

  <div className="cloud-card-bottom"><span>{data?`${(data.variance*100).toFixed(1)}% 3-PC variance`:'Loading projection'}</span>{group===method.paperGroup?<span className="pair-score">{method.specificPairs}/3 specific pairs</span>:<span>{reviewedOnly?`${data?.reviewedIndices?.length??0} reviewed regions`:data?.referenceCount.toLocaleString()+' reference patches'}</span>}</div>

 </div>

}

function Photo({image,data,imageIndex,patch,onPatch,heat,grid,crop,svgRef,live,reviewedPatches}:any){

 const g=image.geometry;const ow=g.original[0],oh=g.original[1],sx=ow/g.resized[0],sy=oh/g.resized[1];const left=g.box[0]*sx,top=g.box[1]*sy;

 const values=live?.norms??frame(data,imageIndex)?.norms??[];

 const pcs=live?.positions??frame(data,imageIndex)?.positions??[];

 const selected={x:left+(patch%14)*16*sx,y:top+Math.floor(patch/14)*16*sy,w:16*sx,h:16*sy};

 const viewBox=crop?`${left} ${top} ${224*sx} ${224*sy}`:`0 0 ${ow} ${oh}`;

 return <svg ref={svgRef} className={'source-photo '+(crop?'square':'')} viewBox={viewBox} role="img" aria-label={'Input image '+image.id+' with group activation patches'}>

  <image href={image.src} x="0" y="0" width={ow} height={oh}/>

  {!crop&&<rect x={left} y={top} width={224*sx} height={224*sy} fill="none" stroke="white" strokeWidth={1.2*sx} strokeDasharray={`${4*sx} ${3*sx}`} opacity=".75"/>}

  {Array.from({length:196},(_,p)=>{

   const marked=!reviewedPatches||reviewedPatches.includes(p);const value=values[p]??0;const alpha=heat&&marked?Math.min(.72,.61*Math.min(1,value/(data?.q99??1))**.65):0;

   const color=data&&pcs[p]?rgb(pcs[p],data):[.1,.8,.65];

   return <rect key={p} x={left+(p%14)*16*sx} y={top+Math.floor(p/14)*16*sy} width={16*sx} height={16*sy} fill={`rgb(${color.map(v=>v*255).join(',')})`} fillOpacity={alpha} stroke={grid?'rgba(255,255,255,.25)':'none'} strokeWidth={.4*sx} className="patch-hit" onClick={marked?()=>onPatch(p):undefined} pointerEvents={marked?'auto':'none'}><title>Patch {p} · response {value.toFixed(3)}</title></rect>

  })}

  {reviewedPatches?.filter((p:number)=>p!==patch).map((p:number)=><rect key={'review-'+p} x={left+(p%14)*16*sx} y={top+Math.floor(p/14)*16*sy} width={16*sx} height={16*sy} fill="none" stroke="#29efb7" strokeWidth={1.1*sx} pointerEvents="none"/>)}
  <rect {...selected} fill="none" stroke="#fff" strokeWidth={2*sx} pointerEvents="none"/><rect x={selected.x+sx} y={selected.y+sy} width={selected.w-2*sx} height={selected.h-2*sy} fill="none" stroke="#0ce1ad" strokeWidth={1.1*sx} pointerEvents="none"/>

 </svg>

}

function PatchThumb({image,patch,selected,onClick}:any){

 const g=image.geometry,sx=g.original[0]/g.resized[0],sy=g.original[1]/g.resized[1];const x=(g.box[0]+(patch%14)*16)*sx,y=(g.box[1]+Math.floor(patch/14)*16)*sy;

 const left=(g.box[0]+Math.max(0,Math.min(128,(patch%14+.5)*16-48)))*sx,top=(g.box[1]+Math.max(0,Math.min(128,(Math.floor(patch/14)+.5)*16-48)))*sy;

 return <button className={'patch-thumb '+(selected?'chosen':'')} onClick={onClick} aria-label={'Select patch '+patch}><svg viewBox={`${left} ${top} ${96*sx} ${96*sy}`}><image href={image.src} width={g.original[0]} height={g.original[1]}/><rect x={x} y={y} width={16*sx} height={16*sy} stroke="#29efb7" strokeWidth={1.5*sx} fill="none"/></svg><span>{patch}</span></button>

}

export default function Dashboard(){

 const [benchmarkOpen,setBenchmarkOpen]=useState(false);
 const [connectionSelection,setConnectionSelection]=useState<{method:string;source:number;target:number}|null>(null);
 const [samePatch,setSamePatch]=useState(false);
 const [patchReference,setPatchReference]=useState<PatchReference|null>(null),[patchFrame,setPatchFrame]=useState<PatchFrame>({}),[patchError,setPatchError]=useState('');
 const [showConnections,setShowConnections]=useState(true),[currentLinks,setCurrentLinks]=useState(true),[overlap,setOverlap]=useState(.15);

 const [manifest,setManifest]=useState<Manifest|null>(null),[fatal,setFatal]=useState('');

 const [mode,setMode]=useState<'compare'|'monitor'>('compare'),[methodId,setMethodId]=useState('ggf');

 const [groups,setGroups]=useState<Record<string,number>>({ggf:32,slics:56,supervised_vanilla:30,vanilla:23});

 const [imageId,setImageId]=useState(502857),[patch,setPatch]=useState(74),[heat,setHeat]=useState(true),[grid,setGrid]=useState(false),[crop,setCrop]=useState(false);

 const [reviewedOnly,setReviewedOnly]=useState(true);const [keepImage,setKeepImage]=useState(false);const [query,setQuery]=useState(''),[about,setAbout]=useState(false),[gallery,setGallery]=useState(false),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(1),[threshold,setThreshold]=useState(.45);

 const [focus,setFocus]=useState<number|null>(null),[detail,setDetail]=useState(false),[resetKey,setResetKey]=useState(0),[backend,setBackend]=useState<any>({ready:false}),[uploading,setUploading]=useState(false),[uploadError,setUploadError]=useState(''),[live,setLive]=useState<any>(null),[events,setEvents]=useState<any[]>([]);

 const [anchor,setAnchor]=useState({x:0,y:0});const workspace=useRef<HTMLDivElement>(null),photoSvg=useRef<SVGSVGElement>(null),uploadInput=useRef<HTMLInputElement>(null);

 const navigation=useRef(0);
 const uploadRequest=useRef(0);
 useEffect(()=>{const url=live?.image?.src;return()=>{if(url?.startsWith('blob:'))URL.revokeObjectURL(url)}},[live]);
 const pointSelection=useRef<{method:string;group:number;imageId:number;patch:number}|null>(null);

 const group=groups[methodId],data=useGroup(methodId,group);

 useEffect(()=>{fetch('/data/manifest.json').then(r=>r.json() as Promise<Manifest>).then(setManifest).catch(()=>setFatal('The recorded dataset could not be loaded. Please reload this page.'));const status=()=>fetch('/api/status').then(r=>r.json()).then(setBackend).catch(()=>setBackend({ready:false}));status();const timer=setInterval(status,30000);return()=>clearInterval(timer)},[]);

 const imageIndex=manifest?.imageIds.indexOf(imageId)??0;

 const method=manifest?.methods.find(m=>m.id===methodId),groupInfo=method?.groups[group];

 const image=live?.image??manifest?.images[imageIndex];

 const scores=useMemo(()=>{if(!manifest||!method)return [];return method.groups.map(g=>{const raw=live?live.methods[methodId].scores[g.id]:(manifest.activity[methodId][imageIndex]?.[g.id]??manifest.activityExtra?.[methodId]?.[imageIndex]?.[g.id]??null);return raw===null?{group:g,raw:0,relative:0,available:false}:{group:g,raw,relative:raw/Math.max(g.q99,1e-8),available:true};})},[manifest,method,methodId,imageIndex,live]);

 const activeCount=scores.filter(v=>v.relative>=threshold).length;const availableCount=scores.filter(v=>v.available!==false).length;

 const strength=useMemo(()=>scores.map(s=>s.relative>=threshold?Math.min(1,s.relative):.04),[scores,threshold]);


 const reference = useMemo(() => manifest ? coactivationReference(manifest) : null, [manifest]);
 useEffect(()=>{
   let active=true;setPatchReference(null);setPatchError('');
   if(samePatch) loadPatchReference(methodId).then(d=>{if(active)setPatchReference(d)}).catch(e=>{if(active)setPatchError(e.message)});
   return()=>{active=false};
 },[samePatch,methodId]);
 useEffect(()=>{
   let active=true;setPatchFrame({});
   if(samePatch&&currentLinks&&!live) loadPatchFrame(methodId,imageId).then(d=>{if(active)setPatchFrame(d)}).catch(e=>{if(active)setPatchError(e.message)});
   return()=>{active=false};
 },[samePatch,currentLinks,methodId,imageId,live]);
 const currentPatchFrame:PatchFrame=live?.methods[methodId]?.norms??patchFrame;
 const patchCoverage=Object.keys(currentPatchFrame).length;
 const pairLinks = useMemo(() => {
   if (!reference || !method) return [];
   if(samePatch) return patchReference ? measurePatchCoactivation(patchReference,reference.rows[methodId],method.groups.map(g=>g.q99),threshold):[];
   return measureCoactivation(reference.rows[methodId], method.groups.map(g => g.q99), threshold);
 }, [reference, method, methodId, threshold, samePatch, patchReference]);
 const eligibleLinks = useMemo(() => {
   const links=visibleCoactivation(pairLinks,overlap,currentLinks?scores.map(s=>s.available&&s.relative>=threshold):null);
   return samePatch&&currentLinks&&method ? links.filter(link=>sharedPatchIndices(currentPatchFrame[link.source],currentPatchFrame[link.target],method.groups[link.source].q99,method.groups[link.target].q99,threshold).length>0):links;
 }, [pairLinks,overlap,currentLinks,scores,threshold,samePatch,currentPatchFrame,method]);
 const displayedLinks = useMemo(() => {
   if (!showConnections) return [];
   const selected = eligibleLinks.filter(link => link.source === group || link.target === group);
   const others = eligibleLinks.filter(link => link.source !== group && link.target !== group);
   return [...selected, ...others].slice(0, 60);
 }, [eligibleLinks, group, showConnections]);
 const selectedPair = connectionSelection?.method === methodId
   ? pairLinks.find(link => link.source === connectionSelection.source && link.target === connectionSelection.target)
   : undefined;
 const sharedIndices = useMemo(() => {
   if (!selectedPair || !reference || !method) return [];
   if(samePatch&&patchReference) return sharedPatchRows(patchReference,reference.rows[methodId],method.groups.map(g=>g.q99),threshold,selectedPair.source,selectedPair.target);
   return coactiveRowIndices(reference.rows[methodId], method.groups.map(g => g.q99),
     threshold, selectedPair.source, selectedPair.target);
 }, [selectedPair, reference, method, methodId, threshold, samePatch, patchReference]);
 const openConnection = (source: number, target: number) => {
   setPlaying(false);
   setConnectionSelection({method:methodId,source:Math.min(source,target),target:Math.max(source,target)});
 };
 const inspectConnectionEvidence = (id:number, selectedImage:number, selectedPatch:number) => {
   navigation.current++;
   pointSelection.current={method:methodId,group:id,imageId:selectedImage,patch:selectedPatch};
   setGroups(previous => ({...previous,[methodId]:id}));
   setLive(null);
   setImageId(selectedImage);
   setPatch(selectedPatch);
   setFocus(id);
   setReviewedOnly(false);
   setDetail(true);
   setPlaying(false);
   setConnectionSelection(null);
 };
 const inspectCoactiveGroup = (id: number) => {
   navigation.current++;
   pointSelection.current = null;
   setGroups(previous => ({...previous, [methodId]: id}));
   setFocus(id);
   setPlaying(false);
 };

 const atlasPoints=useMemo<ScenePoint[]>(()=>{

  if(!manifest)return [];

  const points=(reviewedOnly&&manifest.reviewedOverview?manifest.reviewedOverview[methodId]:manifest.overview[methodId]).map(row=>{const [g,i,x,y,z]=row;const hue=(g*.61803398875)%1;const color=[.18+.45*Math.sin(hue*3.14)**2,.55+.35*Math.cos(hue*2)**2,.58+.32*Math.sin(hue*4)**2];return {id:g*ATLAS_STRIDE+i,group:g,xyz:[x,y,z],color}});

  if(live?.methods[methodId]?.overview)for(let g=0;g<75;g++)for(let p=0;p<196;p++)if(live.methods[methodId].norms[g][p]>0)points.push({id:-(g*196+p+1),group:g,xyz:live.methods[methodId].overview[g][p],color:[.97,.79,.40]});

  return points;

 },[manifest,methodId,live,reviewedOnly]);

 const selectedResponse=live?.methods[methodId]?.norms[group][patch]??frame(data,imageIndex)?.norms[patch]??0;

 const topPatches=live?[...Array(196).keys()].sort((a,b)=>live.methods[methodId].norms[group][b]-live.methods[methodId].norms[group][a]).slice(0,4):frame(data,imageIndex)?.topPatches??[];

 const project=useCallback((x:number,y:number)=>setAnchor(old=>Math.abs(old.x-x)+Math.abs(old.y-y)>.5?{x,y}:old),[]);

 const chooseImage=useCallback((id:number)=>{navigation.current++;pointSelection.current=null;setLive(null);setImageId(id);setPlaying(false)},[]);

 const chooseGroup=async(id:number)=>{const request=++navigation.current;pointSelection.current=null;setGroups(prev=>({...prev,[methodId]:id}));setFocus(id);if(manifest&&!keepImage&&!live&&reviewedOnly){const target:GroupData=await fetch(`/data/groups/${methodId}-${id}.json.gz`).then(readGroup);if(request!==navigation.current)return;if(target.reviewedViews?.length){chooseImage(target.reviewedViews[0].imageId);return;}}if(manifest&&!keepImage&&!live){const values=manifest.activity[methodId].map(row=>row[id]);const best=values.indexOf(Math.max(...values));setImageId(manifest.imageIds[best]);setPlaying(false)}};

 const chooseMethod=async(id:string)=>{const request=++navigation.current;pointSelection.current=null;setMethodId(id);setFocus(null);if(manifest&&!keepImage&&!live&&reviewedOnly){const target:GroupData=await fetch(`/data/groups/${id}-${groups[id]}.json.gz`).then(readGroup);if(request!==navigation.current)return;if(target.reviewedViews?.length){chooseImage(target.reviewedViews[0].imageId);return;}}if(manifest&&!keepImage&&!live){const values=manifest.activity[id].map(row=>row[groups[id]]);const best=values.indexOf(Math.max(...values));setImageId(manifest.imageIds[best]);setPlaying(false)}};

 useEffect(()=>{

  if(!data||live)return;

  const selected=pointSelection.current;

  if(selected&&selected.method===methodId&&selected.group===group&&selected.imageId===imageId){setPatch(selected.patch);return;}

  const row=frame(data,imageIndex);if(row)setPatch(row.topPatches[0]);

 },[data,imageIndex,imageId,methodId,group,live]);

 useEffect(()=>{if(!playing||!manifest||live)return;const sequence=reviewedOnly&&data?.reviewedViews?.length?data.reviewedViews.map(v=>v.imageId):(manifest.playbackIds??manifest.imageIds);const id=setInterval(()=>{pointSelection.current=null;setImageId(previous=>sequence[(sequence.indexOf(previous)+1)%sequence.length])},1600/speed);return()=>clearInterval(id)},[playing,manifest,speed,live,reviewedOnly,data]);

 useEffect(()=>{if(mode!=='monitor'||!scores.length)return;const fired=[...scores].filter(v=>v.relative>=threshold).sort((a,b)=>b.relative-a.relative).slice(0,3);setEvents(previous=>[...fired.map(v=>({key:Date.now()+'-'+v.group.id,group:v.group.id,score:v.raw,imageId,methodId,uploaded:!!live,source:live?'Upload':`Image ${imageId}`})),...previous].slice(0,12));},[imageId,live,methodId,mode,threshold]);

 async function upload(file:File){

  if(file.size>8*1024*1024){setUploadError('Choose an image smaller than 8 MB.');return;}

  const request=++uploadRequest.current;const navigationAtStart=navigation.current;
  setPlaying(false);setUploading(true);setUploadError('');

  try{const form=new FormData();form.append('image',file);const response=await fetch('/api/infer',{method:'POST',body:form});const result=await readInferenceResponse(response);if(request!==uploadRequest.current||navigationAtStart!==navigation.current)return;const url=URL.createObjectURL(file);setEvents(previous=>previous.filter(event=>!event.uploaded));setLive({...result,image:{id:file.name,src:url,geometry:result.geometry}});setPatch(result.methods[methodId].norms[group].indexOf(Math.max(...result.methods[methodId].norms[group])));setMode('monitor');}

  catch(e:any){if(request===uploadRequest.current)setUploadError(e.message)}finally{if(request===uploadRequest.current)setUploading(false)}

 }

 const pick=(id:number,fromMethod:string,g?:number)=>{

  navigation.current++;const selectedGroup=g??groups[fromMethod];

  setMethodId(fromMethod);setPlaying(false);

  if(g!==undefined){setGroups(prev=>({...prev,[fromMethod]:g}));setFocus(g);if(id>=0)id=id%ATLAS_STRIDE;}

  if(id<0){pointSelection.current=null;setPatch(g===undefined?-id-1:(-id-1)%196);return;}

  const selectedImage=manifest!.imageIds[Math.floor(id/196)],selectedPatch=id%196;

  pointSelection.current={method:fromMethod,group:selectedGroup,imageId:selectedImage,patch:selectedPatch};

  setLive(null);setImageId(selectedImage);setPatch(selectedPatch);

 };

 if(!manifest||!method||!image)return <main className="initial-load"><div className="brand-symbol"><Waypoints size={30}/></div><h1>GGF Concept Atlas</h1><p>{fatal||'Opening the measured representation space…'}</p><span className="spinner"/></main>;

 const filtered=method.groups.filter(g=>`${g.id} ${g.theme}`.toLowerCase().includes(query.toLowerCase()));

 const source=live?.methods[methodId];

 const choosePatch=(p:number)=>{navigation.current++;pointSelection.current=live?null:{method:methodId,group,imageId,patch:p};setPatch(p);setPlaying(false)};

 let line='';if(mode==='compare'&&workspace.current&&photoSvg.current&&data){const svg=photoSvg.current;const geom=image.geometry;const point=svg.createSVGPoint();point.x=(geom.box[0]+(patch%14+.5)*16)*geom.original[0]/geom.resized[0];point.y=(geom.box[1]+(Math.floor(patch/14)+.5)*16)*geom.original[1]/geom.resized[1];const matrix=svg.getScreenCTM();if(matrix){const start=point.matrixTransform(matrix),rect=workspace.current.getBoundingClientRect();const x1=start.x-rect.left,y1=start.y-rect.top,x2=anchor.x-rect.left,y2=anchor.y-rect.top;const card=workspace.current.querySelector('.cloud-card-selected')?.getBoundingClientRect();const cards=workspace.current.querySelectorAll('.cloud-card');if(card&&cards.length===4&&card.left<cards[1].getBoundingClientRect().left){const rowGap=(cards[0].getBoundingClientRect().bottom+cards[2].getBoundingClientRect().top)/2-rect.top;const outer=workspace.current.querySelector('.inspector-panel')!.getBoundingClientRect().left-rect.left-10;const inner=card.right-rect.left+6;line=`M ${x1} ${y1} L ${outer} ${y1} L ${outer} ${rowGap} L ${inner} ${rowGap} L ${inner} ${y2} L ${x2} ${y2}`;}else{line=`M ${x1} ${y1} C ${x1-70} ${y1}, ${x2+70} ${y2}, ${x2} ${y2}`;}}}

 return <div className="app-shell">

  <aside className="sidebar"><a className="brand" href="/"><span className="brand-symbol"><Waypoints size={23}/></span><span>GGF<span className="brand-sub">CONCEPT ATLAS</span></span></a>

   <div className="sidebar-caption">WORKSPACE</div>

   <button className={'nav-item '+(mode==='compare'?'nav-selected':'')} onClick={()=>{setMode('compare');setDetail(false)}}><Grid2X2 size={18}/>Compare methods<span>4</span></button>

   <button className={'nav-item '+(mode==='monitor'?'nav-selected':'')} onClick={()=>{setMode('monitor');setDetail(false)}}><Activity size={18}/>Activation monitor</button>

   <button className="nav-item" onClick={()=>setBenchmarkOpen(true)}><Check size={18}/>Benchmark evidence</button>
   <div className="sidebar-rule"/><div className="sidebar-caption">EXPLORE GROUPS <span>75</span></div>

   <label className="group-search"><Search size={15}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Find a concept or group…" aria-label="Search groups"/></label>

   <div className="group-list">{filtered.map(g=><button key={g.id} className={'group-item '+(g.id===group?'group-chosen':'')} onClick={()=>chooseGroup(g.id)}><span className="group-number">{String(g.id).padStart(2,'0')}</span><span><strong>{shortTheme(g.theme)}</strong><small>{g.outcome==='clear'?'Clear theme':g.outcome==='partial'?'Partial theme':'Mixed responses'}</small></span>{g.id===group&&<i/>}</button>)}</div>

   <button className="sidebar-about" onClick={()=>setAbout(true)}><HelpCircle size={17}/>About the research<ArrowUpRight size={14}/></button>

   <div className="sidebar-footer"><span className="status-dot"/>Frozen models · seed 21</div>

  </aside>

  <div className="main-shell">

   <header className="topbar"><div className="breadcrumb">Research <span>/</span> Visual representations</div><div className="topbar-actions"><span className="private-pill"><LockKeyhole size={12}/>Private workspace</span><button className="quiet-btn" onClick={()=>setAbout(true)}>How to explore <ArrowUpRight size={14}/></button></div></header>

   <main>

    <div className="intro"><div><div className="eyebrow"><span/>GUIDED GROUP FEATURIZERS</div><h1>{mode==='compare'?'One group. What does it mean?':'Watch the representation respond.'}</h1><p>{mode==='compare'?'Explore the same four methods from the paper. Every point leads back to an image patch.':'Follow measured group activity across images, or send a new image through the frozen models.'}</p></div><div className="research-spec"><span>DINOv3 <b>ViT-B/16</b></span><span><b>75</b> groups / method</span><span><b>12</b> directions / group</span></div></div>

    <div className="workspace-toolbar"><div className="method-switch">{manifest.methods.map(m=><button key={m.id} className={methodId===m.id?'active':''} onClick={()=>chooseMethod(m.id)}><i style={{background:m.color}}/>{m.name}</button>)}</div><div className="toolbar-end"><label className="response-view"><input type="checkbox" checked={reviewedOnly} onChange={e=>setReviewedOnly(e.target.checked)}/>Reviewed high-response regions</label><button className="quiet-btn" onClick={()=>{setGroups(Object.fromEntries(manifest.methods.map(m=>[m.id,m.paperGroup])));setMethodId('ggf');chooseImage(502857);setPatch(74);setResetKey(k=>k+1)}}><RotateCcw size={14}/>Figure 2 examples</button></div></div>

    <div className="workspace" ref={workspace}>

     <section className="geometry-panel">

      <div className="panel-heading"><div><span className="section-number">01</span><h2>{mode==='compare'?'Group geometry':detail?'Inside group '+group:'Activation atlas'}</h2>{mode==='monitor'&&<span className="data-badge">{live?`LIVE · ${live.seconds.toFixed(2)} s · ${live.device?.toUpperCase()??'MODEL'}`:'RECORDED PLAYBACK'}</span>}</div><div className="panel-tools">{mode==='monitor'&&<><button className="icon-btn" title="Fit all groups" onClick={()=>{setFocus(null);setResetKey(k=>k+1);setDetail(false)}}><Expand size={15}/></button><button className={'quiet-btn '+(detail?'is-on':'')} onClick={()=>setDetail(v=>!v)}><Focus size={14}/>{detail?'Back to atlas':'Inspect group'}</button></>}<span className="gesture-label">Drag to rotate · scroll to zoom</span></div></div>

      {mode==='compare'?<div className="comparison-grid">{manifest.methods.map(m=><GroupCloud key={m.id+'-'+groups[m.id]} method={m} group={groups[m.id]} imageIndex={imageIndex} patch={patch} selected={methodId===m.id} onSelect={()=>chooseMethod(m.id)} onPick={(id:number)=>pick(id,m.id)} onProject={project} live={live?.methods} reviewedOnly={reviewedOnly}/>)}</div>:<div className="atlas-wrap">

       {detail?<GroupCloud key={'detail-'+methodId+'-'+group} method={method} group={group} imageIndex={imageIndex} patch={patch} selected large onPick={(id:number)=>pick(id,methodId)} onProject={project} live={live?.methods} reviewedOnly={reviewedOnly}/>:<Scene points={atlasPoints} connections={displayedLinks}
         selectedConnection={connectionSelection?.method===methodId?connectionSelection:undefined}
         onConnectionPick={openConnection} selectedGroup={group} activity={strength} focusGroup={focus} resetKey={resetKey} large onPick={(id:number,g?:number)=>pick(id,methodId,g)} label="Shared measured contribution-PCA overview of all 75 native groups"/>}

       {!detail&&<><div className="atlas-status"><span className="pulse-dot"/>{activeCount}<small>{availableCount<75?`of ${availableCount} available groups above cutoff`:'groups above cutoff'}</small></div><div className="atlas-key"><span>Lower response</span><i/><span>Higher response</span></div><div className="atlas-hint"><Focus size={14}/>Select a group to focus its response cloud</div></>}

      </div>}


      {mode==='monitor' && !detail && <CoactivationPanel
        samePatch={samePatch} onSamePatch={value=>{setSamePatch(value);setConnectionSelection(null)}}
        patchLoading={samePatch&&!patchReference&&!patchError} patchError={patchError}
        patchCoverage={patchCoverage} onReferenceImage={()=>{setReviewedOnly(false);chooseImage(reference!.imageIds[0])}}
        enabled={showConnections} onEnabled={setShowConnections}
        currentOnly={currentLinks} onCurrentOnly={setCurrentLinks}
        minimum={overlap} onMinimum={setOverlap}
        samples={reference?.imageIds.length ?? 0}
        candidates={eligibleLinks} rendered={displayedLinks.length}
        group={group} groups={method.groups} onGroup={inspectCoactiveGroup}
        connections={displayedLinks} onConnection={openConnection}/>}

      <div className="geometry-footer"><button onClick={()=>setBenchmarkOpen(true)}>Benchmark evidence<ArrowUpRight size={13}/></button><span><i className="tiny-point"/>{mode==='compare'||detail?(reviewedOnly?'24 original reviewed regions per group':'Measured native contribution PCA'):(reviewedOnly?'Original reviewed HIGH regions in one shared PCA frame':'One shared contribution-PCA frame for all groups')}</span><button onClick={()=>setAbout(true)}><HelpCircle size={13}/>{mode==='compare'||detail?'Groups have separate PCA bases':`${(manifest.sharedVariance*100).toFixed(1)}% reference variance shown`}</button></div>

      {mode==='monitor'&&<div className="monitor-controls"><div><Settings2 size={16}/><label>Display cutoff <strong>{Math.round(threshold*100)}%</strong></label><input aria-label="Activation display cutoff" type="range" min="5" max="150" value={threshold*100} onChange={e=>setThreshold(+e.target.value/100)}/><small>relative to each group’s reference scale</small></div><div className="activity-bars">{[...scores].filter(v=>v.available!==false).sort((a,b)=>b.relative-a.relative).slice(0,5).map(v=><button key={v.group.id} onClick={()=>chooseGroup(v.group.id)}><span>G{String(v.group.id).padStart(2,'0')}</span><i><b style={{width:Math.min(v.relative,1)*100+'%'}}/></i><strong>{v.raw.toFixed(1)}</strong></button>)}</div></div>}

     </section>

     <aside className="inspector-panel"><div className="panel-heading"><div><span className="section-number">02</span><h2>Response inspector</h2></div><span className="linked-label"><span/>Linked</span></div>

      <div className="inspector-body"><div className="group-eyebrow"><i style={{background:method.color}}/>{method.name}<span>GROUP {String(group).padStart(2,'0')}</span></div><h3 className="selected-source-title">{live?'Uploaded image':`Image ${imageId}`} · patch {patch}</h3><p className="theme-subtitle"><span className="theme-label">{reviewedOnly?'Reviewed group theme':'Review theme · reference exploration'}</span>{group===method.paperGroup?method.paperTitle:shortTheme(groupInfo?.theme??'Group '+group)}</p>

       <div className="image-follow-control"><label><input type="checkbox" checked={keepImage} onChange={e=>setKeepImage(e.target.checked)}/>Keep image fixed across groups and methods</label></div><div className="image-tabs"><button className={!crop?'active':''} onClick={()=>setCrop(false)}>Full image</button><button className={crop?'active':''} onClick={()=>setCrop(true)}>Model view</button><div/><button className={heat?'active tool-toggle':'tool-toggle'} title="Toggle response overlay" aria-label="Toggle response overlay" onClick={()=>setHeat(v=>!v)}><Layers3 size={15}/></button><button className={grid?'active tool-toggle':'tool-toggle'} title="Toggle patch grid" aria-label="Toggle patch grid" onClick={()=>setGrid(v=>!v)}><Grid2X2 size={14}/></button></div>

       <div className="photo-stage"><Photo image={image} data={data} imageIndex={imageIndex} patch={patch} onPatch={choosePatch} heat={heat} grid={grid} crop={crop} svgRef={photoSvg} reviewedPatches={reviewedOnly&&!live?data?.reviewedViews?.find(v=>v.imageIndex===imageIndex)?.patches:undefined} live={source?{norms:source.norms[group],positions:source.positions[group]}:null}/><span className="photo-caption">{live?'Your image':`COCO ${imageId}`}</span></div>{data&&!source&&!frame(data,imageIndex)&&<p className="missing-response">This image is not in this group’s exported responses. Choose one of its reviewed examples below.</p>}

       <div className="inspector-stats"><div><span>SELECTED PATCH</span><strong>{patch}<small>/ 195</small></strong></div><div><span>NATIVE RESPONSE</span><strong>{data||source?selectedResponse.toFixed(2):'…'}</strong></div><div><span>3-PC VARIANCE</span><strong>{data?(data.variance*100).toFixed(1):'…'}<small>%</small></strong></div></div>

       <div className="patches-title"><h4>{reviewedOnly&&!live?'Reviewed response patches':'Strongest response patches'}</h4><span>Click to locate</span></div><div className="patch-strip">{topPatches.map((p:number)=><PatchThumb key={p} image={image} patch={p} selected={p===patch} onClick={()=>choosePatch(p)}/>)}</div>

       <div className="review-note"><Check size={15}/><div><strong>{groupInfo?.outcome==='clear'?'Clear reviewed theme':groupInfo?.outcome==='partial'?'Partial reviewed theme':'No clear reviewed theme'}</strong><p>{groupInfo?.themeMatches?.join('/6, ')}/6 views matched across three reviews.</p></div></div>

      </div>

     </aside>

     {mode==='compare'&&line&&<svg className="link-layer"><path d={line}/><circle cx={anchor.x-(workspace.current?.getBoundingClientRect().left??0)} cy={anchor.y-(workspace.current?.getBoundingClientRect().top??0)} r="4"/></svg>}

    </div>

    <section className="image-tray"><div className="tray-heading"><div><span className="section-number">03</span><h2>{live?'New-image inference':'Explore the recorded images'}</h2><span className="muted">{reviewedOnly?'6 reviewed images':manifest.recordedImages+' reference images'}</span></div><button className="quiet-btn" onClick={()=>setGallery(true)}>{reviewedOnly?'Browse reviewed images':'Browse reference images'}<ArrowUpRight size={14}/></button></div><div className="tray-content"><button className={'play-btn '+(playing?'playing':'')} onClick={()=>{setLive(null);setMode('monitor');setPlaying(v=>!v)}} aria-label={playing?'Pause recorded playback':'Play recorded activation monitoring'}>{playing?<Pause size={20}/>:<Play size={20}/>}</button><div className="tray-thumbnails">{(reviewedOnly&&data?.reviewedViews?.length?data.reviewedViews.map(v=>v.imageId):method.paperImages).map((id,i)=>{const im=manifest.images.find(x=>x.id===id)!;return <button key={id} className={id===imageId&&!live?'selected':''} onClick={()=>chooseImage(id)}><img src={im.src} alt={`Reviewed example ${i+1}`} loading="lazy"/><span>{String(i+1).padStart(2,'0')}</span></button>})}</div><div className="playback-info"><span>{playing?'Playing recorded responses':live?'Live image response':'Recorded image responses'}</span><strong>{live?'New image':(manifest.playbackIds??manifest.imageIds).includes(imageId)?`${(manifest.playbackIds??manifest.imageIds).indexOf(imageId)+1} / ${manifest.recordedImages}`:'Reviewed example'}</strong><select aria-label="Playback speed" value={speed} onChange={e=>setSpeed(+e.target.value)}><option value="1">1× speed</option><option value="2">2× speed</option><option value="4">4× speed</option></select></div><button className="upload-btn" onClick={()=>uploadInput.current?.click()} disabled={uploading}><Upload size={18}/><span>{uploading?'Running frozen models…':'Try your own image'}<small>{backend.ready?'Model service online':'Live image service is currently offline'}</small></span><span className={'backend-dot '+(backend.ready?'online':'')}/></button><input ref={uploadInput} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={e=>{if(e.target.files?.[0])upload(e.target.files[0]);e.target.value=''}}/></div>{uploadError&&<div className="upload-error" role="alert">{uploadError}</div>}

    </section>

    {mode==='monitor'&&<section className="events-panel"><div><Zap size={15}/><h4>Activation events</h4><span>Measured responses above the display cutoff</span><button onClick={()=>setEvents([])}>Clear</button></div><div className="event-list">{events.length?events.slice(0,6).map((event,i)=><button key={event.key+i} onClick={()=>{pointSelection.current=null;setMethodId(event.methodId);setGroups(prev=>({...prev,[event.methodId]:event.group}));setFocus(event.group);setPlaying(false);if(!event.uploaded){setLive(null);setImageId(event.imageId)}}}><i/><span>Group {event.group}</span><b>{event.score.toFixed(2)}</b><small>{event.source}</small></button>):<p>Play the recorded sequence or upload an image to inspect group activity.</p>}</div></section>}

    {live&&<details className="inference-receipt"><summary>Live inference receipt · {live.backbone} · seed {live.seed}</summary><p>Fresh forward pass through all four frozen methods. Time includes preprocessing, the backbone and group responses.</p><dl><dt>Request</dt><dd>{live.request_id}</dd><dt>Input SHA-256</dt><dd>{live.input_sha256}</dd><dt>Backbone tensors SHA-256</dt><dd>{live.model_identity?.backbone_tensor_sha256}</dd>{Object.entries(live.model_identity?.checkpoint_sha256??{}).map(([name,hash])=><div key={name}><dt>{name} checkpoint SHA-256</dt><dd>{String(hash)}</dd></div>)}</dl><button className="quiet-btn" onClick={()=>{const blob=new Blob([JSON.stringify({request_id:live.request_id,input_sha256:live.input_sha256,seconds:live.seconds,device:live.device,model_identity:live.model_identity,geometry:live.geometry},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='ggf-inference-'+live.request_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}}>Download inference receipt</button></details>}
    <footer className="app-footer"><span>GGF Concept Atlas · Frozen research models</span><span>Reviewed points retain their original image and patch identities.</span><button onClick={()=>setAbout(true)}>Methodology<ArrowUpRight size={12}/></button></footer>

   </main>

  </div>

  {benchmarkOpen&&<BenchmarkEvidence onClose={()=>setBenchmarkOpen(false)}/>}
  {connectionSelection?.method===methodId && selectedPair && reference &&
    <ConnectionEvidence samePatch={samePatch} key={String(samePatch)+methodId+'-'+selectedPair.source+'-'+selectedPair.target}
      method={methodId} methodName={method.name} link={selectedPair} cutoff={threshold}
      images={manifest.images} referenceIds={reference.imageIds} rows={reference.rows[methodId]}
      sharedIndices={sharedIndices} groups={method.groups} currentImageId={live?-1:imageId}
      loadGroup={loadGroup} onClose={()=>setConnectionSelection(null)}
      onInspect={inspectConnectionEvidence}/>}
  {about&&<div className="modal-backdrop" onClick={()=>setAbout(false)}><article className="about-modal" onClick={e=>e.stopPropagation()}><button className="modal-close icon-btn" onClick={()=>setAbout(false)} aria-label="Close methodology"><X size={20}/></button><div className="eyebrow">ABOUT THIS DEMO</div><h2>A map of measured responses.</h2><p>Guided Group Featurizers organize visual representations into groups of learned directions. This demo lets you trace each displayed response back to the image patch that produced it.</p><h3>Two ways to explore</h3><p><b>Compare methods</b> reproduces the four selected native groups in Figure 2. <b>Activation monitor</b> shows all 75 groups in one shared, label-free PCA projection of decoded group contributions. Click a group to focus it, then inspect its separately fitted native contribution PCA.</p><h3>What the geometry means</h3><p>These are 3D projections of higher-dimensional responses. Groups can overlap. The display does not establish intrinsic concept dimension, distinct anatomical regions, or model connectivity. Point colors encode PCA coordinates.</p><h3>Co-activation connections</h3><p>Connections use the common recorded reference images for all four methods. A group fires when its mean response across its four strongest patches reaches the display cutoff relative to its reference scale. Link overlap is the number of images where both groups fire divided by the number where either fires. Each link requires at least three shared images. The cutoff and overlap controls recompute these associations. Lines join reference cloud centers in the shared projection. Live uploads change which links are active; their historical strength stays tied to the recorded sample. Co-activation is an image-level association and does not establish causal influence or anatomical wiring.</p><h3>Recorded and live responses</h3><p>The default group view uses its six original HIGH review images and all four marked patches per image. Only the 24 originally marked regions are plotted in this view. Disable “Reviewed high-response regions” to explore general reference responses, which can include incidental responses outside the reviewed theme. Recorded playback uses 87 common reference images and frozen seed-21 checkpoints. The Figure 2 examples reuse their saved coordinates. Additional group views are read-only exports from the same checkpoints. New-image inference uses the same native interfaces: signed gated BSF codes and independent SLiCS NNLS projections.</p><p>The display cutoff is an exploratory visualization control, not a calibrated detector threshold. The selected Figure 2 examples illustrate differences; the paper’s aggregate comparisons cover every planned group.</p><h3>Private by design</h3><p>The local demo binds to localhost. Uploaded images are processed in memory by the model service and are not saved by the application.</p><a className="paper-link" href="/paper.pdf" target="_blank" rel="noreferrer">Read the research paper <ArrowUpRight size={16}/></a></article></div>}

  {gallery&&<div className="modal-backdrop" onClick={()=>setGallery(false)}><article className="gallery-modal" onClick={e=>e.stopPropagation()}><button className="modal-close icon-btn" onClick={()=>setGallery(false)} aria-label="Close image browser"><X size={20}/></button><div className="eyebrow">RECORDED INPUTS</div><h2>Choose an image to explore.</h2><p>These are the group’s original reviewed high-response images. The marked regions retain their frozen patch selections.</p><div className="gallery-grid">{(reviewedOnly&&data?.reviewedViews?.length?data.reviewedViews.map(v=>manifest.images.find(im=>im.id===v.imageId)!):manifest.images.filter(im=>(manifest.playbackIds??manifest.imageIds).includes(im.id))).map(im=><button key={im.id} onClick={()=>{chooseImage(im.id);setGallery(false)}}><img src={im.src} alt={'COCO image '+im.id} loading="lazy"/><span>{im.id}</span></button>)}</div></article></div>}

 </div>

}

