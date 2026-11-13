'use client';
import {useCallback,useEffect,useMemo,useRef,useState,type CSSProperties} from 'react';
import {ArrowUpRight,Check,ChevronDown,LockKeyhole,RotateCcw,Search,Upload,Waypoints,X} from 'lucide-react';
import Scene,{type ScenePoint} from './scene';
import {focusPair} from '../lib/pair-geometry';
import BenchmarkEvidence from '../components/benchmark-evidence';
import PairEvidence,{PairNavigator,type PairArchive,type PairGroup,type PairView} from '../components/pair-walkthrough';
import {readInferenceResponse} from '../lib/inference-response';
import {strongestGroups,type NativeGroupPca} from '../lib/atlas-geometry';
import {sharedUmapPoints,umapFrame,assertSharedUmapLive,type SharedUmap} from '../lib/shared-umap';
import AtlasTour,{TOUR_STORAGE_KEY} from '../components/atlas-tour';
import {nativeGeometryPoints,nativeBounds,nativeObservation} from '../lib/native-geometry';
import {groupColor} from '../lib/group-colors';
import {UploadSession,uploadDigest,type CachedUpload,type UploadView} from '../lib/upload-session';

type Group={id:number;theme:string;q99:number;outcome:string;themeMatches:number[]};
type Method={id:string;name:string;color:string;paperGroup:number;paperImages:number[];groups:Group[]};
type Photo={id:number|string;src:string;geometry:{original:number[];resized:number[];box:number[];crop:number}};
type Manifest={imageIds:number[];images:Photo[];methods:Method[];activity:Record<string,number[][]>;seed:number;sharedVariance:number;overview:Record<string,number[][]>;reviewedOverview:Record<string,number[][]>};
type NativeData={cloudIndices:number[];referenceCount:number;reviewedIndices?:number[];method:string;group:number;norms:number[];positions:number[][];topPatches:number[][];variance:number;additionalImages?:Record<string,{norms:number[];positions:number[][];topPatches:number[]}>;reviewedViews?:{imageId:number;imageIndex:number;patches:number[];pooled:number}[]};
type NativePcas={methods:Record<string,NativeGroupPca[]>};
const STRIDE=1000000;
const groupCache=new Map<string,Promise<NativeData>>();
function loadGroup(method:string,group:number):Promise<NativeData>{
 const key=method+'-'+group;
 if(!groupCache.has(key))groupCache.set(key,fetch('/data/groups/'+key+'.json.gz').then(async response=>{
  if(!response.ok)throw Error('Group responses could not be loaded.');
  const bytes=await response.arrayBuffer(),header=new Uint8Array(bytes);
  const plain=header[0]===31&&header[1]===139?await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text():new TextDecoder().decode(bytes);
  return JSON.parse(plain) as NativeData;
 }).catch(error=>{groupCache.delete(key);throw error}));
 return groupCache.get(key)!;
}
function frame(data:NativeData|null,index:number){
 if(!data||index<0)return null;
 return (index+1)*196<=data.norms.length?{norms:data.norms.slice(index*196,(index+1)*196),positions:data.positions.slice(index*196,(index+1)*196)}:data.additionalImages?.[index]??null;
}
function box(photo:Photo,patch:number){
 const g=photo.geometry,sx=g.original[0]/g.resized[0],sy=g.original[1]/g.resized[1];
 return {x:(g.box[0]+patch%14*16)*sx,y:(g.box[1]+Math.floor(patch/14)*16)*sy,width:16*sx,height:16*sy};
}
function PatchPhoto({photo,norms,scale,patch,onPatch,marked,color}:{photo:Photo;norms:number[];scale:number;patch:number;onPatch:(patch:number)=>void;marked?:number[];color:string}){
 const g=photo.geometry,sx=g.original[0]/g.resized[0];
 return <svg className="atlas-photo" viewBox={'0 0 '+g.original.join(' ')} role="img" aria-label={'Input image '+photo.id+' with measured patch responses'}>
  <image href={photo.src} width={g.original[0]} height={g.original[1]}/>
  <rect x={g.box[0]*sx} y={g.box[1]*g.original[1]/g.resized[1]} width={224*sx} height={224*g.original[1]/g.resized[1]} fill="none" stroke="white" strokeDasharray={4*sx+' '+3*sx} strokeWidth={sx}/>
  {norms.map((norm,p)=><rect key={p} {...box(photo,p)} fill={color} fillOpacity={marked&&!marked.includes(p)?0:.55*Math.min(1,norm/scale)} stroke={p===patch?'#fff':'none'} strokeWidth={2*sx} onClick={()=>{if(!marked||marked.includes(p))onPatch(p)}} pointerEvents={marked&&!marked.includes(p)?'none':'auto'}><title>Patch {p} · response {norm.toFixed(3)}</title></rect>)}
  {norms.length>0&&<rect {...box(photo,patch)} fill="none" stroke={color} strokeWidth={1.5*sx} pointerEvents="none"/>}
 </svg>;
}
function PatchButton({photo,patch,selected,onClick,color}:{photo:Photo;patch:number;selected:boolean;onClick:()=>void;color:string}){
 const g=photo.geometry,b=box(photo,patch);
 return <button className={'atlas-patch'+(selected?' selected':'')} aria-label={'Select patch '+patch} aria-pressed={selected} onClick={onClick}><svg viewBox={[b.x-b.width,b.y-b.height,b.width*3,b.height*3].join(' ')}><image href={photo.src} width={g.original[0]} height={g.original[1]}/><rect {...b} fill="none" stroke={color} strokeWidth={b.width*.07}/></svg><span>{patch}</span></button>;
}
async function loadSharedUmap():Promise<SharedUmap>{
 const response=await fetch('/data/shared-umap.json.gz');
 if(!response.ok)throw Error('The shared DINOv3 map could not be loaded.');
 const bytes=await response.arrayBuffer(),header=new Uint8Array(bytes);
 const text=header[0]===31&&header[1]===139?await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text():new TextDecoder().decode(bytes);
 return JSON.parse(text);
}
function MethodAtlas({pairEntry,pairIndex=0,photos,pairCrops,onPairPatch,method,map,geometry,evidence,activityView,highlightedGroup,rows,live,liveUmap,selection,imageIndex,patch,onSelect,onPoint,cutoff,reset}:{pairEntry?:PairGroup;pairIndex?:number;photos:Photo[];pairCrops:PairArchive['crops'];onPairPatch:(view:PairView,patch:number)=>void;method:Method;map:SharedUmap;geometry:'native'|'umap';evidence:'review'|'reference';activityView:boolean;highlightedGroup:number;rows:ReturnType<typeof strongestGroups>;live:any;liveUmap?:number[][];selection:{method:string;group:number};imageIndex:number;patch:number;onSelect:(group:number)=>void;onPoint:(id:number,group:number)=>void;cutoff:number;reset:number}){
 const [data,setData]=useState<NativeData|null>(null),[failed,setFailed]=useState(false);
 useEffect(()=>{let alive=true;setData(null);setFailed(false);loadGroup(method.id,highlightedGroup).then(value=>{if(alive)setData(value)}).catch(()=>{if(alive)setFailed(true)});return()=>{alive=false}},[method.id,highlightedGroup]);
 const valid=data?.group===highlightedGroup&&data?.method===method.id?data:null;
 const scale=method.groups[highlightedGroup].q99;
 const color=groupColor(method.id,highlightedGroup);
 const liveNative=useMemo(()=>live?{positions:live.positions[highlightedGroup],norms:live.norms[highlightedGroup]}:undefined,[live,highlightedGroup]);
 const groupBounds=useMemo(()=>valid?nativeBounds(valid):undefined,[valid]);
 const livePoints=useMemo(()=>live&&liveUmap?{positions:liveUmap,norms:live.norms[highlightedGroup]}:undefined,[live,liveUmap,highlightedGroup]);
 const basePoints=useMemo(()=>geometry==='native'?nativeGeometryPoints(valid,evidence,imageIndex,activityView,cutoff,scale,liveNative):sharedUmapPoints(map,evidence,valid,highlightedGroup,scale,imageIndex,activityView,cutoff,livePoints),[geometry,map,evidence,valid,highlightedGroup,scale,imageIndex,activityView,cutoff,livePoints,liveNative]);
 const points=useMemo(()=>pairEntry?focusPair(basePoints,pairEntry.pairs[pairIndex]):basePoints,[basePoints,pairEntry,pairIndex]);
 const annotations=useMemo(()=>{
  const rows=(valid?.reviewedIndices??[]).map(id=>geometry==='native'?nativeObservation(valid!,id).xyz:map.positions[Math.floor(id/196)][id%196]);
  return rows.length?[{group:highlightedGroup,label:method.groups[highlightedGroup].theme,xyz:[0,1,2].map(k=>rows.reduce((sum,row)=>sum+row[k],0)/rows.length)}]:[];
 },[valid,map,geometry,method,highlightedGroup]);
 const selected=selection.method===method.id;
 const selectedId=live&&activityView?-patch-1:imageIndex*196+patch;
 const available=!!live||!!umapFrame(valid,imageIndex);
 return <section className={'method-atlas'+(pairEntry?' pair-method':'')+(selected?' method-selected':'')} style={{'--group-color':color.hex,'--group-ink':color.ink} as CSSProperties} data-group-color={color.hex} data-method={method.id} data-highlighted-group={highlightedGroup} data-geometry={geometry} data-coordinate-sha={geometry==='umap'?map.model_sha256:undefined}>
  <header><button onClick={()=>onSelect(highlightedGroup)}><i style={{background:method.color}}/>{method.name}</button><span>{pairEntry?'G'+String(highlightedGroup).padStart(2,'0')+' · pair '+(pairIndex+1)+'/3':activityView?<><b>{rows.filter(row=>row.relative>=cutoff).length}</b> / 75 active</>:'G'+String(highlightedGroup).padStart(2,'0')+' · 24 reviewed patches'}</span></header>
  {pairEntry&&<p className="pair-group-theme">{method.groups[highlightedGroup].theme}</p>}
  {pairEntry&&<PairEvidence entry={pairEntry} pairIndex={pairIndex} crops={pairCrops} imageId={photos[imageIndex]?.id as number} patch={patch} selected={selected} onPatch={onPairPatch} primary={method.id==='ggf'}/>}
  {points.length>0?<Scene key={geometry+'-'+(geometry==='native'?highlightedGroup:'shared')} points={points} fixedBounds={geometry==='native'?groupBounds:map.bounds} annotations={activityView?undefined:annotations} selectedGroup={highlightedGroup} selected={selected?selectedId:undefined} focusGroup={null} resetKey={reset} onPick={id=>onPoint(id,highlightedGroup)} axisLabels={geometry==='native'?['PC 1','PC 2','PC 3']:['UMAP 1','UMAP 2','UMAP 3']} label={method.name+(geometry==='native'?' learned group response PCA, group ':' shared DINOv3 UMAP, group ')+highlightedGroup}/>:<div className="native-cloud-loading"><span className="spinner"/>{failed?'The group cloud could not be loaded.':'Loading this group’s measured geometry…'}</div>}
  <div className="umap-card-caption"><span>{pairEntry?'Highlighted: this pair’s 8 marked patches':activityView?'Image responses':'Reviewed responses'} · G{String(highlightedGroup).padStart(2,'0')}</span><span>{failed?'Patch data unavailable':!valid?'Loading group…':activityView&&!available?'No recorded patch map for this group/image':geometry==='native'?((valid!.variance*100).toFixed(1)+'% variance retained · own PCA axes'):'Same patch positions in every panel'}</span></div>
  {!pairEntry&&<div className="atlas-top-groups" data-tour={method.id==='ggf'?'method-groups':undefined} aria-label={method.name+' strongest groups'}>{rows.slice(0,3).map(row=><button key={row.group} title={method.groups[row.group].theme} aria-label={'Inspect '+method.name+' group '+row.group} style={{'--row-color':groupColor(method.id,row.group).hex} as CSSProperties} onClick={()=>onSelect(row.group)}><span>G{String(row.group).padStart(2,'0')}</span><small>{method.groups[row.group].theme}</small><b>{row.relative.toFixed(2)}×</b></button>)}</div>}
 </section>;
}
export default function Atlas(){
 const [reviewMode,setReviewMode]=useState<'pairs'|'explore'>('pairs'),[pairCursor,setPairCursor]=useState(0),[pairArchive,setPairArchive]=useState<PairArchive|null>(null);
 const [geometry,setGeometry]=useState<'native'|'umap'>('native');
 const [sharedMap,setSharedMap]=useState<SharedUmap|null>(null),[tour,setTour]=useState(false);
 const tourShown=useRef(false);
 const [manifest,setManifest]=useState<Manifest|null>(null),[nativePcas,setNativePcas]=useState<NativePcas|null>(null),[fatal,setFatal]=useState('');
 const [selection,setSelection]=useState({method:'ggf',group:0}),[imageId,setImageId]=useState(0),[patch,setPatch]=useState(0);
 const [live,setLive]=useState<any>(null),[uploading,setUploading]=useState(false),[error,setError]=useState(''),[backend,setBackend]=useState(false);
 const [groupData,setGroupData]=useState<NativeData|null>(null),[groupError,setGroupError]=useState('');
 const [cutoff,setCutoff]=useState(.45),[allGroups,setAllGroups]=useState(false),[query,setQuery]=useState(''),[benchmark,setBenchmark]=useState(false),[nativePca,setNativePca]=useState(false),[reset,setReset]=useState(0);
 const [highlightedGroups,setHighlightedGroups]=useState<Record<string,number>>({ggf:0,slics:0,supervised_vanilla:0,vanilla:0});
 const [evidenceSet,setEvidenceSet]=useState<'review'|'reference'>('reference'),[activityView,setActivityView]=useState(false);
 const [sessionUploads,setSessionUploads]=useState<CachedUpload[]>([]);
 const uploadSession=useRef<UploadSession|null>(null);
 if(!uploadSession.current)uploadSession.current=new UploadSession();
 const mounted=useRef(true),uploadRequest=useRef<AbortController|null>(null);
 const recordedView=useRef<UploadView>({selection:{method:'ggf',group:32},highlightedGroups:{ggf:32,slics:32,supervised_vanilla:23,vanilla:23},patch:74,imageId:502857,geometry:'native',evidenceSet:'reference',activityView:false,reviewMode:'pairs',pairCursor:0});
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;uploadRequest.current?.abort();uploadSession.current?.clear()}},[]);
 useEffect(()=>{
  const view:UploadView={selection,highlightedGroups,patch,imageId,geometry,evidenceSet,activityView,reviewMode,pairCursor};
  if(live)uploadSession.current!.saveView(live.input_sha256,view);else recordedView.current=view;
 },[live,selection,highlightedGroups,patch,imageId,geometry,evidenceSet,activityView,reviewMode,pairCursor]);
 const navigation=useRef(0);
 const input=useRef<HTMLInputElement>(null),uploadVersion=useRef(0);
 useEffect(()=>{let alive=true;Promise.all([fetch('/data/manifest.json').then(r=>r.json() as Promise<Manifest>),fetch('/data/native-pca.json').then(r=>{if(!r.ok)throw Error();return r.json() as Promise<NativePcas>}),loadSharedUmap(),fetch('/data/pair-reviews.json').then(r=>{if(!r.ok)throw Error();return r.json() as Promise<PairArchive>})]).then(([m,r,u,pairs])=>{if(alive){setSharedMap(u);setManifest(m as Manifest);setNativePcas(r as NativePcas);setPairArchive(pairs);const first=pairs.methods.ggf[0].pairs[0].views[0];pendingPatch.current=first.patches[0];setImageId(first.imageId);setPatch(first.patches[0]);setHighlightedGroups(Object.fromEntries(m.methods.map(method=>[method.id,0])));}}).catch(()=>{if(alive)setFatal('The activation atlas could not be loaded. Please refresh.');});const status=()=>fetch('/api/status').then(r=>r.json()).then((r:any)=>{if(alive)setBackend(!!r.ready)}).catch(()=>{if(alive)setBackend(false)});status();const timer=setInterval(status,30000);return()=>{alive=false;clearInterval(timer)}},[]);
 useEffect(()=>{let alive=true;setGroupData(null);setGroupError('');loadGroup(selection.method,selection.group).then(data=>{if(alive)setGroupData(data)}).catch(e=>{if(alive)setGroupError(e.message)});return()=>{alive=false}},[selection.method,selection.group]);
 useEffect(()=>{
  if(!nativePca)return;
  const prior=document.activeElement as HTMLElement|null,overflow=document.body.style.overflow;
  document.body.style.overflow='hidden';
  const dialog=document.querySelector<HTMLElement>('.native-pca-dialog');dialog?.querySelector<HTMLButtonElement>('button')?.focus();
  const keys=(event:KeyboardEvent)=>{if(event.key==='Escape'){event.stopPropagation();setNativePca(false)}if(event.key==='Tab'){const buttons=dialog?.querySelectorAll<HTMLElement>('button');if(buttons?.length===1){event.preventDefault();buttons[0].focus()}}};
  document.addEventListener('keydown',keys);return()=>{document.removeEventListener('keydown',keys);document.body.style.overflow=overflow;prior?.focus()};
 },[nativePca]);
 const imageIndex=manifest?.imageIds.indexOf(imageId)??-1;
 const bounds=sharedMap?.bounds;
 const ranks=useMemo(()=>manifest?Object.fromEntries(manifest.methods.map(method=>[method.id,strongestGroups(live?.methods[method.id]?.scores??manifest.activity[method.id][imageIndex]??[],method.groups.map(group=>group.q99))])):{},[manifest,live,imageIndex]);
 const method=manifest?.methods.find(method=>method.id===selection.method),group=method?.groups[selection.group];
 const validGroupData=groupData?.method===selection.method&&groupData?.group===selection.group?groupData:null;
 const current=live?.methods[selection.method],responses=current?{norms:current.norms[selection.group],positions:current.positions[selection.group]}:frame(validGroupData,imageIndex);
 const reviewedView=!live?validGroupData?.reviewedViews?.find(view=>view.imageId===imageId):undefined;
 const top=useMemo<number[]>(()=>reviewedView?[...reviewedView.patches]:responses?[...Array(196).keys()].filter(p=>responses.norms[p]>0).sort((a,b)=>responses.norms[b]-responses.norms[a]||a-b).slice(0,4):[],[responses,reviewedView]);
 const responseSource=live?null:groupData;
 const pendingPatch=useRef<number|null>(null);
 useEffect(()=>{if(responses){setPatch(pendingPatch.current??top[0]??0);pendingPatch.current=null}},[responseSource,live,imageIndex,selection.method,selection.group]);
 function goToPair(cursor:number,method=selection.method,view?:PairView,chosenPatch?:number){
  if(!pairArchive)return;
  const safe=Math.max(0,Math.min(224,cursor)),unit=Math.floor(safe/3);
  const selectedView=view??pairArchive.methods[method][unit].pairs[safe%3].views[0];
  navigation.current++;uploadVersion.current++;setLive(null);setReviewMode('pairs');setPairCursor(safe);
  setActivityView(false);setAllGroups(false);setQuery('');setError('');
  setHighlightedGroups(Object.fromEntries(manifest!.methods.map(m=>[m.id,unit])));setSelection({method,group:unit});
  pendingPatch.current=chosenPatch??selectedView.patches[0];setPatch(pendingPatch.current);setImageId(selectedView.imageId);
 }
 function inspectPairPatch(method:string,view:PairView,p:number){goToPair(pairCursor,method,view,p)}
 function showPairWalkthrough(){goToPair(pairCursor)}
 async function chooseGroup(method:string,group:number){
  if(reviewMode==='pairs'&&!live){goToPair(group*3+pairCursor%3,method);return}

  const version=++navigation.current;
  if(method!==selection.method){setQuery('');setAllGroups(false)}pendingPatch.current=null;setHighlightedGroups(previous=>({...previous,[method]:group}));setSelection({method,group});
  if(!live&&!activityView)try{const data=await loadGroup(method,group);if(version===navigation.current&&data.reviewedViews?.length){pendingPatch.current=data.reviewedViews[0].patches[0];setImageId(data.reviewedViews[0].imageId)}}catch(e:any){setError(e.message)}
 }
 function choosePoint(method:string,id:number,group:number){
  if(id>=0&&reviewMode==='pairs'&&pairArchive){
   const imageIndex=Math.floor((id%STRIDE)/196),p=id%STRIDE%196;
   const entry=pairArchive.methods[method][group];
   const pair=entry.pairs.find(pair=>pair.views.some(view=>view.imageIndex===imageIndex&&view.patches.includes(p)));
   if(pair){goToPair(group*3+pair.pair,method,pair.views.find(view=>view.imageIndex===imageIndex),p);return}
   setReviewMode('explore');
  }
  navigation.current++;
  pendingPatch.current=id<0?(-id-1)%196:id%STRIDE%196;setPatch(pendingPatch.current);setHighlightedGroups(previous=>({...previous,[method]:group}));setSelection({method,group});
  if(id>=0){uploadVersion.current++;setLive(null);setImageId(manifest!.imageIds[Math.floor((id%STRIDE)/196)])}
 }
 function restoreView(view:UploadView){
  pendingPatch.current=view.patch;setPatch(view.patch);setSelection(view.selection);
  setHighlightedGroups(view.highlightedGroups);setImageId(view.imageId);
  setReviewMode(view.reviewMode??'explore');setPairCursor(view.pairCursor??0);setGeometry(view.geometry);setEvidenceSet(view.evidenceSet);setActivityView(view.activityView);
  setAllGroups(false);setQuery('');
 }
 function activateUpload(entry:CachedUpload){
  navigation.current++;uploadVersion.current++;setError('');
  setLive(entry.live);setReviewMode('explore');
  if(entry.view){restoreView(entry.view);setReviewMode('explore')}
  else{
   pendingPatch.current=null;setActivityView(true);setAllGroups(false);setQuery('');
   const groups=Object.fromEntries(manifest!.methods.map(method=>[method.id,strongestGroups(entry.live.methods[method.id].scores,method.groups.map(group=>group.q99))[0].group]));
   setHighlightedGroups(groups);setSelection({method:selection.method,group:groups[selection.method]});
  }
 }
 function showRecorded(){
  navigation.current++;uploadVersion.current++;setError('');setLive(null);restoreView(recordedView.current);
 }
 function removeUpload(key:string){
  if(live?.input_sha256===key)showRecorded();
  uploadSession.current!.remove(key);setSessionUploads(uploadSession.current!.list());
 }
 function clearUploads(){
  uploadRequest.current?.abort();uploadRequest.current=null;setUploading(false);
  navigation.current++;uploadVersion.current++;
  if(live){setLive(null);restoreView(recordedView.current)}
  uploadSession.current!.clear();setSessionUploads([]);setError('');
 }
 async function upload(file:File){
  if(file.size>8*1024*1024){setError('Choose an image up to 8 MB.');return}
  const version=++uploadVersion.current,nav=++navigation.current;setUploading(true);setError('');
  const controller=new AbortController();uploadRequest.current=controller;
  try{
   const key=await uploadDigest(file);
   if(!mounted.current||controller.signal.aborted)return;
   const cached=uploadSession.current!.get(key);
   if(cached){if(version===uploadVersion.current&&nav===navigation.current)activateUpload(cached);return}
   const body=new FormData();body.append('image',file);
   const result=await readInferenceResponse(await fetch('/api/infer',{method:'POST',body,signal:controller.signal}));
   if(!mounted.current||controller.signal.aborted)return;
   if(geometry==='umap')assertSharedUmapLive(result,sharedMap!);
   const entry=uploadSession.current!.add(key,file,result);
   setSessionUploads(uploadSession.current!.list());
   if(version===uploadVersion.current&&nav===navigation.current)activateUpload(entry);
  }catch(error:any){if(mounted.current&&!controller.signal.aborted)setError(error.message)}
  finally{if(mounted.current&&uploadRequest.current===controller){uploadRequest.current=null;setUploading(false)}}
 }
 async function example(group:number){
  const version=++uploadVersion.current,nav=++navigation.current;setError('');pendingPatch.current=null;
  try{
   const data=await loadGroup('ggf',group);
   if(version!==uploadVersion.current||nav!==navigation.current)return;
   setLive(null);setReviewMode('explore');setImageId(data.reviewedViews![0].imageId);setSelection({method:'ggf',group});
   setHighlightedGroups(Object.fromEntries(manifest!.methods.map(method=>[method.id,method.id==='ggf'?group:strongestGroups(manifest!.activity[method.id][manifest!.imageIds.indexOf(data.reviewedViews![0].imageId)],method.groups.map(value=>value.q99))[0].group])));setAllGroups(false);
  }catch(error:any){if(nav===navigation.current)setError(error.message)}
 }
 const nativePoints=useMemo<ScenePoint[]>(()=>nativeGeometryPoints(validGroupData,evidenceSet,imageIndex,true,cutoff,group?.q99??1,current?{positions:current.positions[selection.group],norms:current.norms[selection.group]}:undefined),[validGroupData,evidenceSet,imageIndex,cutoff,group,current,selection.group]);
 const expandedBounds=useMemo(()=>validGroupData?nativeBounds(validGroupData):undefined,[validGroupData]);
 const tourStep=useCallback((id:string)=>{
  setNativePca(id==='native-pca');setBenchmark(id==='benchmark');
  setAllGroups(id==='group-browser');
  const display=document.querySelector<HTMLDetailsElement>('[data-tour="display"]');if(display)display.open=id==='display';
 },[]);
 const closeTour=useCallback(()=>{setTour(false);setNativePca(false);setBenchmark(false);setAllGroups(false);const display=document.querySelector<HTMLDetailsElement>('[data-tour="display"]');if(display)display.open=false},[]);
 async function startTour(){setGeometry('native');setActivityView(false);setEvidenceSet('reference');setQuery('');showPairWalkthrough();setTour(true)}
 useEffect(()=>{if(manifest&&sharedMap&&groupData&&!tourShown.current){tourShown.current=true;try{if(!localStorage.getItem(TOUR_STORAGE_KEY))setTour(true)}catch{}}},[manifest,sharedMap,groupData]);
 if(fatal)return <div className="initial-load"><h1>{fatal}</h1></div>;
 if(!manifest||!nativePcas||!sharedMap||!pairArchive||!bounds||!method||!group)return <div className="initial-load"><span className="spinner"/><p>Loading the measured activation atlas…</p></div>;
 const photo:Photo=live?.image??manifest.images[imageIndex];
 const selectedRank=ranks[selection.method]?.find(row=>row.group===selection.group);
 const selectedColor=groupColor(method.id,group.id);
 const colorStyle={'--group-color':selectedColor.hex,'--group-ink':selectedColor.ink} as CSSProperties;
 const list=(ranks[selection.method]??[]).filter(row=>!query||('group '+row.group+' '+method.groups[row.group].theme).toLowerCase().includes(query.toLowerCase()));
 return <div className="simple-atlas">
  <header className="atlas-header"><a href="/" className="atlas-brand"><Waypoints size={25}/><span>GGF<span>CONCEPT ATLAS</span></span></a><div><span className="atlas-private"><LockKeyhole size={12}/>Private</span><button className="atlas-tour-start" onClick={startTour}>Take a tour</button><button onClick={()=>setBenchmark(true)}>Research evidence<ArrowUpRight size={14}/></button><a data-tour="explorer" href="/explore" target="_blank" rel="noreferrer">Detailed explorer<ArrowUpRight size={14}/></a></div></header>
  <main className="atlas-main"><div className="atlas-intro" data-tour="intro"><div><div className="eyebrow">LIVE REPRESENTATION MONITOR</div><h1>{reviewMode==='pairs'?'Coherence. Pair by pair.':'Learned groups. Patch by patch.'}</h1><p>Inspect the original blinded evidence, follow each marked patch into its learned group, then try a new image.</p></div><button data-tour="upload" className="atlas-upload" disabled={uploading} onClick={()=>input.current?.click()}><Upload size={18}/>{uploading?'Processing image…':'Upload an image'}<small>{backend?'Live models ready':'Recorded exploration available'}</small></button><input ref={input} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={event=>{const file=event.target.files?.[0];if(file)upload(file);event.target.value=''}}/></div>
  {error&&<p className="atlas-error" role="alert">{error}</p>}
  <div className="atlas-review-modes" data-tour="highlight" aria-label="Demo mode"><button aria-pressed={reviewMode==='pairs'} onClick={showPairWalkthrough}>Reviewed pairs</button><button aria-pressed={reviewMode==='explore'} onClick={()=>{navigation.current++;uploadVersion.current++;setReviewMode('explore')}}>Explore &amp; uploads</button><span>{reviewMode==='pairs'?'900 original pairs · 3 frozen reviews each':'Recorded examples and your session uploads'}</span></div>
  {reviewMode==='pairs'&&<PairNavigator cursor={pairCursor} onChange={cursor=>goToPair(cursor)}/>}
  <div className="atlas-source-bar"><span className={'atlas-source '+(live?' is-live':'')}><i/>{live?'LIVE UPLOAD':'RECORDED IMAGE'}</span><span>{live?`Cached result · ${live.seconds.toFixed(2)} s · ${live.device?.toUpperCase()??'MODEL'}`:'Try an example'}</span><span className="atlas-examples"><button onClick={()=>{setActivityView(false);example(32)}}>Gloves</button><button onClick={()=>{setActivityView(false);example(69)}}>Books</button></span><div/><label className="pca-evidence-select geometry-select" data-tour="geometry"><span>View</span><select aria-label="Geometry space" value={geometry} onChange={event=>{const value=event.target.value as 'native'|'umap';if(value==='umap'&&live)try{assertSharedUmapLive(live,sharedMap)}catch(error:any){setError(error.message);return}setGeometry(value);setReset(value=>value+1)}}><option value="native">Learned group geometry (PCA)</option><option value="umap">DINOv3 input map (UMAP)</option></select></label><label className="pca-evidence-select" data-tour="evidence-source"><span>Evidence</span><select aria-label="Map evidence set" value={evidenceSet} onChange={event=>setEvidenceSet(event.target.value as 'review'|'reference')}><option value="review">Blinded review patches</option><option value="reference">{geometry==='native'?'Group reference + review patches':'Reference + review patches'}</option></select></label>{reviewMode==='explore'&&<div className="pca-view-switch"><button aria-pressed={!activityView} onClick={()=>setActivityView(false)}>Reviewed responses</button><button aria-pressed={activityView} onClick={()=>setActivityView(true)}>This image</button></div>}<button title="Reset camera views" aria-label="Reset atlas views" onClick={()=>setReset(v=>v+1)}><RotateCcw size={14}/></button><details className="atlas-display" data-tour="display"><summary>Display settings<ChevronDown size={12}/></summary><label>Activation cutoff <strong>{cutoff.toFixed(2)}× reference</strong><input aria-label="Atlas activation cutoff" disabled={reviewMode==='pairs'} type="range" min="5" max="150" value={cutoff*100} onChange={e=>setCutoff(Number(e.target.value)/100)}/></label><p>{reviewMode==='pairs'?'Reviewed pairs always highlights the original marked patches. Switch to Explore & uploads to adjust the activation cutoff.':'Group response is the mean of its four strongest patch norms, divided by its reference scale. These display controls are separate from the paper’s calibrated evaluation.'}</p></details></div>
  {sessionUploads.length>0&&<section className="atlas-session" aria-label="Session uploads" data-tour="session-uploads">
   <div className="session-heading"><div><h2>Your uploads <span>{sessionUploads.length}</span></h2><p>Switch images without reuploading or running the models again. Kept in memory until this page is refreshed or closed.</p></div><div><button aria-pressed={!live} onClick={showRecorded}>Recorded examples</button><button onClick={clearUploads}>Clear uploads</button></div></div>
   <div className="session-upload-list">{sessionUploads.map(entry=><div key={entry.key} className={'session-upload'+(live?.input_sha256===entry.key?' active':'')}><button aria-label={'Open cached upload '+entry.number+': '+entry.name} aria-pressed={live?.input_sha256===entry.key} onClick={()=>activateUpload(entry)}><img src={entry.live.image.src} alt=""/><span><strong>{entry.name}</strong><small>Upload {entry.number} · cached</small></span></button><button className="session-remove" aria-label={'Remove cached upload '+entry.number+': '+entry.name} onClick={()=>removeUpload(entry.key)}><X size={14}/></button></div>)}</div>
  </section>}
  <div className="atlas-workspace"><div className="atlas-maps"><div className="atlas-four" data-tour="maps">{manifest.methods.map(m=><MethodAtlas key={m.id} pairEntry={reviewMode==='pairs'?pairArchive.methods[m.id][Math.floor(pairCursor/3)]:undefined} pairIndex={pairCursor%3} photos={manifest.images} pairCrops={pairArchive.crops} onPairPatch={(view,p)=>inspectPairPatch(m.id,view,p)} method={m} map={sharedMap} geometry={geometry} evidence={evidenceSet} activityView={activityView} highlightedGroup={highlightedGroups[m.id]} rows={ranks[m.id]} live={live?.methods[m.id]} liveUmap={live?.umap.positions} selection={selection} imageIndex={imageIndex} patch={patch} cutoff={cutoff} reset={reset} onSelect={group=>chooseGroup(m.id,group)} onPoint={(id,group)=>choosePoint(m.id,id<0?id:group*STRIDE+id,group)}/>)}</div><div className="atlas-legend">{manifest.methods.map(m=><span key={m.id}><i style={{background:groupColor(m.id,highlightedGroups[m.id]).hex}}/>{m.name} · G{String(highlightedGroups[m.id]).padStart(2,'0')}</span>)}<span>{geometry==='native'?'Faint: additional same-group responses':'Grey: shared input-patch context'}</span><span>{reviewMode==='pairs'?'Bright: current pair · faint: group context':'Larger, brighter: this image'}</span><span>Drag to rotate · scroll to zoom</span></div>
  {reviewMode==='pairs'&&<details className="pair-rubric"><summary>How these pairs were judged</summary><p><strong>Semantic meaning. </strong>{pairArchive.rubrics.semantic_rubric}</p><p><strong>Visual structure. </strong>{pairArchive.rubrics.visual_rubric}</p><p>All 225 planned pairs per method are included, regardless of outcome. A yes or no summary requires two of the three original reviewers; other combinations remain unclear. These are seed21 examples, separate from the seven-order aggregate results under Research evidence. Groups with the same number across methods are not matched concepts, and each method’s pair contains its own selected images.</p><a href="/data/pair-reviews.json" download>Download original pair evidence and source hashes</a></details>}
  {geometry==='native'?<div className="native-geometry-note" data-tour="fidelity"><strong>Each panel shows its selected group’s actual decoded responses.</strong><span>Independent PCA axes and display scale per group. Similar screen positions across panels do not identify a shared concept or brain region. Compare the linked patches, not cloud locations.</span></div>:<div className="umap-fidelity" data-tour="fidelity"><strong>One fixed DINOv3 UMAP · identical input-patch coordinates across methods</strong><span>Neighborhood trustworthiness: {(sharedMap.fidelity.trustworthiness*100).toFixed(1)}%</span><span>15-neighbor recall: {(sharedMap.fidelity.neighbor_recall*100).toFixed(1)}%</span></div>}
  <details className="atlas-layout-note"><summary>What the geometry shows</summary>{geometry==='native'?<p>For each method, a point is the selected group’s measured coefficient vector decoded through its learned directions, then projected through that group’s frozen PCA basis. These are observations within the learned group’s response subspace. Each method/group has a stable display color. Faint tinted points are additional positive responses from that same group; they can include weaker or incidental responses and are not all confirmed matches to the reviewed theme. Saturated points preserve all 24 original marked review patches, including exceptions. Larger, brighter points show measured responses to the selected image above the display cutoff. Colors identify method/group records and do not assert that groups are semantically aligned across methods. No group clusters or surfaces are constructed, and no points are moved to match a theme. A group can retain different poses, textures and lighting conditions, so semantic coherence does not require every response to occupy the same location. Each group has its own PCA axes and isotropic display scale. The reference frame stays fixed during uploads and evidence switches. The displayed variance measures how much reference variation three components retain. The DINOv3 input map remains available as a separate context view.</p>:<p>This optional map places the original normalized DINOv3 inputs, before any group decoding, through one fixed label-free UMAP. All four methods use identical coordinates and initial scale. Group selection changes highlights only. Settings: {sharedMap.parameters.n_neighbors} neighbors, cosine distance, minimum distance {sharedMap.parameters.min_dist}, fixed seed {sharedMap.parameters.random_state}. The fit uses {sharedMap.fit_patch_count.toLocaleString()} patches from {sharedMap.fit_image_ids.length} reference images. Live images are transformed through that same fit. UMAP emphasizes neighborhoods and does not preserve all distances. Its diagnostic sample contains {sharedMap.fidelity.sample_size.toLocaleString()} patches. This input-space context is separate from each method’s learned group geometry.</p>}</details></div>
  <aside style={colorStyle} data-group-color={selectedColor.hex} data-tour="inspector" className="atlas-inspector" aria-label="Selected group response"><div className="atlas-selected-method"><i style={{background:method.color}}/>{method.name}<span>GROUP {String(group.id).padStart(2,'0')}</span></div><div className="atlas-review-label">Reviewed reference theme</div><h2>{group.theme}</h2><div className="atlas-theme-outcome"><Check size={13}/>{'Review: '+group.outcome.replace(/_/g,' ')}<span>{(selectedRank?.relative??0).toFixed(2)}× reference</span></div><div className={'atlas-current-state '+((selectedRank?.relative??0)>=cutoff?'above':'below')}>{(selectedRank?.relative??0)>=cutoff?'Active on this image':'Below the display cutoff on this image'}</div><div className="atlas-source-caption">{live?'Your uploaded image':'Image '+imageId} · patch {patch}</div><PatchPhoto color={selectedColor.hex} photo={photo} norms={responses?.norms??[]} scale={group.q99} patch={patch} onPatch={p=>{pendingPatch.current=p;setPatch(p)}} marked={reviewedView?.patches}/>{groupError&&<p className="atlas-error">{groupError}</p>}{!responses&&!groupError&&<p className="atlas-quiet">{groupData?'This recorded image has no exported patch map for this group.':'Loading patch responses…'}</p>}<div className="atlas-patch-row" data-tour="patches">{top.map(p=><PatchButton color={selectedColor.hex} key={p} photo={photo} patch={p} selected={patch===p} onClick={()=>{pendingPatch.current=p;setPatch(p)}}/>)}</div><div className="atlas-patch-measure"><span>Patch {patch} native response</span><strong>{responses?.norms[patch]?.toFixed(3)??'—'}</strong></div><button className="atlas-pca-button" onClick={()=>setNativePca(true)}>Inspect measured PCA<ArrowUpRight size={14}/></button>
  {validGroupData?.reviewedViews&&<section data-tour="review-gallery" className="atlas-review-evidence" aria-label="Original blinded-review evidence"><h3>Original blinded-review evidence</h3><p>Six HIGH images · four frozen marked patches each</p><div>{validGroupData.reviewedViews.map(view=><button key={view.imageId} aria-label={'View original review image '+view.imageId} aria-pressed={!live&&view.imageId===imageId} onClick={()=>{if(reviewMode==='pairs'){const pair=pairArchive.methods[method.id][group.id].pairs.find(p=>p.views.some(v=>v.imageId===view.imageId))!;goToPair(group.id*3+pair.pair,method.id,view);return}navigation.current++;uploadVersion.current++;setLive(null);setActivityView(false);pendingPatch.current=view.patches[0];setPatch(view.patches[0]);setImageId(view.imageId);setEvidenceSet('review')}}><img src={manifest.images.find(image=>image.id===view.imageId)?.src} alt={'Original review image '+view.imageId}/></button>)}</div><small>Whole-group theme matches: {group.themeMatches.map(count=>typeof count==='number'?count+'/6':'unknown').join(' · ')} across three reviews.</small></section>}
  <div className="atlas-rank-heading" data-tour="group-browser"><h3>{allGroups?'All 75 groups':'Most active groups'}</h3><button onClick={()=>{setAllGroups(v=>!v);setQuery('')}}>{allGroups?'Show strongest':'Browse all'}</button></div>{allGroups&&<label className="atlas-search"><Search size={14}/><input aria-label="Find a group" placeholder="Search a theme or group…" value={query} onChange={e=>setQuery(e.target.value)}/></label>}<div className={'atlas-ranking'+(allGroups?' expanded':'')} aria-label={method.name+' group activity'}>{(allGroups?list:list.slice(0,5)).map(row=><button key={row.group} className={row.group===group.id?'chosen':''} onClick={()=>chooseGroup(method.id,row.group)} aria-label={'Select '+method.name+' group '+row.group} style={{'--row-color':groupColor(method.id,row.group).ink} as CSSProperties}><span>{String(row.group).padStart(2,'0')}</span><div><strong>{method.groups[row.group].theme}</strong><i><b style={{width:Math.min(1,row.relative)*100+'%'}}/></i></div><small>{row.relative.toFixed(2)}×</small></button>)}</div>
  {live&&<details className="atlas-receipt"><summary>Inference details</summary><p>{live.backbone} · seed {live.seed} · {live.seconds.toFixed(2)} seconds</p><span>Input SHA-256</span><code>{live.input_sha256}</code><span>Request</span><code>{live.request_id}</code></details>}</aside></div>
  <footer className="atlas-bottom"><span>Frozen seed 21 · 75 groups per method · 12 directions per group · DINOv3 ViT-B/16</span><span>{geometry==='native'?'Native group responses · frozen PCA projections':'Shared DINOv3 input UMAP · fixed projection'}</span></footer></main>
  <AtlasTour open={tour} onClose={closeTour} onStep={tourStep}/>
  {benchmark&&<BenchmarkEvidence onClose={()=>setBenchmark(false)}/>}
  {nativePca&&<div className="modal-backdrop" onClick={()=>setNativePca(false)}><article style={colorStyle} data-tour="native-pca" className="native-pca-dialog" role="dialog" aria-modal="true" aria-label="Measured group PCA" onClick={e=>e.stopPropagation()}><button className="modal-close icon-btn" aria-label="Close measured PCA" onClick={()=>setNativePca(false)}><X size={20}/></button><div className="eyebrow">{method.name} · GROUP {group.id}</div><h2>Measured native PCA</h2><p>{group.theme}</p><Scene points={nativePoints} fixedBounds={expandedBounds} selected={live?-patch-1:imageIndex*196+patch} onPick={id=>{if(id<0){pendingPatch.current=-id-1;setPatch(-id-1)}else if(!live)choosePoint(method.id,group.id*STRIDE+id,group.id)}} large label={method.name+' group '+group.id+' unarranged native PCA'}/><small>{((nativePcas.methods[method.id][group.id].variance)*100).toFixed(1)}% reference variance in three PCs. Larger, brighter points show the selected image. Axes and display scale belong to this group. This is a projection of its decoded responses.</small></article></div>}
 </div>;
}
