import type {ScenePoint,SceneAnnotation,SceneBounds} from '../app/scene';
export type NativeGroupPca={group:number;points:number[][];variance:number};
export function strongestGroups(scores:number[],scales:number[]){
 return scores.map((raw,group)=>({group,raw,relative:Number.isFinite(raw)&&raw>=0?raw/scales[group]:0,available:Number.isFinite(raw)&&raw>=0})).sort((a,b)=>b.relative-a.relative||a.group-b.group);
}
export function sharedPcaBounds(...collections:Record<string,number[][]>[]):SceneBounds{
 const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
 for(const collection of collections)for(const rows of Object.values(collection))for(const row of rows)for(let k=0;k<3;k++){
  if(!Number.isFinite(row[k+2]))throw Error('Invalid shared PCA coordinate');
  min[k]=Math.min(min[k],row[k+2]);max[k]=Math.max(max[k],row[k+2]);
 }
 return {min,max};
}
export function measuredAnnotations(rows:number[][],themes:{theme:string}[]):SceneAnnotation[]{
 return themes.map((group,index)=>{const points=rows.filter(row=>row[0]===index);return {group:index,label:group.theme,xyz:[0,1,2].map(k=>points.reduce((sum,row)=>sum+row[k+2],0)/Math.max(1,points.length))};});
}
export function measuredOverviewPoints(rows:number[][],live:any,scales:number[],cutoff:number,highlightedGroup?:number):ScenePoint[]{
 const points:ScenePoint[]=rows.map(row=>{
  const group=row[0],hue=(group*.61803398875)%1;
  const channel=(offset:number)=>{const k=(offset+hue*12)%12;return .62-.32*Math.max(-1,Math.min(k-3,9-k,1));};
  const color=highlightedGroup===undefined?[channel(0),channel(8),channel(4)]:group===highlightedGroup?[.22,.93,.72]:[.42,.52,.55];
  return {id:group*1000000+row[1],group,xyz:row.slice(2),color,strength:live?.28:highlightedGroup===undefined?.75:group===highlightedGroup?1:.13,size:group===highlightedGroup?2.6:1,pickable:!live};
 });
 if(live)for(let group=0;group<75;group++)for(let patch=0;patch<196;patch++)if(live.norms[group][patch]>0)points.push({
  id:-(group*196+patch+1),group,xyz:[...live.overview[group][patch]],color:[1,.87,.57],size:2,strength:Math.min(1,live.norms[group][patch]/scales[group]),pickable:live.norms[group][patch]/scales[group]>=cutoff
 });
 return points;
}
