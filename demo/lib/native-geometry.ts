import {groupColor} from './group-colors.ts';
import type {SceneBounds,ScenePoint} from '../app/scene';

export type NativeGeometryData={
  method:string;group:number;variance:number;referenceCount?:number;
  positions:number[][];norms:number[];cloudIndices:number[];reviewedIndices?:number[];
  additionalImages?:Record<string,{positions:number[][];norms:number[]}>;
};
export type NativeFrame={positions:number[][];norms:number[]};

export function nativeFrame(data:NativeGeometryData|null,imageIndex:number):NativeFrame|undefined{
  if(!data||imageIndex<0)return undefined;
  return (imageIndex+1)*196<=data.norms.length
    ?{positions:data.positions.slice(imageIndex*196,(imageIndex+1)*196),norms:data.norms.slice(imageIndex*196,(imageIndex+1)*196)}
    :data.additionalImages?.[imageIndex];
}

export function nativeObservation(data:NativeGeometryData,id:number){
  const imageIndex=Math.floor(id/196),patch=id%196;
  const xyz=id<data.positions.length?data.positions[id]:data.additionalImages?.[imageIndex]?.positions[patch];
  const norm=id<data.norms.length?data.norms[id]:data.additionalImages?.[imageIndex]?.norms[patch];
  if(!xyz||xyz.length!==3||!xyz.every(Number.isFinite)||!Number.isFinite(norm))throw Error('Missing native group observation');
  return {xyz,norm:norm!};
}

/** A frozen frame for this group, independent of camera mode or a new upload. */
export function nativeBounds(data:NativeGeometryData):SceneBounds{
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  for(const id of new Set([...data.cloudIndices,...data.reviewedIndices??[]])){
    const {xyz}=nativeObservation(data,id);
    xyz.forEach((value,k)=>{min[k]=Math.min(min[k],value);max[k]=Math.max(max[k],value)});
  }
  if(!min.every(Number.isFinite))return {min:[-1,-1,-1],max:[1,1,1]};
  return {min,max};
}

/** Pass through the stored projection of actual decoded group contributions. */
export function nativeGeometryPoints(data:NativeGeometryData|null,evidence:'review'|'reference',
  imageIndex:number,imageMode:boolean,cutoff:number,scale:number,live?:NativeFrame):ScenePoint[]{
  if(!data)return [];
  const color=groupColor(data.method,data.group);
  const reviewed=new Set(data.reviewedIndices??[]);
  const ids=new Set(evidence==='review'?[...reviewed]:[...data.cloudIndices,...reviewed]);
  const points:ScenePoint[]=[...ids].map(id=>{
    const {xyz,norm}=nativeObservation(data,id),marked=reviewed.has(id);
    return {id,xyz:[...xyz],group:data.group,
      color:marked?color.rgb:color.context,strength:marked?.95:.29,
      size:marked?3:1,pickable:norm>0};
  });
  if(!imageMode)return points;
  const frame=live??nativeFrame(data,imageIndex);
  if(!frame)return points;
  const byId=new Map(points.map((point,index)=>[point.id,index]));
  frame.norms.forEach((norm,patch)=>{
    if(!(norm>0))return;
    const xyz=frame.positions[patch];
    if(!xyz||xyz.length!==3||!xyz.every(Number.isFinite))throw Error('Invalid live native group position');
    const relative=norm/scale;
    const point:ScenePoint={id:live?-patch-1:imageIndex*196+patch,xyz:[...xyz],group:data.group,
      color:relative>=cutoff?color.active:color.context,strength:relative>=cutoff?.95:.2,
      size:relative>=cutoff?4.5:1,pickable:relative>=cutoff};
    const index=byId.get(point.id);if(index===undefined)points.push(point);else points[index]=point;
  });
  return points;
}
