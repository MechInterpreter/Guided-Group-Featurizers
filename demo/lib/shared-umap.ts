import {groupColor} from './group-colors.ts';
import type {SceneBounds, ScenePoint} from '../app/scene';

export type SharedUmap = {
  version:number; kind:'dinov3-umap'; image_ids:number[]; positions:number[][][];
  reference_indices:number[]; reviewed_indices:number[]; bounds:SceneBounds;
  fit_patch_count:number; fit_image_ids:number[]; model_sha256:string;
  input_space:string; umap_version:string;
  parameters:{n_components:number;n_neighbors:number;min_dist:number;metric:string;random_state:number};
  fidelity:{sample_size:number;neighbors:number;trustworthiness:number;neighbor_recall:number};
};

export type UmapGroup = {
  method?:string;
  norms:number[]; reviewedIndices?:number[];
  reviewedViews?:{imageId:number;imageIndex:number;patches:number[]}[];
  additionalImages?:Record<string,{norms:number[]}>;
};

export function umapFrame(data:UmapGroup|null, imageIndex:number):number[]|undefined {
  if(!data || imageIndex<0) return undefined;
  return (imageIndex+1)*196<=data.norms.length
    ?data.norms.slice(imageIndex*196,(imageIndex+1)*196)
    :data.additionalImages?.[imageIndex]?.norms;
}

/** All methods receive these exact coordinates; membership affects appearance only. */
export function sharedUmapPoints(
  map:SharedUmap, evidence:'review'|'reference', group:UmapGroup|null,
  selectedGroup:number, scale:number, imageIndex:number, imageMode:boolean,
  cutoff:number, live?:{positions:number[][];norms:number[]}
):ScenePoint[] {
  const color=groupColor(group?.method??'ggf',selectedGroup);
  const reviewed=new Set(group?.reviewedIndices??[]);
  const ids=new Set(evidence==='review'?map.reviewed_indices:[...map.reference_indices,...map.reviewed_indices]);
  for(const id of reviewed) ids.add(id);
  const points:ScenePoint[]=[...ids].sort((a,b)=>a-b).map(id=>({
    id, xyz:[...map.positions[Math.floor(id/196)][id%196]],
    group:reviewed.has(id)?selectedGroup:undefined,
    color:reviewed.has(id)&&!imageMode?color.rgb:[.44,.59,.61],
    strength:reviewed.has(id)&&!imageMode?1:.24,
    size:reviewed.has(id)&&!imageMode?4:1,
    pickable:reviewed.has(id)&&!imageMode
  }));
  if(!imageMode) return points;
  const norms=live?.norms??umapFrame(group,imageIndex);
  const positions=live?.positions??map.positions[imageIndex];
  if(!norms || !positions) return points;
  const byId=new Map(points.map((point,index)=>[point.id,index]));
  norms.forEach((norm,patch)=>{
    const relative=norm/scale;
    const point:ScenePoint={id:live?-patch-1:imageIndex*196+patch,
      xyz:[...positions[patch]],group:selectedGroup,
      color:relative>=cutoff?color.active:[.53,.62,.63],
      strength:relative>=cutoff?Math.min(1,.45+relative*.5):.12,
      size:relative>=cutoff?4.5:1,pickable:relative>=cutoff};
    const index=byId.get(point.id);
    if(index===undefined) points.push(point); else points[index]=point;
  });
  return points;
}

export function assertSharedUmapLive(result:any,map:SharedUmap){
  if(result.umap?.model_sha256!==map.model_sha256 ||
    result.umap?.positions?.length!==196 ||
    !result.umap.positions.every((row:any)=>Array.isArray(row)&&row.length===3&&row.every(Number.isFinite))) {
    throw Error('The local model service needs the updated shared UMAP. Restart npm run live.');
  }
}
