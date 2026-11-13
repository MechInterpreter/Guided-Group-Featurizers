import type {ScenePoint} from '../app/scene';

type Pair={views:{imageIndex:number;patches:number[]}[]};

/** Highlight the original eight marked patches; keep all measured coordinates. */
export function focusPair(points:ScenePoint[],pair:Pair):ScenePoint[]{
 const ids=new Set(pair.views.flatMap(view=>view.patches.map(p=>view.imageIndex*196+p)));
 return points.map(point=>ids.has(point.id)?{...point,size:4.5,strength:1,pickable:true}:{...point,size:1,strength:.18,pickable:false});
}
