'use client';

import {useEffect,useLayoutEffect,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {ArrowLeft,ArrowRight,X} from 'lucide-react';

export const TOUR_STORAGE_KEY='ggf-native-geometry-tour-v2';
export const TOUR_STEPS=[
  {id:'welcome',target:'intro',title:'Explore what each group responds to',text:'Start with the original blinded pairs for all four methods, then try your own image. This tour waits for you to press Next at every step.'},
  {id:'maps',target:'maps',title:'Look inside each learned group',text:'Each panel shows a selected group’s actual decoded responses, projected through its own frozen PCA. Every point links to a source patch. Drag to rotate and scroll to zoom. The View menu also offers the shared DINOv3 input map as a separate context view.'},
  {id:'examples',target:'examples',title:'Walk through every original pair',text:'Previous and Next walk through all 225 fixed pairs for each method, covering all 75 groups. Each card shows two original model views and their four marked patches. The methods have their own groups and image selections; equal group numbers are not matched concepts.'},
  {id:'evidence-source',target:'evidence-source',title:'Choose the background evidence',text:'Group reference + review patches shows the selected group’s sampled positive responses, including all 24 original marked patches. Reviewed pairs emphasizes the current eight patches in the group’s stable color, while the others remain context. Blinded review patches removes the general reference background. Coordinates and the reference frame stay fixed; exceptions are kept.'},
  {id:'highlight',target:'highlight',title:'Move between reviews and live exploration',text:'Reviewed pairs highlights the eight marked patches in the current pair without moving any points. Explore & uploads keeps the full response explorer and your session uploads available. Uploading an image opens that mode automatically; you can return to your last pair anytime.'},
  {id:'method-groups',target:'method-groups',title:'Read the original judgments',text:'Semantic meaning and visual structure have separate yes, no or unclear judgments. Each summary uses a two-of-three majority. Open All 3 original judgments & explanations to read the exact notes and any disagreements. No pairs are filtered by outcome.'},
  {id:'inspector',target:'inspector',title:'Follow the selected response',text:'The inspector names the method and the theme recorded in its blinded review. The image, highlighted patch and numeric response follow your selection. A theme describes the reviewed examples, not a guarantee for every new image.'},
  {id:'patches',target:'patches',title:'See exactly which patch fired',text:'Click one of these patch thumbnails, or an active dot in the map. The outline on the full image and the selected map point update together. Recorded review mode uses the exact four patches shown to the reviewers.'},
  {id:'review-gallery',target:'review-gallery',title:'Inspect all six original review images',text:'These are the original HIGH-response images for this method and group. Click each image to inspect its marked patches. The counts below retain all three frozen whole-group reviews, including disagreement or unknown judgments.'},
  {id:'group-browser',target:'group-browser',title:'Explore every group',text:'Browse all 75 groups and search their descriptions. Responses are ranked using each group’s reference scale. This display ranking is separate from the paper’s calibrated concentration measurements.'},
  {id:'native-pca',target:'native-pca',title:'Expand the group geometry',text:'This larger view uses the same actual decoded responses and frozen PCA as the group’s main panel. Click a point to inspect its patch. The retained variance is shown below. Other groups have their own PCA axes, so cloud widths are not a shared-distance comparison.'},
  {id:'upload',target:'upload',title:'Try a new image',text:'With npm run live running locally, upload a JPEG, PNG or WebP up to 8 MB. DINOv3 extracts 196 patches. Each frozen method computes its group coefficients and decoded responses; each group’s existing PCA places those responses in its own cloud. Uploads never refit the projections. Your uploads keeps each image and result in this tab, so you can return after viewing recorded examples without rerunning inference. Refresh or close the page to end that cache.'},
  {id:'display',target:'display',title:'Adjust visibility and reset the camera',text:'The cutoff controls which measured patch responses are highlighted. It does not move points, alter the model or change the research scores. Reset restores the shared initial camera view.'},
  {id:'fidelity',target:'fidelity',title:'Read the projection honestly',text:'Each main panel reports the fraction of that group’s reference variation retained in three PCA dimensions. Its axes and display scale belong to that group: matching screen positions across panels do not identify the same concept or brain region. Rotation changes your viewpoint, not measured distances. Use the linked patches and frozen review judgments to assess semantic coherence.'},
  {id:'benchmark',target:'benchmark',title:'Connect examples to the full results',text:'Research evidence presents the verified aggregate comparisons and uncertainty from the paper. Use these scores alongside the patch examples when explaining the measured coherence and concentration results.'},
  {id:'explorer',target:'explorer',title:'Go further when you are ready',text:'The Detailed explorer retains the additional response and coactivation tools. You can replay this walkthrough at any time using Take a tour. Press Finish to explore.'}
];

export default function AtlasTour({open,onClose,onStep}:{open:boolean;onClose:()=>void;onStep:(id:string)=>void}){
  const [index,setIndex]=useState(0);
  const [rect,setRect]=useState<DOMRect|null>(null);
  const [viewport,setViewport]=useState({width:1200,height:800});
  const card=useRef<HTMLDivElement>(null);
  const previousFocus=useRef<HTMLElement|null>(null);
  const step=TOUR_STEPS[index];
  useEffect(()=>{if(open){setIndex(0);previousFocus.current=document.activeElement as HTMLElement;}},[open]);
  useEffect(()=>{if(open)onStep(step.id)},[open,step.id,onStep]);
  useLayoutEffect(()=>{
    if(!open)return;
    let frame=0,timer:ReturnType<typeof setTimeout>;
    const findTarget=()=>document.querySelector<HTMLElement>(innerWidth<700&&step.id==='maps'?'[data-tour="maps"] .method-atlas':innerWidth<700&&step.id==='inspector'?'.atlas-selected-method':'[data-tour="'+step.target+'"]');
    const update=()=>{setViewport({width:innerWidth,height:innerHeight});setRect(findTarget()?.getBoundingClientRect()??null)};
    const target=findTarget();
    target?.scrollIntoView({block:'center',behavior:'instant'});
    frame=requestAnimationFrame(update);timer=setTimeout(()=>{update();card.current?.querySelector<HTMLButtonElement>('.tour-next')?.focus({preventScroll:true})},350);
    window.addEventListener('resize',update);window.addEventListener('scroll',update,true);
    card.current?.querySelector<HTMLButtonElement>('.tour-next')?.focus({preventScroll:true});
    return()=>{cancelAnimationFrame(frame);clearTimeout(timer);window.removeEventListener('resize',update);window.removeEventListener('scroll',update,true)};
  },[open,index]);
  useEffect(()=>{
    if(!open)return;
    const background=document.querySelector<HTMLElement>('.simple-atlas');if(background)background.inert=true;
    const keys=(event:KeyboardEvent)=>{
      if(event.key==='Escape'){event.preventDefault();event.stopImmediatePropagation();close()}
      if(event.key==='Tab'){
        const buttons=Array.from(card.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')??[]);
        const first=buttons[0],last=buttons[buttons.length-1];
        if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus()}
        else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus()}
      }
    };
    document.addEventListener('keydown',keys,true);
    return()=>{document.removeEventListener('keydown',keys,true);if(background)background.inert=false};
  },[open,onClose]);
  function close(){try{localStorage.setItem(TOUR_STORAGE_KEY,'seen')}catch{}onClose();requestAnimationFrame(()=>previousFocus.current?.focus({preventScroll:true}))}
  if(!open)return null;
  const width=Math.min(368,viewport.width-28),height=280;
  const fitsBelow=rect&&rect.bottom+height+20<viewport.height;
  const top=rect?(fitsBelow?rect.bottom+14:Math.max(14,Math.min(viewport.height-height-14,rect.top-height-14))):Math.max(14,(viewport.height-height)/2);
  const left=rect?Math.max(14,Math.min(viewport.width-width-14,rect.left+rect.width/2-width/2)):(viewport.width-width)/2;
  const spot=rect?{left:Math.max(5,rect.left-5),top:Math.max(5,rect.top-5),width:Math.min(viewport.width-10,rect.width+10),height:Math.min(viewport.height-10,rect.height+10)}:null;
  return createPortal(<div className="atlas-tour" role="dialog" aria-modal="true" aria-labelledby="tour-title" aria-describedby="tour-description">
    <div className="tour-shield"/>
    {spot&&<div className="tour-spotlight" style={spot}/>}
    <div ref={card} className={'tour-bubble '+(fitsBelow?'below':'above')} style={{left,top,width}}>
      <div className="tour-progress"><span>QUICK WALKTHROUGH</span><span aria-live="polite">{index+1} of {TOUR_STEPS.length}</span><button aria-label="Close walkthrough" onClick={close}><X size={17}/></button></div>
      <h2 id="tour-title">{step.title}</h2><p id="tour-description">{step.text}</p>
      <div className="tour-actions"><button className="tour-skip" onClick={close}>Skip tour</button><button disabled={index===0} onClick={()=>setIndex(value=>value-1)}><ArrowLeft size={14}/>Back</button><button className="tour-next" onClick={()=>index===TOUR_STEPS.length-1?close():setIndex(value=>value+1)}>{index===TOUR_STEPS.length-1?'Finish':'Next'}<ArrowRight size={14}/></button></div>
    </div>
  </div>,document.body);
}
