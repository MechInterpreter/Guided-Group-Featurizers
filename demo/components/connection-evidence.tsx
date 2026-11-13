'use client';

import {useEffect, useRef, useState} from 'react';
import {ChevronLeft, ChevronRight, X} from 'lucide-react';
import {sharedPatchIndices} from '../lib/patch-activity';
import type {CoactivationLink} from '../lib/coactivation';

type ImageRecord = {
  id: number;
  src: string;
  geometry: {original: number[]; resized: number[]; box: number[]; crop: number};
};
type Responses = {
  norms: number[];
  topPatches: number[][];
  additionalImages?: Record<string, {norms: number[]; topPatches: number[]}>;
};
type Group = {id: number; theme: string; q99: number};
type Props = {
  samePatch:boolean;
  method: string;
  methodName: string;
  link: CoactivationLink;
  cutoff: number;
  images: ImageRecord[];
  referenceIds: number[];
  rows: number[][];
  sharedIndices: number[];
  groups: Group[];
  currentImageId: number;
  loadGroup: (method: string, group: number) => Promise<Responses>;
  onClose: () => void;
  onInspect: (group: number, imageId: number, patch: number) => void;
};

function responseFrame(data: Responses, index: number) {
  if (index >= 0 && (index + 1) * 196 <= data.norms.length) {
    return {
      norms: data.norms.slice(index * 196, (index + 1) * 196),
      topPatches: data.topPatches[index]
    };
  }
  return data.additionalImages?.[index] ?? null;
}

function patchBox(image: ImageRecord, patch: number) {
  const g = image.geometry;
  const sx = g.original[0] / g.resized[0];
  const sy = g.original[1] / g.resized[1];
  return {
    x: (g.box[0] + (patch % 14) * 16) * sx,
    y: (g.box[1] + Math.floor(patch / 14) * 16) * sy,
    width: 16 * sx,
    height: 16 * sy
  };
}

function ResponseView({image, group, responses, pooled, color, onInspect, shared, selectedPatch, onSharedPatch}: {
  shared?:number[];selectedPatch?:number|null;onSharedPatch?:(patch:number)=>void;
  image: ImageRecord; group: Group;
  responses: ReturnType<typeof responseFrame>;
  pooled: number; color: string;
  onInspect: (patch: number) => void;
}) {
  const patches = shared ?? responses?.topPatches?.filter(p => responses.norms[p] > 0) ?? [];
  const [ownPatch, setOwnPatch] = useState(patches[0] ?? 0);
  const chosen=shared ? (selectedPatch!=null&&patches.includes(selectedPatch)?selectedPatch:patches[0]) : ownPatch;
  const setChosen=(patch:number)=>shared ? onSharedPatch?.(patch) : setOwnPatch(patch);
  const previewPatches=shared ? [chosen,...patches.filter(p=>p!==chosen)].slice(0,4) : patches;
  const g = image.geometry;
  const sx = g.original[0] / g.resized[0];
  if (!responses || !patches.length) return <p role="status">Patch responses are unavailable for this image.</p>;
  return <section className="connection-response" data-group={group.id}>
    <h3><i style={{background: color}}/>Group {group.id}</h3>
    <p className="connection-theme">{group.theme}</p>
    <div className="connection-values">
      <span>Mean top-4 response <strong>{pooled.toFixed(3)}</strong></span>
      <span>Reference-scaled <strong>{(100 * pooled / group.q99).toFixed(1)}%</strong></span>
    </div>
    <svg viewBox={'0 0 ' + g.original.join(' ')} className="connection-photo"
      role="img" aria-label={'Image ' + image.id + ', group ' + group.id + (shared?' shared patches':' strongest patches')}>
      <image href={image.src} width={g.original[0]} height={g.original[1]}/>
      {patches.map(p => <rect key={p} {...patchBox(image, p)}
        fill={color} fillOpacity={p === chosen ? .25 : .08}
        stroke={p === chosen ? '#fff' : color} strokeWidth={sx * (p === chosen ? 2.4 : 1.5)}
        tabIndex={0} role="button" aria-label={'Select group ' + group.id + ' patch ' + p}
        onClick={() => setChosen(p)}
        onKeyDown={event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault(); setChosen(p);
          }
        }}>
        <title>Patch {p} · response {responses.norms[p].toFixed(3)}</title>
      </rect>)}
    </svg>
    {shared&&<label className="shared-patch-select">Shared patch ({patches.length}) <select aria-label={'Shared patch for group '+group.id} value={chosen} onChange={event=>setChosen(Number(event.target.value))}>{patches.map(p=><option key={p} value={p}>Patch {p}</option>)}</select></label>}
    <div className="connection-patches">
      {previewPatches.map(p => {
        const box = patchBox(image, p);
        return <button key={p} className={p === chosen ? 'chosen' : ''}
          aria-label={'Group ' + group.id + ' patch ' + p}
          aria-pressed={p === chosen} data-patch={p}
          onClick={() => setChosen(p)}>
          <svg viewBox={[box.x - box.width, box.y - box.height, box.width * 3, box.height * 3].join(' ')}>
            <image href={image.src} width={g.original[0]} height={g.original[1]}/>
            <rect {...box} stroke={color} strokeWidth={sx} fill="none"/>
          </svg>
          <span>Patch {p}</span>
        </button>;
      })}
    </div>
    <div className="connection-patch-value">Patch {chosen} response <strong>{responses.norms[chosen].toFixed(3)}</strong></div>
    <button className="connection-inspect" onClick={() => onInspect(chosen)}>
      Inspect group {group.id} on this image
    </button>
  </section>;
}

export default function ConnectionEvidence(props: Props) {
  const {link, groups, sharedIndices, referenceIds, images, method, loadGroup} = props;
  const sharedIds = sharedIndices.map(index => referenceIds[index]);
  const [imageId, setImageId] = useState(() =>
    sharedIds.includes(props.currentImageId) ? props.currentImageId : sharedIds[0]
  );
  const [selectedPatch,setSelectedPatch]=useState<number|null>(null);
  const [responses, setResponses] = useState<Responses[] | null>(null);
  const [error, setError] = useState('');
  const dialog = useRef<HTMLElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    let active = true;
    Promise.all([loadGroup(method, link.source), loadGroup(method, link.target)])
      .then(data => {if (active) setResponses(data);})
      .catch(() => {if (active) setError('The recorded patch responses could not be loaded. Please reload and try again.');});
    return () => {active = false;};
  }, [method, link.source, link.target, loadGroup]);
  useEffect(() => {
    const prior = document.activeElement as HTMLElement | null;
    close.current?.focus();
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {document.body.style.overflow = originalOverflow; prior?.focus();};
  }, []);
  const selected = sharedIds.includes(imageId) ? imageId : sharedIds[0];
  const sharedPosition = sharedIds.indexOf(selected);
  const imageIndex = images.findIndex(image => image.id === selected);
  const image = images[imageIndex];
  const referenceIndex = referenceIds.indexOf(selected);
  const shared=props.samePatch&&responses ? sharedPatchIndices(responseFrame(responses[0],imageIndex)?.norms,responseFrame(responses[1],imageIndex)?.norms,groups[link.source].q99,groups[link.target].q99,props.cutoff):undefined;
  const chooseImage=(id:number)=>{setImageId(id);setSelectedPatch(null)};
  return <div className="modal-backdrop connection-backdrop" onClick={props.onClose}>
    <article ref={dialog} className="connection-dialog" role="dialog" aria-modal="true"
      aria-label="Co-activation evidence" onClick={event => event.stopPropagation()}
      onKeyDown={event => {
        if (event.key === 'Escape') {event.stopPropagation(); props.onClose();}
        if (event.key === 'Tab') {
          const focusable = [...dialog.current!.querySelectorAll<HTMLElement>('button:not(:disabled), [tabindex="0"], select')];
          const first = focusable[0], last = focusable.at(-1);
          if (event.shiftKey && document.activeElement === first) {event.preventDefault(); last?.focus();}
          if (!event.shiftKey && document.activeElement === last) {event.preventDefault(); first?.focus();}
        }
      }}>
      <button ref={close} className="modal-close icon-btn" onClick={props.onClose}
        aria-label="Close connection evidence"><X size={20}/></button>
      <div className="eyebrow">{props.methodName} · RECORDED RESPONSES</div>
      <h2>Group {link.source} ↔ Group {link.target}</h2>
      <p className="connection-summary">{sharedIds.length} {props.samePatch?'images with shared patches':'images with both active'} · {link.either} with either active · {link.samples} reference images · {(link.weight * 100).toFixed(1)}% overlap</p>
      <p className="connection-description">Both groups reach the {(props.cutoff * 100).toFixed(0)}% cutoff relative to their reference scales on these images. {props.samePatch?'Both panels highlight the same patch locations where both responses also reach the cutoff. Selecting a shared patch updates both panels.':"Each panel highlights its group's strongest patches, which can occupy different regions."}</p>
      {!sharedIds.length ? <p>No recorded images meet this cutoff for both groups.</p> : <>
        <div className="connection-navigation">
          <button disabled={sharedPosition === 0} aria-label="Previous shared image"
            onClick={() => chooseImage(sharedIds[sharedPosition - 1])}><ChevronLeft size={18}/></button>
          <strong data-image-id={selected}>Image {selected}</strong>
          <span>{sharedPosition + 1} of {sharedIds.length}</span>
          <button disabled={sharedPosition === sharedIds.length - 1} aria-label="Next shared image"
            onClick={() => chooseImage(sharedIds[sharedPosition + 1])}><ChevronRight size={18}/></button>
        </div>
        {error ? <p role="alert">{error}</p> : !responses ? <p role="status">Loading measured patch responses…</p> :
          image && <div className="connection-response-grid">
            {[link.source, link.target].map((group, i) => <ResponseView
              shared={shared} selectedPatch={selectedPatch} onSharedPatch={setSelectedPatch}
              key={group + '-' + selected} image={image} group={groups[group]}
              responses={responseFrame(responses[i], imageIndex)}
              pooled={props.rows[referenceIndex][group]}
              color={i === 0 ? '#0da785' : '#8a56ce'}
              onInspect={patch => props.onInspect(group, selected, patch)}/>)}
          </div>}
        <div className="connection-image-list" aria-label="Shared reference images">
          {sharedIds.map(id => {
            const record = images.find(image => image.id === id);
            return record && <button key={id} aria-label={'View shared image ' + id}
              aria-pressed={selected === id} onClick={() => chooseImage(id)}>
              <img src={record.src} alt="" loading="lazy"/>
              <span>{id}</span>
            </button>;
          })}
        </div>
      </>}
    </article>
  </div>;
}
