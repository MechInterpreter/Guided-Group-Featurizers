'use client';

import {useMemo} from 'react';
import type {CoactivationLink} from '../lib/coactivation';

type Group = {id: number; theme: string};
type Props = {
  samePatch:boolean;
  onSamePatch:(value:boolean)=>void;
  patchLoading:boolean;
  patchError:string;
  patchCoverage:number;
  onReferenceImage:()=>void;
  enabled: boolean;
  onEnabled: (value: boolean) => void;
  currentOnly: boolean;
  onCurrentOnly: (value: boolean) => void;
  minimum: number;
  onMinimum: (value: number) => void;
  samples: number;
  candidates: CoactivationLink[];
  rendered: number;
  group: number;
  groups: Group[];
  onGroup: (id: number) => void;
  onConnection: (source: number, target: number) => void;
  connections: CoactivationLink[];
};

export default function CoactivationPanel(props: Props) {
  const {enabled, currentOnly, minimum, samples, candidates, rendered, group, groups} = props;
  const related = useMemo(() => candidates.filter(link =>
    link.source === group || link.target === group
  ).slice(0, 5), [candidates, group]);
  return <section className="coactivation-panel" aria-label="Group co-activation">
    <div className="coactivation-heading">
      <label><input type="checkbox" checked={enabled}
        onChange={event => props.onEnabled(event.target.checked)}/>
        Co-activation links
      </label>
      <span>{enabled ? rendered : 0} shown · {samples} reference images</span>
    </div>
    <div className="coactivation-controls">
      <label className="connection-scope">Connection scope <select aria-label="Connection scope" value={props.samePatch?'patch':'image'} onChange={event=>props.onSamePatch(event.target.value==='patch')}><option value="image">Same image</option><option value="patch">Same patch</option></select></label>
      <label><input type="checkbox" checked={currentOnly}
        onChange={event => props.onCurrentOnly(event.target.checked)}/>
        Only groups firing on this image
      </label>
      <label className="overlap-control">Minimum overlap <strong>{Math.round(minimum * 100)}%</strong>
        <input type="range" min="5" max="100" step="5"
          aria-label="Minimum co-activation overlap" value={Math.round(minimum * 100)}
          onChange={event => props.onMinimum(Number(event.target.value) / 100)}/>
      </label>
    </div>
    <p className="coactivation-explanation">{props.samePatch?'Same-patch overlap counts images where both groups fire and share at least one patch above the cutoff, divided by images where either group fires.':'Overlap is the share of images where both groups fire among images where either fires.'} Each link needs at least 3 shared images.</p>
    {props.patchLoading&&<p role="status">Loading verified patch responses…</p>}
    {props.patchError&&<p role="alert">{props.patchError}</p>}
    {props.samePatch&&currentOnly&&<p className="patch-coverage">Patch maps available for {props.patchCoverage}/75 groups on this image. {props.patchCoverage<75&&<>Unknown pairs are omitted. <button onClick={props.onReferenceImage}>Use a reference image</button></>}</p>}
    <div className="coactivation-related">
      <h4>Group {group} connections</h4>
      {related.length ? related.map(link => {
        const other = link.source === group ? link.target : link.source;
        return <div key={other} className="coactivation-row"><button onClick={() => props.onGroup(other)}
          aria-label={'Inspect co-active group ' + other}
          title={groups[other]?.theme}>
          <span>G{String(other).padStart(2, '0')} <small>{groups[other]?.theme}</small></span>
          <span><strong>{Math.round(link.weight * 100)}%</strong>
            <small>{link.joint}/{link.either} shared / either</small></span>
        </button><button className="connection-images-button"
          aria-label={'View shared images for groups ' + link.source + ' and ' + link.target}
          onClick={() => props.onConnection(link.source, link.target)}>Images</button></div>;
      }) : <p>No connections meet these settings for this group.</p>}
    </div>
    {props.connections.length > 0 && <details className="connection-index">
      <summary>Browse displayed connections</summary>
      <div>{props.connections.map(link => <button key={link.source + '-' + link.target}
        onClick={() => props.onConnection(link.source, link.target)}
        aria-label={'View shared images for groups ' + link.source + ' and ' + link.target}>
        <span>G{link.source} ↔ G{link.target}</span><span>{link.joint} shared images</span>
      </button>)}</div>
    </details>}
    <p className="coactivation-note">{currentOnly ? 'Lines update with the current image. ' : 'Lines show recorded associations. '}
      Up to 60 qualifying links, with the selected group first. Click a line or Images to see the shared images and both groups' patches.</p>
  </section>;
}

