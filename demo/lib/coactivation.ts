export type CoactivationLink = {
  source: number;
  target: number;
  weight: number;
  joint: number;
  either: number;
  samples: number;
};

type ActivityManifest = {
  imageIds: number[];
  playbackIds?: number[];
  recordedImages: number;
  activity: Record<string, (number | null)[][]>;
};

const measured = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0;

export function coactivationReference(manifest: ActivityManifest) {
  const indices = new Map(manifest.imageIds.map((id, i) => [id, i]));
  const requested = [...new Set(
    manifest.playbackIds ?? manifest.imageIds.slice(0, manifest.recordedImages)
  )];
  const methods = Object.keys(manifest.activity);
  const imageIds = requested.filter(id => {
    const index = indices.get(id);
    return index !== undefined && methods.length > 0 && methods.every(method => {
      const row = manifest.activity[method][index];
      return row?.length === 75 && row.every(measured);
    });
  });
  const rows = Object.fromEntries(methods.map(method => [
    method,
    imageIds.map(id => manifest.activity[method][indices.get(id)!] as number[])
  ]));
  return {imageIds, rows};
}

export function measureCoactivation(
  rows: (number | null)[][],
  scales: number[],
  cutoff: number
): CoactivationLink[] {
  if (!Number.isFinite(cutoff) || cutoff <= 0) {
    throw new Error("The activation cutoff must be positive.");
  }
  if (!scales.length || scales.some(scale => !Number.isFinite(scale) || scale <= 0)) {
    throw new Error("Each group needs a positive reference scale.");
  }
  const complete = rows.filter(row =>
    row.length === scales.length && row.every(measured)
  ) as number[][];
  const firing = complete.map(row =>
    row.map((value, group) => value / scales[group] >= cutoff)
  );
  const links: CoactivationLink[] = [];
  for (let source = 0; source < scales.length; source++) {
    for (let target = source + 1; target < scales.length; target++) {
      let joint = 0;
      let either = 0;
      for (const row of firing) {
        if (row[source] || row[target]) either++;
        if (row[source] && row[target]) joint++;
      }
      if (joint) links.push({
        source, target, joint, either,
        weight: joint / either,
        samples: complete.length
      });
    }
  }
  return links.sort((a, b) =>
    b.weight - a.weight || b.joint - a.joint ||
    a.source - b.source || a.target - b.target
  );
}

export function visibleCoactivation(
  links: CoactivationLink[],
  minimum: number,
  current: boolean[] | null
) {
  return links.filter(link =>
    link.joint >= 3 && link.weight >= minimum &&
    (!current || (current[link.source] && current[link.target]))
  );
}

export function coactiveRowIndices(
  rows: (number | null)[][],
  scales: number[],
  cutoff: number,
  source: number,
  target: number
) {
  if (source === target || !Number.isInteger(source) || !Number.isInteger(target) ||
      source < 0 || target < 0 || source >= scales.length || target >= scales.length ||
      !Number.isFinite(cutoff) || cutoff <= 0) return [];
  if (![scales[source], scales[target]].every(value => Number.isFinite(value) && value > 0)) return [];
  return rows.flatMap((row, index) =>
    row.length === scales.length && row.every(measured) &&
    (row[source] as number) / scales[source] >= cutoff &&
    (row[target] as number) / scales[target] >= cutoff ? [index] : []
  );
}
