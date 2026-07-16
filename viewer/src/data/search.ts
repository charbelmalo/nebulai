/** Pure keyword search over the resident label column. Case-insensitive
 *  substring — honest and predictable for token vocabularies (no stemming,
 *  no fuzzy ranking that would imply semantics the tool doesn't have).
 *  Matching stays O(n) over ~50K labels: fine on every keystroke. */

export interface SearchResults {
  /** matching point ids, ascending (label order = BPE order) */
  matchIds: Uint32Array;
  /** clusterId → matching point ids (clusterId -1 = noise) */
  byCluster: Map<number, number[]>;
  total: number;
}

/** null on an empty/whitespace query — "no query" and "no matches" must stay
 *  distinguishable so the UI can be honest about which one it's showing. */
export function searchLabels(
  labels: string[],
  clusterId: Int32Array,
  text: string,
): SearchResults | null {
  const needle = text.trim().toLowerCase();
  if (!needle) return null;

  const ids: number[] = [];
  const byCluster = new Map<number, number[]>();
  for (let i = 0; i < labels.length; i++) {
    if (!labels[i]!.toLowerCase().includes(needle)) continue;
    ids.push(i);
    const cid = clusterId[i]!;
    const group = byCluster.get(cid);
    if (group) group.push(i);
    else byCluster.set(cid, [i]);
  }
  return { matchIds: Uint32Array.from(ids), byCluster, total: ids.length };
}
