/** atlas/evidence.ts — the words every unit surface uses: the evidence
 *  header, the results table, the inspector and the Learn lesson all read a
 *  map's metadata through these helpers, so "what a point is", "who labelled
 *  it" and "unclustered" cannot be phrased two different ways.
 *
 *  Pure functions of the dataset; no store, no DOM. */

import type { Dataset } from "../../data/loader";
import type { SourceUnit } from "../../data/columns";
import {
  classifyRepresentation,
  datasetLabel,
  modelDisplayName,
  parseCuration,
  REPRESENTATION_COPY,
  type Representation,
} from "../../data/representation";

export type Meta = Record<string, unknown>;

export const metaOf = (ds: Dataset): Meta => ds.columns.meta as Meta;
export const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

export function representationOf(ds: Dataset): Representation {
  return classifyRepresentation(metaOf(ds));
}

const LABEL_SOURCES: Record<string, string> = {
  neuronpedia: "Neuronpedia",
  none: "None — units are unlabelled",
};

/** Who wrote the point labels, in words. */
export function labelSource(meta: Meta): string {
  const raw = str(meta.labels_source);
  if (!raw) return "Not recorded";
  return LABEL_SOURCES[raw] ?? raw;
}

/** A map whose points carry placeholder text rather than labels. */
export function isUnlabelled(meta: Meta): boolean {
  return meta.labels_source === "none";
}

/** The name a row goes by in lists: its label, or — on an unlabelled map —
 *  its unit reference, because "neuron 0 (unlabeled)" is a placeholder, not
 *  evidence about the neuron. */
export function unitTitle(meta: Meta, u: SourceUnit): string {
  if (isUnlabelled(meta)) {
    return u.unitIndex !== null ? `Unit #${u.unitIndex} (unlabelled)` : "Unlabelled unit";
  }
  return u.label || "(empty label)";
}

/** `sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)` → its kind
 *  word only, for a narrow column; the full string is shown in the inspector. */
export function unitKindShort(kind: string | null): string {
  if (!kind) return "unit";
  const i = kind.indexOf("(");
  return (i < 0 ? kind : kind.slice(0, i)).replace(/_/g, " ");
}

export function clusterTitle(ds: Dataset, cid: number): string {
  if (cid < 0) return "Unclustered";
  return ds.columns.clusters.find((c) => c.id === cid)?.title ?? `Cluster ${cid}`;
}

export const fmtInt = (n: number) => n.toLocaleString("en-US");

/** The noise share as the header shows it: "49.46%", from the raw fraction. */
export function pct(fraction: number, digits = 2): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** "4,096 of 24,576 SAE decoder directions" — or the plain count when the map
 *  records no curation. */
export function countLine(ds: Dataset): string {
  const meta = metaOf(ds);
  const rep = representationOf(ds);
  const noun = REPRESENTATION_COPY[rep].pluralNoun;
  const cur = parseCuration(meta.curation);
  const n = ds.columns.count;
  if (cur && cur.kept === n && cur.kept < cur.total) return `${fmtInt(cur.kept)} of ${fmtInt(cur.total)} ${noun}`;
  return `${fmtInt(n)} ${noun}`;
}

export function mapTitle(ds: Dataset, datasetId: string): string {
  return datasetLabel(metaOf(ds), datasetId);
}

export function modelLine(meta: Meta): string {
  const id = str(meta.model);
  if (!id) return "Not recorded";
  const name = modelDisplayName(id);
  return name === id ? id : `${name} (${id})`;
}

export const shortSha = (sha: string | null | undefined) => (sha ? sha.slice(0, 12) : "—");
