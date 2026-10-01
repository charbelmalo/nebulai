/** representation.ts — what one point of an exported map IS, decided from the
 *  artifact's own metadata and never from its dataset id or display label.
 *
 *  The plan's evidence rules (NEBULAI-PLAN §5) say the same map geometry means
 *  different things depending on where the vectors came from: a row of a weight
 *  matrix, an SAE decoder direction, an MLP write direction, or text embedded
 *  by a THIRD-PARTY model that is not the named model at all. Every surface that
 *  explains a point — the evidence header, the inspector, the Learn lesson and
 *  the v1 finding record — asks this one adapter, so the explanation cannot
 *  drift between them.
 *
 *  The raw unit string always travels alongside the class. Classification is a
 *  reading aid, not a replacement: an unrecognised string becomes `unknown`
 *  and is shown verbatim, rather than being guessed into the nearest bucket.
 *
 *  Pure module: no DOM, no fetch, safe for node unit tests and the packager. */

export const REPRESENTATIONS = [
  "token_embedding",
  "token_unembedding",
  "sae_decoder",
  "mlp_neuron",
  "external_text_embedding",
  "unknown",
] as const;

export type Representation = (typeof REPRESENTATIONS)[number];

export function isRepresentation(v: unknown): v is Representation {
  return typeof v === "string" && (REPRESENTATIONS as readonly string[]).includes(v);
}

/** Minimal shape this adapter reads — a NebulaiMeta or a DatasetEntry fits. */
export interface RepresentationSource {
  unit?: unknown;
  weight_key?: unknown;
  geometry?: unknown;
}

const UNEMBED_KEY = /(^|[._])(lm_head|W_U|unembed)/i;

/** Classify a map's unit string (plus the weight key when a token map records
 *  one). Unknown or absent → `unknown`, never a guess. */
export function classifyRepresentation(src: RepresentationSource | null | undefined): Representation {
  const unit = typeof src?.unit === "string" ? src.unit.trim() : "";
  if (!unit) return "unknown";
  if (unit === "token_unembedding" || unit.startsWith("token_unembedding(")) {
    return "token_unembedding";
  }
  if (unit === "token_embedding" || unit.startsWith("token_embedding(")) {
    const key = typeof src?.weight_key === "string" ? src.weight_key : "";
    return UNEMBED_KEY.test(key) ? "token_unembedding" : "token_embedding";
  }
  if (unit.startsWith("sae_decoder(")) return "sae_decoder";
  if (unit.startsWith("mlp_neuron(")) return "mlp_neuron";
  // Both the vocabulary proxy maps and the concept probes are embedded by an
  // external text-embedding model (mxbai, MiniLM …). Their geometry is that
  // model's, not the named source model's — the one mistake the plan names
  // explicitly, so both spellings land here.
  if (unit.startsWith("api_text_embedding(") || unit.startsWith("probe_concept(")) {
    return "external_text_embedding";
  }
  return "unknown";
}

export interface RepresentationCopy {
  /** short noun for a single point: "SAE decoder direction" */
  pointNoun: string;
  /** plural, for headers: "SAE decoder directions" */
  pluralNoun: string;
  /** what one point is, in one sentence */
  meaning: string;
  /** the interpretation limit that must be visible next to the evidence */
  limit: string;
}

export const REPRESENTATION_COPY: Record<Representation, RepresentationCopy> = {
  token_embedding: {
    pointNoun: "token embedding row",
    pluralNoun: "token embedding rows",
    meaning: "A row of the stated model's token-embedding weight matrix.",
    limit:
      "Proximity includes token form and frequency effects; it is not an observed activation or behavior.",
  },
  token_unembedding: {
    pointNoun: "token unembedding row",
    pluralNoun: "token unembedding rows",
    meaning: "A row of the stated model's unembedding (output) weight matrix.",
    limit:
      "Proximity includes token form and frequency effects; it is not an observed activation or behavior.",
  },
  sae_decoder: {
    pointNoun: "SAE decoder direction",
    pluralNoun: "SAE decoder directions",
    meaning: "A decoder direction of the named sparse autoencoder at the named hook and layer.",
    limit:
      "Decoder-vector geometry and descriptive labels do not establish activation on an input or causal control.",
  },
  mlp_neuron: {
    pointNoun: "MLP neuron write direction",
    pluralNoun: "MLP neuron write directions",
    meaning: "The exported write direction of one MLP neuron at the stated layer.",
    limit:
      "Unlabelled neurons stay unlabelled; neighbourhood in weight space is not a measured feature meaning.",
  },
  external_text_embedding: {
    pointNoun: "external text embedding",
    pluralNoun: "external text embeddings",
    meaning: "Text represented by the named external embedding model.",
    limit:
      "External text-embedding space: this is not the named source model's internal geometry.",
  },
  unknown: {
    pointNoun: "exported unit",
    pluralNoun: "exported units",
    meaning: "Unclassified representation — the raw unit kind is shown as exported.",
    limit:
      "The representation is not recorded in a form this viewer recognises; claims that need it are disabled.",
  },
};

/** Parse `first_4096_of_24576` style curation into numbers. null when the
 *  string says something else — shown raw in that case, never reinterpreted. */
export function parseCuration(c: unknown): { kept: number; total: number } | null {
  if (typeof c !== "string") return null;
  const m = /^first_(\d+)_of_(\d+)$/.exec(c);
  if (!m) return null;
  const kept = Number(m[1]);
  const total = Number(m[2]);
  return Number.isSafeInteger(kept) && Number.isSafeInteger(total) && kept <= total
    ? { kept, total }
    : null;
}

/** Display names for the model ids the published maps use. Anything else is
 *  shown exactly as exported — never guessed at. */
const MODEL_NAMES: Record<string, string> = {
  gpt2: "GPT-2",
  "gpt2-small": "GPT-2 Small",
  "gpt2-medium": "GPT-2 Medium",
  distilgpt2: "DistilGPT-2",
  "EleutherAI/pythia-70m": "Pythia 70M",
  "HuggingFaceTB/SmolLM2-135M": "SmolLM2 135M",
  "google/gemma-4-26B-A4B-it": "Gemma 4 26B-A4B Instruct",
  "inclusionai/ling-2.6-flash": "Ling 2.6 Flash",
  "meta-models/Muse-Glimmer-30B": "Muse Glimmer 30B",
  "mistralai/Mistral-Nemo-Instruct-2407": "Mistral Nemo Instruct 2407",
  "Xenova/claude-tokenizer": "Claude tokenizer",
};

export function modelDisplayName(model: string): string {
  return MODEL_NAMES[model] ?? model;
}

/** The one-line name of a map: model · what a point is · layer.
 *  e.g. "GPT-2 Small · SAE decoder directions · Layer 8". Shared by the
 *  release packager (manifest `label`) and the Atlas evidence header. */
export function datasetLabel(meta: {
  model?: unknown;
  unit?: unknown;
  weight_key?: unknown;
  geometry?: unknown;
  layer?: unknown;
}, fallbackId = ""): string {
  const model = typeof meta.model === "string" && meta.model ? meta.model : fallbackId;
  const rep = classifyRepresentation(meta as RepresentationSource);
  const noun = REPRESENTATION_COPY[rep].pluralNoun;
  let what = noun.charAt(0).toUpperCase() + noun.slice(1);
  // an external embedding's geometry belongs to the EMBEDDER, not the model
  // whose vocabulary was embedded — name it, or the label credits the wrong one
  if (rep === "external_text_embedding" && typeof meta.unit === "string") {
    const inner = /\(([^()]+)\)\s*$/.exec(meta.unit)?.[1]?.trim();
    if (inner) what += ` (${inner})`;
  }
  const parts = [modelDisplayName(model), what];
  if (typeof meta.layer === "number" && Number.isInteger(meta.layer)) parts.push(`Layer ${meta.layer}`);
  return parts.filter(Boolean).join(" · ");
}

/** A chooser card's reading of one published map, from its index entry alone
 *  (no map is fetched): a human title, what a point is, the layer when the
 *  unit names one, and the family the choosers group by. The raw id always
 *  travels with it and is shown as the caption. */
export interface DatasetCard {
  title: string;
  subtitle: string;
  layer: number | null;
  family: string;
}

const GPT2_FAMILY = new Set(["gpt2", "gpt2-small", "gpt2-medium", "distilgpt2"]);

export function describeDataset(d: { id: string; model?: unknown; unit?: unknown }): DatasetCard {
  const model = typeof d.model === "string" && d.model ? d.model : d.id;
  const unit = typeof d.unit === "string" ? d.unit : "";
  const rep = classifyRepresentation({ unit });
  // the layer is the digits after a block/layer segment of the unit's hook
  const m = /(?:^|[(,\s.])(?:layers?|h|blocks)\.(\d+)\./.exec(unit);
  const layer = m ? Number(m[1]) : null;
  if (unit.startsWith("probe_concept(")) {
    const embedder = /\(([^()]+)\)\s*$/.exec(unit)?.[1]?.trim();
    return {
      title: `Concept probe: ${model}`,
      subtitle: `Probe texts embedded by ${embedder ?? "an external model"}`,
      layer: null,
      family: "Concept probes",
    };
  }
  const noun = REPRESENTATION_COPY[rep].pluralNoun;
  let subtitle = noun.charAt(0).toUpperCase() + noun.slice(1);
  if (rep === "external_text_embedding") {
    const inner = /\(([^()]+)\)\s*$/.exec(unit)?.[1]?.trim();
    if (inner) subtitle = `Vocabulary embedded by ${inner}`;
  }
  if (layer !== null) subtitle += ` · layer ${layer}`;
  const family = GPT2_FAMILY.has(model)
    ? "GPT-2 family"
    : model === "HuggingFaceTB/SmolLM2-135M"
      ? "SmolLM2"
      : "Other models";
  return { title: modelDisplayName(model), subtitle, layer, family };
}
