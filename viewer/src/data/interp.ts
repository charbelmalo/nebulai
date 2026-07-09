/** Loader + types for the interp bundles (out/<model>/interp/*.json) produced by
 *  `nebulai interp`. These carry the REAL model internals the mechanistic-
 *  interpretability drivers render: weight SVD spectra, positional-embedding
 *  DFT, and per-prompt forward traces (attention, residual norms, logit lens,
 *  next-token distributions). Every bundle's `meta.quantity`/`meta.formula`
 *  states exactly what the numbers are — surfaced in-view and on /guide. */

/** One weight matrix's singular-value spectrum + honest rank summaries. */
export interface SpectrumMatrix {
  name: string;
  kind: "embed" | "pos" | "attn_qkv" | "attn_out" | "mlp_in" | "mlp_out";
  layer: number | null;
  shape: [number, number];
  n_sv: number;
  singular_values: number[]; // top ≤256, descending
  sigma_max: number;
  sigma_min: number;
  stable_rank: number; // ||W||_F^2 / sigma_max^2
  effective_rank: number; // exp(entropy of normalized σ)
  condition: number;
}

export interface WeightsBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
    d: number;
  };
  matrices: SpectrumMatrix[];
}

export interface FourierBundle {
  meta: { model: string; created: string; quantity: string; formula: string; n_ctx: number; d: number };
  freqs: number[]; // cycles per context window
  power_mean: number[]; // mean power over dims, per frequency
  per_dim_dominant: number[]; // dominant frequency index per embedding dim
}

/** PCA projection of the token-embedding matrix W_E (#15 Embedding Constellation).
 *  `coords` is a flat [PC1₀, PC2₀, PC1₁, PC2₁, …] array of length 2·n; `z` is
 *  PC3 (hover only). Positions are exact PC scores, `norm` the exact row L2 norm,
 *  `lead_space` a real orthographic property decoded per token. */
export interface EmbedBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    d: number;
    n_tokens: number;
  };
  n: number;
  dims: number;
  explained_variance_ratio: number[]; // per PC
  total_variance: number;
  coords: number[]; // flat 2·n (PC1, PC2)
  z: number[]; // PC3, length n
  norm: number[]; // length n
  lead_space: number[]; // 0/1, length n
  strs: string[]; // length n
}

/** PCA of every MLP neuron's write direction (#6 Neuron Write-Direction Field).
 *  Neurons are stored in layer order — layer = floor(i / meta.d_mlp), neuron
 *  index within the layer = i % meta.d_mlp. `coords` is flat [PC1₀, PC2₀, …];
 *  `norm` is the exact ‖w_out‖₂ per neuron; `top_*`/`bot_*` are the direct-path
 *  logit readout through the tied unembedding (the token each write direction
 *  most promotes / most suppresses — no downstream-layer effects). */
export interface NeuronsBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
    d_mlp: number;
    d: number;
  };
  n: number;
  dims: number;
  explained_variance_ratio: number[]; // per PC
  total_variance: number;
  coords: number[]; // flat 2·n (PC1, PC2)
  z: number[]; // PC3, length n
  norm: number[]; // ‖w_out‖₂, length n
  top_tok: string[]; // most-promoted token per neuron
  top_val: number[]; // its Δlogit per unit activation (direct path)
  bot_tok: string[]; // most-suppressed token per neuron
  bot_val: number[]; // its Δlogit (negative)
}

/** PCA of every SAE feature's decoder direction (#5 SAE Decoder Constellation).
 *  Rows of W_dec from an open SAE release (meta.sae_repo / meta.hook_point);
 *  `log_sparsity` is the release's MEASURED log₁₀ firing fraction per feature
 *  (−10 = clamp floor, dead). Decoder rows are unit-norm by construction, so
 *  `norm` is hover-only proof, never an encoding. `top_*`/`bot_*` are the
 *  direct-path unembedding readout (enters at the hook layer — skips the
 *  remaining blocks, caveat in meta.note). */
export interface SAEBundle {
  meta: {
    model: string;
    created: string;
    sae_repo: string;
    hook_point: string;
    quantity: string;
    formula: string;
    note: string;
    d_sae: number;
    d_in: number;
    l1_coefficient: number | null;
    training_tokens: number | null;
  };
  n: number;
  dims: number;
  explained_variance_ratio: number[]; // per PC
  total_variance: number;
  coords: number[]; // flat 2·n (PC1, PC2)
  z: number[]; // PC3, length n
  norm: number[]; // ‖W_dec[i]‖₂ (≈1), length n
  log_sparsity: number[]; // measured log10 firing fraction, length n
  top_tok: string[];
  top_val: number[];
  bot_tok: string[];
  bot_val: number[];
}

/** [token string, probability] — the honest readout unit for lens/next-token. */
export type LensTopk = [string, number][];

export interface TraceBundle {
  meta: {
    model: string;
    created: string;
    prompt: string;
    n_layer: number;
    n_head: number;
    d: number;
    T: number;
    quantity: string;
    attn_rounding: number;
  };
  tokens: number[];
  token_strs: string[];
  /** (n_layer, n_head, T, T) post-softmax attention, rounded for transport. */
  attn: number[][][][];
  /** (n_layer+1, T) residual-stream L2 norm per layer/position. */
  resid_norm: number[][];
  /** logit-lens top-k at the LAST position, layer 0 (embed) … n_layer. */
  logit_lens_last: { layer: number; topk: LensTopk }[];
  /** (T) final-layer top-1 prediction after each position. */
  final_pred_per_pos: [string, number][];
  /** final-position next-token distribution, top-12. */
  final_topk: LensTopk;
}

export interface InterpIndex {
  meta: { model: string; created: string };
  bundles: string[];
  traces: { slug: string; prompt: string }[];
}

const cache = new Map<string, unknown>();

async function fetchJSON<T>(url: string): Promise<T> {
  const hit = cache.get(url);
  if (hit) return hit as T;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`interp bundle missing: ${url} (${res.status})`);
  const json = (await res.json()) as T;
  cache.set(url, json);
  return json;
}

/** `model` is the dataset dir id (e.g. "gpt2", "EleutherAI__pythia-70m"). */
export function interpBase(model: string, base = "/out"): string {
  return `${base}/${model}/interp`;
}

export const loadInterpIndex = (model: string, base = "/out") =>
  fetchJSON<InterpIndex>(`${interpBase(model, base)}/index.json`);

export const loadWeights = (model: string, base = "/out") =>
  fetchJSON<WeightsBundle>(`${interpBase(model, base)}/weights.json`);

export const loadFourier = (model: string, base = "/out") =>
  fetchJSON<FourierBundle>(`${interpBase(model, base)}/fourier.json`);

export const loadEmbed = (model: string, base = "/out") =>
  fetchJSON<EmbedBundle>(`${interpBase(model, base)}/embed.json`);

export const loadNeurons = (model: string, base = "/out") =>
  fetchJSON<NeuronsBundle>(`${interpBase(model, base)}/neurons.json`);

export const loadSAE = (model: string, base = "/out") =>
  fetchJSON<SAEBundle>(`${interpBase(model, base)}/sae.json`);

export const loadTrace = (model: string, slug: string, base = "/out") =>
  fetchJSON<TraceBundle>(`${interpBase(model, base)}/trace_${slug}.json`);

/** True if this model has interp bundles (feature nav gates on it, honestly —
 *  a model without bundles simply doesn't offer the interp views). */
export async function hasInterp(model: string, base = "/out"): Promise<boolean> {
  try {
    await loadInterpIndex(model, base);
    return true;
  } catch {
    return false;
  }
}
