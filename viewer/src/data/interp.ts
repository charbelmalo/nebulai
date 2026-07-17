/** Loader + types for the interp bundles (out/<model>/interp/*.json) produced by
 *  `nebulai interp`. These carry the REAL model internals the mechanistic-
 *  interpretability drivers render: weight SVD spectra, positional-embedding
 *  DFT, and per-prompt forward traces (attention, residual norms, logit lens,
 *  next-token distributions). Every bundle's `meta.quantity`/`meta.formula`
 *  states exactly what the numbers are — surfaced in-view and on /guide. */

import { DATA_BASE } from "./base";

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

/** One feature row of the SAE piano-roll: its full activation vector over the
 *  prompt's positions, plus identity (release's measured global log-sparsity,
 *  direct-path top token). `max` is the row's own peak — the driver's per-row
 *  scale, always displayed. */
export interface SAEActsFeature {
  id: number;
  log_sparsity: number;
  top_tok: string;
  top_val: number;
  max: number;
  acts: number[]; // length T, exact encoder activations (3 dp)
}

export interface SAEActsTrace {
  slug: string;
  prompt: string;
  token_strs: string[];
  T: number;
  l0: number[]; // features active per position
  cos: number[]; // reconstruction cosine per position (honesty metric)
  features: SAEActsFeature[]; // top-k by peak over positions ≥ 1
  sink_features: SAEActsFeature[]; // position-0 massive-activation band
}

/** SAE encoder activations on the bundled prompts (#5 Firing Piano-Roll).
 *  acts = ReLU((x̄ − b_dec)·W_enc + b_enc) where x̄ is the residual at
 *  meta.hook_point re-centered per position (TransformerLens
 *  center_writing_weights basis — exact; see meta.formula). */
export interface SAEActsBundle {
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
    hook_layer: number;
    top_k: number;
  };
  traces: SAEActsTrace[];
}

/** Per-attention-head fingerprints (#2 Head Fingerprints). All arrays are
 *  length n = n_layer·n_head, layer-major (index = layer·n_head + head).
 *  `copying`/`eig1_*`/`fro_ov`/`sigma_qk` are weight-circuit quantities
 *  (ln_1 gain folded, biases excluded); `prev`/`sink`/`self`/`entropy` are
 *  MEASURED means over the real forward passes named in meta.prompts —
 *  a stated sample (meta.n_rows query rows), not a property of the weights. */
export interface HeadsBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
    n_head: number;
    d_head: number;
    prompts: string[];
    n_rows: number;
  };
  n: number;
  copying: number[]; // Σ Re λ / Σ |λ| of the OV map, ∈ [−1, 1]
  eig1_re: number[]; // largest-|λ| OV eigenvalue, real part
  eig1_im: number[]; // … imaginary part
  fro_ov: number[]; // ‖diag(γ₁)·W_V·W_O‖_F
  sigma_qk: number[]; // σ_max of the (scaled) QK bilinear form
  prev: number[]; // mean attention to previous token
  sink: number[]; // mean attention to first token
  self: number[]; // mean attention to self
  entropy: number[]; // mean normalized attention entropy ∈ [0, 1]
}

/** Every head's full complex OV spectrum (#2b OV Eigenvalue Constellation).
 *  `re`/`im` are flat n·d_head, head-major (layer-major head index), sorted by
 *  descending |λ| within each head. Conjugate-symmetric per head (real matrix);
 *  `copying` re-exported per head so the bundle is self-contained. */
export interface OVEigsBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
    n_head: number;
    d_head: number;
  };
  n: number; // heads = n_layer·n_head
  d_head: number;
  re: number[]; // flat n·d_head
  im: number[]; // flat n·d_head
  copying: number[]; // per head, Σ Re λ / Σ |λ|
}

/** Q/K/V composition scores between every cross-layer head pair (#2c
 *  Composition Web). `q`/`k`/`v` are flat over layer_pairs × h1 × h2
 *  (pair-major, then h1·n_head + h2). Raw Frobenius composition has a
 *  positive floor for unrelated maps: `baseline_mean ± baseline_std` is that
 *  floor, measured over `baseline_n` seeded random factor pairs — scores are
 *  only meaningful relative to it. Same-layer pairs are excluded (parallel
 *  heads cannot compose). */
export interface CompBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
    n_head: number;
    d_head: number;
    baseline_mean: number;
    baseline_std: number;
    baseline_n: number;
  };
  layer_pairs: [number, number][]; // [earlier, later], all i<j
  q: number[]; // flat n_pairs·n_head² — h1 feeds h2's query
  k: number[]; // — h1 feeds h2's key
  v: number[]; // — h1 feeds h2's value
}

/** One prompt's direct logit attribution (#13). `heads` is flat
 *  n_layer·n_head layer-major; `mlp`/`bias` are per layer (bias = the attention
 *  out-projection b_o — it belongs to no head, so it's its own bucket).
 *  Contributions are to the top-1 vs runner-up logit margin through the final
 *  LayerNorm with σ frozen at the forward's actual value, so they are additive:
 *  emb + lnf_bias + Σheads + Σmlp + Σbias = sum_check ≈ margin (the difference
 *  is the measured float32 accumulation error — displayed, never hidden). */
export interface AttribTrace {
  slug: string;
  prompt: string;
  token_strs: string[];
  T: number;
  top1: [string, number, number]; // token, logit, probability
  top2: [string, number, number];
  margin: number; // top1 logit − top2 logit, from the model's real logits
  sum_check: number; // Σ of all exported contributions
  recon_rel: number; // ‖rebuilt stream − resid[-1]‖ / ‖resid[-1]‖
  emb: number;
  lnf_bias: number; // β_f·(W_U[c1]−W_U[c2]) — the final-LN bias term
  heads: number[]; // flat n_layer·n_head, layer-major
  mlp: number[]; // per layer
  bias: number[]; // per layer (attn out-proj b_o)
  attend_tok: string[]; // per head: argmax-attended token at the final row
  attend_w: number[]; // per head: that attention weight
}

export interface AttribBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
    n_head: number;
  };
  traces: AttribTrace[];
}

/** One matched clean/corrupt prompt pair with its full patching grid. */
export interface PatchPair {
  slug: string;
  clean: string;
  corrupt: string;
  clean_strs: string[];
  corrupt_strs: string[];
  T: number;
  /** positions where the two prompts' tokens differ */
  diff_pos: number[];
  /** [token str, logit in its own run, p in its own run, rank in its own run] */
  ans_clean: [string, number, number, number];
  ans_corrupt: [string, number, number, number];
  ld_clean: number;
  ld_corrupt: number;
  /** raw patched logit-diffs, flat (n_layer+1)·T row-major (layer, pos) */
  ld: number[];
  /** normalized recovery r = (ld − ld_corrupt)/(ld_clean − ld_corrupt), same shape */
  r: number[];
}

export interface PatchBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
  };
  pairs: PatchPair[];
}

/** Full T×T attention pattern (seed A) for one top-scoring head. */
export interface InductionPattern {
  layer: number;
  head: number;
  /** flat T·T row-major (from, to), post-softmax, 4 dp */
  attn: number[];
}

export interface InductionBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
    n_head: number;
    period: number;
    T: number;
    seed_a: number;
    seed_b: number;
    /** chance floor: mean uniform attention 1/(t+1) over scored positions */
    floor: number;
    attn_rowsum_drift: number;
  };
  token_strs: string[];
  /** per-head scores, flat n_layer·n_head row-major (layer, head), seed A */
  ind: number[];
  dup: number[];
  prev: number[];
  /** same three scores measured on seed B — stability is data, not a promise */
  ind_b: number[];
  dup_b: number[];
  prev_b: number[];
  patterns: InductionPattern[];
}

/** One multi-head ablation (top induction heads knocked out together). */
export interface AblationCombo {
  label: string;
  sites: [number, number][];
  d_zero: number;
  d_mean: number;
  /** per-position NLL curves, length T−1, indexed by predicted j−1, nats, 4 dp */
  nll_zero: number[];
  nll_mean: number[];
}

export interface AblationBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_layer: number;
    n_head: number;
    period: number;
    T: number;
    seed: number;
    /** induction window: predicted-token indices [first, last], inclusive */
    window: [number, number];
    /** baseline mean NLL over the window (nats) */
    base_window: number;
    /** baseline mean NLL over first-repeat predictions (nats) */
    base_first: number;
    /** max |Δlogit| of the unablated replicated forward vs the baseline */
    ident_drift: number;
    n_forward: number;
  };
  token_strs: string[];
  /** seed-0 induction score per head (same formula as induction.json) */
  ind: number[];
  /** Δ mean window NLL per head, flat n_layer·n_head row-major, nats */
  d_zero: number[];
  d_mean: number[];
  /** baseline per-position NLL, length T−1, indexed by predicted j−1 */
  nll_base: number[];
  /** per-head curves, flat (n_layer·n_head)·(T−1) row-major (head, position) */
  nll_zero: number[];
  nll_mean: number[];
  combos: AblationCombo[];
}

/** #12 Decoder Cosine Web — nearest-neighbor cosine structure of W_dec.
 *  Joined with SAEBundle (labels + measured firing sparsity) in the driver. */
export interface SAEWebBundle {
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
    /** features that are their nearest neighbor's nearest neighbor */
    mutual_count: number;
    /** measured random-pair cosine baseline — the yardstick for "close" */
    baseline: {
      n_pairs: number;
      seed: number;
      mean: number;
      std: number;
      p99: number;
      p999: number;
      max: number;
    };
  };
  nn_idx: number[]; // length d_sae
  nn_cos: number[]; // length d_sae, 4 dp
  mutual: number[]; // 0/1, length d_sae
  pairs: { i: number; j: number; cos: number; mutual: boolean }[];
}

/** One clock face: token rows of the trained W1 projected onto the
 *  orthonormalized (cos 2πka/p, sin 2πka/p) Fourier pair of frequency k. */
export interface GrokClock {
  k: number;
  /** measured phase alignment |mean exp(i(θ_a ∓ 2πka/p))| — selection criterion */
  circ: number;
  /** radius coefficient of variation — how un-circular the ring is */
  radius_cv: number;
  /** hidden units whose dominant frequency is k */
  n_units: number;
  /** this frequency's share of total spectral power */
  power_frac: number;
  /** flat [x0,y0,x1,y1,…], length 2p */
  xy: number[];
}

/** Grokking run of a toy modular-addition MLP — trained from scratch in
 *  numpy, NOT derived from GPT-2 (meta.note says so). */
export interface GrokBundle {
  meta: {
    model: string;
    task: string;
    created: string;
    note: string;
    quantity: string;
    formula: string;
    p: number;
    n_hidden: number;
    train_frac: number;
    n_train: number;
    n_test: number;
    steps_run: number;
    ckpt_every: number;
    /** first checkpoint with train acc > threshold */
    tr100_step: number;
    /** first checkpoint with test acc > threshold — the grok */
    grok_step: number;
    acc_threshold: number;
    purity_init: number;
    /** purity at the last memorized-but-not-generalized checkpoint */
    purity_at_memorized: number;
    purity_final: number;
    top5_mass_final: number;
    init_max_frac: number;
    n_freq: number;
    spread_note: string;
  };
  steps: number[]; // n_ckpt checkpoint steps
  train_acc: number[]; // 4 dp
  test_acc: number[];
  train_loss: number[]; // 6 dp MSE
  test_loss: number[];
  /** median / quartiles of per-unit single-frequency purity per checkpoint */
  purity_med: number[];
  purity_q1: number[];
  purity_q3: number[];
  /** flat n_ckpt × n_freq power fraction per frequency (row-sums ≈ 1) */
  fpower: number[];
  /** per hidden unit: dominant non-DC frequency of the final network */
  unit_freq: number[];
  unit_frac: number[];
  clocks: GrokClock[];
  n_ckpt: number;
}

/** #22 Direction Compass — per SAE feature, exact max cosine of its decoder
 *  direction against ALL MLP-neuron write directions (c_proj rows) and ALL
 *  token embeddings (W_E rows), plus a measured random-direction baseline. */
export interface CompassBundle {
  meta: {
    model: string;
    created: string;
    sae_repo: string;
    hook_point: string;
    quantity: string;
    formula: string;
    note: string;
    d_sae: number;
    d: number;
    n_neurons: number;
    n_tokens: number;
    d_mlp: number;
    /** fraction of best-neuron matches in layers 0–7 (before the hook) */
    upstream_frac: number;
    baseline: {
      n_dirs: number;
      seed: number;
      neuron: { mean: number; p99: number; max: number };
      token: { mean: number; p99: number; max: number };
    };
  };
  /** (d_sae) max signed cos vs neuron write directions, 4 dp */
  nc: number[];
  /** (d_sae) flat best-neuron index: layer = i // d_mlp, unit = i % d_mlp */
  ni: number[];
  /** (d_sae) max signed cos vs token embeddings, 4 dp */
  tc: number[];
  /** (d_sae) best-token id */
  ti: number[];
  /** (d_sae) index into tok_strs for the best token's string */
  ti_u: number[];
  tok_strs: string[];
  /** (n_layer) how many features' best neuron lives in each layer */
  layer_counts: number[];
  exemplars: { f: number; kind: "neuron" | "token"; cos: number }[];
}

/** One co-firing chip: an exemplar pair with its exact counts. */
export interface CofireChip {
  i: number;
  j: number;
  /** exact joint count */
  c: number;
  /** expected count under independence, 2 dp */
  e: number;
  lift: number;
  /** decoder-direction cosine, 4 dp */
  cos: number;
  /** most common token at co-firing positions (null when c = 0) */
  tok: string | null;
  /** that token's share of co-firing positions, 3 dp */
  share: number;
  /** joint count after a seeded shuffle of one feature's rows */
  shuf: number;
}

export interface CofireBundle {
  meta: {
    model: string;
    created: string;
    sae_repo: string;
    hook_point: string;
    corpus: {
      title: string;
      source: string;
      sha256: string;
      n_chars: number;
      n_tokens: number;
      window: number;
      n_windows: number;
      /** counted positions (position 0 of each window dropped) */
      n_pos: number;
    };
    quantity: string;
    formula: string;
    note: string;
    d_sae: number;
    layer: number;
    /** support floor: pairs considered only when c >= this */
    min_count: number;
    /** export = top n_pairs by Dunning's G² among the n_support support pairs */
    selection: string;
    /** smallest exported G² — the truncation boundary */
    g2_min: number;
    /** total pairs with c >= min_count (before truncation) */
    n_support: number;
    /** Pearson r(decoder cos, PMI) over ALL support pairs — the global answer */
    pearson_cos_pmi: number;
    /** features that fired at least once */
    n_active: number;
    /** features with marginal count >= min_count */
    n_eligible: number;
    n_pairs: number;
    l0_mean: number;
    l0_median: number;
    recon_cos_mean: number;
    recon_cos_n: number;
    shuffle: {
      seed: number;
      n_sampled: number;
      shuf_sum: number;
      e_sum: number;
      /** aggregate shuffled/expected — ≈1 confirms the independence yardstick */
      agg_ratio: number;
    };
  };
  /** counted corpus positions */
  N: number;
  /** (n_pairs) feature ids, sorted by c desc */
  pi: number[];
  pj: number[];
  /** (n_pairs) exact joint counts */
  c: number[];
  /** (n_pairs) decoder cosine, 4 dp */
  cos: number[];
  /** (n_pairs) index into ctok_strs: top co-firing token */
  tt: number[];
  /** (n_pairs) top co-token's share of co-firing positions, 3 dp */
  tshare: number[];
  ctok_strs: string[];
  /** marginal counts for every feature appearing in the export */
  f_ids: number[];
  f_n: number[];
  chips: { assoc: CofireChip[]; count: CofireChip[]; avoid: CofireChip[] };
}

/** One occlusion mode's per-position results (arrays indexed by position). */
export interface OcclusionMode {
  /** drop in the baseline top-1's log-prob when this position is occluded, nats, 4 dp */
  drop_lp: number[];
  /** same difference in raw logits */
  drop_logit: number[];
  /** the occluded run's own top-1 per position: [token string, probability] */
  new_top: [string, number][];
}

export interface OcclusionPrompt {
  slug: string;
  prompt: string;
  token_strs: string[];
  T: number;
  /** baseline top-1 at the final position: [token string, probability, logit] */
  top1: [string, number, number];
  sub: OcclusionMode;
  del: OcclusionMode;
}

export interface OcclusionBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    n_forward: number;
    /** max |Δlogit| of del-final-token vs the baseline's T−2 row (causality) */
    causal_drift: number;
  };
  prompts: OcclusionPrompt[];
}

/** One layer's lens-fidelity stats over the held-out eval positions. */
export interface TunedCurvePoint {
  /** mean KL(p_final ‖ p_lens), bits */
  mean: number;
  p25: number;
  p50: number;
  p75: number;
  /** fraction of eval positions where lens top-1 == final top-1 */
  agree: number;
}

/** One grid cell: [tok_strs index of the lens top-1, its probability (3 dp),
 *  KL(p_final ‖ p_lens) at this position in bits (3 dp)]. */
export type TunedCell = [number, number, number];

export interface TunedGrid {
  slug: string;
  prompt: string;
  token_strs: string[];
  T: number;
  /** final top-1 per position: [tok_strs index, probability] */
  final_top: [number, number][];
  /** (n_layer+1, T) cells — row L = stream entering block L; row 12 = final */
  logit: TunedCell[][];
  tuned: TunedCell[][];
}

export interface TunedBundle {
  meta: {
    model: string;
    created: string;
    quantity: string;
    formula: string;
    note: string;
    corpus: {
      title: string;
      source: string;
      sha256: string;
      n_tokens: number;
      window: number;
      n_windows: number;
    };
    split: string;
    n_train_pos: number;
    n_eval_pos: number;
    eval_windows: number;
    eval_seed: number;
    solve_resid: number;
    /** train R² of the affine translator per layer 0..11 */
    r2_train: number[];
    kl_direction: string;
  };
  n_layer: number;
  /** curves indexed 0..n_layer (12 = final residual, KL exactly 0) */
  logit: TunedCurvePoint[];
  tuned: TunedCurvePoint[];
  tok_strs: string[];
  grids: TunedGrid[];
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

/** Live capture sets: while a set is registered, every bundle URL that passes
 *  through fetchJSON (cache hits included) is recorded into it. The Internals
 *  page wraps each driver load in one so its "download data" affordance can
 *  offer exactly the files the active view was computed from — no guessing. */
const captures = new Set<Set<string>>();

export function startBundleCapture(): Set<string> {
  const s = new Set<string>();
  captures.add(s);
  return s;
}

export function takeBundleCapture(s: Set<string>): string[] {
  captures.delete(s);
  return [...s].sort();
}

/** The parsed JSON for an already-loaded bundle URL (undefined if evicted). */
export function cachedBundle(url: string): unknown {
  return cache.get(url);
}

async function fetchJSON<T>(url: string): Promise<T> {
  for (const s of captures) s.add(url);
  const hit = cache.get(url);
  if (hit) return hit as T;
  // live:// URLs are cache-only — the payload was computed by the local live
  // server and inserted via putLiveTrace; there is nothing to fetch. Missing
  // means a permalink or reload outlived the in-memory result.
  if (url.startsWith("live://"))
    throw new Error("live trace not loaded — type the prompt again (results live in memory only)");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`interp bundle missing: ${url} (${res.status})`);
  const json = (await res.json()) as T;
  cache.set(url, json);
  return json;
}

/** `model` is the dataset dir id (e.g. "gpt2", "EleutherAI__pythia-70m"). */
export function interpBase(model: string, base = DATA_BASE): string {
  return `${base}/${model}/interp`;
}

export const loadInterpIndex = (model: string, base = DATA_BASE) =>
  fetchJSON<InterpIndex>(`${interpBase(model, base)}/index.json`);

export const loadWeights = (model: string, base = DATA_BASE) =>
  fetchJSON<WeightsBundle>(`${interpBase(model, base)}/weights.json`);

export const loadFourier = (model: string, base = DATA_BASE) =>
  fetchJSON<FourierBundle>(`${interpBase(model, base)}/fourier.json`);

export const loadEmbed = (model: string, base = DATA_BASE) =>
  fetchJSON<EmbedBundle>(`${interpBase(model, base)}/embed.json`);

export const loadNeurons = (model: string, base = DATA_BASE) =>
  fetchJSON<NeuronsBundle>(`${interpBase(model, base)}/neurons.json`);

export const loadSAE = (model: string, base = DATA_BASE) =>
  fetchJSON<SAEBundle>(`${interpBase(model, base)}/sae.json`);

export const loadSAEActs = (model: string, base = DATA_BASE) =>
  fetchJSON<SAEActsBundle>(`${interpBase(model, base)}/sae_acts.json`);

export const loadOVEigs = (model: string, base = DATA_BASE) =>
  fetchJSON<OVEigsBundle>(`${interpBase(model, base)}/ov_eigs.json`);

export const loadAttrib = (model: string, base = DATA_BASE) =>
  fetchJSON<AttribBundle>(`${interpBase(model, base)}/attrib.json`);

export const loadComp = (model: string, base = DATA_BASE) =>
  fetchJSON<CompBundle>(`${interpBase(model, base)}/comp.json`);

export const loadPatch = (model: string, base = DATA_BASE) =>
  fetchJSON<PatchBundle>(`${interpBase(model, base)}/patch.json`);

export const loadInduction = (model: string, base = DATA_BASE) =>
  fetchJSON<InductionBundle>(`${interpBase(model, base)}/induction.json`);

export const loadAblation = (model: string, base = DATA_BASE) =>
  fetchJSON<AblationBundle>(`${interpBase(model, base)}/ablation.json`);

export const loadOcclusion = (model: string, base = DATA_BASE) =>
  fetchJSON<OcclusionBundle>(`${interpBase(model, base)}/occlusion.json`);

export const loadTuned = (model: string, base = DATA_BASE) =>
  fetchJSON<TunedBundle>(`${interpBase(model, base)}/tuned.json`);

export const loadSAEWeb = (model: string, base = DATA_BASE) =>
  fetchJSON<SAEWebBundle>(`${interpBase(model, base)}/sae_web.json`);

export const loadGrok = (model: string, base = DATA_BASE) =>
  fetchJSON<GrokBundle>(`${interpBase(model, base)}/grok.json`);

export const loadCompass = (model: string, base = DATA_BASE) =>
  fetchJSON<CompassBundle>(`${interpBase(model, base)}/compass.json`);

export const loadCofire = (model: string, base = DATA_BASE) =>
  fetchJSON<CofireBundle>(`${interpBase(model, base)}/cofire.json`);

export const loadHeads = (model: string, base = DATA_BASE) =>
  fetchJSON<HeadsBundle>(`${interpBase(model, base)}/heads.json`);

/** 2b — custom prompts. A trace computed by the local live server
 *  (POST /live/trace returns EXACTLY the offline trace_<slug>.json shape, same
 *  producer function) is inserted into the same bundle cache under a synthetic
 *  live:// URL, so every trace-driven driver — and the ⤓ data download —
 *  works on typed prompts with zero driver changes. Live slugs are prefixed
 *  so chrome can tell them from bundled ones (and keep them out of permalinks:
 *  the payload lives in this tab's memory only). */
export const LIVE_TRACE_PREFIX = "live-";

export const isLiveTrace = (slug: string): boolean => slug.startsWith(LIVE_TRACE_PREFIX);

export const liveTraceUrl = (model: string, slug: string): string =>
  `live://${model}/trace_${slug}.json`;

export function putLiveTrace(model: string, slug: string, bundle: TraceBundle): void {
  cache.set(liveTraceUrl(model, slug), bundle);
}

export const loadTrace = (model: string, slug: string, base = DATA_BASE) =>
  fetchJSON<TraceBundle>(
    isLiveTrace(slug)
      ? liveTraceUrl(model, slug)
      : `${interpBase(model, base)}/trace_${slug}.json`,
  );

/** 2c — live SAE lens. The prompt text behind each live slug, so a driver that
 *  needs a fresh live-server computation (the Piano-Roll's /live/sae) can make
 *  it from the slug alone. Registered by the tracebar's runPrompt. */
const livePrompts = new Map<string, string>();

export function registerLivePrompt(model: string, slug: string, prompt: string): void {
  livePrompts.set(`${model}/${slug}`, prompt);
}

export function livePromptFor(model: string, slug: string): string | undefined {
  return livePrompts.get(`${model}/${slug}`);
}

/** One SAEActsTrace computed by POST /live/sae (same producer as the offline
 *  sae_acts.json traces), cached under a live:// URL like putLiveTrace so the
 *  ⤓ data download captures it too. */
export const liveSAEActsUrl = (model: string, slug: string): string =>
  `live://${model}/sae_acts_${slug}.json`;

export function putLiveSAEActs(model: string, slug: string, trace: SAEActsTrace): void {
  cache.set(liveSAEActsUrl(model, slug), trace);
}

export const loadLiveSAEActs = (model: string, slug: string) =>
  fetchJSON<SAEActsTrace>(liveSAEActsUrl(model, slug));

/** True if this model has interp bundles (feature nav gates on it, honestly —
 *  a model without bundles simply doesn't offer the interp views). */
export async function hasInterp(model: string, base = DATA_BASE): Promise<boolean> {
  try {
    await loadInterpIndex(model, base);
    return true;
  } catch {
    return false;
  }
}
