/** The Internals feature rail — the single source of truth for which
 *  mechanistic-interpretability features are LIVE. A feature appears here only
 *  once its driver renders a real computed quantity end-to-end and has passed
 *  the three review passes (numerical correctness, visual truthfulness,
 *  performance/interaction). The 25-feature spec is the roadmap; this list is
 *  the honest subset that actually works. `/guide` reads the same registry so
 *  documentation can never drift from what's shipped.
 *
 *  Groups map to data source, which keeps the honesty contract legible:
 *    weights  — raw weight tensors only (no forward pass)
 *    forward  — a real forward pass on a curated prompt (trace bundle)
 *    sae      — sparse-autoencoder features (downloaded SAE weights)
 *    trained  — a small model trained offline (e.g. grokking toy model)
 *    live     — in-browser forward pass on user text (capstone)
 */

import { AblationDriver } from "./AblationDriver";
import { AttentionFlowDriver } from "./AttentionFlowDriver";
import { AttentionRolloutDriver } from "./AttentionRolloutDriver";
import { CompositionWebDriver } from "./CompositionWebDriver";
import { EmbeddingConstellationDriver } from "./EmbeddingConstellationDriver";
import { FourierAtlasDriver } from "./FourierAtlasDriver";
import { HeadFingerprintDriver } from "./HeadFingerprintDriver";
import { InductionDriver } from "./InductionDriver";
import type { InterpFeature } from "./InterpDriver";
import { LogitAttribDriver } from "./LogitAttribDriver";
import { LogitLensTunnelDriver } from "./LogitLensTunnelDriver";
import { NeuronFieldDriver } from "./NeuronFieldDriver";
import { OcclusionDriver } from "./OcclusionDriver";
import { OVEigenDriver } from "./OVEigenDriver";
import { PatchingMapDriver } from "./PatchingMapDriver";
import { ProbabilitySimplexDriver } from "./ProbabilitySimplexDriver";
import { ResidualRibbonDriver } from "./ResidualRibbonDriver";
import { SAEConstellationDriver } from "./SAEConstellationDriver";
import { SAEPianoRollDriver } from "./SAEPianoRollDriver";
import { WeightSpectrumDriver } from "./WeightSpectrumDriver";

export const INTERP_FEATURES: InterpFeature[] = [
  {
    id: "fourier-atlas",
    n: 1,
    label: "Fourier Atlas",
    group: "weights",
    blurb:
      "DFT of GPT-2's learned positional embeddings W_pos along the position axis. " +
      "Angle = log frequency (cycles per 1024-token window); gold radius = log mean " +
      "power P(f); cyan = how many of the 768 dims peak at that frequency.",
    math:
      "X[d,f] = Σ_p W_pos[p,d]·exp(−2πi·f·p/n_ctx); P(f) = mean_d |X[d,f]|². " +
      "Angle θ(f) = π·log₂(f)/log₂(n_ctx/2); radius = normalized log P(f).",
    source:
      "fourier.json — real DFT of W_pos (n_ctx×768) computed offline in float64 " +
      "(numpy.fft.rfft along the position axis). No windowing, no smoothing.",
    legend: [
      { label: "mean power P(f) — log radius, outward", rgb: "245,195,59" },
      { label: "dims dominant at f — count, inward", rgb: "70,200,235" },
    ],
    note: "angle = log frequency, f=1 at top → f=512 clockwise",
    legendCorner: "bl",
    create: () => new FourierAtlasDriver(),
  },
  {
    id: "weight-spectrum",
    n: 21,
    label: "Weight Spectrum",
    group: "weights",
    blurb:
      "Singular-value spectra of every weight matrix (float64 SVD). x = index, " +
      "y = log₁₀σ; hover reads exact σ, stable rank, effective rank, condition κ.",
    math:
      "σ = svd(W) (descending). stable rank = ‖W‖_F²/σ₁²; effective rank = " +
      "exp(−Σ pₖ ln pₖ) with pₖ = σₖ/Σσ; condition κ = σ₁/σ_min.",
    source:
      "weights.json — float64 SVD of every weight matrix (W_E, W_pos, per-layer " +
      "attn QKV/out, MLP in/out), top ≤256 σ stored. Raw rows are NOT exported.",
    legend: [
      { label: "W_E (token embed)", rgb: "234,79,134" },
      { label: "W_pos (positional)", rgb: "245,195,59" },
      { label: "attn QKV", rgb: "70,200,235" },
      { label: "attn out", rgb: "90,230,180" },
      { label: "MLP in", rgb: "150,130,240" },
      { label: "MLP out", rgb: "139,59,240" },
    ],
    note: "brightness ↑ with layer depth",
    create: () => new WeightSpectrumDriver(),
  },
  {
    id: "embedding-constellation",
    n: 15,
    label: "Embedding Constellation",
    group: "weights",
    blurb:
      "Every one of GPT-2's 50,257 token embeddings W_E[i], placed at its exact " +
      "score on the top two principal axes of the mean-centered embedding matrix. " +
      "Star size = the real row norm ‖W_E[i]‖₂; color = a real orthographic " +
      "property (does the token begin with a space). The leading axes organize " +
      "tokens by SURFACE FORM — leading space, case, digits, the commonest " +
      "function words at the PC1 extreme — far more than by meaning. PC1+PC2 " +
      "explain only ~2.6% of the variance, so this is honestly a low-dimensional " +
      "shadow of a 768-D space; hover any star for its exact PC1/PC2/PC3 and norm.",
    math:
      "Wc = W_E − mean_row(W_E); eigendecompose the 768×768 covariance WcᵀWc for " +
      "its top axes V; coords = Wc·V (exact PC scores). size ∝ ‖W_E[i]‖₂. " +
      "Axes drawn to one isometric scale, so on-screen distance is faithful.",
    source:
      "embed.json — PCA of the token embedding matrix W_E (50257×768) computed " +
      "offline in float64 (covariance eigendecomposition). Coords rounded to 3 dp; " +
      "leading-space and norm are exact per-token quantities. No layout synthesis.",
    legend: [
      { label: "token with leading space · ␣word", rgb: "245,190,92" },
      { label: "token with no leading space · word", rgb: "92,198,236" },
      { label: "star size = embedding norm ‖W_E‖₂", rgb: "205,210,224" },
    ],
    note: "PC1+PC2 ≈ 2.6% of variance — a shadow, not the whole space",
    legendCorner: "tr",
    create: () => new EmbeddingConstellationDriver(),
  },
  {
    id: "neuron-field",
    n: 6,
    label: "Neuron Write-Direction Field",
    group: "weights",
    blurb:
      "All 36,864 MLP neurons (12 layers × 3072), each placed at the exact PCA " +
      "score of its write direction — the row of mlp.c_proj the neuron adds to " +
      "the residual stream, scaled by its activation. Dot size = the real write " +
      "norm ‖w_out‖₂; the median norm grows monotonically with depth (≈2.2 at " +
      "layer 0 → ≈5.2 at layer 11 — late layers write hardest). Hover any neuron " +
      "for its exact PCs, norm, and direct-path logit readout: the token its " +
      "direction most promotes and most suppresses through the tied unembedding. " +
      "PC1+PC2 explain only ~3.3% of variance — a low-D shadow, not the space.",
    math:
      "rows = stack_L(W_out^L) ((n_layer·d_mlp)×d); Rc = rows − mean_row; " +
      "eig(RcᵀRc) → top axes V; coords = Rc·V (exact PC scores). readout " +
      "ℓ = ((w − mean(w)) ⊙ γ_f)·W_Eᵀ — final-LN centering + gain, dropping only " +
      "the positive 1/σ scalar (rank-preserving). size = ‖w_out‖₂.",
    source:
      "neurons.json — PCA of all MLP write directions (rows of mlp.c_proj, " +
      "36864×768) computed offline in float64 (covariance eigendecomposition), " +
      "plus each neuron's most-promoted/suppressed token through the tied W_E. " +
      "Direct path only — no downstream-layer effects, positive activation assumed.",
    legend: [
      { label: "layer 0 neuron", rgb: "59,82,138" },
      { label: "layer 6", rgb: "54,181,120" },
      { label: "layer 11", rgb: "253,231,37" },
      { label: "dot size = write norm ‖w_out‖₂", rgb: "205,210,224" },
    ],
    note: "direct-path readout only · PC1+PC2 ≈ 3.3% var · click a chip to isolate a layer",
    legendCorner: "tr",
    create: () => new NeuronFieldDriver(),
  },
  {
    id: "head-fingerprints",
    n: 2,
    label: "Head Fingerprints",
    group: "weights",
    blurb:
      "All 144 attention heads (12 layers × 12 heads) on two honest, bounded " +
      "axes. y — the OV copying score: eigenvalues of the head's residual-space " +
      "OV map say whether it writes back the directions it reads with positive " +
      "sign (+1, a copying head) or inverts them (−1). x — how much of its " +
      "attention actually goes to the previous token, measured over real forward " +
      "passes of the 5 bundled prompts. Textbook GPT-2-small structure falls out " +
      "of the raw numbers: L4H11 is THE previous-token head (≈100% prev-token " +
      "attention, copying +0.96) and near-pure copying heads concentrate in " +
      "layers 9–11. Dot size = ‖OV‖_F, how strongly the head writes. Hover any " +
      "head for its exact circuit stats and measured attention shares.",
    math:
      "copying = Σ Re λ / Σ |λ| over eig(W_O·diag(γ₁)·W_V) (= nonzero eig of " +
      "the d×d OV map, computed exactly at d_head×d_head); σ_qk = " +
      "σ_max(diag(γ₁)·W_Q·W_Kᵀ·diag(γ₁))/√d_head; prev/first/self = mean of " +
      "a[i,i−1] / a[i,0] / a[i,i] over all query rows i≥1; entropy = mean " +
      "H(a[i,·])/ln(i+1) ∈ [0,1]. ln₁ gain folded; biases excluded.",
    source:
      "heads.json — OV/QK circuit stats from the weights (float64) + attention " +
      "behavior measured over 5 real forward passes (40 query rows, unrounded " +
      "attention — the trace bundles round to 4 dp, these stats don't). Both " +
      "axes are independent scales; no distance claims across them.",
    legend: [
      { label: "layer 0 head", rgb: "59,82,138" },
      { label: "layer 6", rgb: "54,181,120" },
      { label: "layer 11", rgb: "253,231,37" },
      { label: "dot size = OV write strength ‖OV‖_F", rgb: "205,210,224" },
    ],
    note: "x is a measured sample (5 prompts, 40 rows — stated) · y is pure weights",
    legendCorner: "br",
    create: () => new HeadFingerprintDriver(),
  },
  {
    id: "ov-eigen",
    n: 2,
    label: "OV Eigenvalue Constellation",
    group: "weights",
    blurb:
      "Every complex eigenvalue of every head's residual-space OV map — all " +
      "144 heads × 64 eigenvalues in ℂ. Along its eigendirection a head " +
      "writes back λ× what it reads: positive real copies, negative real " +
      "inverts, |λ| > 1 amplifies (the emphasized ring is |λ| = 1). This is " +
      "the full spectrum behind the fingerprints' copying scalar, and it " +
      "shows what the scalar hides: L11H8 scores a mild +0.29 yet contains a " +
      "single λ = −87.5 — one massively inverted direction canceled by an " +
      "otherwise-positive spectrum — while L11H3's entire spectrum sits in " +
      "[+3.3, +9.6], a uniform copying amplifier. Hover brightens the whole " +
      "spectrum of a head; click isolates it.",
    math:
      "λ = eig(W_O·diag(γ₁)·W_V) per head — the nonzero eigenvalues of the " +
      "d×d OV map diag(γ₁)·W_V·W_O, computed exactly at d_head×d_head since " +
      "eig(AB) = eig(BA). Real matrix → conjugate-symmetric spectrum (the " +
      "mirror across the real axis is the math, not decoration). Plot is " +
      "log-polar: angle = arg λ exactly; radius = log₁₀|λ| clamped to " +
      "[−2, +2] (stated window; 0.2% of points clamp at the center).",
    source:
      "ov_eigs.json — float64 eigendecomposition of every head's OV circuit " +
      "(ln_1 gain folded, biases excluded), verified identical to the full " +
      "768×768 eigendecomposition. copying re-exported per head; cross-checked " +
      "against heads.json to 4 dp.",
    legend: [
      { label: "layer 0 eigenvalue", rgb: "59,82,138" },
      { label: "layer 6", rgb: "54,181,120" },
      { label: "layer 11", rgb: "253,231,37" },
      { label: "|λ| = 1 ring — amplify/attenuate boundary", rgb: "166,173,200" },
    ],
    note: "log-polar: angle = arg λ · radius = log₁₀|λ| ∈ [−2,2] clamped · conjugate symmetry is the math",
    legendCorner: "bl",
    create: () => new OVEigenDriver(),
  },
  {
    id: "comp-web",
    n: 2,
    label: "Composition Web",
    group: "weights",
    blurb:
      "Which heads feed which heads — measured from the weights alone, no " +
      "forward pass. For every cross-layer head pair, the Q/K/V composition " +
      "score asks how much of head 1's OV write survives inside head 2's " +
      "query, key, or value channel. Raw composition has a positive floor even " +
      "for unrelated heads, so the measured random floor is shipped with the " +
      "data and arcs are drawn only above a stated multiple of it. The " +
      "induction circuit is visible and directional: the prev-token head " +
      "L4H11 composes into L5H1/L5H5 through their KEYS (K-comp 2.7–2.8× " +
      "floor) but not their queries — match the previous token, then copy. " +
      "Hover an arc for all three scores; click a head to isolate its web.",
    math:
      "M_ov¹ = diag(γ₁)W_V¹W_O¹, M_qk² = diag(γ₁)W_Q²W_K²ᵀdiag(γ₁). " +
      "Q-comp = ‖M_ov¹M_qk²‖_F/(‖M_ov¹‖‖M_qk²‖); K-comp = ‖M_qk²M_ov¹ᵀ‖_F/(same); " +
      "V-comp = ‖M_ov¹M_ov²‖_F/(‖M_ov¹‖‖M_ov²‖) (Elhage et al. 2021). Rank ≤ 64 " +
      "→ computed exactly via d_head-sized Grams; verified against explicit " +
      "768×768 products. Random floor measured over 200 seeded Gaussian factor " +
      "pairs; scores read relative to it.",
    source:
      "comp.json — float64 weight-only composition scores for all 9,504 " +
      "cross-layer head pairs × {Q,K,V}, plus the measured random baseline. " +
      "Same-layer pairs excluded (parallel heads cannot compose). Inter-layer " +
      "LayerNorm folded as γ gain only — the data-dependent 1/σ is not a " +
      "weight and is stated as unfolded.",
    legend: [
      { label: "arc at the display threshold (× floor)", rgb: "64,66,96" },
      { label: "arc at ≥6× the random floor", rgb: "245,195,59" },
      { label: "head in ≥1 arc at threshold", rgb: "166,173,200" },
      { label: "head with no arc at threshold", rgb: "118,126,158" },
    ],
    note: "arc shape is layout, not data · floor measured, stated, and drawn only above it",
    legendCollapsed: true,
    legendCorner: "br",
    create: () => new CompositionWebDriver(),
  },
  {
    id: "induction-microscope",
    n: 2,
    label: "Induction Microscope",
    group: "forward",
    blurb:
      "Behavioral proof of the induction circuit the Composition Web predicted " +
      "from weights alone. Feed 48 uniform-random tokens repeated twice: on " +
      "the second repeat, a head doing induction ([A][B] … [A] → attend to " +
      "[B]) must attend from each position t to t−47. L5H1 and L5H5 — the " +
      "heads L4H11's K-composition pointed at — score ~0.90, about 64× the " +
      "uniform-attention floor, alongside L7H10 and L6H9. L4H11 itself tops " +
      "the prev-token score at 0.979. Click any head to see its real T×T " +
      "attention pattern: induction is a literal stripe at offset −47. Which " +
      "top head ranks first swaps between seeds — both seeds' scores ship, " +
      "and hover shows them side by side.",
    math:
      "induction[l,h] = mean over second-repeat positions t of " +
      "attn[l,h][t, t−period+1]; duplicate: target t−period; prev: target " +
      "t−1. Chance floor = mean_t 1/(t+1) — what uniform attention would " +
      "score. Sequence = <|endoftext|> + 48 uniform-random tokens ×2 " +
      "(Olsson et al. 2022 diagnostic; random tokens isolate copying from " +
      "semantics).",
    source:
      "induction.json — two real forwards (seeds 0 and 1) over the repeated " +
      "sequence; all 144 heads' scores exported for BOTH seeds, full T×T " +
      "post-softmax patterns for the top-8 heads (144 would be ~8 MB). " +
      "Verified: sequence repeats exactly; attention rows sum to 1 (drift " +
      "2.4e-7); scores recomputed from the exported patterns match at 4 dp; " +
      "the pattern's upper triangle is exactly zero (causal mask).",
    legend: [
      { label: "high attention to the structural target", rgb: "245,195,59" },
      { label: "≤ uniform-attention floor", rgb: "118,126,158" },
    ],
    note:
      "grid ramp spans floor→max of the selected score (both stated in the " +
      "header) · stripe alpha is linear in attention — no gamma, no smoothing",
    legendCollapsed: true,
    legendCorner: "br",
    ownPrompts: true,
    create: () => new InductionDriver(),
  },
  {
    id: "ablation-ghosts",
    n: 17,
    label: "Ablation Ghosts",
    group: "forward",
    blurb:
      "The causal closer for the induction arc: every head knocked out, one " +
      "real ablated forward each, on the same repeated random-token sequence " +
      "as #2d. The grid is the rise in second-repeat loss; clicking a head " +
      "draws its ghost — the ablated loss curve over the baseline's ×60 " +
      "in-context drop. The headline is redundancy: ablating the TOP-scoring " +
      "induction head (L7H10) changes nothing (−0.01 nats), but the top-4 " +
      "together cost +2.0 (zero) / +3.2 (mean) nats — 3× the sum of their " +
      "singles, backup heads caught compensating. Mean-ablation ranks the " +
      "circuit #2c predicted: L5H1 +0.65, L4H11 +0.38, L6H9 +0.27. And " +
      "L10H7/L11H10 — the negative movers from #13 — HELP when removed.",
    math:
      "Δ[l,h] = mean NLL over predicted positions 50–96 (second-repeat tokens " +
      "whose context token has occurred before) minus baseline; NLL(j) = " +
      "−log p(s_j | s_<j) in nats. Ablation replaces head (l,h)'s a@v slice " +
      "before c_proj with 0 (zero mode) or its per-run positional mean (mean " +
      "mode). Zero is off-distribution, mean keeps the head's average signal " +
      "— they disagree by up to 1.3 nats and both ship.",
    source:
      "ablation.json — 293 real forwards (baseline + 144 heads × 2 modes + " +
      "2 combos × 2). Verified: the unablated replicated forward reproduces " +
      "the baseline logits bit-for-bit (drift 0.0); Δ recomputed from the " +
      "exported per-position curves matches at 4 dp; sequence and induction " +
      "scores identical to induction.json's seed-0 run.",
    legend: [
      { label: "Δ > 0 — ablation raises window loss (head needed)", rgb: "245,195,59" },
      { label: "Δ < 0 — ablation lowers loss (head hurts)", rgb: "96,165,250" },
      { label: "baseline NLL curve (unablated)", rgb: "205,210,228" },
    ],
    note:
      "grid ramp spans 0→max |Δ| of the selected mode (stated in header) · " +
      "curves join 96 discrete per-position values — no smoothing · " +
      "single-head ablation understates redundant circuits (see combo chips)",
    legendCollapsed: true,
    legendCorner: "br",
    ownPrompts: true,
    create: () => new AblationDriver(),
  },
  {
    id: "occlusion-vignette",
    n: 19,
    label: "Occlusion Vignette",
    group: "forward",
    blurb:
      "Leave-one-token-out: each prompt position is occluded — substituted " +
      "with <|endoftext|> or deleted outright — and one real forward measures " +
      "how far the baseline top-1's log-probability falls. On the IOI prompt, " +
      "deleting “ Mary” flips the prediction to “ John” (+4.22 nats) and " +
      "deleting the duplicated “ John” makes it fall back to “ them” (+1.79) " +
      "— name-mover evidence at the input level. Occluding the “ E” of Eiffel " +
      "makes the model guess “ London”. Honest surprise: syntax dominates — " +
      "“ is”/“ of”/“ to” post the biggest drops (up to +18.9 nats), because " +
      "occlusion measures total contribution, scaffolding included. Completes " +
      "the causal lenses: input (here) · residual patch (#14) · head ablation " +
      "(#17) · direct attribution (#13).",
    math:
      "drop[p] = log p_base(c) − log p_occluded(c) at the final position, " +
      "where c is the BASELINE forward's own argmax next token; drop_logit is " +
      "the same difference in raw logits. Sub mode keeps positions " +
      "(<|endoftext|> in place); del mode deletes the token, so later " +
      "positions shift and their positional embeddings move — a bigger, " +
      "disclosed intervention. Each occluded run's own top-1 is exported, so " +
      "prediction flips are shown, not inferred.",
    source:
      "occlusion.json — 100 real forwards (5 prompts × (baseline + T sub + " +
      "T del + causality check)). Verified: the replicated baseline matches " +
      "Trace.logits exactly; deleting the FINAL token reproduces the " +
      "baseline's second-to-last logits bit-for-bit (causal drift 0.0); the " +
      "IOI del-“ John” drop recomputed from scratch matches at 4 dp.",
    legend: [
      { label: "drop > 0 — token supported the prediction", rgb: "245,195,59" },
      { label: "drop < 0 — token was suppressing it", rgb: "96,165,250" },
      { label: "white outline — occlusion flips the top-1", rgb: "255,255,255" },
    ],
    note:
      "one linear scale both directions — nats-per-px identical above and " +
      "below the zero axis, extents stated on the axis · del mode shifts " +
      "positional embeddings (disclosed) · drops include syntactic " +
      "scaffolding, not just semantics",
    legendCollapsed: true,
    legendCorner: "br",
    create: () => new OcclusionDriver(),
  },
  {
    id: "logit-lens-tunnel",
    n: 3,
    label: "Logit-Lens Tunnel",
    group: "forward",
    blurb:
      "The logit lens: decode the last-position residual stream through the " +
      "model's own unembedding at every layer. Each row is one layer's real " +
      "next-token distribution (layer 0 = raw embedding, bottom → final layer, " +
      "top); every cell's width is that token's probability. The faint track is " +
      "the full 0..1 scale, so the unfilled part is the tail mass beyond the " +
      "top-k. The final answer is traced in gold as it sharpens up the stack.",
    math:
      "For each layer ℓ, logits_ℓ = LayerNorm_f(x_ℓ[last])·W_U; P_ℓ = softmax(logits_ℓ). " +
      "Row ℓ shows top-6 of P_ℓ; cell width = P. Layer 0 lenses the raw embedding.",
    source:
      "trace_*.json → logit_lens_last: the last-position residual at every layer " +
      "(0=embed…12) decoded through the model's own final LN + unembedding W_U.",
    legend: [
      { label: "final-layer top-1 token (the answer)", rgb: "245,195,59" },
      { label: "any other top-k token · width = P", rgb: "96,165,224" },
    ],
    note: "layer 0 (embed) at bottom → final layer at top",
    legendCorner: "br",
    create: () => new LogitLensTunnelDriver(),
  },
  {
    id: "attention-flow",
    n: 7,
    label: "Attention-Head Flow",
    group: "forward",
    blurb:
      "Post-softmax attention of one head from a real forward pass. Left = query " +
      "tokens, right = key tokens; each line's width & opacity scale with the " +
      "attention probability attn[i][j] (rows are a causal softmax → sum to 1, " +
      "j ≤ i). Pick any of the 12×12 heads — cells are tinted by attention focus " +
      "(1 − mean normalized row entropy).",
    math:
      "attn[l,h,i,j] = softmax_j(QKᵀ/√d_head + causal mask)[i,j], Σ_j attn = 1, j ≤ i. " +
      "Head focus = 1 − mean_i H(attn[l,h,i,:])/log(i+1) (normalized row entropy).",
    source:
      "trace_*.json → attn (n_layer×n_head×T×T), the actual post-softmax attention " +
      "probabilities captured on the forward pass (rounded for transport, not smoothed).",
    legend: [
      { label: "query→key · width+opacity ∝ attn", rgb: "245,195,59" },
      { label: "key column", rgb: "70,200,235" },
    ],
    note: "hover a line for the exact weight · hover a token to isolate it",
    legendCorner: "br",
    create: () => new AttentionFlowDriver(),
  },
  {
    id: "attention-rollout",
    n: 23,
    label: "Attention-Rollout Waterfall",
    group: "forward",
    blurb:
      "Attention rollout (Abnar & Zuidema 2020): the cumulative product of the " +
      "residual-augmented, head-averaged attention maps, Ã_l = ½·mean_h A_l + ½·I, " +
      "R_d = Ã_d···Ã_0. Cell (i,j) = how much source token j reaches destination " +
      "token i after layers 0..d — each row is a distribution (sums to 1), strictly " +
      "causal (j ≤ i). Scrub or play the depth to watch mixing cascade onto the " +
      "first-token sink. Color is log₁₀ (values span orders of magnitude).",
    math:
      "A_l = mean_h attn[l]; Ã_l = row_normalize(½·A_l + ½·I); R_d = Ã_d·Ã_{d−1}···Ã_0. " +
      "Each R_d row sums to 1 and is causal; R_d[i,j] = j's contribution to i thru layer d.",
    source:
      "trace_*.json → attn, head-averaged then rolled out in-browser (float64). " +
      "Method: Abnar & Zuidema, “Quantifying Attention Flow in Transformers” (2020).",
    legend: [
      { label: "high rollout weight (log₁₀ color)", rgb: "245,205,90" },
      { label: "low / zero", rgb: "46,52,96" },
    ],
    note: "click a row to isolate one token's provenance · hover for exact value",
    legendCorner: "br",
    create: () => new AttentionRolloutDriver(),
  },
  {
    id: "residual-ribbon",
    n: 8,
    label: "Residual-Stream Ribbon",
    group: "forward",
    blurb:
      "The L2 norm ‖x‖₂ of every token's residual-stream vector at each layer of a " +
      "real forward pass (resid_norm[layer][token]; layer 0 = token+position " +
      "embedding, 1..12 = after each block). One ribbon per token, left→right " +
      "across depth. y is log₁₀‖x‖₂ — the norm grows geometrically with depth and " +
      "one token usually balloons into a massive activation that dwarfs the rest, " +
      "so log-y keeps every trajectory legible while preserving order.",
    math:
      "y = log₁₀‖x_ℓ(t)‖₂ where x_ℓ(t) is token t's residual vector at layer ℓ " +
      "(ℓ=0 embedding, 1..12 after each block). Growth factor = ‖x_final‖/‖x_embed‖.",
    source:
      "trace_*.json → resid_norm ((n_layer+1)×T), the exact Euclidean norm of the " +
      "residual stream captured per token per layer on the forward pass.",
    legend: [
      { label: "one ribbon per token · hue = position", rgb: "245,195,59" },
      { label: "decade gridline ‖x‖ = 10ᵏ", rgb: "148,140,165" },
    ],
    note: "log₁₀ y · hover a node for the exact norm + embed→final growth",
    legendCorner: "tr",
    create: () => new ResidualRibbonDriver(),
  },
  {
    id: "probability-simplex",
    n: 18,
    label: "Probability Simplex",
    group: "forward",
    blurb:
      "The final next-token distribution on a true 2-simplex (ternary plot). " +
      "Corners are the top-1 token, the top-2 token, and “all other tokens”; the " +
      "point's barycentric coordinates are the EXACT probabilities (p₁, p₂, p_rest) " +
      "with no renormalization — so a confident prediction sits in a corner and an " +
      "unconfident one sits toward “other”. GPT-2 is often unconfident: on the " +
      "Eiffel prompt the top-3 hold only ~14% of the mass, so the point sits near " +
      "“other”, not near “ Paris”. The bars give the full top-12 + the ranks-13+ tail.",
    math:
      "P = softmax(final logits); p₁,p₂ = its two largest, p_rest = 1−p₁−p₂. " +
      "Barycentric point = p₁·A + p₂·B + p_rest·C on triangle A,B,C — no renormalization.",
    source:
      "trace_*.json → final_topk (top-12 (token,prob) at the last position). The " +
      "tail (ranks 13+) = 1 − Σ top-12 is shown explicitly; nothing is hidden.",
    legend: [
      { label: "top-1 token corner · p₁", rgb: "245,195,59" },
      { label: "top-2 token corner · p₂", rgb: "96,165,224" },
      { label: "all other tokens (rank ≥3) · p_rest", rgb: "123,130,156" },
    ],
    note: "no renormalization — position IS the true probability · switch prompts to compare",
    legendCorner: "br",
    create: () => new ProbabilitySimplexDriver(),
  },
  {
    id: "logit-attrib",
    n: 13,
    label: "Logit Attribution",
    group: "forward",
    blurb:
      "Who wrote the answer? The final residual decomposes exactly into " +
      "everything ever added to it — the embedding, every attention head's " +
      "write, every MLP block, every bias. Each piece is projected into the " +
      "margin between the model's top-1 and runner-up next-token logits, so " +
      "the cells are additive: they sum to the real margin (the float32 drift " +
      "between Σ and the true margin is printed in the header). Amber pushed " +
      "the prediction, blue pushed the runner-up. The right gutter accumulates " +
      "the margin layer by layer — watch the decision form. On the IOI prompt " +
      "this recovers the published circuit: name-movers L9H6/L9H9 dominate " +
      "(reading “ Mary”), negative mover L10H7 fights them.",
    math:
      "x_final = emb + Σ_L(Σ_h head_out + b_o + mlp_out), recomputed per head " +
      "from the unrounded forward. contrib(v) = ((v − mean(v)) ⊙ γ_f)·" +
      "(W_U[top1] − W_U[top2]) / σ, with σ FROZEN at this forward's actual " +
      "final-LN normalizer (standard DLA linearization — the one thing not " +
      "attributed is each piece's effect on σ itself).",
    source:
      "attrib.json — per-head/MLP/bias margin contributions on the bundled " +
      "prompts, additivity verified per trace (|Σ − margin| ≤ 0.0006) and the " +
      "rebuilt stream checked against resid[-1] (rel err < 1e-6). Per head, " +
      "the argmax-attended token at the final row is what the head READ — " +
      "shown on hover, not a causal claim.",
    legend: [
      { label: "pushes the prediction (+, at color clamp)", rgb: "245,195,59" },
      { label: "pushes the runner-up (−, at color clamp)", rgb: "96,150,255" },
      { label: "≈0 — below 4dp export floor", rgb: "118,126,158" },
    ],
    note: "additive by construction · b_o column = bias no head owns · frozen-σ DLA",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new LogitAttribDriver(),
  },
  {
    id: "causal-patching",
    n: 14,
    label: "Causal Patching Map",
    group: "forward",
    blurb:
      "Where does the answer causally live? Run a matched clean/corrupt " +
      "prompt pair (swap one name, one country, one word), then for every " +
      "(layer, position) copy a single clean residual row into the corrupt " +
      "forward and let it finish — every cell is a real intervention, a full " +
      "patched forward pass. Color = how much of the clean-vs-corrupt answer " +
      "logit gap that one patch recovers. Unlike attribution (#13), this is " +
      "causal: r = 1 means that row alone flips the run. The IOI pair " +
      "reproduces the published circuit shape by intervention — the swapped " +
      "name position carries everything through L8, then the signal hands " +
      "off to the final position at L9, exactly where #13's name-mover heads " +
      "write the margin. Fact recall (Paris/Rome) hands off later, at L10.",
    math:
      "r[i,p] = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt), where " +
      "LD = logit(ans_clean) − logit(ans_corrupt) at the final position and " +
      "the patch copies resid_i[p] (the residual entering block i) from the " +
      "clean forward into the corrupt one, resuming blocks i…11 + ln_f in " +
      "the model's own float32. Answers are designated single-token " +
      "contrasts (published convention); each answer's rank in its own run " +
      "is printed in the header.",
    source:
      "patch.json — 312 real patched forward resumes over 3 matched pairs. " +
      "Resume path verified: unpatched resume from every layer reproduces " +
      "the corrupt logits (worst |Δ| 3e-5); full patch at layer 0 reproduces " +
      "the clean logits; patching an identical token's row is a no-op. Raw " +
      "patched LDs exported next to r — nothing hides in normalization.",
    legend: [
      { label: "recovers the clean answer (r → 1)", rgb: "245,195,59" },
      { label: "pushes further corrupt (r < 0)", rgb: "96,150,255" },
      { label: "≈0 — the patch did nothing", rgb: "118,126,158" },
    ],
    note: "true interventions, not attribution · color clamps |r| ≤ 1, exact value in hover · single-site patches can understate redundant circuits",
    legendCorner: "br",
    legendCollapsed: true,
    ownPrompts: true,
    create: () => new PatchingMapDriver(),
  },
  {
    id: "sae-decoder",
    n: 5,
    label: "SAE Decoder Constellation",
    group: "sae",
    blurb:
      "All 24,576 features of an open sparse autoencoder trained on GPT-2's " +
      "layer-8 residual stream (Joseph Bloom's res-jb release), each placed at " +
      "the exact PCA score of its decoder direction W_dec[i] — the vector the " +
      "feature adds back to the residual stream. Same math as the embedding and " +
      "neuron constellations, so the three views are directly comparable. " +
      "Decoder rows are unit-norm by construction, so size and brightness " +
      "encode the release's MEASURED firing sparsity — how often each feature " +
      "actually fires — never the norm. Hover any feature for its exact PCs, " +
      "log₁₀ sparsity, and the token its direction most promotes/suppresses.",
    math:
      "Rc = W_dec − mean_row (24576×768); eig(RcᵀRc) → top axes V; coords = " +
      "Rc·V (exact PC scores). readout ℓ = ((w − mean(w)) ⊙ γ_f)·W_Eᵀ " +
      "(rank-preserving). sparsity = log₁₀(firing fraction), measured by the " +
      "release over its evaluation set (−10 = clamp floor, dead feature).",
    source:
      "sae.json — PCA of W_dec from jbloom/GPT2-Small-SAEs-Reformatted " +
      "(blocks.8.hook_resid_pre, d_sae=24576) in float64; sparsity from the " +
      "release's sparsity.safetensors. Readout is direct-path only — decoder " +
      "directions enter at layer 8 and skip blocks 8–11 on the way to W_E.",
    legend: [
      { label: "fires ~10% of tokens (10⁻¹, ramp top)", rgb: "253,231,37" },
      { label: "fires ~0.1% (10⁻³, mid ramp)", rgb: "71,189,110" },
      { label: "fires ≤10⁻⁶ · incl. dead (ramp floor)", rgb: "59,82,138" },
      { label: "unit-norm decoder — size ≠ norm", rgb: "205,210,224" },
    ],
    note: "color/size span firing fraction 10⁻⁶–10⁻¹ (clamped) · direct path skips blocks 8–11",
    legendCorner: "tr",
    create: () => new SAEConstellationDriver(),
  },
  {
    id: "sae-piano-roll",
    n: 4,
    label: "SAE Firing Piano-Roll",
    group: "sae",
    perTrace: true,
    blurb:
      "The same SAE's ENCODER run on a real forward pass: which features fire " +
      "on which tokens of the prompt, with exact activation values. Rows are " +
      "the top features by peak activation, ordered by where they peak; each " +
      "row is scaled to its own peak, printed at the right edge. GPT-2's " +
      "first-token massive-activation outlier drives a few features 60–100× " +
      "everything else — those sit in a separate labeled band instead of " +
      "silently flooding the board. The strip below prints the exact " +
      "per-position L0 and reconstruction cosine: how much of the residual " +
      "stream the SAE actually explains at each token.",
    math:
      "x̄ = x − mean(x) per position (the SAE's training basis — TransformerLens " +
      "center_writing_weights; LayerNorm-invariant, exact). acts = " +
      "ReLU((x̄ − b_dec)·W_enc + b_enc); recon = acts·W_dec + b_dec; " +
      "cos = cosine(recon, x̄); L0 = #{acts > 0}. Cell brightness = act / row " +
      "peak (or board peak via the toggle).",
    source:
      "sae_acts.json — the res-jb encoder (jbloom/GPT2-Small-SAEs-Reformatted, " +
      "blocks.8.hook_resid_pre) applied to the real layer-8 residuals of the " +
      "bundled prompts. Feeding the uncentered HF-basis residual gives L0≈2700 " +
      "and cos≈0.76; the exact re-centering restores the published regime " +
      "(L0≈30–100, cos≈0.93–0.9999) — verified both ways.",
    legend: [
      { label: "act = row peak (printed at right)", rgb: "245,195,59" },
      { label: "act = ½ row peak", rgb: "155,131,78" },
      { label: "act = 0 — never smoothed", rgb: "118,126,158" },
    ],
    note: "per-row scale (toggle for board scale) · outlier band always own scale · L0/cos printed exactly",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new SAEPianoRollDriver(),
  },
];

export function findFeature(id: string): InterpFeature | undefined {
  return INTERP_FEATURES.find((f) => f.id === id);
}

export const GROUP_LABEL: Record<InterpFeature["group"], string> = {
  weights: "Weights",
  forward: "Forward pass",
  sae: "SAE features",
  trained: "Trained probe",
  live: "Live prompt",
};
