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
 *    live     — a real forward pass on user text via a local probe server
 *               (weights stay on-machine; nothing precomputed) — capstone
 */

import { AblationDriver } from "./AblationDriver";
import { AttentionFlowDriver } from "./AttentionFlowDriver";
import { AttentionRolloutDriver } from "./AttentionRolloutDriver";
import { CofireDriver } from "./CofireDriver";
import { CompassDriver } from "./CompassDriver";
import { CompositionWebDriver } from "./CompositionWebDriver";
import { EmbeddingConstellationDriver } from "./EmbeddingConstellationDriver";
import { FourierAtlasDriver } from "./FourierAtlasDriver";
import { GrokClockDriver } from "./GrokClockDriver";
import { HeadFingerprintDriver } from "./HeadFingerprintDriver";
import { InductionDriver } from "./InductionDriver";
import type { InterpFeature } from "./InterpDriver";
import { LiveNebulaDriver } from "./LiveNebulaDriver";
import { LogitAttribDriver } from "./LogitAttribDriver";
import { LogitLensTunnelDriver } from "./LogitLensTunnelDriver";
import { NeuronFieldDriver } from "./NeuronFieldDriver";
import { OcclusionDriver } from "./OcclusionDriver";
import { OVEigenDriver } from "./OVEigenDriver";
import { PatchingMapDriver } from "./PatchingMapDriver";
import { TunedLensDriver } from "./TunedLensDriver";
import { ProbabilitySimplexDriver } from "./ProbabilitySimplexDriver";
import { ResidualRibbonDriver } from "./ResidualRibbonDriver";
import { SAEConstellationDriver } from "./SAEConstellationDriver";
import { SAEPianoRollDriver } from "./SAEPianoRollDriver";
import { SAEWebDriver } from "./SAEWebDriver";
import { WeightSpectrumDriver } from "./WeightSpectrumDriver";

export const INTERP_FEATURES: InterpFeature[] = [
  {
    id: "fourier-atlas",
    n: 1,
    label: "Position Patterns by Frequency",
    subtitle: "Repeating patterns in how GPT-2 represents token positions",
    group: "weights",
    blurb:
      "This view looks for repeating patterns in GPT-2’s 1,024 position embeddings. " +
      "Angle shows how often a pattern repeats across the context window. Angles are " +
      "logarithmically spaced, so equal angular steps represent multiplication rather " +
      "than equal additions in frequency. Gold extends outward with average pattern " +
      "strength. Cyan extends inward from the baseline ring; its length is the count of " +
      "embedding dimensions that peak at that frequency, scaled to the largest count " +
      "on this chart. Hover for the exact frequency, period, power, and count.",
    math:
      "First, subtract each embedding dimension’s average across positions. A frequency " +
      "analysis called a discrete Fourier transform (DFT) then separates each of the " +
      "768 position sequences into repeat rates. The chart averages squared signal " +
      "strength across dimensions at each rate and uses logarithmic frequency and power " +
      "scales so weak and strong patterns remain visible.",
    source:
      "fourier.json — generated offline from GPT-2’s 1,024 × 768 position-embedding " +
      "matrix with 64-bit calculations. The data is not smoothed, and no edge-tapering " +
      "window is applied.",
    legend: [
      { label: "average pattern strength · farther out means stronger", rgb: "245,195,59" },
      {
        label: "number of dimensions whose strongest pattern is at this frequency",
        rgb: "70,200,235",
      },
    ],
    note:
      "Frequency increases clockwise: 1 cycle is at the top and 512 is the maximum. " +
      "This is a static weight pattern, not evidence that the model uses a frequency " +
      "on a prompt; averaging can also hide differences between dimensions.",
    legendCorner: "bl",
    create: () => new FourierAtlasDriver(),
  },
  {
    id: "weight-spectrum",
    n: 21,
    label: "Weight-Matrix Strength",
    subtitle: "How strongly each stored weight matrix acts along its main directions",
    group: "weights",
    blurb:
      "Each line shows one of GPT-2’s 50 analyzed weight matrices. Read left to right " +
      "from the strongest toward weaker directions among the largest 256 displayed; " +
      "higher points mean a stronger direction. The vertical scale is logarithmic, so " +
      "large and small values can share one chart. Hover for exact values and three " +
      "full-spectrum summaries: two estimates of how many directions matter, plus the " +
      "strongest-to-weakest ratio. A very large ratio can make some calculations " +
      "sensitive to small numerical errors.",
    math:
      "Singular value decomposition (SVD) separates each matrix into its main " +
      "directions and sorts their strengths from largest to smallest. Stable rank and " +
      "effective rank summarize how many directions matter in different ways. The " +
      "condition number divides the strongest value by the weakest nonzero value.",
    source:
      "weights.json — created offline with 64-bit SVD for the token and position " +
      "embeddings and four matrices per layer: one combined attention query/key/value " +
      "matrix, attention output, feed-forward input, and feed-forward output. This is " +
      "50 matrices in total: two embedding matrices and four matrices for each of 12 layers. " +
      "Up to the 256 largest singular values are stored for " +
      "each; raw rows are not included. Summary measures use every nonzero value.",
    legend: [
      { label: "token embeddings", rgb: "234,79,134" },
      { label: "position embeddings", rgb: "245,195,59" },
      { label: "combined attention query, key, and value weights", rgb: "70,200,235" },
      { label: "attention output weights", rgb: "90,230,180" },
      { label: "feed-forward input weights", rgb: "150,130,240" },
      { label: "feed-forward output weights", rgb: "139,59,240" },
    ],
    note:
      "Among layer-specific lines, brighter means a later layer; the two embedding " +
      "lines use fixed middle brightness. These spectra describe stored transformations, " +
      "not whether a direction is used on a prompt or causes a behavior.",
    create: () => new WeightSpectrumDriver(),
  },
  {
    id: "embedding-constellation",
    n: 15,
    label: "Token Embedding Map",
    subtitle: "A two-dimensional map of GPT-2’s token representations",
    group: "weights",
    blurb:
      "This map places all 50,257 GPT-2 tokens using the two directions with the " +
      "most variation in their 768-dimensional embeddings. In this projection, visible " +
      "groupings often follow written form—such as leading spaces, capitalization, " +
      "digits, and some common words. This is a visual observation, not a quantified " +
      "test of spelling versus meaning. Star size shows embedding length; color shows " +
      "whether the token begins with a space. Hover for the token’s first three PCA " +
      "scores and length.",
    math:
      "Center the token-embedding matrix, run principal component analysis (PCA), and " +
      "use each token’s scores on the first two components as its coordinates. Both " +
      "axes use the same scale, so distances are comparable within this two-axis " +
      "projection. Star size is the embedding vector’s ordinary Euclidean length.",
    source:
      "embed.json — PCA was computed offline in 64-bit precision from the 50,257 × 768 " +
      "token-embedding matrix. Coordinates and embedding lengths are rounded to three " +
      "decimal places; the leading-space flag is measured per token. No artificial " +
      "layout was added.",
    legend: [
      { label: "token starts with a space · ␣word", rgb: "245,190,92" },
      { label: "token has no leading space · word", rgb: "92,198,236" },
      { label: "larger star · longer embedding vector", rgb: "205,210,224" },
      { label: "stronger glow · higher embedding-length rank", rgb: "205,210,224" },
      { label: "brighter area · more tokens overlap", rgb: "205,210,224" },
    ],
    note:
      "The two axes explain about 2.6% of variation, so nearby points can be far apart " +
      "in the full space. Vector length alone does not measure importance or causal influence.",
    legendCorner: "tr",
    create: () => new EmbeddingConstellationDriver(),
  },
  {
    id: "neuron-field",
    n: 6,
    label: "Neuron Output-Vector Map",
    subtitle: "A two-dimensional map of the stored signal each feed-forward neuron can add",
    group: "weights",
    blurb:
      "This map places all 36,864 feed-forward (MLP) neurons by their stored output " +
      "vectors—the signal each would add to the model’s shared working state per unit " +
      "of positive activation. The PCA uses unnormalized vectors, so both direction " +
      "and length can affect position. Larger dots have longer output vectors. Median " +
      "length grows from about 2.2 in layer 0 to 5.2 in layer 11. Hover for coordinates, " +
      "length, and the token the vector would most raise or lower through the direct " +
      "output path.",
    math:
      "Stack every neuron’s unnormalized output vector, center them, run PCA, and plot " +
      "the first two component scores. The direct token readout uses the centering and " +
      "learned gain from the model’s final normalization, then the tied token-output " +
      "matrix. It omits only a positive normalization scale, so token ranking is " +
      "preserved, but it does not include effects from later layers. Dot size is " +
      "ordinary Euclidean length.",
    source:
      "neurons.json — PCA was computed offline in 64-bit precision from all 36,864 × " +
      "768 neuron output directions. The file also stores each neuron’s most promoted " +
      "and suppressed token through the direct output path. Coordinates and lengths " +
      "are rounded to three decimal places. This assumes a positive activation and " +
      "excludes downstream-layer effects.",
    legend: [
      { label: "layer 0 neuron · color shows layer", rgb: "59,82,138" },
      { label: "layer 6", rgb: "54,181,120" },
      { label: "layer 11", rgb: "253,231,37" },
      { label: "larger dot · longer output vector", rgb: "205,210,224" },
      { label: "stronger glow · higher length rank, not a later layer", rgb: "205,210,224" },
      { label: "brighter area · more neurons overlap", rgb: "205,210,224" },
    ],
    note:
      "The two axes explain about 3.3% of variation. This weight map does not show how " +
      "often neurons fire or their full causal effect; the token readout is direct-path " +
      "only. Select a layer chip to isolate it.",
    legendCorner: "tr",
    create: () => new NeuronFieldDriver(),
  },
  {
    id: "head-fingerprints",
    n: 2,
    label: "Attention-Head Behavior Map",
    subtitle: "How strongly each head copies information and looks at the previous token",
    group: "weights",
    blurb:
      "This chart compares all 144 attention heads. The horizontal axis is the share " +
      "of attention placed on the previous token, measured from five sample prompts. " +
      "The vertical axis estimates whether a head copies the directions it reads (+1) " +
      "or reverses them (−1) in the model’s shared working state. Larger dots write " +
      "more strongly. In this sample, L4H11 attends almost entirely to the previous " +
      "token and has a copying score of about 0.96; many scores near +1 appear in " +
      "layers 9–11. Hover for copying, previous/first/self attention, attention " +
      "spread, query-key sharpness, and write strength. These are descriptive " +
      "measurements, not effects of intervening on a head.",
    math:
      "The copying score combines the signs and sizes of the eigenvalues of each " +
      "head’s output-value (OV) transformation. Previous-token, first-token, and " +
      "self-attention are averaged over all 40 query positions after the first token " +
      "across five prompts. Attention entropy measures spread and is normalized to a " +
      "0–1 scale. Query-key sharpness is an additional hover statistic, not an axis. " +
      "The model’s first normalization gain is included; biases are not.",
    source:
      "heads.json — the vertical copying score and hover-only query-key statistic were " +
      "computed from weights in 64-bit precision. The horizontal value is unrounded " +
      "attention from five complete runs: Eiffel Tower location, a Mary/John name " +
      "prompt, a number sequence, hot/opposite, and capital of France (40 positions " +
      "after each prompt’s first token in total). The axes use different units, so " +
      "compare each separately, not by straight-line distance. The copying score covers " +
      "the shared-state OV map, not the full token-output circuit.",
    legend: [
      { label: "layer 0 head · color shows layer", rgb: "59,82,138" },
      { label: "layer 6", rgb: "54,181,120" },
      { label: "layer 11", rgb: "253,231,37" },
      { label: "larger dot · stronger output-value (OV) write", rgb: "205,210,224" },
    ],
    note:
      "The horizontal value is a small measured sample: five prompts and 40 rows. " +
      "The vertical value comes from the weights. Neither is a causal effect estimate " +
      "or proof that the pattern generalizes beyond these prompts.",
    legendCorner: "br",
    linksTo: ["head"],
    create: () => new HeadFingerprintDriver(),
  },
  {
    id: "ov-eigen",
    linksTo: ["head"],
    n: 2,
    label: "Attention-Head Copying Spectrum",
    subtitle: "Weight-only view of how each head can transform what it reads",
    group: "weights",
    blurb:
      "Each attention head has 64 dots, one for each direction measured in this view. " +
      "A dot on the right means the head can copy that direction; a dot on the left " +
      "means it can reverse it. A dot away from the horizontal axis also includes a " +
      "rotation. Dots outside the unit-magnitude ring strengthen a direction; dots inside " +
      "weaken it. Select a head to see all 64 dots. For example, L11H8 looks mildly " +
      "copy-like on average (+0.29), yet it contains one strongly reversing direction " +
      "(an eigenvalue of −87.5). The full pattern can therefore reveal what one average hides.",
    math:
      "For each head, the first equation gives the 64 values plotted below. " +
      "They are the nonzero eigenvalues of the full 768 × 768 output-value map; " +
      "the second equality makes the smaller calculation exact. The last equation " +
      "sets the angle and the clipped logarithmic radius. " +
      "About 0.2% of points below that window meet at the center. Because the " +
      "matrix is real, complex values appear in mirrored conjugate pairs.",
    formulas: [
      {
        ariaLabel:
          "lambda equals the eigenvalues of W sub O times diagonal gamma sub 1 times W sub V; the eigenvalues of A B equal those of B A; angle equals argument lambda and radius equals base-10 logarithm of absolute lambda.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><mi>λ</mi><mo>=</mo><mi>eig</mi><mo>(</mo><msub><mi>W</mi><mi>O</mi></msub><mo>·</mo><mi>diag</mi><mo>(</mo><msub><mi>γ</mi><mn>1</mn></msub><mo>)</mo><mo>·</mo><msub><mi>W</mi><mi>V</mi></msub><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mi>eig</mi><mo>(</mo><mi>A</mi><mi>B</mi><mo>)</mo><mo>=</mo><mi>eig</mi><mo>(</mo><mi>B</mi><mi>A</mi><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mo>(</mo><mi>θ</mi><mo>,</mo><mi>r</mi><mo>)</mo><mo>=</mo><mo>(</mo><mi>arg</mi><mo>⁡</mo><mi>λ</mi><mo>,</mo><msub><mi>log</mi><mn>10</mn></msub><mo>|</mo><mi>λ</mi><mo>|</mo><mo>)</mo></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "ov_eigs.json — every head's OV spectrum was calculated " +
      "offline in 64-bit precision, with the first LayerNorm gain included " +
      "and biases excluded. The smaller calculation was checked against the " +
      "full 768×768 eigendecomposition; each head's copying score was also " +
      "cross-checked with heads.json to 4 decimal places.",
    legend: [
      { label: "eigenvalue from layer 0", rgb: "59,82,138" },
      { label: "eigenvalue from layer 6", rgb: "54,181,120" },
      { label: "eigenvalue from layer 11", rgb: "253,231,37" },
      { label: "|λ| = 1 ring: weaken inside, amplify outside", rgb: "166,173,200" },
    ],
    note:
      "Weight-only association, not observed behavior or a causal test · angle " +
      "shows arg λ · radius shows log₁₀|λ| in a clipped [−2,+2] window",
    legendCorner: "bl",
    create: () => new OVEigenDriver(),
  },
  {
    id: "comp-web",
    linksTo: ["head"],
    n: 2,
    label: "Attention-Head Connection Map",
    subtitle: "Potential paths between heads estimated from weights alone",
    group: "weights",
    blurb:
      "This map asks whether an earlier attention head's output is aligned " +
      "with a later head's query, key, or value channel. It uses weights only: " +
      "an arc is a possible route, not a record that information flowed during " +
      "a prompt. In the GPT-2-small data, previous-token head L4H11 aligns " +
      "with the key channels of L5H1 and L5H5 at 2.8× and 2.7× the measured " +
      "random baseline. Their query scores are only 1.9× and 1.2×, below the " +
      "default 2× display threshold. This pattern motivated the separate " +
      "repeated-sequence behavior test. Hover an arc for all three scores; " +
      "click a head to isolate its visible connections.",
    math:
      "Compare each earlier/later head pair with the three normalized matrix-overlap " +
      "scores defined below. Rank is at most 64, so smaller Gram " +
      "matrices give the exact result. The reference floor comes from 200 " +
      "seeded random Gaussian factor pairs (Elhage et al., 2021).",
    formulas: [
      {
        ariaLabel:
          "The output-value matrix M sub O V equals diagonal gamma sub 1 times W sub V times W sub O; the query-key matrix M sub Q K equals diagonal gamma sub 1 times W sub Q times W sub K transpose times diagonal gamma sub 1; Q, K and V are the normalized Frobenius overlaps shown.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><msub><mi>M</mi><mrow><mi>ov</mi></mrow></msub><mo>=</mo><mi>diag</mi><mo>(</mo><msub><mi>γ</mi><mn>1</mn></msub><mo>)</mo><msub><mi>W</mi><mi>V</mi></msub><msub><mi>W</mi><mi>O</mi></msub></mrow></mtd></mtr><mtr><mtd><mrow><msub><mi>M</mi><mrow><mi>qk</mi></mrow></msub><mo>=</mo><mi>diag</mi><mo>(</mo><msub><mi>γ</mi><mn>1</mn></msub><mo>)</mo><msub><mi>W</mi><mi>Q</mi></msub><msubsup><mi>W</mi><mi>K</mi><mo>T</mo></msubsup><mi>diag</mi><mo>(</mo><msub><mi>γ</mi><mn>1</mn></msub><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mi>Q</mi><mo>=</mo><mfrac><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>ov</mi><mn>1</mn></msubsup><msubsup><mi>M</mi><mi>qk</mi><mn>2</mn></msubsup><mo>‖</mo></mrow><mi>F</mi></msub><mrow><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>ov</mi><mn>1</mn></msubsup><mo>‖</mo></mrow><mi>F</mi></msub><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>qk</mi><mn>2</mn></msubsup><mo>‖</mo></mrow><mi>F</mi></msub></mrow></mfrac><mo>,</mo><mspace width=\"0.7em\"/><mi>K</mi><mo>=</mo><mfrac><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>qk</mi><mn>2</mn></msubsup><msubsup><mi>M</mi><mi>ov</mi><mn>1</mn></msubsup><mo>T</mo><mo>‖</mo></mrow><mi>F</mi></msub><mrow><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>ov</mi><mn>1</mn></msubsup><mo>‖</mo></mrow><mi>F</mi></msub><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>qk</mi><mn>2</mn></msubsup><mo>‖</mo></mrow><mi>F</mi></msub></mrow></mfrac></mrow></mtd></mtr><mtr><mtd><mrow><mi>V</mi><mo>=</mo><mfrac><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>ov</mi><mn>1</mn></msubsup><msubsup><mi>M</mi><mi>ov</mi><mn>2</mn></msubsup><mo>‖</mo></mrow><mi>F</mi></msub><mrow><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>ov</mi><mn>1</mn></msubsup><mo>‖</mo></mrow><mi>F</mi></msub><msub><mrow><mo>‖</mo><msubsup><mi>M</mi><mi>ov</mi><mn>2</mn></msubsup><mo>‖</mo></mrow><mi>F</mi></msub></mrow></mfrac></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "comp.json — contains 64-bit Q, K, and V scores for all " +
      "9,504 cross-layer head pairs, plus the random baseline (0.0361 ± " +
      "0.0004 for GPT-2-small). Same-layer pairs are excluded because those " +
      "heads run in parallel. LayerNorm gain is included, but its " +
      "input-dependent 1/σ scaling is not a weight and is not included. The " +
      "smaller calculations were checked against explicit 768×768 products.",
    legend: [
      { label: "faint arc: at the selected baseline multiple", rgb: "64,66,96" },
      { label: "gold arc: at least 6× the random baseline", rgb: "245,195,59" },
      { label: "light node: has a visible connection", rgb: "166,173,200" },
      { label: "dark node: none above the selected threshold", rgb: "118,126,158" },
    ],
    note:
      "Weight-only alignment, not measured behavior or causation · arc shape " +
      "is layout only · connections below the selected baseline multiple are hidden",
    legendCollapsed: true,
    legendCorner: "br",
    create: () => new CompositionWebDriver(),
  },
  {
    id: "induction-microscope",
    linksTo: ["head"],
    n: 2,
    label: "Repeated-Pattern Attention Test",
    subtitle: "Measured attention on two repeated random-token sequences",
    group: "forward",
    blurb:
      "The model receives an end-of-text token, 48 uniformly sampled random " +
      "tokens, and then the same 48 tokens again. During the second copy, an " +
      "induction-style head looks one step after the " +
      "earlier matching token. L5H1 scores 0.8937 on both seeds; L5H5 scores " +
      "0.9008 and 0.9486—about 64× the 0.0141 uniform-attention baseline. " +
      "L7H10 ranks first on seed 0 (0.9189), while L5H5 ranks first on seed 1; " +
      "L6H9 is also high (0.8953 and 0.9050). L4H11 instead leads previous-" +
      "token attention at 0.9788. Scores are available for all 144 heads; " +
      "click one of the eight heads with an exported grid to see its measured " +
      "97 × 97 attention pattern and the stripe 47 positions back.",
    math:
      "Average each head's attention over second-repeat positions. The equations " +
      "below identify the induction target, comparison targets, and uniform-attention " +
      "baseline. Random tokens reduce meaning-based cues " +
      "so this Olsson et al. (2022) diagnostic focuses on repeated structure.",
    formulas: [
      {
        ariaLabel:
          "Induction target equals t minus period plus 1; duplicate target equals t minus period; previous-token target equals t minus 1; uniform causal attention baseline equals the mean over t of 1 divided by t plus 1, or 0.0141.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><mtext>induction target</mtext><mo>=</mo><mi>t</mi><mo>−</mo><mtext>period</mtext><mo>+</mo><mn>1</mn></mrow></mtd></mtr><mtr><mtd><mrow><mtext>duplicate target</mtext><mo>=</mo><mi>t</mi><mo>−</mo><mtext>period</mtext><mo>,</mo><mspace width=\"0.7em\"/><mtext>previous target</mtext><mo>=</mo><mi>t</mi><mo>−</mo><mn>1</mn></mrow></mtd></mtr><mtr><mtd><mrow><mtext>baseline</mtext><mo>=</mo><mi>mean</mi><mspace width=\"0.2em\"/><mfrac><mn>1</mn><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></mfrac><mo>=</mo><mn>0.0141</mn></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "induction.json — records two complete forward passes " +
      "(seeds 0 and 1), both seeds' three scores for all 144 heads, and seed " +
      "0's full post-softmax grids for the top eight selected heads; all 144 " +
      "grids would be about 8 MB. Checks confirm exact repetition, causal " +
      "masking, score reproduction to 4 decimal places, and attention-row " +
      "sums within 2.4e−7 of 1.",
    legend: [
      { label: "gold: more attention to the selected structural target", rgb: "245,195,59" },
      { label: "dark: at or below the 0.0141 uniform baseline", rgb: "118,126,158" },
    ],
    note:
      "Measured behavior on two constructed sequences, not a general ability " +
      "claim · color spans the displayed baseline→maximum · no smoothing or gamma",
    legendCollapsed: true,
    legendCorner: "br",
    ownPrompts: true,
    create: () => new InductionDriver(),
  },
  {
    id: "ablation-ghosts",
    linksTo: ["head"],
    n: 17,
    label: "Attention-Head Removal Test",
    subtitle: "Causal loss change after replacing each head's output",
    group: "forward",
    blurb:
      "Each of GPT-2-small's 144 heads is changed in a separate run on the " +
      "seed-0 repeated-token sequence. Baseline loss falls from 13.1927 nats " +
      "on the first repeat to 0.2200 on the scored second-repeat window—about " +
      "60× lower. Removing top induction head L7H10 alone has little measured " +
      "effect (−0.0137 nats with zero replacement; +0.0047 with mean " +
      "replacement). Removing the top four together raises loss by +2.0356 " +
      "and +3.1894 nats, 2.8× and 3.2× their single-head sums. That " +
      "super-additivity is consistent with redundancy or interaction, but " +
      "does not identify the mechanism. Under mean replacement, L5H1 adds " +
      "+0.6520, L4H11 +0.3761, and L6H9 +0.2684 nats. Removing L10H7 or " +
      "L11H10 lowers loss by roughly 0.05–0.07 nats in both modes on this task.",
    math:
      "Compare average prediction loss in the changed and baseline runs over predicted " +
      "positions 50–96. The equations below define the loss difference and each " +
      "position's negative log probability, measured in nats. " +
      "The intervention replaces one head's a@v slice before c_proj either " +
      "with zero or with that run's mean across positions. Zero replacement " +
      "is off-distribution; mean replacement preserves the average signal but " +
      "removes position-specific variation. The two modes differ by as much " +
      "as 1.3005 nats, so both are shown.",
    formulas: [
      {
        ariaLabel:
          "Delta equals mean changed-run negative log likelihood minus mean baseline negative log likelihood. Negative log likelihood at position j equals negative logarithm of the probability of token s sub j given previous tokens.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><mi>Δ</mi><mo>=</mo><mi>mean</mi><mo>(</mo><mtext>changed-run NLL</mtext><mo>)</mo><mo>−</mo><mi>mean</mi><mo>(</mo><mtext>baseline NLL</mtext><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mi>NLL</mi><mo>(</mo><mi>j</mi><mo>)</mo><mo>=</mo><mo>−</mo><mi>log</mi><mo>⁡</mo><mi>p</mi><mo>(</mo><msub><mi>s</mi><mi>j</mi></msub><mo>|</mo><mrow><msub><mi>s</mi><mo>&lt;</mo></msub><mi>j</mi></mrow><mo>)</mo></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "ablation.json — its reported 293 runs are one baseline, 144 " +
      "heads × two replacement modes, and two head combinations × two modes. " +
      "An unchanged replay matches the baseline logits bit-for-bit (drift " +
      "0.0); exported 96-position curves reproduce Δ to 4 decimal places; the " +
      "sequence and induction scores match induction.json's seed-0 run.",
    legend: [
      { label: "gold: replacement raises loss on this sequence", rgb: "245,195,59" },
      { label: "blue: replacement lowers loss on this sequence", rgb: "96,165,250" },
      { label: "light line: unmodified loss curve", rgb: "205,210,228" },
    ],
    note:
      "Causal only for this intervention and sequence · color spans 0→the " +
      "selected mode's stated max |Δ| · 96 measured curve values, no smoothing",
    legendCollapsed: true,
    legendCorner: "br",
    ownPrompts: true,
    create: () => new AblationDriver(),
  },
  {
    id: "occlusion-vignette",
    n: 19,
    label: "Prompt-Token Intervention Test",
    subtitle: "Causal prediction change after replacing or deleting each input token",
    group: "forward",
    blurb:
      "Each prompt token is changed one at a time, followed by a complete " +
      "model run. The chart measures how much less likely the original top " +
      "prediction becomes. In the name-repetition prompt, deleting “ Mary” " +
      "changes the top prediction to “ John” and produces a +4.2202-nat drop; " +
      "deleting the second “ John” changes it to “ them” (+1.7927). Replacing " +
      "or deleting the “ E” in “Eiffel” changes the prediction from “ Paris” " +
      "to “ London”. Function words such as “ is”, “ of”, and “ to” can cause " +
      "the largest drops—up to +18.8979 nats—because this intervention measures " +
      "all support for the prediction, including grammar and position, not " +
      "meaning alone. It does not identify which internal mechanism used the token.",
    math:
      "At the final position, the equation below compares the baseline run's own top " +
      "next token with its probability after a change. " +
      "Positive means the token supported c; negative means it suppressed c. " +
      "Substitution puts <|endoftext|> in the same position. Deletion removes " +
      "the token, shifts every later token left, and changes those positional " +
      "embeddings, so the modes answer different questions. Raw-logit drops " +
      "and each changed run's own top prediction are also exported.",
    formulas: [
      {
        ariaLabel:
          "Drop at token position p equals logarithm of the baseline probability of c minus logarithm of the changed-run probability of c.",
        mathml:
          "<mrow><mtext>drop</mtext><mo>[</mo><mi>p</mi><mo>]</mo><mo>=</mo><mi>log</mi><mo>⁡</mo><msub><mi>p</mi><mtext>base</mtext></msub><mo>(</mo><mi>c</mi><mo>)</mo><mo>−</mo><mi>log</mi><mo>⁡</mo><msub><mi>p</mi><mtext>changed</mtext></msub><mo>(</mo><mi>c</mi><mo>)</mo></mrow>",
      },
    ],
    source:
      "occlusion.json — records 100 complete runs across five prompts and 45 " +
      "prompt tokens: five baselines, 90 interventions (one substitution and " +
      "one deletion per token), and five unchanged-baseline checks. Each check " +
      "matches Trace.logits exactly. Deleting " +
      "the final token reproduces the baseline's second-to-last logits " +
      "bit-for-bit (causal drift 0.0), and the name-prompt deletion result was " +
      "recalculated independently to 4 decimal places.",
    legend: [
      { label: "gold: token supported the original prediction", rgb: "245,195,59" },
      { label: "blue: token suppressed the original prediction", rgb: "96,165,250" },
      { label: "white outline: the top prediction changed", rgb: "255,255,255" },
    ],
    note:
      "Input-level intervention, not a mechanism explanation · one linear " +
      "scale in both directions · deletion shifts later positions · final-token " +
      "occlusion also changes which token the model predicts from",
    legendCollapsed: true,
    legendCorner: "br",
    create: () => new OcclusionDriver(),
  },
  {
    id: "logit-lens-tunnel",
    n: 3,
    label: "Prediction at Every Layer",
    subtitle: "How next-token candidates change from the input embedding to the final layer",
    group: "forward",
    blurb:
      "Each row asks what a direct readout would predict from the model’s working state " +
      "at that depth. Start at the bottom with the token-and-position embedding, then " +
      "move upward through the transformer blocks to the final output. Each box shows " +
      "one of the six leading next-token candidates, and its width is that token’s " +
      "stored probability. Empty space is probability assigned to every other token. " +
      "Gold marks the final winner. Intermediate rows are diagnostic estimates made " +
      "with the model’s final normalization, not predictions the model actually emitted.",
    math:
      "Apply the model’s final readout at each depth using the two equations below. " +
      "Row ℓ shows the six largest probabilities, and " +
      "each box width equals its stored probability. Layer 0 is the token-and-position " +
      "embedding. This is a direct logit lens, not a trained correction for intermediate states.",
    formulas: [
      {
        ariaLabel:
          "Logits sub ell equal final layer normalization of x sub ell at the last position times W sub U. P sub ell equals softmax of logits sub ell.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><msub><mi>logits</mi><mi>ℓ</mi></msub><mo>=</mo><msub><mi>LayerNorm</mi><mi>f</mi></msub><mo>(</mo><msub><mi>x</mi><mi>ℓ</mi></msub><mo>[</mo><mtext>last</mtext><mo>]</mo><mo>)</mo><mo>·</mo><msub><mi>W</mi><mi>U</mi></msub></mrow></mtd></mtr><mtr><mtd><mrow><msub><mi>P</mi><mi>ℓ</mi></msub><mo>=</mo><mi>softmax</mi><mo>(</mo><msub><mi>logits</mi><mi>ℓ</mi></msub><mo>)</mo></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "trace_*.json, field logit_lens_last — the NumPy GPT-2 forward " +
      "pass applies the model's final layer normalization and tied output-token matrix " +
      "to the last-position state from the embedding through the final block. It stores " +
      "the top six token probabilities rounded to four decimal places; the viewer does " +
      "not renormalize, smooth, or interpolate them.",
    legend: [
      { label: "gold: final layer's top prediction", rgb: "245,195,59" },
      { label: "blue: another top-six candidate; width = stored probability", rgb: "96,165,224" },
    ],
    note:
      "Read bottom to top: token-and-position embedding → final layer · empty track = " +
      "probability outside the stored top six, subject to four-decimal rounding",
    legendCorner: "br",
    create: () => new LogitLensTunnelDriver(),
  },
  {
    id: "attention-flow",
    n: 7,
    label: "Attention Connections",
    subtitle: "Where one attention head looks in a real prompt",
    group: "forward",
    blurb:
      "Choose a layer and attention head. Tokens on the left are queries asking " +
      "where to look; tokens on the right are possible source keys. A thicker, " +
      "brighter line means the selected head assigned more attention to that source. " +
      "Before values are rounded for storage, each query's attention is a probability " +
      "distribution " +
      "over itself and earlier tokens only; future tokens are blocked by the causal mask. " +
      "The head grid is shaded by focus, from diffuse attention to attention concentrated " +
      "on a few legal source positions.",
    math:
      "A softmax turns each query–key score into a probability. The equations below " +
      "also give the causal constraints and the normalized focus summary. The driver " +
      "uses the stored rounded values directly and draws only links with attn ≥ 0.008.",
    formulas: [
      {
        ariaLabel:
          "Attention at layer l, head h, query i and key j equals softmax over j of Q K transpose divided by the square root of d sub head plus the causal mask. Attention sums to 1 over j and is zero when j is greater than i. Focus equals 1 minus mean over i of attention entropy divided by base-2 log of i plus 1.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><mi>attn</mi><mo>[</mo><mi>l</mi><mo>,</mo><mi>h</mi><mo>,</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo>]</mo><mo>=</mo><msub><mi>softmax</mi><mi>j</mi></msub><mo>(</mo><mfrac><mrow><mi>Q</mi><msup><mi>K</mi><mi>T</mi></msup></mrow><msqrt><msub><mi>d</mi><mtext>head</mtext></msub></msqrt></mfrac><mo>+</mo><mtext>causal mask</mtext><mo>)</mo><mo>[</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo>]</mo></mrow></mtd></mtr><mtr><mtd><mrow><munderover><mo>Σ</mo><mi>j</mi><mi></mi></munderover><mi>attn</mi><mo>=</mo><mn>1</mn><mo>,</mo><mspace width=\"0.7em\"/><mi>attn</mi><mo>=</mo><mn>0</mn><mspace width=\"0.2em\"/><mtext>for</mtext><mspace width=\"0.2em\"/><mi>j</mi><mo>&gt;</mo><mi>i</mi></mrow></mtd></mtr><mtr><mtd><mrow><mi>focus</mi><mo>=</mo><mn>1</mn><mo>−</mo><mfrac><msub><mi>mean</mi><mi>i</mi></msub><mi>H</mi><mo>(</mo><mi>attn</mi><mo>[</mo><mi>l</mi><mo>,</mo><mi>h</mi><mo>,</mo><mi>i</mi><mo>,</mo><mo>:</mo><mo>]</mo><mo>)</mo><msub><mi>log</mi><mn>2</mn></msub><mo>(</mo><mi>i</mi><mo>+</mo><mn>1</mn><mo>)</mo></mfrac></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "trace_*.json, field attn, with shape " +
      "n_layer×n_head×T×T. These post-softmax attention probabilities come from " +
      "the selected NumPy GPT-2 forward pass and are stored to four decimal places. " +
      "They are not smoothed; rounding can make a stored row differ slightly from 1.",
    legend: [
      { label: "gold line: query → source; thicker + brighter = more attention", rgb: "245,195,59" },
      { label: "cyan: source-token column", rgb: "70,200,235" },
    ],
    note:
      "Hover a line for its stored probability; hover a token to isolate connections · " +
      "links below 0.008 are hidden · attention is not proof of causal effect",
    legendCorner: "br",
    create: () => new AttentionFlowDriver(),
  },
  {
    id: "attention-rollout",
    n: 23,
    label: "Attention Flow Across Layers",
    subtitle: "An approximate summary of how attention paths accumulate through the model",
    group: "forward",
    blurb:
      "Attention can connect tokens through several layers. This view combines those " +
      "connections into an attention-rollout estimate: a summary of how much aggregated " +
      "attention-path weight connects each source token to each destination token. Sources " +
      "run across the lattice and destinations run along its rows; every destination row " +
      "adds to 1. Taller, brighter columns have more estimated weight. The blank region " +
      "contains pairs where the source lies later than the destination, which the causal " +
      "mask forbids. This method averages all " +
      "heads and assumes an equal split between attention and the unchanged residual path, " +
      "so it is a useful summary—not causal proof or a direct measure of information flow.",
    math:
      "Average the heads at each layer, blend attention equally with the unchanged " +
      "residual path, then multiply those maps across layers as shown below. Every " +
      "rollout row remains a causal probability distribution. The final expression " +
      "sets the plotted logarithmic scale, so powers of ten land at evenly spaced ticks.",
    formulas: [
      {
        ariaLabel:
          "A sub l equals mean over heads of attention at layer l. A tilde sub l equals row normalize of one half A sub l plus one half identity. R sub d equals A tilde sub d times A tilde sub d minus 1 through A tilde sub 0. R sub d at i j is zero when j is greater than i. Display equals clamp of base-10 log R sub d at i j plus 4, divided by 4, between 0 and 1.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><msub><mi>A</mi><mi>l</mi></msub><mo>=</mo><msub><mi>mean</mi><mi>h</mi></msub><mi>attn</mi><mo>[</mo><mi>l</mi><mo>]</mo></mrow></mtd></mtr><mtr><mtd><mrow><msub><mi>Ã</mi><mi>l</mi></msub><mo>=</mo><mi>row_normalize</mi><mo>(</mo><mn>0.5</mn><mo>·</mo><msub><mi>A</mi><mi>l</mi></msub><mo>+</mo><mn>0.5</mn><mo>·</mo><mi>I</mi><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><msub><mi>R</mi><mi>d</mi></msub><mo>=</mo><msub><mi>Ã</mi><mi>d</mi></msub><mo>·</mo><msub><mi>Ã</mi><mrow><mi>d</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>·</mo><mo>…</mo><mo>·</mo><msub><mi>Ã</mi><mn>0</mn></msub><mo>,</mo><mspace width=\"0.7em\"/><msub><mi>R</mi><mi>d</mi></msub><mo>[</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo>]</mo><mo>=</mo><mn>0</mn><mspace width=\"0.2em\"/><mtext>for</mtext><mspace width=\"0.2em\"/><mi>j</mi><mo>&gt;</mo><mi>i</mi></mrow></mtd></mtr><mtr><mtd><mrow><mtext>display</mtext><mo>=</mo><mi>clamp</mi><mo>(</mo><mfrac><mrow><msub><mi>log</mi><mn>10</mn></msub><msub><mi>R</mi><mi>d</mi></msub><mo>[</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo>]</mo><mo>+</mo><mn>4</mn></mrow><mn>4</mn></mfrac><mo>,</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>)</mo></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "trace_*.json, field attn — stored to four decimal places. " +
      "The browser averages heads, restores each residual-augmented row to a " +
      "distribution, and multiplies the maps using 64-bit numbers. Method: Abnar and " +
      "Zuidema, “Quantifying Attention Flow in Transformers” (2020).",
    legend: [
      { label: "gold: higher estimated rollout weight on the log scale", rgb: "245,205,90" },
      { label: "dark: low, zero, or at/below the 10⁻⁴ display floor", rgb: "46,52,96" },
    ],
    note:
      "Drag to orbit · scroll to zoom · click a row to isolate one destination's " +
      "estimated sources · rollout summarizes attention paths; it is not causal evidence",
    legendCorner: "br",
    create: () => new AttentionRolloutDriver(),
  },
  {
    id: "residual-ribbon",
    n: 8,
    label: "Signal Strength by Token and Layer",
    subtitle: "How the size of each token's internal working state changes through the model",
    group: "forward",
    blurb:
      "Each column shows the Euclidean length of one token's residual-stream vector " +
      "at one stage of a real forward pass. Stages run from the token-and-position " +
      "embedding through every transformer block, while prompt tokens run toward " +
      "you. Height and glow use the same base-10 log scale from an absolute length " +
      "of 1 to the smallest full decade at or above the selected run's largest value. This " +
      "keeps small and large magnitudes visible together. Color identifies token " +
      "position only. The view measures vector size, not direction, meaning, or " +
      "whether a large state mattered to the prediction.",
    math:
      "The equations below define each state’s Euclidean length, the scale ceiling, " +
      "the displayed height and glow, and the growth factor. Layer 0 is the " +
      "token-and-position embedding; later layers are after each transformer block. " +
      "Stored lengths at or below 1, including zero, are placed at the scale base. " +
      "when that denominator is zero, the driver displays 0 as an unavailable-value " +
      "placeholder, not as a literal growth ratio.",
    formulas: [
      {
        ariaLabel:
          "The Euclidean length of x sub ell at token t equals the square root of the sum over k of x sub ell at t comma k squared. E equals the maximum of 1 and the ceiling of the base-10 logarithm of the maximum norm. Display equals clamp of the base-10 logarithm of the norm divided by E between 0 and 1. Growth equals the final norm divided by the embedding norm.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><msub><mrow><mo>‖</mo><msub><mi>x</mi><mi>ℓ</mi></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>‖</mo></mrow><mn>2</mn></msub><mo>=</mo><msqrt><munderover><mo>Σ</mo><mi>k</mi><mi></mi></munderover><msubsup><mi>x</mi><mrow><mi>ℓ</mi><mo>,</mo><mi>t</mi><mo>,</mo><mi>k</mi></mrow><mn>2</mn></msubsup></msqrt></mrow></mtd></mtr><mtr><mtd><mrow><mi>E</mi><mo>=</mo><mi>max</mi><mo>(</mo><mn>1</mn><mo>,</mo><mo>⌈</mo><msub><mi>log</mi><mn>10</mn></msub><mo>(</mo><mi>max</mi><mo>‖</mo><mi>x</mi><mo>‖</mo><msub><mo>₂</mo></msub><mo>)</mo><mo>⌉</mo><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mtext>display</mtext><mo>=</mo><mi>clamp</mi><mo>(</mo><mfrac><msub><mi>log</mi><mn>10</mn></msub><msub><mrow><mo>‖</mo><msub><mi>x</mi><mi>ℓ</mi></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>‖</mo></mrow><mn>2</mn></msub><mi>E</mi></mfrac><mo>,</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mtext>growth</mtext><mo>=</mo><mfrac><msub><mrow><mo>‖</mo><msub><mi>x</mi><mtext>final</mtext></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>‖</mo></mrow><mn>2</mn></msub><msub><mrow><mo>‖</mo><msub><mi>x</mi><mtext>embed</mtext></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>‖</mo></mrow><mn>2</mn></msub></mfrac></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "trace_*.json, field resid_norm, with shape " +
      "(n_layer+1)×T. The NumPy GPT-2 forward pass takes each float32 residual " +
      "vector's Euclidean norm at the embedding and after every block, then stores " +
      "the result rounded to three decimal places.",
    legend: [
      { label: "color: token position in the prompt", rgb: "245,195,59" },
      { label: "height + glow: residual length on a base-10 log scale", rgb: "148,140,165" },
    ],
    note:
      "Scale base = ‖x‖₂ 1; ceiling snaps up to a full decade · drag to orbit · " +
      "scroll to zoom · click a row to follow one token through the blocks",
    legendCorner: "br",
    create: () => new ResidualRibbonDriver(),
  },
  {
    id: "probability-simplex",
    n: 18,
    label: "Next-Token Probability Split",
    subtitle: "How probability is split between the top two predictions and everything else",
    group: "forward",
    blurb:
      "The triangle shows how the model divides its next-token probability. One corner " +
      "is the most likely token, one is the runner-up, and the third represents every " +
      "other token. A point near a named token means the model gives that token a large " +
      "share. A point near “all other” means tokens ranked third or lower have a large " +
      "combined share; that share may be spread across many tokens or concentrated in " +
      "one lower-ranked candidate. The chart uses the stored probabilities as they are; " +
      "it does not " +
      "stretch the top two to fill the triangle. The list beside it shows the top 12 " +
      "tokens separately and combines everything ranked 13th or lower.",
    math:
      "The producer calculates a full-vocabulary probability distribution and stores the " +
      "12 largest values to four decimal places. The equations below derive the remaining " +
      "probability and place the point in the triangle. If rounding makes the two largest " +
      "values exceed one, the remainder becomes zero and the point remains on the top-two " +
      "edge. In that edge case, the position preserves the stored runner-up coordinate and " +
      "uses the remaining top-one coordinate only to keep the point inside the " +
      "triangle; the printed stored probabilities remain unchanged. The ranks-13-and-lower " +
      "tail uses the same nonnegative remainder rule, so it likewise becomes zero if rounding pushes that sum " +
      "above 1. Neither calculation renormalizes the stored top values.",
    formulas: [
      {
        ariaLabel:
          "P equals softmax of final logits. P rest equals the maximum of zero and one minus p sub 1 minus p sub 2. X equals p sub 2 plus one half p rest; y equals square root of 3 divided by 2 times p rest. The barycentric point equals p sub 1 times A plus p sub 2 times B plus p rest times C.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><mi>P</mi><mo>=</mo><mi>softmax</mi><mo>(</mo><mtext>final logits</mtext><mo>)</mo><mo>,</mo><mspace width=\"0.7em\"/><msub><mi>p</mi><mtext>rest</mtext></msub><mo>=</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>−</mo><msub><mi>p</mi><mn>1</mn></msub><mo>−</mo><msub><mi>p</mi><mn>2</mn></msub><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mi>x</mi><mo>=</mo><msub><mi>p</mi><mn>2</mn></msub><mo>+</mo><mn>0.5</mn><mo>·</mo><msub><mi>p</mi><mtext>rest</mtext></msub><mo>,</mo><mspace width=\"0.7em\"/><mi>y</mi><mo>=</mo><mfrac><msqrt><mn>3</mn></msqrt><mn>2</mn></mfrac><mo>·</mo><msub><mi>p</mi><mtext>rest</mtext></msub></mrow></mtd></mtr><mtr><mtd><mrow><mi>point</mi><mo>=</mo><msub><mi>p</mi><mn>1</mn></msub><mo>·</mo><mi>A</mi><mo>+</mo><msub><mi>p</mi><mn>2</mn></msub><mo>·</mo><mi>B</mi><mo>+</mo><msub><mi>p</mi><mtext>rest</mtext></msub><mo>·</mo><mi>C</mi></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "trace_*.json, field final_topk — the NumPy GPT-2 forward pass " +
      "applies the model's final normalization and tied output-token matrix at the " +
      "last prompt position, softmaxes across the full vocabulary, and stores the top " +
      "12 token probabilities rounded to four decimal places. The viewer derives both " +
      "combined remainder values from those stored numbers.",
    legend: [
      { label: "gold corner: most likely token (p₁)", rgb: "245,195,59" },
      { label: "blue corner: second-most likely token (p₂)", rgb: "96,165,224" },
      { label: "gray corner: every token ranked third or lower (p_rest)", rgb: "123,130,156" },
    ],
    note:
      "Triangle coordinates use the stored four-decimal probabilities without rescaling · " +
      "bar lengths are scaled to the largest displayed bar, so read their printed percentages · " +
      "switch prompts to compare",
    legendCorner: "tl",
    create: () => new ProbabilitySimplexDriver(),
  },
  {
    id: "logit-attrib",
    linksTo: ["head"],
    n: 13,
    label: "Which Parts Supported the Prediction",
    subtitle: "A direct-path estimate of each component’s share of the top-two score gap",
    group: "forward",
    blurb:
      "This view breaks the winning next-token lead in the model’s internal score " +
      "(called a logit, before conversion to probabilities) into direct contributions. " +
      "The pieces are the input embedding, each attention head, each feed-forward (MLP) " +
      "block, and the model’s added biases. Gold pieces widen the winner’s lead; blue " +
      "pieces favor the runner-up. The running total shows how the gap builds by layer. " +
      "The pieces add to approximately the real score gap because the calculation keeps " +
      "the final normalization scale fixed at the value measured in this run. It cannot " +
      "show how a piece changes that scale or affects later processing, so this is a " +
      "direct-path breakdown—not a complete causal explanation.",
    math:
      "The first equation sums the final state from its direct pieces. The next two " +
      "equations calculate each piece's contribution and the final-normalization bias. " +
      "The value of σ is frozen at this forward pass’s measured final-LayerNorm " +
      "normalizer, which makes the contributions additive but excludes every " +
      "piece’s effect on σ itself. The final-LayerNorm bias contributes one " +
      "additional direct term.",
    formulas: [
      {
        ariaLabel:
          "The final state equals the embedding plus the sum of layer pieces v sub ell. Each layer piece is the sum over heads of head output plus attention-output bias plus MLP output. D is winner readout minus runner-up readout. Contribution of v equals v minus its mean, elementwise multiplied by final layer normalization gain, dot d, divided by sigma. Bias contribution equals beta dot d.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><msub><mi>x</mi><mtext>final</mtext></msub><mo>=</mo><mtext>embedding</mtext><mo>+</mo><munderover><mo>Σ</mo><mi>ℓ</mi><mi></mi></munderover><msub><mi>v</mi><mi>ℓ</mi></msub></mrow></mtd></mtr><mtr><mtd><mrow><msub><mi>v</mi><mi>ℓ</mi></msub><mo>=</mo><munderover><mo>Σ</mo><mi>h</mi><mi></mi></munderover><msub><mtext>head output</mtext><mrow><mi>ℓ</mi><mo>,</mo><mi>h</mi></mrow></msub><mo>+</mo><msub><mi>b</mi><mtext>attn</mtext></msub><mo>+</mo><msub><mi>MLP</mi><mi>ℓ</mi></msub></mrow></mtd></mtr><mtr><mtd><mrow><mi>d</mi><mo>=</mo><mtext>winner readout</mtext><mo>−</mo><mtext>runner-up readout</mtext></mrow></mtd></mtr><mtr><mtd><mrow><mtext>contribution</mtext><mo>(</mo><mi>v</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mo>(</mo><mi>v</mi><mo>−</mo><mi>mean</mi><mo>(</mo><mi>v</mi><mo>)</mo><mo>)</mo><mo>⊙</mo><mi>γ</mi><mo>·</mo><mi>d</mi></mrow><mi>σ</mi></mfrac></mrow></mtd></mtr><mtr><mtd><mrow><mtext>bias contribution</mtext><mo>=</mo><mi>β</mi><mo>·</mo><mi>d</mi></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "attrib.json — contains per-head, per-MLP, attention-output-bias, " +
      "and final-LayerNorm-bias " +
      "contributions for the bundled prompts. For every trace, the exported " +
      "pieces sum to the measured top-two score gap within 0.0006, and the " +
      "rebuilt final state matches the recorded state with relative error below " +
      "10⁻⁶. A head’s displayed “read” token is simply its highest-attended token " +
      "at the final query position, not evidence that the token caused its output.",
    legend: [
      { label: "supports the winning token (+; color clips at the scale limit)", rgb: "245,195,59" },
      { label: "supports the runner-up (−; color clips at the scale limit)", rgb: "96,150,255" },
      { label: "approximately zero after four-decimal export", rgb: "118,126,158" },
    ],
    note: "Additive with the measured final normalization scale held fixed. The b_o column is an attention-output bias that belongs to no individual head; final-LayerNorm β is shown separately.",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new LogitAttribDriver(),
  },
  {
    id: "causal-patching",
    n: 14,
    label: "Single-State Causal Patch Test",
    subtitle: "Where one clean internal state restores the clean answer’s score advantage",
    group: "forward",
    blurb:
      "This test starts with a clean prompt and a matched version in which one " +
      "name, country, or word is changed. For each layer and token position, it " +
      "copies one internal vector from the clean run into the changed run and " +
      "then reruns the rest of the model. Each cell is therefore a real " +
      "intervention. Its value reports how much of the clean answer’s score " +
      "advantage over the designated changed answer returns. A value of 0 means " +
      "the patch had no effect on that two-answer gap; 1 means it recovered the " +
      "entire clean-versus-changed gap. These one-location tests can still miss " +
      "information stored redundantly across several locations.",
    math:
      "The first equation measures how much of the clean answer's score gap returns; " +
      "the second defines that two-answer score gap at the final prompt position. " +
      "The patch replaces the residual state at " +
      "position p entering block i, then runs blocks i through 11 and the final " +
      "LayerNorm in the model’s own float32 path. The two answers are designated " +
      "single-token contrasts and are not necessarily each run’s top prediction; " +
      "their ranks are printed in the header.",
    formulas: [
      {
        ariaLabel:
          "Recovery r at layer i and position p equals logit difference after patching minus logit difference changed, divided by logit difference clean minus logit difference changed. Logit difference equals score of the clean answer minus score of the changed answer.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><mi>r</mi><mo>[</mo><mi>i</mi><mo>,</mo><mi>p</mi><mo>]</mo><mo>=</mo><mfrac><mrow><msub><mi>LD</mi><mtext>patched</mtext></msub><mo>−</mo><msub><mi>LD</mi><mtext>changed</mtext></msub></mrow><mrow><msub><mi>LD</mi><mtext>clean</mtext></msub><mo>−</mo><msub><mi>LD</mi><mtext>changed</mtext></msub></mrow></mfrac></mrow></mtd></mtr><mtr><mtd><mrow><mi>LD</mi><mo>=</mo><mtext>score(clean answer)</mtext><mo>−</mo><mtext>score(changed answer)</mtext></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "patch.json — contains 312 full patched continuations across " +
      "three matched prompt pairs. An unpatched resume from every layer " +
      "reproduces the changed-run scores with worst absolute error 3×10⁻⁵; a " +
      "full layer-0 replacement reproduces the clean run; and replacing an " +
      "identical token’s vector has no effect. Raw patched score differences " +
      "are exported beside normalized recovery r.",
    legend: [
      { label: "restores the clean answer’s advantage (toward r = 1)", rgb: "245,195,59" },
      { label: "moves farther toward the changed answer (r < 0)", rgb: "96,150,255" },
      { label: "little or no change in the two-answer gap", rgb: "118,126,158" },
    ],
    note: "Each cell is an intervention, not an attribution score. Color clips at |r| = 1, while hover shows values outside that range. One-site patches can understate redundant circuits.",
    legendCorner: "br",
    legendCollapsed: true,
    ownPrompts: true,
    create: () => new PatchingMapDriver(),
  },
  {
    id: "tuned-lens",
    n: 20,
    label: "Translated vs Raw Predictions by Layer",
    subtitle: "A least-squares translation compared with direct layer readout",
    group: "forward",
    blurb:
      "Early layers may store information in a different form from the final layer. A " +
      "raw readout ignores that difference and sends an early state straight through " +
      "the model’s final output machinery. The translated readout first applies a " +
      "layer-specific conversion learned from separate text. On held-out Alice text, " +
      "that conversion reduces the average layer-0 distribution difference (KL divergence) " +
      "from 71.7 bits to 2.6 bits. Its top token matches the final top token on 50% of " +
      "held-out positions by layer 8 instead of layer 11. This shows that the fitted " +
      "conversion matches the eventual output earlier; it does not prove what the model “knows.” The grid shows " +
      "both methods’ top token and exact difference from the final distribution.",
    math:
      "The first formula translates a layer state before the final readout. The next " +
      "formula gives the least-squares objective used to learn that translation; the " +
      "last is the raw comparison readout. Float64 normal equations give solve residual " +
      "4 × 10⁻¹⁵ and training R² from 0.44 at L0 to 0.94 at L11. Evaluation uses KL " +
      "divergence, in bits, over the complete 50,257-token softmax at 3,048 " +
      "held-out positions. The translator is trained on residual-state squared " +
      "error, not on KL divergence.",
    formulas: [
      {
        ariaLabel:
          "Translated sub L of h equals final layer normalization of A sub L times h plus b sub L, times W sub E transpose. A sub L and b sub L minimize the sum of squared norm of A times h sub L plus b minus h final. Raw readout of h equals final layer normalization of h times W sub E transpose.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><msub><mi>translated</mi><mi>L</mi></msub><mo>(</mo><mi>h</mi><mo>)</mo><mo>=</mo><mtext>final-LN</mtext><mo>(</mo><msub><mi>A</mi><mi>L</mi></msub><mo>·</mo><mi>h</mi><mo>+</mo><msub><mi>b</mi><mi>L</mi></msub><mo>)</mo><mo>·</mo><msup><msub><mi>W</mi><mi>E</mi></msub><mi>T</mi></msup></mrow></mtd></mtr><mtr><mtd><mrow><mo>(</mo><msub><mi>A</mi><mi>L</mi></msub><mo>,</mo><msub><mi>b</mi><mi>L</mi></msub><mo>)</mo><mo>=</mo><mi>arg min</mi><munderover><mo>Σ</mo><mtext>training positions</mtext><mi></mi></munderover><msup><mrow><mo>‖</mo><mi>A</mi><mo>·</mo><msub><mi>h</mi><mi>L</mi></msub><mo>+</mo><mi>b</mi><mo>−</mo><msub><mi>h</mi><mtext>final</mtext></msub><mo>‖</mo></mrow><mn>2</mn></msup></mrow></mtd></mtr><mtr><mtd><mrow><mtext>raw</mtext><mo>(</mo><mi>h</mi><mo>)</mo><mo>=</mo><mtext>final-LN</mtext><mo>(</mo><mi>h</mi><mo>)</mo><mo>·</mo><msup><msub><mi>W</mi><mi>E</mi></msub><mi>T</mi></msup></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "tuned.json — built from Alice’s Adventures in Wonderland " +
      "(Project Gutenberg #11; corpus SHA-256 recorded in metadata) in 128-token " +
      "windows. Every fourth window is held out and position 0 is dropped. An " +
      "independent layer-5 SVD least-squares refit matches the reported R², KL, " +
      "and agreement to four decimals; a prompt-grid cell was recomputed through " +
      "the model readout; both lenses are asserted to have exactly zero KL at " +
      "the final layer; and a repeat build is byte-identical.",
    legend: [
      { label: "least-squares translated readout", rgb: "245,195,59" },
      { label: "raw layer readout", rgb: "96,150,255" },
      { label: "cell color: KL to final, dark at 0 and amber at the clip limit", rgb: "143,119,60" },
      { label: "white outline: top token matches the final top token", rgb: "255,255,255" },
    ],
    note: "This is a residual least-squares approximation, not the KL-trained tuned lens of Belrose et al. (2023). The raw L0 point is drawn off-scale with 71.7 bits printed; hover shows exact KL.",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new TunedLensDriver(),
  },
  {
    id: "sae-decoder",
    linksTo: ["saeFeature"],
    n: 5,
    label: "Map of SAE Feature Directions",
    subtitle: "A two-dimensional PCA projection of 24,576 learned decoder directions",
    group: "sae",
    blurb:
      "A sparse autoencoder (SAE) is a separate analysis tool that learns recurring " +
      "patterns in model activity. This map shows all 24,576 patterns it learned from " +
      "GPT-2’s working state before block 8. Each dot marks the direction that one " +
      "feature adds when the SAE rebuilds that state. The map uses the first two PCA " +
      "directions, as in the token and neuron maps. Because the original directions " +
      "have 768 dimensions, nearby dots can still differ in ways this flat view hides. " +
      "The decoder directions are nearly equal in length, so size and brightness show " +
      "how often a feature fires—not vector length.",
    math:
      "Center the 24,576 × 768 decoder matrix by subtracting its mean row, find the " +
      "leading eigenvectors of its centered covariance matrix, and plot each row's first " +
      "two PCA scores, as shown below. Firing rate is shown as the base-10 logarithm of the fraction of " +
      "evaluation tokens on which the feature fires, with −10 used as the " +
      "release floor for inactive features. The token readout projects a centered " +
      "decoder direction through the final LayerNorm gain and tied output matrix; " +
      "it preserves token rank but models only the direct output path.",
    formulas: [
      {
        ariaLabel:
          "R sub c equals R minus its row mean. V are the leading eigenvectors of R sub c transpose R sub c. Coordinates equal R sub c times V sub 1 colon 2. Firing-rate display equals base-10 logarithm of the fraction of evaluation tokens on which the feature fires.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><msub><mi>R</mi><mi>c</mi></msub><mo>=</mo><mi>R</mi><mo>−</mo><mtext>row mean</mtext><mo>(</mo><mi>R</mi><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mi>V</mi><mo>=</mo><mi>eigenvectors</mi><mo>(</mo><msubsup><mi>R</mi><mi>c</mi><mi>T</mi></msubsup><msub><mi>R</mi><mi>c</mi></msub><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mtext>coordinates</mtext><mo>=</mo><msub><mi>R</mi><mi>c</mi></msub><msub><mi>V</mi><mrow><mn>1</mn><mo>:</mo><mn>2</mn></mrow></msub></mrow></mtd></mtr><mtr><mtd><mrow><mtext>firing-rate display</mtext><mo>=</mo><msub><mi>log</mi><mn>10</mn></msub><mo>(</mo><mtext>fraction that fire</mtext><mo>)</mo></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "sae.json — computed in float64 from W_dec in Joseph Bloom’s " +
      "jbloom/GPT2-Small-SAEs-Reformatted release at blocks.8.hook_resid_pre " +
      "(d_sae = 24,576). Firing rates come from the release’s " +
      "sparsity.safetensors evaluation measurements. Hover token effects are " +
      "direct-path projections: they skip GPT-2 blocks 8–11 and therefore are " +
      "not the feature’s full effect after later processing.",
    legend: [
      { label: "fires on about 10% of tokens (10⁻¹; top of ramp)", rgb: "253,231,37" },
      { label: "fires on about 0.1% of tokens (10⁻³; middle of ramp)", rgb: "71,189,110" },
      { label: "fires on 10⁻⁶ or fewer, including inactive features", rgb: "59,82,138" },
      { label: "decoder length is about 1; size shows firing rate", rgb: "205,210,224" },
    ],
    note: "Color and size clip firing rates to 10⁻⁶–10⁻¹. Position is a two-axis PCA projection, and hover token readouts skip blocks 8–11.",
    legendCorner: "tr",
    create: () => new SAEConstellationDriver(),
  },
  {
    id: "sae-piano-roll",
    linksTo: ["saeFeature"],
    n: 4,
    label: "SAE Features Active on Each Token",
    subtitle: "Measured sparse-autoencoder activations across one prompt",
    group: "sae",
    perTrace: true,
    blurb:
      "This view runs the same SAE encoder on GPT-2’s measured block-8 input for " +
      "each token in a prompt. The main board shows up to 32 features with the " +
      "largest peak after the first model token (not necessarily a whole word), " +
      "ordered by where that peak occurs. " +
      "By default, every row has its own brightness scale and prints its peak " +
      "activation on the right. GPT-2’s unusually large first-token state makes " +
      "a few features activate 60–100 times more strongly than the rest; those " +
      "features appear in a separate labeled band with their own scale. The lower " +
      "strip reports how many of all 24,576 features are active and the cosine " +
      "similarity between the SAE reconstruction and the centered model state at " +
      "each position.",
    math:
      "For each position, center the model state to match the SAE's TransformerLens " +
      "training basis without changing LayerNorm behavior. The equations below give the " +
      "encoder activations, reconstruction, similarity, and active-feature count. Cell " +
      "brightness is activation divided by that row’s peak, or by the " +
      "main board’s peak when the shared-scale toggle is selected.",
    formulas: [
      {
        ariaLabel:
          "x bar equals x minus mean of x. z equals x bar minus b decoder. Activations equal ReLU of z times W encoder plus b encoder. Reconstruction equals activations times W decoder plus b decoder. Similarity equals cosine of reconstruction and x bar. L zero equals the number of activations above zero.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><mover accent=\"true\"><mi>x</mi><mo>¯</mo></mover><mo>=</mo><mi>x</mi><mo>−</mo><mi>mean</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>,</mo><mspace width=\"0.7em\"/><mi>z</mi><mo>=</mo><mover accent=\"true\"><mi>x</mi><mo>¯</mo></mover><mo>−</mo><msub><mi>b</mi><mtext>dec</mtext></msub></mrow></mtd></mtr><mtr><mtd><mrow><mtext>activations</mtext><mo>=</mo><mi>ReLU</mi><mo>(</mo><mi>z</mi><mo>·</mo><msub><mi>W</mi><mtext>enc</mtext></msub><mo>+</mo><msub><mi>b</mi><mtext>enc</mtext></msub><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><mtext>reconstruction</mtext><mo>=</mo><mtext>activations</mtext><mo>·</mo><msub><mi>W</mi><mtext>dec</mtext></msub><mo>+</mo><msub><mi>b</mi><mtext>dec</mtext></msub></mrow></mtd></mtr><mtr><mtd><mrow><mtext>similarity</mtext><mo>=</mo><mi>cosine</mi><mo>(</mo><mtext>reconstruction</mtext><mo>,</mo><mover accent=\"true\"><mi>x</mi><mo>¯</mo></mover><mo>)</mo></mrow></mtd></mtr><mtr><mtd><mrow><msub><mi>L</mi><mn>0</mn></msub><mo>=</mo><mo>#</mo><mo>{</mo><mtext>activations</mtext><mo>&gt;</mo><mn>0</mn><mo>}</mo></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "sae_acts.json — produced by the res-jb encoder from " +
      "jbloom/GPT2-Small-SAEs-Reformatted at blocks.8.hook_resid_pre, applied to " +
      "the bundled prompts’ measured GPT-2 states. As a basis check, uncentered " +
      "Hugging Face states give about 2,700 active features and reconstruction " +
      "cosine about 0.76; exact per-position centering restores the release’s " +
      "expected range of about 30–100 active features and cosine 0.93–0.9999.",
    legend: [
      { label: "strongest activation in that row (value printed at right)", rgb: "245,195,59" },
      { label: "half of that row’s peak activation", rgb: "155,131,78" },
      { label: "zero activation; values are not smoothed", rgb: "118,126,158" },
    ],
    note: "Default brightness is per row; a toggle uses one main-board scale. The first-token outlier band always keeps its own scale. Stored L0 and reconstruction cosine are printed exactly.",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new SAEPianoRollDriver(),
  },
  {
    id: "decoder-cosine-web",
    linksTo: ["saeFeature"],
    n: 12,
    label: "Similar Learned Features",
    subtitle: "How closely each sparse-autoencoder feature resembles its nearest neighbor",
    group: "sae",
    blurb:
      "Each dot is one of the SAE’s 24,576 learned features. Its height shows how " +
      "closely its output direction matches its most similar neighbor; its horizontal " +
      "position shows the base-10 logarithm of the fraction of evaluation tokens where " +
      "the feature fires. Every feature is compared with every other feature. At least " +
      "eight features have a nearest-neighbor cosine of 1.0000 and a newline as their " +
      "top token through the direct output readout; 24 have a score above 0.9. The median " +
      "is 0.525, higher than 99.9% of the sampled random pairs. Close directions may " +
      "reveal duplicated or split features, but they do not prove that the features " +
      "activate together or behave identically.",
    math:
      "Treat each feature’s decoder-weight vector—the direction it adds back to the " +
      "model—as its output direction. Normalize every such vector, compare each " +
      "feature with all 24,575 other features, and keep the largest cosine " +
      "similarity. A pair is mutual when each feature is the other's closest " +
      "match. The guide lines come from a seeded sample of 199,991 distinct " +
      "random feature pairs (mean 0.0041, 99th percentile 0.1712, 99.9th " +
      "percentile 0.3357, maximum 0.6965); this baseline is sampled, while " +
      "the nearest-neighbor search is exhaustive.",
    source:
      "sae_web.json joined with sae.json — uses the open res-jb " +
      "SAE at GPT-2's layer-8 input. The cosine scan uses 32-bit matrix " +
      "calculations; seeded checks against 64-bit dot products differ by less " +
      "than 0.00001. The horizontal axis uses the firing rate published with " +
      "the SAE release.",
    legend: [
      { label: "Gold: two features are each other's closest match", rgb: "245,195,59" },
      { label: "Gray: only one feature chooses the other", rgb: "138,146,178" },
      { label: "Guide lines: cosine from 199,991 sampled random pairs", rgb: "118,126,158" },
    ],
    note:
      "Both axes are measured values; there is no projection. Point order is " +
      "shuffled reproducibly to avoid draw-order bias, and the pair list shows " +
      "each feature only once. The firing-rate axis runs from 10⁻¹⁰ to 1; values " +
      "outside that range are clipped to its dead-feature and 100% endpoints.",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new SAEWebDriver(),
  },
  {
    id: "direction-compass",
    linksTo: ["saeFeature"],
    n: 22,
    label: "What Learned Features Resemble",
    subtitle: "Whether each SAE feature is closer to a neuron output or a token embedding",
    group: "sae",
    blurb:
      "Each dot is one of 24,576 SAE features. The horizontal axis shows its " +
      "closest token-embedding vector; this is geometric resemblance, not evidence " +
      "that the feature fires on that token. The vertical axis shows its closest MLP " +
      "neuron-output match. Both searches are exhaustive across all 50,257 " +
      "tokens and 36,864 neurons. About 98.4% of features fall above the " +
      "diagonal, so their closest neuron is more similar than their closest " +
      "token; 32 have a neuron match above 0.9. Of the closest neurons, 93.82% " +
      "are in layers 0–7, before this layer-8 SAE measurement. Similarity is " +
      "geometric evidence, not proof that a neuron created or caused a feature.",
    math:
      "Normalize every SAE decoder direction, neuron " +
      "output direction, and token embedding, then keep the largest signed " +
      "cosine in each comparison set. Cosine is not replaced by its absolute " +
      "value, so a strong opposite direction is not treated as a match; any " +
      "negative maximum would be clipped to zero on the displayed 0–1 axes. For a " +
      "chance reference, 2,000 seeded random 768-dimensional directions are " +
      "scanned in the same way; their average best match is 0.1486 for neurons " +
      "and 0.1304 for tokens, with 99th percentiles of 0.1783 and 0.1777.",
    source:
      "compass.json joined with sae.json — uses the open res-jb " +
      "SAE at the input to GPT-2 block 8. Seeded 64-bit checks reproduce both " +
      "the best partner and its score; 32-bit scan drift is below 0.00001. A " +
      "match in layers 8–11 occurs after the SAE measurement, so the viewer " +
      "labels it as geometric only, not an upstream source.",
    legend: [
      { label: "Blue end of scale: closest neuron is in layer 0", rgb: "59,82,138" },
      { label: "Green middle of scale: closest neuron is around layer 6", rgb: "54,181,120" },
      { label: "Yellow end of scale: layer 11, after the SAE measurement", rgb: "253,231,37" },
      { label: "Guide lines: best matches for 2,000 random directions", rgb: "118,126,158" },
    ],
    note:
      "Both axes are exhaustive best-match cosines on the same 0–1 scale; " +
      "there is no projection. Color covers every neuron layer from 0 through " +
      "11. Point order is shuffled reproducibly, and repeated best-matching " +
      "partners are listed once.",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new CompassDriver(),
  },
  {
    id: "cofire-venn",
    linksTo: ["saeFeature"],
    n: 24,
    label: "Which Learned Features Activate Together",
    subtitle: "How direction similarity relates to shared activations in one book",
    group: "sae",
    blurb:
      "Each point is a pair of SAE features measured across 44,179 token positions " +
      "from Alice’s Adventures in Wonderland. The horizontal axis shows how similar " +
      "their output directions are. The vertical axis shows whether they fire together " +
      "more or less often than chance would predict from their separate firing rates. " +
      "Among all 854,237 pairs that fired together at least 20 times, direction " +
      "cosine and co-firing PMI have a correlation of 0.50. They are related, but " +
      "far from identical, and the association does not show that one feature causes " +
      "the other. The chart displays the 20,000 pairs with the largest G² association " +
      "statistic among those supported pairs, not the highest-PMI pairs. Select one " +
      "to see exact firing-position counts: each circle’s area shows one feature’s " +
      "count, and the overlap area shows their joint count.",
    math:
      "A feature counts as firing when its centered SAE activation is above zero. For " +
      "features i and j, the equations below give the independent expected joint count " +
      "and pointwise mutual information (PMI). The " +
      "export keeps the top 20,000 pairs by Dunning's G² statistic from all " +
      "pairs with at least 20 joint activations. Zero-co-firing examples cannot " +
      "pass that support rule, so separate avoidance examples come from the " +
      "300 most active features.",
    formulas: [
      {
        ariaLabel:
          "Expected joint count for features i and j equals count i times count j divided by 44,179. Pointwise mutual information equals base-2 logarithm of observed joint count divided by expected joint count.",
        mathml:
          "<mtable columnalign=\"left\"><mtr><mtd><mrow><mi>E</mi><mo>[</mo><msub><mi>n</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>]</mo><mo>=</mo><mfrac><msub><mi>n</mi><mi>i</mi></msub><msub><mi>n</mi><mi>j</mi></msub><mn>44,179</mn></mfrac></mrow></mtd></mtr><mtr><mtd><mrow><mi>PMI</mi><mo>=</mo><msub><mi>log</mi><mn>2</mn></msub><mo>(</mo><mfrac><mtext>observed joint count</mtext><mtext>expected joint count</mtext></mfrac><mo>)</mo></mrow></mtd></mtr></mtable>",
      },
    ],
    source:
      "cofire.json joined with sae.json — the text is public-" +
      "domain Project Gutenberg eBook #11, and its checksum is stored in the " +
      "bundle. Counts from sparse matrix multiplication match independent row " +
      "intersections for every exported pair. A seeded 200-pair shuffle gives " +
      "0.9869 times the independent expectation, providing a measured chance " +
      "check. The 128-token window matches the SAE's training context.",
    legend: [
      { label: "Point color: joint activation count on a log scale", rgb: "54,181,120" },
      { label: "Blue circle: token positions where feature i fires", rgb: "96,165,250" },
      { label: "Pink circle: token positions where feature j fires", rgb: "244,114,182" },
      { label: "PMI = 0 line: co-firing expected under independence", rgb: "118,126,158" },
    ],
    note:
      "The first position of each 128-token window is excluded because it is " +
      "both a chunk boundary and a known activation outlier. This is a ranked " +
      "20,000-pair view, not a complete display of all supported pairs; rare " +
      "pairs can dominate high PMI values. The measured relationship is specific " +
      "to this book, its tokenization, and this windowing procedure.",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new CofireDriver(),
  },
  {
    id: "grokking-clock",
    n: 16,
    label: "When a Small Model Starts to Generalize",
    subtitle: "A separate modular-addition toy model, not GPT-2",
    group: "trained",
    blurb:
      "This view follows a small network trained to add numbers modulo 97; it " +
      "is a separate toy experiment, not GPT-2. The first stored checkpoint above " +
      "99.9% training accuracy is step 3,800; the first held-out checkpoint above " +
      "99.9% is step 7,900. Held-out accuracy begins " +
      "rising sharply well before that second threshold: it climbs from 0.60% at " +
      "step 3,000 to 40.48% at 3,800, 86.21% at 4,000, and 98.53% at 4,500. " +
      "At the same time, individual hidden units’ input weights concentrate on one " +
      "frequency; this timing does not show that frequency structure causes the " +
      "accuracy rise. Projecting the 97 possible values of the first addend onto a " +
      "frequency pair places them around a repeated circle, or 'clock.'",
    math:
      "A two-layer, 128-hidden-unit network learns the modular-addition target shown " +
      "below, with squared activations, mean-squared-error loss, " +
      "full-batch AdamW, and weight decay 0.3. A unit's frequency purity is " +
      "the largest non-constant Fourier frequency's share of that unit's " +
      "input-weight power. Clock quality measures how closely a projected " +
      "frequency follows a circle; the best measured frequency is 24 with a " +
      "score of 0.964.",
    formulas: [
      {
        ariaLabel: "Target equals a plus b modulo 97.",
        mathml:
          "<mrow><mtext>target</mtext><mo>=</mo><mo>(</mo><mi>a</mi><mo>+</mo><mi>b</mi><mo>)</mo><mtext> mod </mtext><mn>97</mn></mrow>",
      },
    ],
    source:
      "grok.json — the toy model was trained from scratch in " +
      "NumPy with seed 0 on all 9,409 ordered pairs (a, b), with a and b each " +
      "from 0 to 96. Inputs are two joined 97-place one-hot vectors and the " +
      "target is a 97-place one-hot answer. A seeded fixed permutation assigns " +
      "2,069 pairs (22%) to training and 7,340 to held-out testing. The file " +
      "stores 120 checkpoints every 100 steps through 11,900; complete splits " +
      "are evaluated without smoothing. Reruns with the same seed and software " +
      "are identical, and clock scores were recalculated from exported coordinates. " +
      "The five largest non-constant frequencies hold 15.35% of total input-weight " +
      "spectral power across all hidden units and frequencies.",
    legend: [
      { label: "Gold: test accuracy", rgb: "245,195,59" },
      { label: "Gray: training accuracy", rgb: "138,146,178" },
      { label: "Cyan: median hidden-unit frequency purity and middle 50%", rgb: "70,200,235" },
      { label: "Heat map: input-weight power at each frequency", rgb: "155,131,78" },
    ],
    note:
      "This is an offline toy model, not GPT-2. Axes are linear and curves are " +
      "unsmoothed. The heat scale clips at the largest non-constant value, so " +
      "the constant-frequency cells can saturate. Clock choices are ranked by " +
      "their measured circularity. This single seeded run does not establish the " +
      "same transition for other seeds, splits, or training settings.",
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new GrokClockDriver(),
  },
  {
    id: "live-nebula",
    n: 25,
    label: "Live Predictions by Layer",
    subtitle: "Final-readout probes of GPT-2's intermediate states for text you type",
    group: "live",
    blurb:
      "Type a prompt to run GPT-2 through the configured probe server. The grid " +
      "applies GPT-2’s final output readout to each intermediate state and shows " +
      "which next token that probe ranks first at every layer and token position. " +
      "These probes are not native intermediate predictions and do not explain why " +
      "the result changes. Brighter cells mean lower entropy across the complete " +
      "distribution—not merely a higher probability for the displayed token—and " +
      "an outline marks agreement with the final layer. After each edit, the " +
      "viewer waits 450 ms before " +
      "requesting a fresh calculation and displays the server's measured " +
      "compute time. The default endpoint is local, so prompts and about 0.5 GB " +
      "of model weights stay on this computer. The bundled local server does not " +
      "save or upload prompt text. If you configure a remote endpoint, the prompt " +
      "is sent there and its handling depends on that server. Errors are shown " +
      "rather than replaced with sample data.",
    math:
      "Apply GPT-2’s normal final readout to the working " +
      "state entering each block, plus the final state after block 11, then " +
      "calculate the full 50,257-token " +
      "probability distribution with 64-bit entropy and KL divergence. Entropy " +
      "measures uncertainty on an absolute 0-to-15.617-bit scale; KL measures " +
      "distance from the final distribution. Row 12 is that final post-block " +
      "state and is checked against the model's actual output: its top token " +
      "matches and measured KL is below " +
      "0.0001 bits.",
    source:
      "A POST to /live/forward, handled by src/nebulai/backend/interp/" +
      "live_server.py — by default this is a loopback-only local Python " +
      "server using the same NumPy GPT-2 forward pass as the offline bundles; " +
      "the endpoint can be changed in Settings → Model Probing. Independent " +
      "cell calculations match the response at export precision, the HTTP and " +
      "direct-function results match, and repeated runs are deterministic.",
    legend: [
      { label: "Bright gold: lower entropy across the full token distribution", rgb: "245,195,59" },
      { label: "Dark: high uncertainty, close to uniform", rgb: "40,42,60" },
      { label: "White outline: top prediction matches the final layer", rgb: "255,255,255" },
    ],
    note:
      "Requests run after a 450 ms pause and prompts are limited to 96 tokens; " +
      "longer prompts are truncated and labeled. The displayed time measures " +
      "server computation, not network travel. The first token can produce a " +
      "known large-activation effect that makes early layers look almost " +
      "uniform.",
    ownPrompts: true,
    legendCorner: "br",
    legendCollapsed: true,
    create: () => new LiveNebulaDriver(),
  },
];

export function findFeature(id: string): InterpFeature | undefined {
  return INTERP_FEATURES.find((f) => f.id === id);
}

export const GROUP_LABEL: Record<InterpFeature["group"], string> = {
  weights: "Model weights",
  forward: "Prompt analysis",
  sae: "Learned features (SAE)",
  trained: "Small experiment",
  live: "Your prompt",
};
