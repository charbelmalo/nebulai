export interface ResearchPaper {
  title: string;
  citation: string;
  url: string;
}

const paper = (title: string, citation: string, url: string): ResearchPaper => ({ title, citation, url });

/** Curated paper-level references for every live Internals guide view. */
export const GUIDE_RESEARCH = {
  "fourier-atlas": [
    paper("On the Spectral Bias of Neural Networks", "Rahaman et al. (2019)", "https://arxiv.org/abs/1806.08734"),
    paper("Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", "Tancik et al. (2020)", "https://arxiv.org/abs/2006.10739"),
    paper("On the Frequency Bias of Neural Networks", "Xu et al. (2019)", "https://arxiv.org/abs/1807.01251"),
  ],
  "weight-spectrum": [
    paper("The Heavy-Tailed Theory of Neural Networks", "Martin and Mahoney (2019)", "https://arxiv.org/abs/1805.08210"),
    paper("Traditional and Heavy-Tailed Self Regularization in Neural Network Models", "Martin and Mahoney (2019)", "https://arxiv.org/abs/1901.08278"),
    paper("Implicit Self-Regularization in Deep Neural Networks", "Martin et al. (2021)", "https://arxiv.org/abs/2012.00173"),
  ],
  "embedding-constellation": [
    paper("The Geometry of Categorical and Hierarchical Concepts in Large Language Models", "Park et al. (2024)", "https://arxiv.org/abs/2406.01506"),
    paper("Uncovering Hidden Geometry in Transformers via Disentangling Position and Context", "Song and Zhong (2023)", "https://arxiv.org/abs/2310.04861"),
    paper("The Geometry of Hidden Representations of Large Transformer Models", "Valeriani et al. (2023)", "https://arxiv.org/abs/2302.00294"),
  ],
  "neuron-field": [
    paper("Transformer Feed-Forward Layers Are Key-Value Memories", "Geva et al. (2021)", "https://arxiv.org/abs/2012.14913"),
    paper("Knowledge Neurons in Pretrained Transformers", "Dai et al. (2022)", "https://arxiv.org/abs/2104.08696"),
    paper("Locating and Editing Factual Associations in GPT", "Meng et al. (2022)", "https://arxiv.org/abs/2202.05262"),
  ],
  "head-fingerprints": [
    paper("In-context Learning and Induction Heads", "Olsson et al. (2022)", "https://arxiv.org/abs/2209.11895"),
    paper("Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small", "Wang et al. (2022)", "https://arxiv.org/abs/2211.00593"),
    paper("Copy Suppression: Comprehensively Understanding an Attention Head", "McDougall et al. (2023)", "https://arxiv.org/abs/2310.04625"),
  ],
  "ov-eigen": [
    paper("A Mathematical Framework for Transformer Circuits", "Elhage et al. (2021)", "https://transformer-circuits.pub/2021/framework/index.html"),
    paper("In-context Learning and Induction Heads", "Olsson et al. (2022)", "https://arxiv.org/abs/2209.11895"),
    paper("Copy Suppression: Comprehensively Understanding an Attention Head", "McDougall et al. (2023)", "https://arxiv.org/abs/2310.04625"),
  ],
  "comp-web": [
    paper("Talking Heads: Understanding Inter-layer Communication in Transformer Language Models", "Merullo, Eickhoff, and Pavlick (2024)", "https://arxiv.org/abs/2406.09519"),
    paper("In-context Learning and Induction Heads", "Olsson et al. (2022)", "https://arxiv.org/abs/2209.11895"),
    paper("Out-of-distribution Generalization via Composition: a Lens through Induction Heads in Transformers", "Song, Xu, and Zhong (2024)", "https://arxiv.org/abs/2408.09503"),
  ],
  "induction-microscope": [
    paper("In-context Learning and Induction Heads", "Olsson et al. (2022)", "https://arxiv.org/abs/2209.11895"),
    paper("What Needs to Go Right for an Induction Head?", "Singh et al. (2024)", "https://arxiv.org/abs/2404.07129"),
    paper("Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers", "Chen et al. (2024)", "https://arxiv.org/abs/2409.10559"),
  ],
  "ablation-ghosts": [
    paper("Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning", "Crosbie and Shutova (2024)", "https://arxiv.org/abs/2407.07011"),
    paper("Copy Suppression: Comprehensively Understanding an Attention Head", "McDougall et al. (2023)", "https://arxiv.org/abs/2310.04625"),
    paper("Towards Best Practices of Activation Patching in Language Models", "Zhang and Nanda (2023)", "https://arxiv.org/abs/2309.16042"),
  ],
  "occlusion-vignette": [
    paper("Considering Likelihood in NLP Classification Explanations with Occlusion and Language Modeling", "Harbecke and Alt (2020)", "https://arxiv.org/abs/2004.09890"),
    paper("Explaining Natural Language Processing Classifiers with Occlusion and Language Modeling", "Harbecke (2021)", "https://arxiv.org/abs/2101.11889"),
    paper("RISE: Randomized Input Sampling for Explanation of Black-box Models", "Petsiuk, Das, and Saenko (2018)", "https://arxiv.org/abs/1806.07421"),
  ],
  "logit-lens-tunnel": [
    paper("Eliciting Latent Predictions from Transformers with the Tuned Lens", "Belrose et al. (2023)", "https://arxiv.org/abs/2303.08112"),
    paper("Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small", "Wang et al. (2022)", "https://arxiv.org/abs/2211.00593"),
    paper("Do Language Models Have Beliefs? Methods for Detecting, Updating, and Visualizing Model Beliefs", "Azaria and Mitchell (2023)", "https://arxiv.org/abs/2205.11129"),
  ],
  "attention-flow": [
    paper("Quantifying Attention Flow in Transformers", "Abnar and Zuidema (2020)", "https://arxiv.org/abs/2005.00928"),
    paper("Attention Flows are Shapley Value Explanations", "Ethayarajh and Jurafsky (2021)", "https://arxiv.org/abs/2105.14652"),
    paper("Attention Flows for General Transformers", "Metzger et al. (2022)", "https://arxiv.org/abs/2205.15389"),
  ],
  "attention-rollout": [
    paper("Quantifying Attention Flow in Transformers", "Abnar and Zuidema (2020)", "https://arxiv.org/abs/2005.00928"),
    paper("Transformer Interpretability Beyond Attention Visualization", "Chefer, Gur, and Wolf (2020)", "https://arxiv.org/abs/2012.09838"),
    paper("GMAR: Gradient-Driven Multi-Head Attention Rollout for Vision Transformer Interpretability", "Jo, Jang, and Park (2025)", "https://arxiv.org/abs/2504.19414"),
  ],
  "residual-ribbon": [
    paper("A Mathematical Framework for Transformer Circuits", "Elhage et al. (2021)", "https://transformer-circuits.pub/2021/framework/index.html"),
    paper("Transformer Feed-Forward Layers Are Key-Value Memories", "Geva et al. (2021)", "https://arxiv.org/abs/2012.14913"),
    paper("Residual Stream Analysis with Multi-Layer SAEs", "Lawson et al. (2025)", "https://arxiv.org/abs/2409.04185"),
  ],
  "probability-simplex": [
    paper("Attention Is All You Need", "Vaswani et al. (2017)", "https://arxiv.org/abs/1706.03762"),
    paper("Normalized Attention Without Probability Cage", "Richter and Wattenhofer (2020)", "https://arxiv.org/abs/2005.09561"),
    paper("Constrained Belief Updates Explain Geometric Structures in Transformer Representations", "Piotrowski et al. (2025)", "https://arxiv.org/abs/2502.01954"),
  ],
  "logit-attrib": [
    paper("An Adversarial Example for Direct Logit Attribution: Memory Management in GELU-4L", "Janiak et al. (2023)", "https://arxiv.org/abs/2310.07325"),
    paper("Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small", "Wang et al. (2022)", "https://arxiv.org/abs/2211.00593"),
    paper("Copy Suppression: Comprehensively Understanding an Attention Head", "McDougall et al. (2023)", "https://arxiv.org/abs/2310.04625"),
  ],
  "causal-patching": [
    paper("Towards Best Practices of Activation Patching in Language Models", "Zhang and Nanda (2023)", "https://arxiv.org/abs/2309.16042"),
    paper("Attribution Patching Outperforms Automated Circuit Discovery", "Syed, Rager, and Conmy (2024)", "https://aclanthology.org/2024.blackboxnlp-1.25/"),
    paper("Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers", "Dumas et al. (2024)", "https://arxiv.org/abs/2411.08745"),
  ],
  "tuned-lens": [
    paper("Eliciting Latent Predictions from Transformers with the Tuned Lens", "Belrose et al. (2023)", "https://arxiv.org/abs/2303.08112"),
    paper("Does Transformer Interpretability Transfer to RNNs?", "Paulo, Marshall, and Belrose (2024)", "https://arxiv.org/abs/2404.05971"),
    paper("On the Effect of Uncertainty on Layer-wise Inference Dynamics", "Kim, Yoo, and Oh (2025)", "https://arxiv.org/abs/2507.06722"),
  ],
  "sae-decoder": [
    paper("The Geometry of Concepts: Sparse Autoencoder Feature Structure", "Li et al. (2024)", "https://arxiv.org/abs/2410.19750"),
    paper("Improving Dictionary Learning with Gated Sparse Autoencoders", "Rajamanoharan et al. (2024)", "https://arxiv.org/abs/2404.16014"),
    paper("Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control", "Makelov, Lange, and Nanda (2024)", "https://arxiv.org/abs/2405.08366"),
  ],
  "sae-piano-roll": [
    paper("Sparse Autoencoders Find Highly Interpretable Directions in Language Model Activation Space", "Cunningham et al. (2023)", "https://arxiv.org/abs/2309.08600"),
    paper("Improving Dictionary Learning with Gated Sparse Autoencoders", "Rajamanoharan et al. (2024)", "https://arxiv.org/abs/2404.16014"),
    paper("Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", "Templeton et al. (2024)", "https://arxiv.org/abs/2406.04093"),
  ],
  "decoder-cosine-web": [
    paper("The Geometry of Concepts: Sparse Autoencoder Feature Structure", "Li et al. (2024)", "https://arxiv.org/abs/2410.19750"),
    paper("Sparse Autoencoders Find Highly Interpretable Directions in Language Model Activation Space", "Cunningham et al. (2023)", "https://arxiv.org/abs/2309.08600"),
    paper("Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", "Templeton et al. (2024)", "https://arxiv.org/abs/2406.04093"),
  ],
  "direction-compass": [
    paper("Representation Engineering: A Top-Down Approach to AI Transparency", "Zou et al. (2023)", "https://arxiv.org/abs/2310.01405"),
    paper("Activation Addition: Steering Language Models Without Optimization", "Turner et al. (2023)", "https://arxiv.org/abs/2308.10248"),
    paper("Inference-Time Intervention: Eliciting Truthful Answers from a Language Model", "Li et al. (2023)", "https://arxiv.org/abs/2306.03341"),
  ],
  "cofire-venn": [
    paper("Sparse Autoencoders Find Highly Interpretable Directions in Language Model Activation Space", "Cunningham et al. (2023)", "https://arxiv.org/abs/2309.08600"),
    paper("Improving Dictionary Learning with Gated Sparse Autoencoders", "Rajamanoharan et al. (2024)", "https://arxiv.org/abs/2404.16014"),
    paper("Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", "Templeton et al. (2024)", "https://arxiv.org/abs/2406.04093"),
  ],
  "grokking-clock": [
    paper("Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets", "Power et al. (2022)", "https://arxiv.org/abs/2201.02177"),
    paper("Progress Measures for Grokking via Mechanistic Interpretability", "Nanda, Lee, and Wattenberg (2023)", "https://arxiv.org/abs/2301.05217"),
    paper("Omnigrok: Grokking Beyond Algorithmic Data", "Liu et al. (2022)", "https://arxiv.org/abs/2210.01117"),
  ],
  "live-nebula": [
    paper("Attention Is All You Need", "Vaswani et al. (2017)", "https://arxiv.org/abs/1706.03762"),
    paper("Towards Best Practices of Activation Patching in Language Models", "Zhang and Nanda (2023)", "https://arxiv.org/abs/2309.16042"),
    paper("Eliciting Latent Predictions from Transformers with the Tuned Lens", "Belrose et al. (2023)", "https://arxiv.org/abs/2303.08112"),
  ],
} satisfies Record<string, readonly ResearchPaper[]>;

export type GuideResearchId = keyof typeof GUIDE_RESEARCH;

/** Keep the guide's research section as a shipping invariant. A newly
 * registered Internals view must add its references here before the guide can
 * render it; silently returning an empty list would make the documentation
 * look complete while dropping its evidence. */
export function guideResearchFor(featureId: GuideResearchId): readonly ResearchPaper[] {
  return GUIDE_RESEARCH[featureId];
}
