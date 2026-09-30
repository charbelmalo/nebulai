import { describe, expect, it } from "vitest";
import { describeDataset } from "../../src/data/representation";

describe("describeDataset", () => {
  it("names a GPT-2 SAE map with its layer and family", () => {
    const c = describeDataset({
      id: "gpt2-small__sae__blocks.8.hook_resid_pre",
      model: "gpt2-small",
      unit: "sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)",
    });
    expect(c).toEqual({
      title: "GPT-2 Small",
      subtitle: "SAE decoder directions · layer 8",
      layer: 8,
      family: "GPT-2 family",
    });
  });

  it("reads neuron layers from h.N and layers.N hooks", () => {
    expect(describeDataset({ id: "a", model: "gpt2", unit: "mlp_neuron(gpt2, h.8.mlp.c_proj)" }).layer).toBe(8);
    expect(
      describeDataset({
        id: "b",
        model: "HuggingFaceTB/SmolLM2-135M",
        unit: "mlp_neuron(HuggingFaceTB/SmolLM2-135M, model.layers.21.mlp.down_proj)",
      }),
    ).toMatchObject({ title: "SmolLM2 135M", layer: 21, family: "SmolLM2" });
  });

  it("credits the embedder of an external embedding, not the named model", () => {
    const c = describeDataset({ id: "c", model: "gpt2", unit: "api_text_embedding(mxbai-embed-large)" });
    expect(c.subtitle).toBe("Vocabulary embedded by mxbai-embed-large");
  });

  it("titles a concept probe by its concept", () => {
    const c = describeDataset({ id: "probe__grief", model: "grief", unit: "probe_concept(mxbai-embed-large)" });
    expect(c).toMatchObject({ title: "Concept probe: grief", family: "Concept probes", layer: null });
  });

  it("shows an unknown model exactly as exported", () => {
    expect(describeDataset({ id: "x", model: "acme/thing-1", unit: "token_embedding" }).title).toBe("acme/thing-1");
  });
});
