import { describe, expect, it } from "vitest";
import {
  classifyRepresentation,
  datasetLabel,
  parseCuration,
  REPRESENTATION_COPY,
  REPRESENTATIONS,
} from "../../src/data/representation";

describe("classifyRepresentation", () => {
  it("classifies every exported unit string", () => {
    expect(classifyRepresentation({ unit: "token_embedding", weight_key: "wte.weight" })).toBe("token_embedding");
    expect(classifyRepresentation({ unit: "token_embedding", weight_key: "lm_head.weight" })).toBe("token_unembedding");
    expect(classifyRepresentation({ unit: "token_embedding", weight_key: "model.W_U" })).toBe("token_unembedding");
    expect(classifyRepresentation({ unit: "sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)" })).toBe("sae_decoder");
    expect(classifyRepresentation({ unit: "mlp_neuron(gpt2, h.8.mlp.c_proj)" })).toBe("mlp_neuron");
    expect(classifyRepresentation({ unit: "api_text_embedding(mxbai-embed-large)" })).toBe("external_text_embedding");
    expect(classifyRepresentation({ unit: "probe_concept(mxbai-embed-large)" })).toBe("external_text_embedding");
  });

  it("never guesses: unknown and missing units are unknown", () => {
    expect(classifyRepresentation({ unit: "attention_head(gpt2, 3)" })).toBe("unknown");
    expect(classifyRepresentation({})).toBe("unknown");
    expect(classifyRepresentation(null)).toBe("unknown");
  });

  it("does not read 'unembed' out of an unrelated weight key", () => {
    expect(classifyRepresentation({ unit: "token_embedding", weight_key: "embed_in.weight" })).toBe("token_embedding");
  });

  it("has copy for every class", () => {
    for (const r of REPRESENTATIONS) {
      const c = REPRESENTATION_COPY[r];
      expect(c.pointNoun && c.pluralNoun && c.meaning && c.limit).toBeTruthy();
    }
  });
});

describe("parseCuration", () => {
  it("reads first_N_of_M", () => {
    expect(parseCuration("first_4096_of_24576")).toEqual({ kept: 4096, total: 24576 });
  });
  it("rejects anything else", () => {
    for (const v of ["first_10_of_5", "top_4096", "", null, 3, "first_-1_of_5"]) expect(parseCuration(v)).toBeNull();
  });
});

describe("datasetLabel", () => {
  it("matches the plan's starter label exactly", () => {
    expect(
      datasetLabel({ model: "gpt2-small", unit: "sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)", layer: 8 }),
    ).toBe("GPT-2 Small · SAE decoder directions · Layer 8");
  });
  it("credits the embedder of an external embedding, not the vocabulary's model", () => {
    expect(datasetLabel({ model: "gpt2", unit: "api_text_embedding(mxbai-embed-large)" })).toBe(
      "GPT-2 · External text embeddings (mxbai-embed-large)",
    );
  });
  it("shows unknown model ids verbatim and falls back to the dataset id", () => {
    expect(datasetLabel({ model: "acme/unlisted-70m", unit: "token_embedding" })).toBe(
      "acme/unlisted-70m · Token embedding rows",
    );
    expect(datasetLabel({ unit: "token_embedding" }, "toy")).toBe("toy · Token embedding rows");
  });
});
