import { describe, expect, it } from "vitest";
import { encodeSegment } from "../../src/data/segment";

describe("encodeSegment", () => {
  it("keeps the @ of a pinned-revision id, which the dev server does not decode", () => {
    expect(encodeSegment("smollm2-135m-instruct@12fd25f77366.v1.L19")).toBe(
      "smollm2-135m-instruct@12fd25f77366.v1.L19",
    );
  });
  it("still encodes everything that is not path-safe", () => {
    expect(encodeSegment("a b/c?d#e")).toBe("a%20b%2Fc%3Fd%23e");
  });
});
