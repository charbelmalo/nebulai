import { describe, expect, it } from "vitest";
import { promoteSearchToHash } from "../../src/app/searchToHash";

const at = (search: string, hash = "") => ({ pathname: "/learn/", search, hash });

describe("promoteSearchToHash", () => {
  it("moves the lesson keys from the query into the hash", () => {
    expect(promoteSearchToHash(at("?lesson=what-is-a-point&lesson_step=2"))).toBe(
      "/learn/#lesson=what-is-a-point&lesson_step=2",
    );
  });

  it("keeps a hash value over the query's and leaves unknown keys in the query", () => {
    expect(promoteSearchToHash(at("?frozen&lesson_step=1", "#lesson_step=3"))).toBe("/learn/?frozen=#lesson_step=3");
  });

  it("does nothing when the query names no view key", () => {
    expect(promoteSearchToHash(at("?view=chord", "#page=map"))).toBeNull();
  });
});
