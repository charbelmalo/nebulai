/** Atlas's pin and inspector state (app/slices/atlas.ts). A verified pin is
 *  a claim about ONE row of ONE loaded artifact, so every transition that
 *  changes the row, the map or the view must drop it — and an error must
 *  survive the loads a recovery triggers, so the reader can still read why. */

import { beforeEach, describe, expect, it } from "vitest";
import { appStore } from "../../src/app/store";
import type { Dataset } from "../../src/data/loader";
import type { UnitPin } from "../../src/data/finding";

const ds = {} as Dataset;
const pin: UnitPin = { datasetId: "m", sha256: "a".repeat(64), pointId: 3, unitKind: "k", unitIndex: 3, dims: 2 };
const st = () => appStore.getState();

beforeEach(() => {
  st().setDataset("m", ds);
  st().setSelection(null);
  st().setPin({ status: "none" });
});

describe("inspector", () => {
  it("opens for a newly selected point and closes for a cluster or nothing", () => {
    st().setSelection({ kind: "point", id: 1 });
    expect(st().inspectorOpen).toBe(true);
    st().setSelection({ kind: "cluster", id: 2 });
    expect(st().inspectorOpen).toBe(false);
    st().setSelection({ kind: "point", id: 1 });
    st().setSelection(null);
    expect(st().inspectorOpen).toBe(false);
  });

  it("re-selecting the same point keeps a panel the reader folded away", () => {
    st().setSelection({ kind: "point", id: 1 });
    st().setInspectorOpen(false); // narrow screen: back to the list
    st().setSelection({ kind: "point", id: 1 });
    expect(st().inspectorOpen).toBe(false);
    st().setSelection({ kind: "point", id: 2 });
    expect(st().inspectorOpen).toBe(true);
  });

  it("a map switch closes it", () => {
    st().setSelection({ kind: "point", id: 1 });
    st().setDataset("n", ds);
    expect(st().inspectorOpen).toBe(false);
  });
});

describe("pin", () => {
  it("a verified pin lasts while its row stays selected, and no longer", () => {
    st().setPin({ status: "ok", pin, row: 3, source: "link" });
    st().setSelection({ kind: "point", id: 3 });
    expect(st().pin.status).toBe("ok");
    st().setSelection({ kind: "point", id: 4 });
    expect(st().pin.status).toBe("none");
  });

  it("a verified pin is dropped by a map switch and by leaving the atlas view", () => {
    st().setPin({ status: "ok", pin, row: 3, source: "link" });
    st().setDataset("m", ds);
    expect(st().pin.status).toBe("none");
    st().setPin({ status: "ok", pin, row: 3, source: "import" });
    st().setViewMode("chord");
    expect(st().pin.status).toBe("none");
    st().setViewMode("atlas");
  });

  it("a pending pin survives loading its own map, not another one", () => {
    st().setPin({ status: "pending", pin, source: "link" });
    st().setDataset("m", ds);
    expect(st().pin.status).toBe("pending");
    st().setDataset("other", ds);
    expect(st().pin.status).toBe("none");
  });

  it("an error survives the recovery load it offers", () => {
    st().setPin({ status: "error", pin, source: "link", code: "wrong-artifact", title: "t", message: "m" });
    st().setDataset("m", ds);
    st().setSelection({ kind: "point", id: 0 });
    expect(st().pin.status).toBe("error");
  });
});
