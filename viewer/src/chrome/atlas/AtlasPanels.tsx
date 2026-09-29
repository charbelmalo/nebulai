/** AtlasPanels — the Atlas experience's map page. Desktop: workspace on the
 *  left, the map in the middle, the inspector on the right (the legend steps
 *  aside while it is open). Narrow screens: ONE panel at a time behind a
 *  Map / Results switch, and the inspector full width with a way back.
 *
 *  Without a renderer the results list is the whole interface: Map is
 *  disabled with its reason, never an empty canvas. */

import { AxisRail } from "../AxisRail";
import { HandRig } from "../HandRig";
import { LegendCard } from "../LegendCard";
import { appStore } from "../../app/store";
import { $atlasPanel, $dataset, $inspectorOpen, $renderer, $selection } from "../state";
import { AtlasWorkspace, PinNotice } from "./AtlasWorkspace";
import { SelectedInspector } from "./Inspector";
import { $narrow } from "./layout";

function PanelSwitch({ noGpu }: { noGpu: boolean }) {
  const panel = noGpu ? "results" : $atlasPanel.value;
  const set = (p: "map" | "results") => appStore.getState().setAtlasPanel(p);
  return (
    <section class="atlas-switch" aria-label="Panel switch">
      <div role="group" aria-label="Show">
        <button
          type="button"
          class={panel === "map" ? "aw-seg is-on" : "aw-seg"}
          aria-pressed={panel === "map"}
          disabled={noGpu}
          title={noGpu ? "The map needs WebGL or WebGPU" : undefined}
          onClick={() => set("map")}
        >
          Map
        </button>
        <button
          type="button"
          class={panel === "results" ? "aw-seg is-on" : "aw-seg"}
          aria-pressed={panel === "results"}
          onClick={() => set("results")}
        >
          Results
        </button>
      </div>
    </section>
  );
}

export function AtlasPanels() {
  const narrow = $narrow.value;
  const noGpu = $renderer.value === "unavailable";
  const sel = $selection.value;
  const inspecting = sel?.kind === "point" && $inspectorOpen.value;
  const panel = noGpu ? "results" : $atlasPanel.value;
  const hasData = $dataset.value !== null;

  if (narrow) {
    const back = () => appStore.getState().setInspectorOpen(false);
    return (
      <>
        <PinNotice />
        {inspecting ? (
          <SelectedInspector onBack={back} backLabel={panel === "map" ? "Back to map" : "Back to results"} />
        ) : (
          <AtlasWorkspace hidden={panel === "map"} />
        )}
        {!inspecting && panel === "map" && <AxisRail />}
        {!inspecting && hasData && <PanelSwitch noGpu={noGpu} />}
      </>
    );
  }
  return (
    <>
      <PinNotice />
      <AtlasWorkspace />
      {/* the legend explains colours on a canvas; with no canvas it has
          nothing to explain */}
      {inspecting ? <SelectedInspector /> : !noGpu && <LegendCard />}
      {!inspecting && <AxisRail />}
      <HandRig />
      {noGpu && hasData && (
        <p class="atlas-stage-note" role="note">
          Map unavailable: this browser has neither WebGPU nor WebGL. The results list has every unit and the
          same evidence.
        </p>
      )}
    </>
  );
}
