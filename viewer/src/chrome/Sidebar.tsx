/** Left settings panel — the video's Settings/Additional sidebar. Collapsible
 *  to a gear pill. Dataset/Type/Dimensions selects on the Settings tab plus
 *  the layer toggles; render-quality knobs live under Additional. */

import { useSignal } from "@preact/signals";
import { requestDataset, requestViewMode } from "../app/actions";
import { appStore, type Toggles, type ViewMode } from "../app/store";
import {
  $capabilities,
  $compareData,
  $dataset,
  $datasetId,
  $datasets,
  $dims,
  $loading,
  $settings,
  $toggles,
  $viewMode,
} from "./state";
import { SelectRow, SliderRow, Tabs, ToggleRow } from "./controls";

const TOGGLE_ROWS: { key: keyof Toggles; label: string }[] = [
  { key: "territories", label: "Territories" },
  { key: "labels", label: "Labels" },
  { key: "beams", label: "Connections" },
  { key: "noise", label: "Noise" },
  { key: "legend", label: "Legend" },
];

export function Sidebar() {
  const open = useSignal(true);
  const tab = useSignal("Settings");
  const caps = $capabilities.value;
  const toggles = $toggles.value;
  const settings = $settings.value;

  if (!open.value) {
    return (
      <button
        type="button"
        class="sidebar-fab"
        aria-label="Open settings"
        onClick={() => (open.value = true)}
      >
        ⚙
      </button>
    );
  }

  return (
    <aside class="sidebar" aria-label="Settings">
      <header class="sidebar-head">
        <Tabs
          tabs={["Settings", "Additional"]}
          active={tab.value}
          onChange={(t) => (tab.value = t)}
        />
        <button
          type="button"
          class="sidebar-collapse"
          aria-label="Collapse settings"
          onClick={() => (open.value = false)}
        >
          ‹
        </button>
      </header>

      {tab.value === "Settings" ? (
        <div class="sidebar-body" role="tabpanel">
          <SelectRow
            label="Dataset"
            value={$datasetId.value ?? ""}
            disabled={$loading.value.active || $viewMode.value === "compare"}
            options={$datasets.value.map((d) => ({ value: d.id, label: d.id }))}
            onChange={(id) => requestDataset(id)}
          />
          <SelectRow
            label="Type"
            value={$viewMode.value}
            options={[
              { value: "atlas", label: "Atlas" },
              { value: "chord", label: "Chord" },
              {
                value: "hierarchy",
                label: "Hierarchical",
                disabled: !$dataset.value?.columns.edges,
                hint: !$dataset.value?.columns.edges ? "needs edges (v2 export)" : undefined,
              },
              {
                value: "compare",
                label: "Compare",
                disabled: caps?.tier !== "webgpu" || !$compareData.value,
                hint:
                  caps?.tier !== "webgpu"
                    ? "webgpu only"
                    : !$compareData.value
                      ? "run `nebulai compare`"
                      : undefined,
              },
            ]}
            onChange={(v) => requestViewMode(v as ViewMode)}
          />
          {$viewMode.value === "atlas" && (
            <SelectRow
              label="Dimensions"
              value={String($dims.value)}
              options={[
                { value: "2", label: "2D map" },
                { value: "3", label: "3D flythrough" },
              ]}
              onChange={(v) => appStore.getState().setDims(v === "3" ? 3 : 2)}
            />
          )}
          {$viewMode.value !== "compare" && (
            <>
              <div class="sidebar-sep" />
              {TOGGLE_ROWS.map((r) => (
                <ToggleRow
                  key={r.key}
                  label={r.label}
                  checked={toggles[r.key]}
                  onChange={(v) => appStore.getState().setToggle(r.key, v)}
                />
              ))}
            </>
          )}
        </div>
      ) : (
        <div class="sidebar-body" role="tabpanel">
          <SliderRow
            label="Point scale"
            value={settings.pointScale}
            min={0.5}
            max={2}
            step={0.05}
            format={(v) => `${v.toFixed(2)}×`}
            onChange={(v) => appStore.getState().setSetting("pointScale", v)}
          />
          <SliderRow
            label="Confidence floor"
            value={settings.confidenceFloor}
            min={0}
            max={1}
            step={0.01}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(v) => appStore.getState().setSetting("confidenceFloor", v)}
          />
          <ToggleRow
            label="Bloom"
            checked={settings.bloom}
            disabled={caps?.tier !== "webgpu"}
            hint={caps?.tier !== "webgpu" ? "(webgpu only)" : undefined}
            onChange={(v) => appStore.getState().setSetting("bloom", v)}
          />
        </div>
      )}
    </aside>
  );
}
