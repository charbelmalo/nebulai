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
  $experience,
  $loading,
  $settings,
  $sidebarOpen,
  $toggles,
  $viewMode,
  openPanel,
} from "./state";
import { handControlUnavailableReason } from "../hands/types";
import { SelectRow, SliderRow, Tabs, ToggleRow } from "@psychix/viz/controls";
import { APP_ROOT } from "../data/base";

/** Research's Comparisons page for a given map: Chord, Hierarchical and
 *  Compare live there, with their label-space caveat, not in Atlas. */
export function advancedHref(datasetId: string | null): string {
  const params = new URLSearchParams({ page: "map" });
  if (datasetId) params.set("model", datasetId);
  return `${new URL("research/", APP_ROOT).href}#${params.toString()}`;
}

/** Atlas's stand-in for the Type select: the one map view it hosts, and an
 *  explicit way to the advanced ones rather than options that silently vanish. */
export function AdvancedHandoff() {
  return (
    <div class="sidebar-handoff">
      <span class="sidebar-handoff-label">View</span>
      <span class="sidebar-handoff-value">Atlas map</span>
      <a class="sidebar-handoff-link" href={advancedHref($datasetId.value)}>
        Chord, hierarchy and model comparison are in Research
        <span aria-hidden="true"> ↗</span>
      </a>
    </div>
  );
}

const TOGGLE_ROWS: { key: keyof Toggles; label: string }[] = [
  { key: "territories", label: "Territories" },
  { key: "labels", label: "Labels" },
  { key: "beams", label: "Connections" },
  { key: "halos", label: "Halos" },
  { key: "noise", label: "Noise" },
  { key: "legend", label: "Legend" },
];

export function Sidebar() {
  const tab = useSignal("Settings");
  const caps = $capabilities.value;
  const toggles = $toggles.value;
  const settings = $settings.value;
  const handReason = handControlUnavailableReason();

  if (!$sidebarOpen.value) {
    return (
      <button
        type="button"
        class="sidebar-fab"
        aria-label="Open settings"
        onClick={() => openPanel("sidebar")}
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
          onClick={() => ($sidebarOpen.value = false)}
        >
          ‹
        </button>
      </header>

      {tab.value === "Settings" ? (
        <div class="sidebar-body" role="tabpanel">
          <SelectRow
            label="Dataset"
            value={$datasetId.value ?? ""}
            disabled={$viewMode.value === "compare"}
            options={$datasets.value.map((d) => ({ value: d.id, label: d.id }))}
            onChange={(id) => requestDataset(id)}
          />
          {$experience.value === "atlas" ? (
            <AdvancedHandoff />
          ) : (
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
                // no tier gate: the field is TSL, so the forceWebGL rung draws
                // the same scene without bloom (verified on ?gpu=webgl)
                disabled: !$compareData.value,
                hint: !$compareData.value ? "run `nebulai compare`" : undefined,
              },
            ]}
            onChange={(v) => requestViewMode(v as ViewMode)}
          />
          )}
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
          {/* Atlas-gated because the rig only steers the atlas — the same
              condition that mounts <HandRig/> in apps/nebulai.tsx. Kept on the
              Settings tab rather than under Additional: a control nobody can
              find is a feature nobody has, and this one is off by default and
              needs a camera grant, so it has to be visible where the operator
              is already looking. */}
          {$viewMode.value === "atlas" && (
            <ToggleRow
              label="Hand control"
              checked={settings.handTracking}
              disabled={handReason !== null}
              hint={handReason ?? "webcam gestures — video never leaves this machine"}
              onChange={(v) => appStore.getState().setSetting("handTracking", v)}
            />
          )}
          {/* Only once the rig is on, and off by default when it is. Navigation
              is what the rig is for; the two visual casts are a thing to opt
              into afterwards, and they are the reason this is a toggle rather
              than always-on — they ride the hand that is NOT steering, so with
              one hand raised they cost nothing and with two they are the whole
              difference between a spare hand and an accident. */}
          {$viewMode.value === "atlas" && settings.handTracking && (
            <ToggleRow
              label="Hand effects"
              checked={settings.handEffects}
              hint="your free hand can throw a shockwave or snap the cloud bright"
              onChange={(v) => appStore.getState().setSetting("handEffects", v)}
            />
          )}
          <div class="sidebar-sep" />
          <button
            type="button"
            class="sidebar-more"
            onClick={() => appStore.getState().setSettingsOpen(true)}
          >
            All settings…
          </button>
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
          <div class="sidebar-sep" />
          <button
            type="button"
            class="sidebar-more"
            onClick={() => appStore.getState().setSettingsOpen(true)}
          >
            All settings…
          </button>
        </div>
      )}
    </aside>
  );
}
