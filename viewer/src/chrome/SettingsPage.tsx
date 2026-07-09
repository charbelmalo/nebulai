/** Full-page Settings overlay — the central home for every customizable
 *  option in the viewer. Any new user-facing knob (driver setting, probe
 *  config, chrome preference) MUST land in a tab here. See AGENTS.md /
 *  the `nebulai` skill's SETTINGS_HOME rule. */

import type { ComponentChildren } from "preact";
import { useSignal } from "@preact/signals";
import { requestDataset, requestViewMode } from "../app/actions";
import { appStore, type ViewMode } from "../app/store";
import { probeEndpoint, startBuildProbe, cancelBuildProbe } from "./probe";
import {
  $appearance,
  $capabilities,
  $compareData,
  $dataset,
  $datasetId,
  $datasets,
  $dims,
  $loading,
  $probing,
  $progress,
  $sessions,
  $settings,
  $settingsOpen,
  $snapshot,
  $viewMode,
} from "./state";
import {
  clearSessionAnalyses as clearPersistedSessions,
  deleteSessionAnalysis,
} from "./sessionStore";
import { SelectRow, SliderRow, Tabs, TextRow, ToggleRow } from "./controls";

const TABS = ["General", "Appearance", "Model Probing", "Snapshot", "Sessions", "Data", "About"];

const STAGE_ORDER: readonly string[] = [
  "probing",
  "loading",
  "reducing",
  "clustering",
  "naming",
  "exporting",
  "rendering",
  "done",
];

const STAGE_LABEL: Record<string, string> = {
  idle: "Idle",
  probing: "Probing endpoint",
  loading: "Loading units",
  reducing: "UMAP reduce",
  clustering: "HDBSCAN cluster",
  naming: "Cluster naming",
  exporting: "Writing nebulai.json",
  rendering: "Rendering map",
  done: "Complete",
  error: "Error",
};

export function SettingsPage() {
  const tab = useSignal("General");
  if (!$settingsOpen.value) return null;

  const close = () => appStore.getState().setSettingsOpen(false);

  return (
    <div class="settings-scrim" role="dialog" aria-label="Settings" aria-modal="true">
      <div class="settings-panel">
        <header class="settings-head">
          <div class="settings-head-title">
            <span class="settings-eyebrow">Settings</span>
            <h1>Preferences &amp; customization</h1>
          </div>
          <button type="button" class="settings-close" aria-label="Close settings" onClick={close}>
            ✕
          </button>
        </header>

        <nav class="settings-nav" aria-label="Settings sections">
          {TABS.map((t) => (
            <button
              key={t}
              type="button"
              class={`settings-nav-item${tab.value === t ? " is-active" : ""}`}
              aria-current={tab.value === t}
              onClick={() => (tab.value = t)}
            >
              {t}
            </button>
          ))}
        </nav>

        <div class="settings-body">
          {tab.value === "General" && <GeneralTab />}
          {tab.value === "Appearance" && <AppearanceTab />}
          {tab.value === "Model Probing" && <ProbingTab />}
          {tab.value === "Snapshot" && <SnapshotTab />}
          {tab.value === "Sessions" && <SessionsTab />}
          {tab.value === "Data" && <DataTab />}
          {tab.value === "About" && <AboutTab />}
        </div>
      </div>
    </div>
  );
}

// ── General ────────────────────────────────────────────────────────────────

function GeneralTab() {
  const settings = $settings.value;
  const caps = $capabilities.value;
  return (
    <>
      <SettingsSection
        title="Chrome"
        hint="Theme and motion preferences apply to the whole viewer."
      >
        <SelectRow
          label="Theme"
          value={settings.theme}
          options={[
            { value: "dark", label: "Dark (default)" },
            { value: "light", label: "Light" },
            { value: "auto", label: "Match system" },
          ]}
          onChange={(v) => appStore.getState().setSetting("theme", v as "dark" | "light" | "auto")}
        />
        <ToggleRow
          label="Reduced motion"
          checked={settings.reducedMotion}
          onChange={(v) => appStore.getState().setSetting("reducedMotion", v)}
          hint="disables halos and orbit"
        />
        <SliderRow
          label="Animation speed"
          value={settings.animationSpeed}
          min={0.25}
          max={2}
          step={0.05}
          format={(v) => `${v.toFixed(2)}×`}
          onChange={(v) => appStore.getState().setSetting("animationSpeed", v)}
        />
      </SettingsSection>

      <SettingsSection title="Rendering" hint="Live-applied across all graph types.">
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
        <SliderRow
          label="Label density"
          value={settings.labelDensity}
          min={0.2}
          max={2}
          step={0.05}
          format={(v) => `${v.toFixed(2)}×`}
          onChange={(v) => appStore.getState().setSetting("labelDensity", v)}
        />
        <ToggleRow
          label="Bloom post-processing"
          checked={settings.bloom}
          disabled={caps?.tier !== "webgpu"}
          hint={caps?.tier !== "webgpu" ? "webgpu only" : undefined}
          onChange={(v) => appStore.getState().setSetting("bloom", v)}
        />
      </SettingsSection>
    </>
  );
}

// ── Appearance ─────────────────────────────────────────────────────────────

function AppearanceTab() {
  const sub = useSignal<"atlas" | "chord" | "hierarchy" | "compare">("atlas");
  const a = $appearance.value;

  return (
    <>
      <div class="settings-subtabs">
        <Tabs
          tabs={["atlas", "chord", "hierarchy", "compare"]}
          active={sub.value}
          onChange={(t) => (sub.value = t as typeof sub.value)}
        />
      </div>

      {sub.value === "atlas" && (
        <SettingsSection title="Atlas — the 2D/3D concept map">
          <SliderRow
            label="Territory opacity"
            value={a.atlas.hullOpacity}
            min={0}
            max={1}
            step={0.01}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(v) => appStore.getState().setAppearance("atlas", "hullOpacity", v)}
          />
          <SliderRow
            label="Connection width"
            value={a.atlas.beamWidth}
            min={0.25}
            max={3}
            step={0.05}
            format={(v) => `${v.toFixed(2)}×`}
            onChange={(v) => appStore.getState().setAppearance("atlas", "beamWidth", v)}
          />
          <SliderRow
            label="Halo intensity"
            value={a.atlas.haloIntensity}
            min={0}
            max={1}
            step={0.01}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(v) => appStore.getState().setAppearance("atlas", "haloIntensity", v)}
          />
          <SelectRow
            label="Background"
            value={a.atlas.background}
            options={[
              { value: "vignette", label: "Vignette (default)" },
              { value: "flat", label: "Flat" },
              { value: "grid", label: "Grid" },
            ]}
            onChange={(v) =>
              appStore
                .getState()
                .setAppearance("atlas", "background", v as "vignette" | "flat" | "grid")
            }
          />
          <ToggleRow
            label="3D orbit"
            checked={a.atlas.orbitEnabled}
            onChange={(v) => appStore.getState().setAppearance("atlas", "orbitEnabled", v)}
          />
          <SliderRow
            label="Orbit speed"
            value={a.atlas.orbitSpeed}
            min={0.1}
            max={3}
            step={0.05}
            format={(v) => `${v.toFixed(2)}×`}
            onChange={(v) => appStore.getState().setAppearance("atlas", "orbitSpeed", v)}
          />
        </SettingsSection>
      )}

      {sub.value === "chord" && (
        <SettingsSection title="Chord — cluster-to-cluster ribbons">
          <SliderRow
            label="Ribbon opacity"
            value={a.chord.ribbonOpacity}
            min={0}
            max={1}
            step={0.01}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(v) => appStore.getState().setAppearance("chord", "ribbonOpacity", v)}
          />
          <SliderRow
            label="Curve tension"
            value={a.chord.curveTension}
            min={0}
            max={1}
            step={0.01}
            format={(v) => v.toFixed(2)}
            onChange={(v) => appStore.getState().setAppearance("chord", "curveTension", v)}
          />
          <ToggleRow
            label="Rotate rim labels"
            checked={a.chord.labelRotation}
            onChange={(v) => appStore.getState().setAppearance("chord", "labelRotation", v)}
          />
          <ToggleRow
            label="Show rim ticks"
            checked={a.chord.showTicks}
            onChange={(v) => appStore.getState().setAppearance("chord", "showTicks", v)}
          />
        </SettingsSection>
      )}

      {sub.value === "hierarchy" && (
        <SettingsSection title="Hierarchy — the radial dendrogram">
          <SliderRow
            label="Link stroke"
            value={a.hierarchy.linkStroke}
            min={0.5}
            max={3}
            step={0.05}
            format={(v) => `${v.toFixed(2)}×`}
            onChange={(v) => appStore.getState().setAppearance("hierarchy", "linkStroke", v)}
          />
          <SliderRow
            label="Node size"
            value={a.hierarchy.nodeSize}
            min={0.5}
            max={3}
            step={0.05}
            format={(v) => `${v.toFixed(2)}×`}
            onChange={(v) => appStore.getState().setAppearance("hierarchy", "nodeSize", v)}
          />
          <SliderRow
            label="Fan angle"
            value={a.hierarchy.fanAngle}
            min={60}
            max={360}
            step={5}
            format={(v) => `${v.toFixed(0)}°`}
            onChange={(v) => appStore.getState().setAppearance("hierarchy", "fanAngle", v)}
          />
          <SelectRow
            label="Color by"
            value={a.hierarchy.colorBy}
            options={[
              { value: "cluster", label: "Cluster" },
              { value: "depth", label: "Depth" },
              { value: "confidence", label: "Confidence" },
            ]}
            onChange={(v) =>
              appStore
                .getState()
                .setAppearance("hierarchy", "colorBy", v as "cluster" | "depth" | "confidence")
            }
          />
        </SettingsSection>
      )}

      {sub.value === "compare" && (
        <SettingsSection title="Compare — cross-model overlay">
          <SliderRow
            label="Swatch size"
            value={a.compare.swatchSize}
            min={4}
            max={20}
            step={1}
            format={(v) => `${v.toFixed(0)}px`}
            onChange={(v) => appStore.getState().setAppearance("compare", "swatchSize", v)}
          />
          <ToggleRow
            label="Highlight stroke on hover"
            checked={a.compare.strokeOnHover}
            onChange={(v) => appStore.getState().setAppearance("compare", "strokeOnHover", v)}
          />
          <ToggleRow
            label="Dim non-selected models"
            checked={a.compare.dimOthers}
            onChange={(v) => appStore.getState().setAppearance("compare", "dimOthers", v)}
          />
        </SettingsSection>
      )}
    </>
  );
}

// ── Model Probing ──────────────────────────────────────────────────────────

function ProbingTab() {
  const p = $probing.value;
  const pg = $progress.value;

  return (
    <>
      <SettingsSection
        title="Endpoint"
        hint="Point the naming/embedding chain at a custom OpenAI-compatible endpoint, or route through the M4 worker bridge."
      >
        <ToggleRow
          label="Route through M4 worker bridge"
          checked={p.useM4Worker}
          onChange={(v) => appStore.getState().setProbing("useM4Worker", v)}
          hint="192.168.0.200:8100"
        />
        <TextRow
          label="Base URL"
          type="url"
          value={p.endpoint}
          placeholder="https://api.openai.com/v1"
          onChange={(v) => appStore.getState().setProbing("endpoint", v)}
        />
        <TextRow
          label="API key"
          type="password"
          value={p.apiKey}
          placeholder="sk-…"
          onChange={(v) => appStore.getState().setProbing("apiKey", v)}
          hint="in-memory only"
        />
        <TextRow
          label="Model id"
          value={p.model}
          placeholder="llama3.2:3b or gpt-4o-mini"
          onChange={(v) => appStore.getState().setProbing("model", v)}
        />
      </SettingsSection>

      <SettingsSection title="Live probing">
        <ToggleRow
          label="Ping endpoint on change"
          checked={p.liveProbe}
          onChange={(v) => appStore.getState().setProbing("liveProbe", v)}
        />
        <SliderRow
          label="Probe interval"
          value={p.probeIntervalMs / 1000}
          min={5}
          max={60}
          step={1}
          format={(v) => `${v.toFixed(0)}s`}
          onChange={(v) => appStore.getState().setProbing("probeIntervalMs", v * 1000)}
        />
        <ToggleRow
          label="Auto-rebuild map on config change"
          checked={p.autoRun}
          onChange={(v) => appStore.getState().setProbing("autoRun", v)}
        />
        <div class="settings-actions">
          <button type="button" class="btn-primary" onClick={() => probeEndpoint()}>
            Probe now
          </button>
          <button
            type="button"
            class="btn-ghost"
            onClick={() => startBuildProbe()}
            disabled={pg.stage !== "idle" && pg.stage !== "done" && pg.stage !== "error"}
          >
            Rebuild map
          </button>
          <button
            type="button"
            class="btn-ghost"
            onClick={() => cancelBuildProbe()}
            disabled={pg.stage === "idle" || pg.stage === "done" || pg.stage === "error"}
          >
            Cancel
          </button>
        </div>
      </SettingsSection>

      <SettingsSection title="Progress" hint="Live view of the pipeline as the map builds.">
        <ProgressStrip />
        <ProgressLog />
      </SettingsSection>
    </>
  );
}

function ProgressStrip() {
  const pg = $progress.value;
  return (
    <div class="progress-strip">
      <div class="progress-bar" role="progressbar" aria-valuenow={Math.round(pg.pct * 100)}>
        <div class="progress-fill" style={{ width: `${Math.min(1, Math.max(0, pg.pct)) * 100}%` }} />
      </div>
      <div class="progress-meta">
        <span class="progress-stage" data-stage={pg.stage}>
          {STAGE_LABEL[pg.stage] ?? pg.stage}
        </span>
        <span class="progress-msg">{pg.message || "—"}</span>
        {pg.latencyMs != null && (
          <span class="progress-latency">{pg.latencyMs.toFixed(0)}ms</span>
        )}
      </div>
      <ol class="progress-stages">
        {STAGE_ORDER.map((s) => {
          const currentIdx = STAGE_ORDER.indexOf(pg.stage);
          const idx = STAGE_ORDER.indexOf(s);
          const state =
            pg.stage === "error"
              ? idx <= currentIdx
                ? "error"
                : "pending"
              : idx < currentIdx
                ? "done"
                : idx === currentIdx
                  ? "active"
                  : "pending";
          return (
            <li key={s} class={`progress-node is-${state}`}>
              <span class="progress-node-dot" />
              <span class="progress-node-label">{STAGE_LABEL[s]}</span>
            </li>
          );
        })}
      </ol>
      {pg.error && <div class="progress-error">{pg.error}</div>}
    </div>
  );
}

function ProgressLog() {
  const pg = $progress.value;
  if (pg.history.length === 0) {
    return <p class="settings-empty">No events yet. Press “Rebuild map” to start.</p>;
  }
  return (
    <ul class="progress-log">
      {pg.history
        .slice()
        .reverse()
        .map((h) => (
          <li key={h.id}>
            <span class="progress-log-time">{formatTime(h.t)}</span>
            <span class="progress-log-stage" data-stage={h.stage}>
              {h.stage}
            </span>
            <span class="progress-log-msg">{h.message}</span>
          </li>
        ))}
    </ul>
  );
}

function formatTime(t: number) {
  const d = new Date(t);
  return `${d.getHours().toString().padStart(2, "0")}:${d
    .getMinutes()
    .toString()
    .padStart(2, "0")}:${d.getSeconds().toString().padStart(2, "0")}`;
}

// ── Snapshot ───────────────────────────────────────────────────────────────

function SnapshotTab() {
  const snap = $snapshot.value;
  const newName = useSignal("");
  return (
    <>
      <SettingsSection
        title="Topic presets"
        hint="Each preset is a case-insensitive keyword list the Snapshot Map watches for. Edit inline; changes apply immediately."
      >
        <ul class="settings-preset-list">
          {snap.topics.map((t) => (
            <li key={t.id} class="settings-preset">
              <div class="settings-preset-head">
                <input
                  class="ctl-input"
                  type="text"
                  value={t.name}
                  onInput={(e) =>
                    appStore
                      .getState()
                      .updateTopicPreset(t.id, {
                        name: (e.currentTarget as HTMLInputElement).value,
                      })
                  }
                />
                <button
                  type="button"
                  class="btn-ghost"
                  onClick={() => appStore.getState().removeTopicPreset(t.id)}
                >
                  remove
                </button>
              </div>
              <textarea
                class="settings-preset-kws"
                rows={3}
                value={t.keywords.join(", ")}
                onInput={(e) =>
                  appStore
                    .getState()
                    .updateTopicPreset(t.id, {
                      keywords: (e.currentTarget as HTMLTextAreaElement).value
                        .split(",")
                        .map((k) => k.trim())
                        .filter(Boolean),
                    })
                }
              />
            </li>
          ))}
        </ul>
        <div class="settings-actions">
          <input
            class="ctl-input"
            type="text"
            placeholder="new topic name"
            value={newName.value}
            onInput={(e) => (newName.value = (e.currentTarget as HTMLInputElement).value)}
          />
          <button
            type="button"
            class="btn-primary"
            disabled={!newName.value.trim()}
            onClick={() => {
              const name = newName.value.trim();
              if (!name) return;
              appStore.getState().addTopicPreset({
                id: `topic-${Date.now().toString(36)}`,
                name,
                keywords: [],
              });
              newName.value = "";
            }}
          >
            Add preset
          </button>
        </div>
      </SettingsSection>
    </>
  );
}

// ── Sessions ─────────────────────────────────────────────────────────────────

function SessionsTab() {
  const sess = $sessions.value;
  const bytes = sess.analyses.reduce((n, a) => n + JSON.stringify(a).length, 0);
  const kb = (bytes / 1024).toFixed(bytes >= 10240 ? 0 : 1);
  return (
    <>
      <SettingsSection
        title="Persisted analyses"
        hint="Analysed sessions are saved in this browser (IndexedDB) and restored on your next visit — this is what carries analysis from session to session. Only the DERIVED summary is stored (token totals, the per-turn trajectory, tool histogram, file paths); raw transcript text is never saved or transmitted."
      >
        <div class="settings-kv">
          <div class="settings-kv-row">
            <span>Stored sessions</span>
            <b>{sess.analyses.length}</b>
          </div>
          <div class="settings-kv-row">
            <span>Shown on plot</span>
            <b>{sess.activeIds.length}</b>
          </div>
          <div class="settings-kv-row">
            <span>Approx. size</span>
            <b>{kb} KB</b>
          </div>
        </div>
        {sess.analyses.length > 0 && (
          <ul class="settings-preset-list">
            {sess.analyses.map((a) => (
              <li key={a.id} class="settings-preset">
                <div class="settings-preset-head">
                  <span class="settings-session-name">{a.name}</span>
                  <label class="settings-session-toggle">
                    <input
                      type="checkbox"
                      checked={sess.activeIds.includes(a.id)}
                      onChange={() => appStore.getState().toggleSessionActive(a.id)}
                    />
                    show
                  </label>
                  <button
                    type="button"
                    class="btn-ghost"
                    onClick={() => {
                      appStore.getState().removeSessionAnalysis(a.id);
                      deleteSessionAnalysis(a.id).catch(() => {});
                    }}
                  >
                    remove
                  </button>
                </div>
                <span class="settings-session-meta">
                  {a.model ?? "—"} · {a.nAssistant} turns · {a.toolTotal} tools ·{" "}
                  {a.errorCount} errors
                </span>
              </li>
            ))}
          </ul>
        )}
        <div class="settings-actions">
          <button
            type="button"
            class="btn-ghost"
            disabled={sess.analyses.length === 0}
            onClick={() => {
              appStore.getState().clearSessionAnalyses();
              clearPersistedSessions().catch(() => {});
            }}
          >
            Clear all persisted sessions
          </button>
        </div>
      </SettingsSection>
    </>
  );
}

// ── Data ───────────────────────────────────────────────────────────────────

function DataTab() {
  const caps = $capabilities.value;
  return (
    <SettingsSection
      title="Dataset & view"
      hint="Choose which export is loaded and how the units are drawn."
    >
      <SelectRow
        label="Dataset"
        value={$datasetId.value ?? ""}
        disabled={$loading.value.active || $viewMode.value === "compare"}
        options={$datasets.value.map((d) => ({ value: d.id, label: d.id }))}
        onChange={(id) => requestDataset(id)}
      />
      <SelectRow
        label="View type"
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
    </SettingsSection>
  );
}

// ── About ──────────────────────────────────────────────────────────────────

function AboutTab() {
  const caps = $capabilities.value;
  const ds = $dataset.value;
  return (
    <SettingsSection title="Provenance" hint="What was measured and how.">
      <dl class="settings-dl">
        <div>
          <dt>GPU tier</dt>
          <dd>{caps?.tier ?? "probing…"}</dd>
        </div>
        <div>
          <dt>Reduced motion</dt>
          <dd>{caps?.reducedMotion ? "yes" : "no"}</dd>
        </div>
        {ds && (
          <>
            <div>
              <dt>Model</dt>
              <dd>{ds.columns.meta.model ?? "—"}</dd>
            </div>
            <div>
              <dt>Unit type</dt>
              <dd>{ds.columns.meta.unit}</dd>
            </div>
            <div>
              <dt>Points</dt>
              <dd>{ds.columns.meta.n_points.toLocaleString("en-US")}</dd>
            </div>
            <div>
              <dt>Clusters</dt>
              <dd>{ds.columns.meta.n_clusters}</dd>
            </div>
            <div>
              <dt>Namer</dt>
              <dd>{ds.columns.meta.namer}</dd>
            </div>
            <div>
              <dt>Noise fraction</dt>
              <dd>{(ds.columns.meta.noise_fraction * 100).toFixed(1)}%</dd>
            </div>
          </>
        )}
      </dl>
    </SettingsSection>
  );
}

// ── Shared shell ───────────────────────────────────────────────────────────

function SettingsSection(props: {
  title: string;
  hint?: string;
  children: ComponentChildren;
}) {
  return (
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>{props.title}</h2>
        {props.hint && <p>{props.hint}</p>}
      </header>
      <div class="settings-section-body">{props.children}</div>
    </section>
  );
}
