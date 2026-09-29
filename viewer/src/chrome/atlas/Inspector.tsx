/** Inspector — everything the loaded artifact says about ONE unit, and
 *  nothing it does not say. It is the same component whether the unit was
 *  picked on the canvas, reached from the results table, opened from a
 *  pinned link or imported from a saved finding, so all four routes show
 *  identical evidence.
 *
 *  What it never shows: an activation, a "confidence" in the label, or a
 *  coordinate presented as a measurement. Coordinates are labelled as the
 *  projection they are; membership is labelled as clustering, not truth. */

import { signal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";
import { appStore } from "../../app/store";
import { APP_ROOT } from "../../data/base";
import {
  buildFinding,
  exportEligibility,
  findingFileName,
  findingLimitations,
  findingPin,
  FINDING_NOTE_MAX,
  modelRevision,
  pinParams,
} from "../../data/finding";
import { sourceUnit } from "../../data/columns";
import type { Dataset } from "../../data/loader";
import { parseCuration, REPRESENTATION_COPY } from "../../data/representation";
import { $dataset, $datasetId, $dims, $inspectorOpen, $pin, $selection } from "../state";
import { restoreFocus } from "./AtlasWorkspace";
import {
  clusterTitle,
  fmtInt,
  isUnlabelled,
  labelSource,
  metaOf,
  modelLine,
  representationOf,
  shortSha,
  str,
  unitTitle,
} from "./evidence";
import { $narrow } from "./layout";

/** the save form's draft note survives closing and reopening the panel for
 *  the same unit, and is dropped when the unit changes */
const $draft = signal<{ row: number; note: string; open: boolean }>({ row: -1, note: "", open: false });
const $copyState = signal<"idle" | "copied" | "failed">("idle");

const fmtCoord = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "—");

export function closeInspector(): void {
  appStore.getState().setSelection(null);
  restoreFocus();
}

function download(name: string, json: string): void {
  const blob = new Blob([json + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function unitLink(ds: Dataset, datasetId: string, row: number, dims: 2 | 3): string | null {
  const r = buildFinding(ds, datasetId, row, { dims });
  if (!r.ok) return null;
  return `${new URL("atlas/", APP_ROOT).href}#${pinParams(findingPin(r.finding)).toString()}`;
}

export function Inspector({
  row,
  onBack,
  backLabel,
  actions = true,
  closable = true,
}: {
  row: number;
  /** narrow screens: return to the list without dropping the selection */
  onBack?: () => void;
  backLabel?: string;
  /** the Learn lesson shows evidence only until its save step */
  actions?: boolean;
  /** the Learn lesson keeps its unit on screen: no close button, and Escape
   *  does not drop the unit the lesson is about */
  closable?: boolean;
}) {
  const ds = $dataset.value;
  const id = $datasetId.value;
  const headRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    // a newly opened unit takes focus so its evidence is read next; the
    // canvas keeps focus if the reader is still dragging the map
    const active = document.activeElement;
    if (!active || active === document.body || active.closest(".atlas-workspace, .inspector, .pin-notice"))
      headRef.current?.focus({ preventScroll: true });
    $copyState.value = "idle";
  }, [row]);
  if (!ds || !id) return null;
  const u = sourceUnit(ds.columns, row);
  if (!u) return null;
  const meta = metaOf(ds);
  const rep = representationOf(ds);
  const copy = REPRESENTATION_COPY[rep];
  const revision = modelRevision(meta);
  const cur = parseCuration(meta.curation);
  const pin = $pin.value;
  const verified = pin.status === "ok" && pin.row === row;
  const elig = exportEligibility(ds);
  const draft = $draft.value.row === row ? $draft.value : { row, note: "", open: false };
  const dims = $dims.value;

  const save = () => {
    const r = buildFinding(ds, id, row, { dims, note: draft.note });
    if (!r.ok) return;
    download(findingFileName(r.finding), JSON.stringify(r.finding, null, 2));
    $draft.value = { row, note: draft.note, open: false };
  };
  const copyLink = async () => {
    const link = unitLink(ds, id, row, dims);
    if (!link) return;
    try {
      await navigator.clipboard.writeText(link);
      $copyState.value = "copied";
    } catch {
      $copyState.value = "failed";
    }
  };

  return (
    <aside
      class="inspector"
      aria-labelledby="inspector-title"
      onKeyDown={(e) => {
        if (closable && e.key === "Escape" && !(e.target as HTMLElement).closest("textarea")) {
          e.stopPropagation();
          closeInspector();
        }
      }}
    >
      <div class="inspector-top">
        {onBack && (
          <button type="button" class="aw-btn aw-btn-quiet inspector-back" onClick={onBack}>
            <span aria-hidden="true">← </span>
            {backLabel ?? "Back"}
          </button>
        )}
        <p class="inspector-kicker">{copy.pointNoun}</p>
        {closable && (
          <button type="button" class="inspector-close" aria-label="Close unit details" onClick={closeInspector}>
            <span aria-hidden="true">×</span>
          </button>
        )}
      </div>
      <h2 id="inspector-title" class="inspector-title" tabIndex={-1} ref={headRef}>
        {unitTitle(meta, u)}
      </h2>
      {isUnlabelled(meta) && (
        <p class="inspector-note">This map has no labels. The placeholder text is not shown as evidence.</p>
      )}
      {verified && (
        <p class="inspector-badge" role="status">
          Verified against artifact sha256 {shortSha(ds.sha256)}…
          {pin.source === "import" ? " from a saved finding" : " from a unit link"}
        </p>
      )}
      {verified && pin.note && (
        <div class="inspector-imported">
          <p class="inspector-label">Note in the saved finding</p>
          <p class="inspector-note-text">{pin.note}</p>
        </div>
      )}

      <dl class="inspector-dl">
        <dt>Unit</dt>
        <dd>
          <span class="aw-mono">{u.unitKind ?? "Not recorded"}</span>
          <br />
          index {u.unitIndex ?? "—"} · point ID {u.pointId ?? "—"}
        </dd>
        <dt>Model</dt>
        <dd>
          {modelLine(meta)}
          <br />
          <span class="aw-dim">revision {revision ?? "not recorded"}</span>
        </dd>
        <dt>Layer</dt>
        <dd>{u.layer ?? (typeof meta.layer === "number" ? `${meta.layer} (map-wide)` : "Not recorded")}</dd>
        <dt>Cluster</dt>
        <dd>
          {clusterTitle(ds, u.clusterId)}
          {u.clusterId < 0 && (
            <span class="aw-dim">
              {" "}
              — the clustering left this point in no group. It is still a real {copy.pointNoun}.
            </span>
          )}
        </dd>
        {u.clusterId >= 0 && (
          <>
            <dt>Membership</dt>
            <dd>
              {u.membership.toFixed(3)}
              <span class="aw-dim"> — how firmly the clustering placed it; not confidence in the label</span>
            </dd>
          </>
        )}
        <dt>Label source</dt>
        <dd>{labelSource(meta)}</dd>
        {str(meta.geometry) && (
          <>
            <dt>Geometry</dt>
            <dd>{str(meta.geometry)}</dd>
          </>
        )}
        {cur && (
          <>
            <dt>Curation</dt>
            <dd>
              first {fmtInt(cur.kept)} of {fmtInt(cur.total)}
            </dd>
          </>
        )}
        <dt>Position</dt>
        <dd>
          <span class="aw-mono">
            ({fmtCoord(u.xyz[0])}, {fmtCoord(u.xyz[1])}, {fmtCoord(u.xyz[2])})
          </span>
          <br />
          <span class="aw-dim">projection coordinates, not an activation or a measurement</span>
        </dd>
      </dl>

      <div class="inspector-limits">
        <p class="inspector-label">What this does not show</p>
        <ul>
          {findingLimitations(meta, rep, revision).map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </div>

      {actions && (
        <div class="inspector-actions">
          {!elig.ok && <p class="inspector-note" id="export-reason">{elig.reason}</p>}
          {draft.open ? (
            <form
              class="inspector-save"
              onSubmit={(e) => {
                e.preventDefault();
                save();
              }}
            >
              <label for="finding-note" class="inspector-label">
                Note (optional)
              </label>
              <textarea
                id="finding-note"
                rows={3}
                maxLength={FINDING_NOTE_MAX}
                value={draft.note}
                onInput={(e) => ($draft.value = { row, open: true, note: (e.currentTarget as HTMLTextAreaElement).value })}
                onKeyDown={(e) => {
                  if (e.key === "Escape") {
                    e.stopPropagation();
                    $draft.value = { ...draft, open: false };
                  }
                }}
              />
              <p class="aw-dim">
                Saves this unit's source evidence as a JSON file on this device. Reopening it uses the plain
                atlas; the note stays in the file and is never put in a link.
              </p>
              <div class="inspector-row">
                <button type="submit" class="aw-btn aw-btn-primary">
                  Download finding
                </button>
                <button type="button" class="aw-btn aw-btn-quiet" onClick={() => ($draft.value = { ...draft, open: false })}>
                  Cancel
                </button>
              </div>
            </form>
          ) : (
            <div class="inspector-row">
              <button
                type="button"
                class="aw-btn aw-btn-primary"
                disabled={!elig.ok}
                aria-describedby={elig.ok ? undefined : "export-reason"}
                onClick={() => ($draft.value = { row, note: draft.note, open: true })}
              >
                Save finding…
              </button>
              <button
                type="button"
                class="aw-btn"
                disabled={!elig.ok}
                aria-describedby={elig.ok ? undefined : "export-reason"}
                onClick={() => void copyLink()}
              >
                Copy unit link
              </button>
            </div>
          )}
          <p class="aw-dim inspector-copy-state" role="status" aria-live="polite">
            {$copyState.value === "copied"
              ? "Link copied. It opens this exact unit, verified against the same artifact."
              : $copyState.value === "failed"
                ? "The clipboard is blocked here. The link is in the address bar."
                : ""}
          </p>
        </div>
      )}
    </aside>
  );
}

/** The selected point's inspector, if the reader has one open. */
export function SelectedInspector(props: { onBack?: () => void; backLabel?: string }) {
  const sel = $selection.value;
  if (sel?.kind !== "point") return null;
  if (!$inspectorOpen.value) return null;
  return <Inspector row={sel.id} {...props} />;
}

export { $narrow };
