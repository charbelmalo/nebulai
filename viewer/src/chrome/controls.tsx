/** Sidebar/legend control primitives, styled after the reference video's
 *  settings panel. Every control is keyboard-first: native elements where
 *  possible, ARIA switch pattern for toggles, real radios for radio rows. */

import type { ComponentChildren } from "preact";

export interface Option {
  value: string;
  label: string;
  disabled?: boolean;
  /** short hint shown for disabled options, e.g. "M3" */
  hint?: string;
}

export function SelectRow(props: {
  label: string;
  value: string;
  options: Option[];
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const id = `sel-${props.label.replace(/\s+/g, "-").toLowerCase()}`;
  return (
    <label class="ctl-row" for={id}>
      <span class="ctl-label">{props.label}</span>
      <select
        id={id}
        class="ctl-select"
        value={props.value}
        disabled={props.disabled}
        onChange={(e) => props.onChange((e.currentTarget as HTMLSelectElement).value)}
      >
        {props.options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
            {o.disabled && o.hint ? ` — ${o.hint}` : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ToggleRow(props: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  hint?: string;
}) {
  return (
    <div class="ctl-row">
      <span class="ctl-label">
        {props.label}
        {props.hint ? <span class="ctl-hint"> {props.hint}</span> : null}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={props.checked}
        aria-label={props.label}
        class="ctl-switch"
        disabled={props.disabled}
        onClick={() => props.onChange(!props.checked)}
      >
        <span class="ctl-switch-thumb" />
      </button>
    </div>
  );
}

export function SliderRow(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (value: number) => void;
}) {
  const id = `sld-${props.label.replace(/\s+/g, "-").toLowerCase()}`;
  return (
    <label class="ctl-row ctl-row-slider" for={id}>
      <span class="ctl-label">
        {props.label}
        <span class="ctl-value">{props.format(props.value)}</span>
      </span>
      <input
        id={id}
        class="ctl-slider"
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onInput={(e) => props.onChange(Number((e.currentTarget as HTMLInputElement).value))}
      />
    </label>
  );
}

export function RadioRow(props: {
  name: string;
  value: string;
  options: Option[];
  onChange: (value: string) => void;
}) {
  return (
    <div class="ctl-radios" role="radiogroup" aria-label={props.name}>
      {props.options.map((o) => (
        <label key={o.value} class={`ctl-radio${o.disabled ? " is-disabled" : ""}`}>
          <input
            type="radio"
            name={props.name}
            value={o.value}
            checked={props.value === o.value}
            disabled={o.disabled}
            onChange={() => props.onChange(o.value)}
          />
          <span class="ctl-radio-dot" />
          <span>
            {o.label}
            {o.disabled && o.hint ? <span class="ctl-hint"> {o.hint}</span> : null}
          </span>
        </label>
      ))}
    </div>
  );
}

export function Tabs(props: {
  tabs: string[];
  active: string;
  onChange: (tab: string) => void;
  children?: ComponentChildren;
}) {
  return (
    <div class="ctl-tabs" role="tablist">
      {props.tabs.map((t) => (
        <button
          key={t}
          type="button"
          role="tab"
          aria-selected={props.active === t}
          class={`ctl-tab${props.active === t ? " is-active" : ""}`}
          onClick={() => props.onChange(t)}
        >
          {t}
        </button>
      ))}
    </div>
  );
}
