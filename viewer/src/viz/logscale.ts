/** A log₁₀ axis, as a pure function.
 *
 *  Every extruded chart on this surface has the same problem: the quantity
 *  spans decades, so a linear height leaves everything except the outlier flat
 *  on the floor — which is exactly the structure the view exists to show. The
 *  answer is always a log axis, and a log axis is only honest if the ticks come
 *  from the SAME expression as the columns. Two functions that happen to agree
 *  today will silently stop agreeing, and every bar is then mislabelled by the
 *  gap with nothing to catch it. */

/** Position of `v` on a log₁₀ axis spanning `10^loExp … 10^hiExp`, as 0..1.
 *
 *  Clamped at both ends. Zero, negative and NaN all return 0 — NaN in
 *  particular must not fall through to the log branch: a NaN height fed to an
 *  instance matrix is singular under inversion and silently kills picking for
 *  the entire mesh. */
export function logSpan(v: number, loExp: number, hiExp: number): number {
  const span = hiExp - loExp;
  if (!(span > 0)) return 0;
  if (!(v > 0)) return 0;
  const t = (Math.log10(v) - loExp) / span;
  return t <= 0 ? 0 : t >= 1 ? 1 : t;
}

/** Where the decade `10^e` sits on that same axis. Not clamped: a caller
 *  ticking outside the span wants to know it landed outside, not to have the
 *  tick quietly stack up on the cage edge. */
export function decadeOn(e: number, loExp: number, hiExp: number): number {
  const span = hiExp - loExp;
  return span > 0 ? (e - loExp) / span : 0;
}
