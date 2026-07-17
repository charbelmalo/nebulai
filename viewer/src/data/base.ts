/** Runtime data root for the baked `out/` artifacts.
 *
 *  The app shell is served under Vite's BASE_URL — which may be the domain
 *  root ("/") during dev or a sub-path (e.g. "/psychiX/nebulai-maps/") in a
 *  static deploy. The baked artifact tree is served as `out/` directly beneath
 *  that base, so we derive the data root from BASE_URL rather than hardcoding
 *  an absolute "/out". This makes every static fetch (index.json, per-dataset
 *  nebulai.json, interp bundles, compare.json) resolve correctly whether the
 *  viewer is hosted at the root or under an arbitrary sub-path.
 *
 *  Dev: BASE_URL="./"  -> DATA_BASE="./out"                    (served by the
 *       serveOut() vite middleware at /out)
 *  Prod (VITE_BASE=/psychiX/nebulai-maps/):
 *       BASE_URL="/psychiX/nebulai-maps/" -> "/psychiX/nebulai-maps/out"
 */
export const DATA_BASE =
  (import.meta.env.BASE_URL || "/").replace(/\/+$/, "") + "/out";
