/** Encode one path segment of an artefact id for a URL. `encodeURIComponent`
 *  alone turns the `@` in every pinned-revision id
 *  (`smollm2-135m-instruct@12fd25f77366.v1.L19`) into `%40`, which the Vite dev
 *  server does not decode before looking the file up — it falls through to
 *  index.html, and the loader then reports "`<!doctype` is not valid JSON" for
 *  a study that is sitting right there. `@` is legal in a path segment, so it
 *  is kept as written; everything else still encodes. */
export function encodeSegment(id: string): string {
  return encodeURIComponent(id).replace(/%40/g, "@");
}
