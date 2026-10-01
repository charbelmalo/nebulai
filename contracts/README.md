# Data contracts

These are the specifications and source-backed fixtures for the three-experience release. The viewer implements them: `viewer/src/data/experience.ts` validates the manifest, and `viewer/src/data/finding.ts` validates, resolves and exports findings. The live manifest is `out/experience.json`, written by `viewer/scripts/package-experience.ts` (see `docs/DEPLOY-STATIC.md` §10).

- `finding.schema.json`: JSON Schema Draft 2020-12 for one v1 atlas-unit finding. All named fields are required; unknown model revisions are null. Extra top-level fields are rejected; raw source metadata intentionally allows fields from different pipelines.
- `finding.example.json`: actual point 0 and metadata from the local 4,096-direction starter. Its artifact digest was independently computed from the source bytes. No user note or invented model revision is present.
- `experience-manifest.example.json`: a **minimal example**, not a complete deployable index. A release packager must enumerate all supported current maps and all previously published pinned map digests, add their immutable paths, and include the pinned Research intro bundle. On the live host every listed digest exists as `artifacts/<digest>/nebulai.json`; one named only in this example does not.

The experience manifest is versioned separately from findings and the existing map schema. It must have a supported schema version, unique `(dataset_id, sha256)` entries, a valid SHA-256 and byte count for each entry, normalized relative same-origin paths under DATA_BASE, and defaults that resolve to listed entries. Reject traversal, absolute paths and conflicting duplicate entries. `legacy_path` and `sidecar_base` identify existing resources; no sidecar may be inferred from a content-addressed map directory. The Research intro is a separate bundle reference, not an atlas point artifact.

Finding structure validation is only the first gate. A resolver must additionally:

1. Enforce the 256 KiB input size and finite/safe numeric values before applying state. JSON with NaN/Infinity is invalid. The note limit is 4,000 characters.
2. Resolve dataset/digest using the trusted manifest. `artifact.source_path` is provenance text, not a fetch target. Do not accept a matching hash from an arbitrary imported URL as permission to fetch it.
3. Hash exact decoded response bytes and match the expected digest; identify exactly one source point by ID plus raw unit kind/index. Row ordering and display labels are not identities.
4. Compare model ID/revision when recorded, representation, point evidence and raw metadata to the verified source. Unknown revision remains null. An edit to a note is allowed; an edit to scientific evidence produces a conflicting-record state.
5. Preserve source coordinates and membership in full precision on the CPU; the Float32/Uint8 renderer buffers are insufficient to regenerate exact source evidence. Source JSON numbers must round-trip through JavaScript without new quantization.
6. Open the plain atlas with its source unit highlighted, or show an explicit failure. Never apply a half-restored selection. Honor the user's local list/no-GPU preference. A link cannot contain notes or claim to reproduce a research-sidecar analysis.

### Finding URL example

For the reference record, the decoded logical fields are:

```text
experience=atlas
page=map
model=gpt2-small__sae__blocks.8.hook_resid_pre
view=atlas
dims=2
artifact=d36402d8503e82df71a9c64e48a4f9561f35ad2ac666c43eee127ff71eac10f7
point=0
unit_kind=sae_decoder(gpt2-small-res-jb, blocks.8.hook_resid_pre)
unit_index=0
```

Encode with URLSearchParams under the Atlas entry. The Atlas entry implements these keys; `viewer/scripts/smoke-release.mjs` reopens this exact link against the published bytes. A valid pinned link verifies artifact and identity, but contains no saved note/evidence object to compare. Imported JSON additionally compares its saved evidence. New links open Atlas explicitly even when created from a Learn lesson.

### Precision and version policy

The SHA pins the full map, not just the selected point. Any relabeling, reordering, renaming or regenerated coordinates creates a new digest even if a unit's semantic identity is unchanged. Do not automatically migrate a finding to that new digest. Provide a separately labelled “Open latest map” action; exporting from it creates a new record.

Keep legacy v1/v2 maps renderable when their fields permit it. Missing/ambiguous stable identity disables verified finding export but does not erase the original map. New unsupported finding versions are rejected with an explanation. View replay does not promise that rerunning the original scientific pipeline produces the same bytes.
