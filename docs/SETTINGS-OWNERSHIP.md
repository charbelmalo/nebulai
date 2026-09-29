# Settings ownership

Generated from `viewer/src/chrome/settingsScope.ts` by `npm run docs:settings`. Do not edit by hand.

Settings is the one canonical home for every preference. Each context shows only the tabs, sections and rows it owns. A hidden option keeps its stored value and still applies wherever it is used. Rows not listed inherit their section's owners.

| Setting | Seer | Learn | Atlas | Research |
| --- | --- | --- | --- | --- |
| **General** | shown | shown | shown | shown |
|   Chrome | shown | shown | shown | shown |
|     Cross-view linking | – | shown | – | shown |
|   Rendering | – | – | shown | shown |
|   Hand control | – | – | shown | shown |
| **Appearance** | shown | – | shown | shown |
|   atlas | – | – | shown | shown |
|   chord | – | – | – | shown |
|   hierarchy | – | – | – | shown |
|   compare | – | – | – | shown |
|   sessions | shown | – | – | – |
| **Behavior** | – | – | – | shown |
| **Model Probing** | shown | – | shown | shown |
|   Map builder | – | – | shown | shown |
|   Endpoint | shown | – | shown | shown |
|     Naming chain | – | – | shown | shown |
|     Live nebula server | – | – | – | shown |
|     SessionSeer server | shown | – | – | – |
|   Live probing | – | – | shown | shown |
|   Progress | – | – | shown | shown |
| **Snapshot** | shown | – | – | – |
| **Sessions** | shown | – | – | – |
| **Data** | – | – | shown | shown |
|   Dataset & view | – | – | shown | shown |
|     View type | – | – | – | shown |
| **About** | shown | shown | shown | shown |
