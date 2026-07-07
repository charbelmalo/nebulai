# Reading a token map

A token-embedding map mixes two kinds of cluster. Distinguishing them is the
core interpretive skill — and the caveat the whole project is honest about.

## Semantic clusters (meaning)

Tokens grouped by what they refer to. Observed on GPT-2 full vocab:

- First names split by type: male (`Michael · Chris · Brian`), female
  (`Jennifer · Rebecca · Julie`), surnames (`Thompson · Johnson · Miller`),
  Chinese surnames (`Zhang · Zheng · Huang`).
- Topic families: legislators/officials, animals, theology, storms/hurricanes,
  autoimmune diseases, clothing, weapons, body parts, planetary/spacecraft,
  musicians/songs, days & parts of the day.
- Number families separate cleanly: years (`2011 2012 2013`) vs two-digit
  (`10 12 50`) vs zero-padded (`00 01 000`).

## Code clusters (training-data fingerprint)

GPT-2 saw a lot of web + code, so programming tokens pool: `PHP · MySQL · SQL`,
`Array · integer · iterator · Buffer`, `connectors · plugin · plugins`. These
are real semantic clusters but they're a fingerprint of the corpus, worth
calling out as such.

## Orthographic / subword clusters (spelling, not meaning)

Tokens grouped by *form* — shared suffixes or letter shapes rather than meaning:
`acked · acking · uck`, `ol · om · op · ot`, `gg · ged · ging`, `ing · ers · ess`.
These sit nearer the dense core. Their existence is exactly why the honest
framing says "token-embedding structure is partly frequency/orthography" —
mean-centering + cosine reduce but do not remove it. When you present the map,
label these as orthographic so no one reads them as concepts.

## The glitch-token island (a real phenomenon reproduced)

GPT-2's anomalous "unspeakable" tokens — ` externalToEVA`, `rawdownload`,
`quickShip`, and the `SolidGoldMagikarp` family — form their own isolated
cluster far from the main cloud. These are under-trained tokens whose embeddings
never got pulled into the manifold. Reproducing the known interpretability
result is a strong, honest talking point: the tool surfaces a documented
phenomenon without being told to look for it.

## Sanity checks when a run looks off

- **Very few clusters, ~0% noise** → you're on `eom`; switch to `leaf`.
- **Hundreds of tiny clusters, very high noise** → `min_cluster_size` too low;
  raise it (default scales as `max(15, n//1000)`).
- **Everything orthographic, no semantics** → check centering is on; anisotropy
  is dominating.
- **Titles are all `x · y · z`** → namer fell to centroid; bring up ollama or
  set an OpenRouter key and re-run (reductions cached).

Use `scripts/inspect_map.py out/<model>/nebulai.json --top 40 --tail` to read
titles + sample members fast, and `scripts/sweep_hdbscan.py out/<model>/reduced.npz`
to retune clustering without recomputing UMAP.
