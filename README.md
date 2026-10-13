# Pilot Corpus Setup — Temporal Hybrid GraphRAG Project

Builds your Week 2 deliverable: a pilot corpus of ~40 arXiv papers
(full PDFs + version history + citation graph), ready to feed into the
extraction pipeline starting Week 3.

## What you get at the end

- `pilot_corpus_metadata.json` — title, authors, abstract, categories,
  and full **version history** (v1, v2, v3… each with a date) for every
  paper in the pilot set.
- `pdfs/` — the actual PDF for each paper.
- `pilot_corpus_citations.json` — the same records, enriched with
  **citation graph data** (what each paper cites, what cites it,
  publication venue, fields of study) from Semantic Scholar.

## Setup (one-time)

1. **Get the arXiv metadata file.**
   Create a free Kaggle account, then download the dataset from
   <https://www.kaggle.com/datasets/Cornell-University/arxiv>.
   You need `arxiv-metadata-oai-snapshot.json` — it's several GB.
   Place it in this folder (or edit `RAW_METADATA_PATH` in script 1 to
   point wherever you saved it).

2. **(Recommended) Get a free Semantic Scholar API key.**
   <https://www.semanticscholar.org/product/api> — takes a few minutes.
   Without a key the scripts still work, just more slowly, since the
   shared anonymous rate limit is low. Paste your key into
   `S2_API_KEY` near the top of script 2 if you get one.

3. No package installs are needed — both scripts use only Python's
   standard library. Python 3.8+ is fine.

## Running it

```bash
python 01_build_pilot_corpus.py
python 02_enrich_citation_graph.py
```

Run them in that order — the second script reads the output of the first.

## Adjusting the pilot corpus

Open `01_build_pilot_corpus.py` and change the values near the top:

- `CATEGORY_FILTER` — which arXiv category to pull from (default `cs.CL`;
  use `cs.AI`, `cs.LG`, etc. if your topic focus shifts)
- `DATE_FROM` / `DATE_TO` — the time window (default 2020–2025; a 3–5
  year span is enough to show real research evolution)
- `PILOT_SIZE` — how many papers to keep (default 40; the timeline's
  Week 2 target was 30–50)

## A few things worth knowing

- **Runtime**: scanning the full metadata file takes a few minutes
  (it's large); downloading 40 PDFs at the default 3-second delay
  between each takes about 2 minutes. This is deliberately unhurried —
  see the note below on why.
- **Why the delays exist**: arXiv's terms permit downloading individual
  papers via their public PDF links, but not unattended bulk scraping.
  A slow, deliberate pull of ~40 papers for a research pilot is normal,
  reasonable use — the delays just keep it that way rather than hammering
  their servers. If you later scale beyond a pilot to hundreds/thousands
  of papers, switch to the free bulk GCS bucket
  (`gs://arxiv-dataset`, linked from the Kaggle page) instead of
  downloading one-by-one.
- **Some papers may not be found on Semantic Scholar** — this is normal
  for a handful of papers (very recent preprints especially) and is
  handled gracefully; they're kept in the output with
  `"semantic_scholar": null` rather than dropped.
- **Reproducibility**: the sample is controlled by `RANDOM_SEED` (default
  42), so re-running script 1 with the same settings gives you the same
  40 papers each time — useful when you need to explain/defend your
  corpus selection later.

## Not run/tested in this environment

These scripts were written and reviewed carefully against verified,
current API documentation, but this sandbox has no network access, so
they could not be executed end-to-end here. If something errors on your
first run (a changed field name, a rate-limit surprise), it's most
likely a small fix, not a structural problem — let me know what the
error says and I'll help you fix it.
