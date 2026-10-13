"""
scripts/08_add_connector_papers.py

Corpus-growth script: adds 8 specific, named "connector" papers to the
pilot corpus — foundational works (BERT, GPT-3, "Attention Is All You
Need", RoBERTa, Sentence-BERT, T5, InstructGPT, Llama 2) identified by
directly analyzing this project's own already-fetched Semantic Scholar
data: each is referenced by or cites several papers already in the
40-paper corpus. Adding them creates real, internal `RESOLVES_TO` edges
in the knowledge graph (previously 0 — see project notes), without which
graph_retriever.py has nothing to traverse on this corpus.

Why named targets, not another random sample
--------------------------------------------------
01_build_pilot_corpus.py's random sampling is why the corpus had zero
internal citation overlap in the first place — a 40-paper random sample
out of all of cs.CL essentially never citation-links to itself by chance.
This is the standard citation-network research fix for that: deliberately
add the "hub" papers a random sample already points to, rather than
hoping a bigger random sample would eventually do it by luck. The 8
targets below were chosen by counting, across all 40 existing papers'
already-fetched references+citations (citations/pilot_corpus_citations.
json), which outside papers show up most often — see this project's own
analysis output for the exact counts (BERT: 15, GPT-3: 12, Attention:
10, RoBERTa: 6, SBERT: 5, T5: 4, InstructGPT: 4, Llama 2: 4).

Why this is additive-only, unlike replace_corpus_paper.py
------------------------------------------------------------------
replace_corpus_paper.py's whole design is remove-one/add-one, keeping
the corpus at a fixed PILOT_SIZE. This script never removes anything —
it grows the corpus from 40 to 48. The two scripts solve different
problems (fixing a bad paper vs. deliberately strengthening graph
structure) and shouldn't be forced into one shape.

Why failures are per-paper, not all-or-nothing
---------------------------------------------------
replace_corpus_paper.py only ever handles ONE paper, so "did the whole
thing succeed" and "did that one paper succeed" are the same question.
This script handles 8, so it follows this project's usual multi-item
convention instead (05_build_embeddings.py, 06_build_knowledge_graph.py):
one paper's PDF failing to download shouldn't cost you the other 7.

Why a paper is only added once BOTH its PDF and its metadata are in hand
-------------------------------------------------------------------------
metadata/pilot_corpus_metadata.json without a matching pdfs/*.pdf would
make 03_parse_documents.py fail on that entry later, with a less obvious
error than catching it here. Each candidate paper is downloaded AND
enriched before being written to metadata/citations at all — a paper
that fails either step is skipped and reported, never half-added.

Resumable
-------------
Like every other build script here: already-added arxiv_ids (checked
against the current metadata file) are skipped, so re-running after a
partial failure only retries what's still missing.

BEFORE RUNNING:
- Same prerequisites as scripts 01-03: the raw Kaggle arXiv metadata
  snapshot present at data/arxiv-metadata-oai-snapshot.json, S2_API_KEY
  in .env (optional but recommended — unauthenticated Semantic Scholar
  calls are heavily rate-limited).
- Takes a few minutes: scanning the full raw metadata file for 8 specific
  ids is a single pass, but it's still a multi-GB file to read through.

AFTER RUNNING:
Re-run, in order: 03_parse_documents.py (parses only the new papers —
the other 40 are skipped), 05_build_embeddings.py (same), then
06_build_knowledge_graph.py and 07_build_vector_index.py (both always
rebuild fully from whatever's on disk, so they'll naturally pick up the
new papers — no special-casing needed there).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG — edit these ----------------
# (arxiv_id, human-readable label for logging only — not used for lookup)
TARGET_PAPERS = [
    ("1810.04805", "BERT"),
    ("2005.14165", "GPT-3 (Language Models are Few-Shot Learners)"),
    ("1706.03762", "Attention Is All You Need"),
    ("1907.11692", "RoBERTa"),
    ("1908.10084", "Sentence-BERT"),
    ("1910.10683", "T5"),
    ("2203.02155", "InstructGPT"),
    ("2307.09288", "Llama 2"),
]

RAW_METADATA_PATH = "data/arxiv-metadata-oai-snapshot.json"
METADATA_PATH = "metadata/pilot_corpus_metadata.json"
CITATIONS_PATH = "citations/pilot_corpus_citations.json"
PDF_DIR = "pdfs"

S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = (
    "title,year,publicationDate,venue,citationCount,referenceCount,"
    "fieldsOfStudy,references.externalIds,references.title,references.year,"
    "citations.externalIds,citations.title,citations.year"
)
# -------------------------------------------------------


def _trim(record: dict[str, Any]) -> dict[str, Any]:
    """Identical field set to 01_build_pilot_corpus.py / replace_corpus_
    paper.py's trimmed metadata — kept in sync manually, same reasoning
    those files already give for why this small a duplication beats an
    import."""
    return {
        "arxiv_id": record.get("id"),
        "title": (record.get("title") or "").strip().replace("\n", " "),
        "authors": record.get("authors"),
        "abstract": (record.get("abstract") or "").strip().replace("\n", " "),
        "categories": record.get("categories"),
        "doi": record.get("doi"),
        "journal_ref": record.get("journal-ref"),
        "comments": record.get("comments"),
        "versions": record.get("versions"),
        "update_date": record.get("update_date"),
    }


def find_target_records(target_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Single pass over the raw metadata file, collecting only the
    records whose id is in `target_ids`. Stops early once every target
    has been found — no need to read the rest of a multi-GB file once
    there's nothing left to look for."""
    if not os.path.exists(RAW_METADATA_PATH):
        raise FileNotFoundError(
            f"Could not find '{RAW_METADATA_PATH}' — the same file "
            f"01_build_pilot_corpus.py needs. It's gitignored on purpose; "
            f"re-download from Kaggle if it's no longer on disk."
        )

    print(f"Scanning {RAW_METADATA_PATH} for {len(target_ids)} target paper(s)...")
    remaining = set(target_ids)
    found: dict[str, dict[str, Any]] = {}

    with open(RAW_METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            arxiv_id = record.get("id")
            if arxiv_id in remaining:
                found[arxiv_id] = record
                remaining.discard(arxiv_id)

    if remaining:
        print(f"  WARNING: {len(remaining)} target(s) not found in the raw metadata file: {sorted(remaining)}")
    return found


def download_pdf(arxiv_id: str) -> None:
    safe_id = arxiv_id.replace("/", "_")
    out_path = Path(PDF_DIR) / f"{safe_id}.pdf"
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-pilot-corpus-script/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def query_batch(arxiv_ids: list[str], retries: int = 3) -> dict[str, Optional[dict[str, Any]]]:
    """One batched Semantic Scholar call for every id at once (the batch
    endpoint accepts up to 500 — 8 is trivial for it), rather than 8
    separate requests. Returns a dict keyed by arxiv_id; a value of None
    means S2 didn't have that paper (normal, not an error — see
    replace_corpus_paper.py's identical stance on this)."""
    if not arxiv_ids:
        return {}

    payload = json.dumps({"ids": [f"ARXIV:{aid}" for aid in arxiv_ids]}).encode("utf-8")
    url = f"{S2_BATCH_URL}?fields={S2_FIELDS}"
    headers = {"Content-Type": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                results = json.loads(resp.read().decode("utf-8"))
                # S2's batch endpoint returns results in the SAME ORDER as
                # the input ids list, with null for anything not found.
                return dict(zip(arxiv_ids, results))
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                wait = 5 * attempt
                reason = "rate-limited" if e.code == 429 else f"server error {e.code}"
                print(f"  {reason}, waiting {wait}s and retrying...")
                time.sleep(wait)
                continue
            print(f"  HTTP error {e.code}: {e.reason}")
            return {aid: None for aid in arxiv_ids}
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error: {e}, retrying...")
            time.sleep(3)

    print("  giving up after retries")
    return {aid: None for aid in arxiv_ids}


def _to_enriched_entry(trimmed: dict[str, Any], s2_result: Optional[dict[str, Any]]) -> dict[str, Any]:
    entry = dict(trimmed)
    if s2_result is None:
        entry["semantic_scholar"] = None
    else:
        entry["semantic_scholar"] = {
            "title": s2_result.get("title"),
            "year": s2_result.get("year"),
            "publicationDate": s2_result.get("publicationDate"),
            "venue": s2_result.get("venue"),
            "citationCount": s2_result.get("citationCount"),
            "referenceCount": s2_result.get("referenceCount"),
            "fieldsOfStudy": s2_result.get("fieldsOfStudy"),
            "references": s2_result.get("references", []),
            "citations": s2_result.get("citations", []),
        }
    return entry


def main() -> None:
    metadata: list[dict[str, Any]] = json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))
    citations: list[dict[str, Any]] = json.loads(Path(CITATIONS_PATH).read_text(encoding="utf-8"))

    current_ids = {p["arxiv_id"] for p in metadata}
    to_add = [(aid, label) for aid, label in TARGET_PAPERS if aid not in current_ids]

    if not to_add:
        print("All target papers are already in the corpus — nothing to do.")
        return

    print(f"{len(to_add)}/{len(TARGET_PAPERS)} target paper(s) still need adding:")
    for aid, label in to_add:
        print(f"  {aid}  {label}")

    target_ids = {aid for aid, _ in to_add}
    raw_records = find_target_records(target_ids)

    # -- Download PDFs, one at a time, tolerating individual failures ------
    downloaded: list[str] = []
    for aid in raw_records:
        try:
            download_pdf(aid)
            downloaded.append(aid)
            print(f"  downloaded PDF for {aid}")
        except Exception as e:
            print(f"  FAILED to download PDF for {aid}: {e} — skipping this paper")

    if not downloaded:
        print("\nNo PDFs downloaded successfully — nothing to add. Check your network connection.")
        return

    # -- One batched S2 call for everything that has a PDF in hand ---------
    print(f"\nQuerying Semantic Scholar for {len(downloaded)} paper(s)...")
    s2_results = query_batch(downloaded)

    added = 0
    for aid in downloaded:
        trimmed = _trim(raw_records[aid])
        enriched = _to_enriched_entry(trimmed, s2_results.get(aid))
        metadata.append(trimmed)
        citations.append(enriched)
        found_note = "found on S2" if s2_results.get(aid) else "not found on S2 (normal, handled as null)"
        print(f"  added {aid} — {found_note}")
        added += 1

    Path(METADATA_PATH).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path(CITATIONS_PATH).write_text(json.dumps(citations, indent=2), encoding="utf-8")

    print(f"\nDone. {added} paper(s) added — corpus is now {len(metadata)} papers.")
    print(f"Updated {METADATA_PATH} and {CITATIONS_PATH}.")
    print("\nNext, run in order:")
    print("  python scripts/03_parse_documents.py     (only parses the new papers)")
    print("  python scripts/05_build_embeddings.py    (only embeds the new papers)")
    print("  python scripts/06_build_knowledge_graph.py   (rebuilds the full graph)")
    print("  python scripts/07_build_vector_index.py      (rebuilds the full FAISS index)")


if __name__ == "__main__":
    main()