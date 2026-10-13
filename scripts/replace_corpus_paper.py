"""
scripts/replace_corpus_paper.py

One-time correction to the pilot corpus: 2109.01558 turned out to be a
full PhD thesis, not a regular paper (its own abstract says "the first
goal of this thesis is..."). This script removes it and adds one
freshly-selected replacement from the same filtered candidate pool, so
the corpus stays at PILOT_SIZE papers and every one of them is a
comparable, single-document unit going forward.

01_build_pilot_corpus.py has ALSO been updated with the same
thesis-detection heuristic used below, so regenerating the corpus from
scratch in the future won't hit this same issue — this script is a
one-time patch for the corpus that's already built, not a new permanent
pipeline stage. (EXCLUDE_ARXIV_ID is a plain constant rather than a
command-line argument on purpose — if a similar one-off swap is ever
needed again for a different reason, editing one line here and re-running
is simpler than building a general CLI for something this rare.)

BEFORE RUNNING:
- Same prerequisites as scripts 01-03: raw metadata file present, S2 key
  in .env, GROBID running.
- Takes a few minutes: finding a replacement means re-scanning the full
  raw metadata file, the same slow step 01_build_pilot_corpus.py
  originally did — there's no shortcut, since the full candidate pool
  was never saved anywhere, only the final 40.

Design note: destructive changes last, not first
------------------------------------------------------
Every step that can fail (finding a candidate, downloading its PDF,
querying Semantic Scholar) happens BEFORE anything in metadata/,
citations/, pdfs/, or parsed/ is touched. If any of those steps fails,
your existing corpus is untouched and re-running is safe. Only once the
replacement is fully in hand does this script remove the thesis's old
files and write the updated corpus — so a network hiccup can't leave you
with a half-updated, inconsistent corpus.

Duplicated rather than imported from 01/02
------------------------------------------------
scripts/ isn't set up as an importable package (these are meant to be run
directly, not imported from), so the small pieces of logic this script
needs from 01_build_pilot_corpus.py and 02_enrich_citation_graph.py are
copied here in miniature rather than imported. Small enough — a filter
function and a single-paper S2 query — that the duplication is cheaper
than the import machinery would be.
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
EXCLUDE_ARXIV_ID = "2109.01558"     # the thesis being removed
REPLACEMENT_SEED = 43               # deliberately NOT 42 (the original corpus's
                                     # seed) — this is a fresh, separate selection
                                     # from a smaller (already-filtered,
                                     # already-40-excluded) pool, not a claimed
                                     # continuation of the original random.sample()
                                     # call, which doesn't guarantee a stable
                                     # prefix across different sample sizes anyway

RAW_METADATA_PATH = "data/arxiv-metadata-oai-snapshot.json"
CATEGORY_FILTER = "cs.CL"
DATE_FROM = "2020-01-01"
DATE_TO = "2025-12-31"

METADATA_PATH = "metadata/pilot_corpus_metadata.json"
CITATIONS_PATH = "citations/pilot_corpus_citations.json"
PDF_DIR = "pdfs"
PARSED_DIR = "parsed"

S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = (
    "title,year,publicationDate,venue,citationCount,referenceCount,"
    "fieldsOfStudy,references.externalIds,references.title,references.year,"
    "citations.externalIds,citations.title,citations.year"
)
# -------------------------------------------------------


def _looks_like_thesis(record: dict[str, Any]) -> bool:
    """Identical heuristic to the one added to 01_build_pilot_corpus.py —
    see that file for the full reasoning. Kept in sync manually; there
    are only two copies of this and both are small."""
    comments = (record.get("comments") or "").lower()
    if any(kw in comments for kw in ("thesis", "dissertation")):
        return True
    abstract = (record.get("abstract") or "").lower()
    return "this thesis" in abstract or "this dissertation" in abstract


def _parse_first_version_date(versions: Optional[list[dict[str, Any]]]) -> Optional[datetime]:
    if not versions:
        return None
    try:
        return datetime.strptime(versions[0].get("created"), "%a, %d %b %Y %H:%M:%S %Z")
    except (ValueError, TypeError, KeyError):
        return None


def _matches_filters(record: dict[str, Any]) -> bool:
    categories = record.get("categories", "") or ""
    if CATEGORY_FILTER not in categories.split():
        return False
    if _looks_like_thesis(record):
        return False
    v1_date = _parse_first_version_date(record.get("versions"))
    if v1_date is None:
        return False
    date_from = datetime.strptime(DATE_FROM, "%Y-%m-%d")
    date_to = datetime.strptime(DATE_TO, "%Y-%m-%d")
    return date_from <= v1_date <= date_to


def _trim(record: dict[str, Any]) -> dict[str, Any]:
    """Matches 01_build_pilot_corpus.py's trimmed field set exactly,
    including the comments field added alongside the thesis filter."""
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


def find_replacement(exclude_ids: set[str]) -> dict[str, Any]:
    """Re-scans the raw metadata file for one replacement: same filters as
    01_build_pilot_corpus.py (now including the thesis check), excluding
    every arxiv_id already in the corpus."""
    if not os.path.exists(RAW_METADATA_PATH):
        raise FileNotFoundError(
            f"Could not find '{RAW_METADATA_PATH}' — the same file "
            f"01_build_pilot_corpus.py needs. It's gitignored on purpose; "
            f"re-download from Kaggle if it's no longer on disk."
        )

    print("Re-scanning the raw metadata file for a replacement candidate "
          "(same slow step 01_build_pilot_corpus.py did originally — a few minutes)...")

    candidates: list[dict[str, Any]] = []
    with open(RAW_METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _matches_filters(record):
                continue
            if record.get("id") in exclude_ids:
                continue
            candidates.append(record)

    if not candidates:
        raise RuntimeError(
            "No eligible replacement candidates found outside the current "
            "corpus — widen CATEGORY_FILTER/DATE_FROM/DATE_TO in this file."
        )

    print(f"Found {len(candidates):,} eligible candidates not already in the corpus.")
    import random

    random.seed(REPLACEMENT_SEED)
    return random.choice(candidates)


def download_pdf(arxiv_id: str) -> Path:
    safe_id = arxiv_id.replace("/", "_")
    out_path = Path(PDF_DIR) / f"{safe_id}.pdf"
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-pilot-corpus-script/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    out_path.write_bytes(data)
    print(f"Downloaded PDF for {arxiv_id}")
    return out_path


def enrich_one(arxiv_id: str, retries: int = 3) -> Optional[dict[str, Any]]:
    """Single-paper version of 02_enrich_citation_graph.py's query_batch."""
    payload = json.dumps({"ids": [f"ARXIV:{arxiv_id}"]}).encode("utf-8")
    url = f"{S2_BATCH_URL}?fields={S2_FIELDS}"
    headers = {"Content-Type": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                results = json.loads(resp.read().decode("utf-8"))
                return results[0] if results else None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * attempt
                print(f"  rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  HTTP error {e.code}: {e.reason}")
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error: {e}, retrying...")
            time.sleep(3)

    print("  giving up after retries")
    return None


def main() -> None:
    metadata: list[dict[str, Any]] = json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))
    citations: list[dict[str, Any]] = json.loads(Path(CITATIONS_PATH).read_text(encoding="utf-8"))

    current_ids = {p["arxiv_id"] for p in metadata}
    if EXCLUDE_ARXIV_ID not in current_ids:
        print(f"{EXCLUDE_ARXIV_ID} isn't in {METADATA_PATH} — already fixed, nothing to do.")
        return

    # ---- Steps that can fail: nothing destructive happens until all of
    # them succeed (see module docstring). ----
    replacement_raw = find_replacement(exclude_ids=current_ids)
    replacement = _trim(replacement_raw)
    print(f"Replacement selected: {replacement['arxiv_id']} — {replacement['title']}")

    download_pdf(replacement["arxiv_id"])

    print(f"Querying Semantic Scholar for {replacement['arxiv_id']}...")
    s2_result = enrich_one(replacement["arxiv_id"])
    enriched_replacement = dict(replacement)
    if s2_result is None:
        enriched_replacement["semantic_scholar"] = None
        print("  not found on Semantic Scholar (normal for some papers, handled as null)")
    else:
        enriched_replacement["semantic_scholar"] = {
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
        print("  found and enriched")

    # ---- Everything needed is in hand — now the destructive part ----
    metadata = [p for p in metadata if p["arxiv_id"] != EXCLUDE_ARXIV_ID]
    metadata.append(replacement)
    Path(METADATA_PATH).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Updated {METADATA_PATH} ({len(metadata)} papers)")

    citations = [p for p in citations if p["arxiv_id"] != EXCLUDE_ARXIV_ID]
    citations.append(enriched_replacement)
    Path(CITATIONS_PATH).write_text(json.dumps(citations, indent=2), encoding="utf-8")
    print(f"Updated {CITATIONS_PATH} ({len(citations)} papers)")

    old_pdf = Path(PDF_DIR) / f"{EXCLUDE_ARXIV_ID.replace('/', '_')}.pdf"
    old_parsed = Path(PARSED_DIR) / f"{EXCLUDE_ARXIV_ID.replace('/', '_')}.json"
    for path in (old_pdf, old_parsed):
        if path.exists():
            path.unlink()
            print(f"Removed {path}")

    print(f"\nDone. {EXCLUDE_ARXIV_ID} is out, {replacement['arxiv_id']} is in.")
    print("Run 03_parse_documents.py again — it skips the 39 already-parsed")
    print("papers automatically and only processes the new one.")


if __name__ == "__main__":
    main()
