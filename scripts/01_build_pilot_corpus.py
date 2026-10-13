"""
01_build_pilot_corpus.py

Builds a filtered, manageable pilot corpus (default ~40 papers) from the
Cornell University arXiv Dataset, and downloads the corresponding PDFs.

BEFORE RUNNING THIS SCRIPT:
1. Create a free Kaggle account: https://www.kaggle.com
2. Download the dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv
   (You need the file `arxiv-metadata-oai-snapshot.json` — it's a large
   file, several GB, one JSON object per line.)
3. Place it in the same folder as this script, or update RAW_METADATA_PATH
   below to point to wherever you saved it.

WHAT THIS SCRIPT DOES:
- Streams through the metadata file line-by-line (it's too large to load
  into memory all at once).
- Keeps only papers matching your chosen arXiv category and date range.
- Takes a deterministic sample of PILOT_SIZE papers from the matches.
- Saves their metadata (including full version history) to
  pilot_corpus_metadata.json.
- Downloads each paper's PDF (politely rate-limited) into ./pdfs/.

This targets a SMALL, curated pilot set (tens of papers), which is normal,
reasonable use of arXiv's public PDF links. It deliberately avoids bulk /
unattended scraping of the whole corpus — see README.md for arXiv's reuse
policy notes.
"""

import json
import os
import time
import random
import urllib.request
import urllib.error
from datetime import datetime

# ---------------- CONFIG — edit these ----------------
RAW_METADATA_PATH = "data/arxiv-metadata-oai-snapshot.json"
CATEGORY_FILTER = "cs.CL"          # arXiv category to focus on (change as needed)
DATE_FROM = "2020-01-01"           # inclusive, based on first version's submission date
DATE_TO = "2025-12-31"             # inclusive
PILOT_SIZE = 40                    # how many papers to keep
RANDOM_SEED = 42                   # for a reproducible sample
OUTPUT_METADATA_PATH = "metadata/pilot_corpus_metadata.json"
PDF_DIR = "pdfs"
DOWNLOAD_DELAY_SECONDS = 3         # politeness delay between PDF downloads
# -------------------------------------------------------


def parse_first_version_date(versions):
    """The 'versions' field is a list of {version, created} dicts.
    Returns the created date of v1 as a datetime, or None if unparseable."""
    if not versions:
        return None
    try:
        # Example format: 'Mon, 2 Jan 2020 18:30:00 GMT'
        raw = versions[0].get("created")
        return datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %Z")
    except (ValueError, TypeError, KeyError):
        return None


def _looks_like_thesis(record):
    """Added after 2109.01558 (a full PhD thesis) slipped through this
    filter — category+date alone can't distinguish a thesis from a paper.
    Checks 'comments' first: many arXiv thesis submissions self-report
    this there (e.g. "PhD thesis, University of X, 214 pages"), the most
    reliable signal available at metadata-scan time since we don't have
    the PDF yet to check page count directly. Falls back to the abstract's
    own wording, since not every thesis uses comments this way — 2109.01558
    didn't; its abstract said "the first goal of this thesis is..." instead."""
    comments = (record.get("comments") or "").lower()
    if any(kw in comments for kw in ("thesis", "dissertation")):
        return True
    abstract = (record.get("abstract") or "").lower()
    return "this thesis" in abstract or "this dissertation" in abstract


def matches_filters(record):
    categories = record.get("categories", "") or ""
    if CATEGORY_FILTER not in categories.split():
        return False

    if _looks_like_thesis(record):
        return False

    v1_date = parse_first_version_date(record.get("versions"))
    if v1_date is None:
        return False

    date_from = datetime.strptime(DATE_FROM, "%Y-%m-%d")
    date_to = datetime.strptime(DATE_TO, "%Y-%m-%d")
    return date_from <= v1_date <= date_to


def stream_filtered_records(path):
    """Yields matching records one at a time so we never hold the full
    multi-GB file in memory."""
    matched = 0
    scanned = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned % 200000 == 0:
                print(f"  ...scanned {scanned:,} records, {matched:,} matches so far")
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if matches_filters(record):
                matched += 1
                yield record


def build_pilot_corpus():
    if not os.path.exists(RAW_METADATA_PATH):
        raise FileNotFoundError(
            f"Could not find '{RAW_METADATA_PATH}'. Download it from "
            f"https://www.kaggle.com/datasets/Cornell-University/arxiv "
            f"and place it next to this script, or update RAW_METADATA_PATH."
        )

    print(f"Scanning {RAW_METADATA_PATH} for category={CATEGORY_FILTER}, "
          f"dates {DATE_FROM} to {DATE_TO} ...")

    matches = list(stream_filtered_records(RAW_METADATA_PATH))
    print(f"Found {len(matches):,} papers matching the filters.")

    if len(matches) == 0:
        raise RuntimeError(
            "No papers matched — try a broader CATEGORY_FILTER or wider date range."
        )

    random.seed(RANDOM_SEED)
    sample_size = min(PILOT_SIZE, len(matches))
    pilot = random.sample(matches, sample_size)

    # Keep only the fields we actually need downstream
    trimmed = []
    for r in pilot:
        trimmed.append({
            "arxiv_id": r.get("id"),
            "title": (r.get("title") or "").strip().replace("\n", " "),
            "authors": r.get("authors"),
            "abstract": (r.get("abstract") or "").strip().replace("\n", " "),
            "categories": r.get("categories"),
            "doi": r.get("doi"),
            "journal_ref": r.get("journal-ref"),
            "comments": r.get("comments"),
            "versions": r.get("versions"),      # full version history, with dates
            "update_date": r.get("update_date"),
        })

    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2)
    print(f"Saved {len(trimmed)} papers' metadata to {OUTPUT_METADATA_PATH}")
    return trimmed


def download_pdfs(pilot_papers):
    os.makedirs(PDF_DIR, exist_ok=True)
    print(f"\nDownloading {len(pilot_papers)} PDFs to ./{PDF_DIR}/ "
          f"(waiting {DOWNLOAD_DELAY_SECONDS}s between each, please be patient)...")

    succeeded, failed = 0, []
    for i, paper in enumerate(pilot_papers, 1):
        arxiv_id = paper["arxiv_id"]
        out_path = os.path.join(PDF_DIR, f"{arxiv_id.replace('/', '_')}.pdf")
        if os.path.exists(out_path):
            print(f"  [{i}/{len(pilot_papers)}] {arxiv_id} already downloaded, skipping")
            succeeded += 1
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-pilot-corpus-script/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"  [{i}/{len(pilot_papers)}] downloaded {arxiv_id}")
            succeeded += 1
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  [{i}/{len(pilot_papers)}] FAILED {arxiv_id}: {e}")
            failed.append(arxiv_id)

        time.sleep(DOWNLOAD_DELAY_SECONDS)

    print(f"\nDone. {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed IDs (you can retry these individually):", failed)


if __name__ == "__main__":
    pilot_papers = build_pilot_corpus()
    download_pdfs(pilot_papers)
    print("\nNext step: run 02_enrich_citation_graph.py to pull citation-graph "
          "data for these same papers from Semantic Scholar.")
