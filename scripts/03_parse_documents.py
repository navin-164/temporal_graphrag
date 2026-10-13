"""
scripts/03_parse_documents.py

Turns the pilot corpus (metadata + citation enrichment from Weeks 1-2,
PDFs already downloaded) into one structured JSON file per paper — the
Paper entity from core/entities.py, now fully populated: sections,
citations, figures, tables, and equations, all from ONE call to GROBID
per paper.

No separate 04_extract_multimodal.py
----------------------------------------
The original plan (see this file's own earlier version) had figures/
tables/equations added by a later, separate script that would re-load
these same parsed/ JSON files. Building them in here instead, in the same
pass as sections/citations, avoids calling GROBID a second time per paper
— the TEI tree itself is never saved anywhere, only the final Paper JSON,
so a later script would have had no choice but to re-process every PDF
from scratch. Given how long GROBID actually took per paper in practice,
that's not a cost worth paying just to keep the original file split.

BEFORE RUNNING THIS SCRIPT:
1. `pip install -e .` from the project root, if you haven't yet — this
   script imports `temporal_graphrag...` (the short form), which only
   resolves once the package is installed, not just sitting in src/.
2. GROBID's CRF Docker container needs to be running:
   docker run --rm --init -p 8070:8070 grobid/grobid:<pinned-tag>-crf
3. 01_build_pilot_corpus.py and 02_enrich_citation_graph.py must have
   already run — this script reads THEIR output, not the raw Kaggle file.
4. IMPORTANT if you already ran an earlier version of this script: clear
   out parsed/ first (rename it, or delete it — `ren parsed parsed_v1` on
   Windows keeps the old output around for comparison). The skip-if-
   exists resume logic below can't tell "this paper genuinely has zero
   figures" apart from "this file predates figure extraction entirely" —
   both look like an empty `"figures": []` in the saved JSON. Rather than
   guess, this script just documents the gotcha: without clearing
   parsed/, every paper will show as "already parsed, skipping" and
   silently keep its old, incomplete data forever.

WHAT THIS SCRIPT DOES:
- Loads pilot_corpus_citations.json (Week 2's output — richer than
  pilot_corpus_metadata.json alone, since it already has the Semantic
  Scholar enrichment folded in).
- For each paper: builds a Paper entity from that existing metadata
  (title/authors/abstract/categories/versions/venue/citation counts —
  all already available, no GROBID needed for any of it), sends the PDF
  to GROBID once, and fills in sections/citations/figures/tables/
  equations from that one result.
- Crops figures and tables to PNGs using their GROBID coordinates
  (CROP_IMAGES below — on by default, easy to turn off).
- Writes each paper to parsed/{arxiv_id}.json.
- Skips papers already parsed (see the gotcha above), and stops the whole
  run (rather than grinding through 39 more identical failures) if GROBID
  itself becomes unreachable mid-batch, as opposed to just failing on one
  bad PDF.

Why authors need a parser here, not just a field lookup
--------------------------------------------------------
The Kaggle arXiv dataset's `authors` field is a single free-text string
("A, B and C"), not a structured list — there's no way around some
heuristic splitting. Good enough for a pilot corpus; genuinely imperfect
for edge cases (a hyphenated surname containing "and", unusual
punctuation). Worth a proper name-parsing library if this turns out to
matter later, not before.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError as PydanticValidationError

from temporal_graphrag.core.entities import Author, Paper, PaperVersion, Venue
from temporal_graphrag.core.exceptions import ParsingError, ServiceUnavailableError, TemporalGraphRAGError
from temporal_graphrag.ingestion import (
    GrobidClient,
    crop_figure_image,
    crop_table_image,
    extract_citations,
    extract_equations,
    extract_figures,
    extract_sections,
    extract_tables,
)

# ---------------- CONFIG — edit these ----------------
INPUT_METADATA_PATH = "citations/pilot_corpus_citations.json"  # Week 2's output, not the plainer
                                                                 # pilot_corpus_metadata.json — this
                                                                 # one already has the S2 enrichment
PDF_DIR = "pdfs"
OUTPUT_DIR = "parsed"
IMAGES_DIR = "extracted_images"  # cropped figure/table PNGs; flat, not per-paper —
                                   # filenames already encode the paper (crop_page_region
                                   # names them from the entity id, e.g. 2003.08964_figure_0.png)
CROP_IMAGES = True  # set False to skip cropping (faster) if you just want text/citations/
                     # captions first, or while troubleshooting the coordinate math flagged
                     # in figure_extractor.py's docstring
DATE_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"  # matches 01_build_pilot_corpus.py's
                                            # parse_first_version_date exactly
# -------------------------------------------------------


def load_pilot_corpus() -> list[dict[str, Any]]:
    path = Path(INPUT_METADATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{INPUT_METADATA_PATH}'. Run 01_build_pilot_corpus.py "
            f"and 02_enrich_citation_graph.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_authors(raw_authors: Optional[str]) -> list[Author]:
    """Best-effort split of arXiv's free-text authors string into Author
    entities — see module docstring for why this can't be exact."""
    if not raw_authors:
        return []
    normalized = raw_authors.replace(" and ", ", ")
    names = [n.strip() for n in normalized.split(",")]
    return [Author(name=n) for n in names if n]


def _parse_versions(raw_versions: list[dict[str, Any]]) -> list[PaperVersion]:
    """Parses arXiv's version history into PaperVersion entities. Skips
    (rather than fails on) any single malformed entry — every paper in
    this pilot corpus already had its v1 date validated by script 01's
    own filtering step, so an empty result here would mean something
    more specific went wrong, not "some papers just don't have dates"."""
    versions: list[PaperVersion] = []
    for v in raw_versions:
        try:
            created = datetime.strptime(v["created"], DATE_FORMAT)
        except (ValueError, KeyError, TypeError):
            continue
        versions.append(PaperVersion(version=v.get("version", "v?"), created=created))
    return versions


def _parse_s2_fields(raw_record: dict[str, Any]) -> dict[str, Any]:
    """Pulls 02_enrich_citation_graph.py's semantic_scholar sub-object into
    the shape Paper's constructor wants. Returns {} — not defaults — when
    semantic_scholar is None, which that script already documents as
    normal for a handful of recent/unindexed papers, not a failure.

    Note: semantic_scholar_id stays unset here on purpose. 02_enrich_
    citation_graph.py's FIELDS constant never requested S2's own paperId,
    only these derived fields — there's nothing to read, not a bug here."""
    s2 = raw_record.get("semantic_scholar")
    if not s2:
        return {}

    publication_date = None
    if s2.get("publicationDate"):
        try:
            publication_date = date.fromisoformat(s2["publicationDate"])
        except ValueError:
            pass

    return {
        "venue": Venue(name=s2["venue"]) if s2.get("venue") else None,
        "publication_date": publication_date,
        "citation_count": s2.get("citationCount"),
        "reference_count": s2.get("referenceCount"),
        "fields_of_study": s2.get("fieldsOfStudy") or [],
    }


async def parse_one_paper(raw_record: dict[str, Any], pdf_path: Path, client: GrobidClient) -> Paper:
    """Builds a fully-populated Paper — existing metadata plus this run's
    GROBID extraction — for one pilot corpus record."""
    arxiv_id = raw_record["arxiv_id"]

    versions = _parse_versions(raw_record.get("versions") or [])
    if not versions:
        raise ParsingError(arxiv_id, "no parseable version history in pilot_corpus_citations.json")

    try:
        paper = Paper(
            arxiv_id=arxiv_id,
            title=raw_record["title"],
            authors=_parse_authors(raw_record.get("authors")),
            abstract=raw_record["abstract"],
            categories=(raw_record.get("categories") or "").split(),
            doi=raw_record.get("doi"),
            versions=versions,
            **_parse_s2_fields(raw_record),
        )
    except PydanticValidationError as e:
        # Wrapped so a batch run over 40 papers reports "2003.08964: metadata
        # failed validation: ..." instead of a bare pydantic traceback that
        # doesn't say which paper it was about.
        raise ParsingError(arxiv_id, f"metadata failed validation: {e}") from e

    tei_tree = await client.process_fulltext_document(pdf_path=str(pdf_path), arxiv_id=arxiv_id)
    paper.sections = extract_sections(tei_tree, paper_id=arxiv_id)
    paper.citations = extract_citations(tei_tree, paper_id=arxiv_id)
    paper.figures = extract_figures(tei_tree, paper_id=arxiv_id)
    paper.tables = extract_tables(tei_tree, paper_id=arxiv_id)
    paper.equations = extract_equations(tei_tree, paper_id=arxiv_id)

    if CROP_IMAGES:
        _crop_all_images(paper, pdf_path)

    return paper


def _crop_all_images(paper: Paper, pdf_path: Path) -> None:
    """Crops every figure/table that has a bounding_box, setting
    image_path on each in place. Crop failures are caught per item and
    logged, not raised — a bad coordinate on one figure has nothing to do
    with whether this paper's sections/citations/other figures parsed
    correctly, and shouldn't discard all of that. Broad `except Exception`
    here is deliberate, unlike the specific exception types used
    elsewhere in this project: PyMuPDF can raise several different error
    types for a malformed PDF region, and the goal is simply "don't let
    ANY crop problem take down an otherwise-good paper," not to
    distinguish between them.
    """
    for fig in paper.figures:
        if fig.bounding_box is None:
            continue
        try:
            fig.image_path = crop_figure_image(fig, str(pdf_path), IMAGES_DIR)
        except Exception as e:
            print(f"    warning: couldn't crop {fig.id}: {e}")

    for tbl in paper.tables:
        if tbl.bounding_box is None:
            continue
        try:
            tbl.image_path = crop_table_image(tbl, str(pdf_path), IMAGES_DIR)
        except Exception as e:
            print(f"    warning: couldn't crop {tbl.id}: {e}")


async def parse_all_papers() -> None:
    pilot_papers = load_pilot_corpus()
    print(f"Loaded {len(pilot_papers)} papers from {INPUT_METADATA_PATH}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    succeeded = 0
    failed: list[tuple[str, str]] = []

    async with GrobidClient() as client:
        if not await client.is_alive():
            print(f"\nERROR: GROBID isn't reachable. Start it first:")
            print("  docker run --rm --init -p 8070:8070 grobid/grobid:<pinned-tag>-crf")
            return

        for i, raw_record in enumerate(pilot_papers, 1):
            arxiv_id = raw_record["arxiv_id"]
            safe_id = arxiv_id.replace("/", "_")  # matches script 01's PDF filename convention
            out_path = output_dir / f"{safe_id}.json"
            pdf_path = Path(PDF_DIR) / f"{safe_id}.pdf"

            if out_path.exists():
                print(f"  [{i}/{len(pilot_papers)}] {arxiv_id} already parsed, skipping")
                succeeded += 1
                continue

            if not pdf_path.exists():
                print(f"  [{i}/{len(pilot_papers)}] SKIPPED {arxiv_id}: no PDF at {pdf_path}")
                failed.append((arxiv_id, "PDF file not found"))
                continue

            try:
                paper = await parse_one_paper(raw_record, pdf_path, client)
            except ServiceUnavailableError as e:
                # GROBID itself is down, not just unhappy with this one PDF --
                # every remaining paper would fail identically, so stop now
                # rather than grinding through the rest for no reason.
                print(f"\nSTOPPING: GROBID became unreachable while processing {arxiv_id}: {e}")
                print(f"Fix it and re-run — already-parsed papers in {OUTPUT_DIR}/ are")
                print("skipped automatically, so this resumes rather than starts over.")
                break
            except TemporalGraphRAGError as e:
                print(f"  [{i}/{len(pilot_papers)}] FAILED {arxiv_id}: {e}")
                failed.append((arxiv_id, str(e)))
                continue

            out_path.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
            tables_with_cells = sum(1 for t in paper.tables if t.raw_data)
            images_cropped = sum(1 for f in paper.figures if f.image_path) + sum(
                1 for t in paper.tables if t.image_path
            )
            print(
                f"  [{i}/{len(pilot_papers)}] parsed {arxiv_id}: "
                f"{len(paper.sections)} sections, {len(paper.citations)} citations, "
                f"{len(paper.figures)} figures, {len(paper.tables)} tables "
                f"({tables_with_cells} with cell data), {len(paper.equations)} equations, "
                f"{images_cropped} images cropped"
            )
            succeeded += 1

    print(f"\nDone. {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed papers:")
        for arxiv_id, reason in failed:
            print(f"  {arxiv_id}: {reason}")


if __name__ == "__main__":
    asyncio.run(parse_all_papers())
    print("\nIngestion is now complete: every parsed/*.json has sections, citations,")
    print("figures, tables, and equations. Next: Week 6's embeddings/ modules build")
    print("on top of this — no more GROBID calls needed from here on.")
