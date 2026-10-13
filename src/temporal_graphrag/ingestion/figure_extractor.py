"""
ingestion/figure_extractor.py

Turns GROBID's <figure> elements into Figure entities, and crops the
actual figure image out of the original PDF using the coordinates GROBID
provides — the "2.3 Figure Understanding" extraction half of Week 4-5.
Captioning the image (BLIP-2) and embedding it (CLIP) are separate, later
steps in embeddings/ — this file's job ends at "here's a clean PNG of
just this figure," which is what those steps need as input.

Two research-driven design decisions worth knowing about
------------------------------------------------------------
1. Tables use the SAME <figure> element as real figures, distinguished
   only by a `type="table"` attribute (confirmed in GROBID's own training
   documentation — this isn't a guess). `type="box"` (boxed/callout
   content) is excluded the same way. Skipping this check would mean
   figure_extractor.py silently also extracting every table as if it
   were a figure.
2. pdf_parser.py now requests `teiCoordinates=figure` and `teiCoordinates=
   table` from GROBID (a small addition made alongside this file) —
   without that parameter, GROBID's response has captions but no
   <graphic coords="..."> element at all, and every bounding_box below
   would come back None.

The one thing here I genuinely cannot verify without a real PDF + GROBID
+ PyMuPDF environment: that the coordinate math in crop_figure_image()
lines up correctly. What IS confirmed, not guessed, from GROBID's own
"Coordinates in PDF" documentation: coordinates are in PDF points with
the origin at the page's upper-left corner, which matches PyMuPDF's
default page coordinate system directly (no axis flip needed). The same
documentation also flags a specific known failure mode: GROBID measures
against the PDF's CropBox, falling back to MediaBox only if no CropBox is
set. If your first few crops come out consistently offset by a fixed
amount rather than looking randomly wrong, that mismatch — not a logic
bug in this function — is the first thing to check. Open one or two
saved crops next to the actual PDF page before trusting a full batch run.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from lxml import etree

from temporal_graphrag.core.entities import BoundingBox, Figure

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_NS_ID = "{http://www.w3.org/XML/1998/namespace}id"
_WHITESPACE_RE = re.compile(r"\s+")

# Confirmed via GROBID's own training documentation: a table is encoded
# as a <figure type="table">, and boxed/callout content as type="box" —
# neither is a figure in the sense this file cares about.
_EXCLUDED_FIGURE_TYPES = {"table", "box"}


def extract_figures(tei_tree: etree._Element, paper_id: str) -> list[Figure]:
    """Extract Figure entities (caption, number, position — not yet the
    cropped image; see crop_figure_image() for that) from a paper's TEI
    tree, in document order.

    Skips: <figure type="table"> and <figure type="box"> (see module
    docstring), and any <figure> that yields neither a figure_number nor
    a caption — nothing usable to record about it.
    """
    body = tei_tree.find(".//tei:text/tei:body", namespaces=TEI_NS)
    if body is None:
        return []

    figures: list[Figure] = []
    index = 0

    for fig_el in body.findall(".//tei:figure", namespaces=TEI_NS):
        if fig_el.get("type") in _EXCLUDED_FIGURE_TYPES:
            continue

        head_el = fig_el.find("tei:head", namespaces=TEI_NS)
        figure_number = _clean_text(head_el) or None

        desc_el = fig_el.find("tei:figDesc", namespaces=TEI_NS)
        caption = _clean_text(desc_el) or None

        if not figure_number and not caption:
            continue

        graphic_el = fig_el.find("tei:graphic", namespaces=TEI_NS)
        bounding_box = _parse_bounding_box(graphic_el)

        figures.append(
            Figure(
                id=Figure.build_id(paper_id, index),
                paper_id=paper_id,
                figure_number=figure_number,
                caption=caption,
                page_number=bounding_box.page if bounding_box else None,
                bounding_box=bounding_box,
                grobid_ref_id=fig_el.get(XML_NS_ID),
            )
        )
        index += 1

    return figures


def crop_figure_image(
    figure: Figure, pdf_path: str, output_dir: str, dpi: int = 150
) -> Optional[str]:
    """Crops one figure's region out of the original PDF and saves it as
    a PNG. Returns the saved path, or None if this figure has no
    bounding_box to crop from (GROBID recognized the figure but couldn't
    place coordinates for it — happens occasionally, not an error).

    Thin wrapper around crop_page_region() (below) — the actual
    coordinate math is entity-agnostic, and table_extractor.py needs the
    identical logic for tables, so it lives in one place, not two.
    """
    if figure.bounding_box is None:
        return None
    return crop_page_region(figure.bounding_box, figure.id, pdf_path, output_dir, dpi=dpi)


def crop_page_region(
    bounding_box: BoundingBox, entity_id: str, pdf_path: str, output_dir: str, dpi: int = 150
) -> str:
    """Crops one page region out of a PDF and saves it as a PNG named
    after `entity_id`. Entity-agnostic on purpose — both Figure and Table
    need the exact same coordinate-to-crop math (see module docstring for
    the coordinate-system reasoning), so this is the one place it lives.

    Deliberately lets a bad `pdf_path` or corrupted PDF raise rather than
    returning None — a PDF that won't open is the same problem for every
    figure/table in that paper, worth surfacing clearly at the caller
    rather than silently looking identical to "no bounding box."
    """
    import fitz  # PyMuPDF — imported here, not at module level, so calling
                  # extract_figures() alone (the common case: TEI parsing
                  # only) never requires opening an actual PDF file.

    doc = fitz.open(pdf_path)
    try:
        if bounding_box.page < 1 or bounding_box.page > len(doc):
            raise ValueError(
                f"bounding_box.page={bounding_box.page} is out of range for "
                f"{pdf_path} ({len(doc)} pages)"
            )
        page = doc[bounding_box.page - 1]  # GROBID pages are 1-indexed, fitz's are 0-indexed
        clip_rect = fitz.Rect(
            bounding_box.x,
            bounding_box.y,
            bounding_box.x + bounding_box.width,
            bounding_box.y + bounding_box.height,
        )
        pixmap = page.get_pixmap(clip=clip_rect, dpi=dpi)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{entity_id.replace('::', '_')}.png")
        pixmap.save(out_path)
        return out_path
    finally:
        doc.close()


def _parse_bounding_box(graphic_el: Optional[etree._Element]) -> Optional[BoundingBox]:
    """Parses GROBID's "page,x,y,width,height" coords string (confirmed
    format, from GROBID's own documentation examples) into a BoundingBox.
    Fails soft to None on anything malformed — a figure with an
    unparseable coords string still gets a Figure entity with its
    caption, just without a crop-able position."""
    if graphic_el is None:
        return None
    coords = graphic_el.get("coords")
    if not coords:
        return None
    parts = coords.split(",")
    if len(parts) != 5:
        return None
    try:
        page_str, x, y, width, height = parts
        return BoundingBox(
            page=int(page_str), x=float(x), y=float(y), width=float(width), height=float(height)
        )
    except (ValueError, TypeError):
        return None


def _clean_text(element: Optional[etree._Element]) -> str:
    """Same flattening as text_extractor.py's and citation_extractor.py's
    helper of the same name — duplicated for the same reason: these files
    are meant to stay independent, and this is four lines."""
    if element is None:
        return ""
    raw = "".join(element.itertext())
    return _WHITESPACE_RE.sub(" ", raw).strip()
