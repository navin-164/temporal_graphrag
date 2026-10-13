"""
ingestion/table_extractor.py

Turns GROBID's <figure type="table"> elements into Table entities. Shares
its basic shape with figure_extractor.py (both come from the same
<figure> element, just filtered the opposite way — this file wants ONLY
type="table", that one excludes it), but table content itself is
genuinely uncertain in a way figures weren't, and this file is built to
be honest about that rather than assume.

The real uncertainty: does <row>/<cell> come back at all?
--------------------------------------------------------------
GROBID's own docs say a table's actual <table> structure "appears in the
table model applied in cascade" — i.e. cell-level recognition is a
SEPARATE model from the one that marks the region as <figure type=
"table"> in the first place. Whether calling processFulltextDocument
(what pdf_parser.py already does) runs that cascade automatically, or
whether it needs a different call, isn't something documentation alone
settled — and there's no real GROBID instance in the environment this
was written in to just check.

So extract_tables() below is built to degrade gracefully rather than
guess: it looks for <row>/<cell> anywhere inside the table figure, and if
they're there, raw_data comes back as a proper row-major grid. If they're
NOT there, raw_data stays None — but table_number, caption, page_number,
and bounding_box (and from that, a cropped image via crop_table_image())
still get captured, since none of those depend on cell structure existing.
A table with only a caption and a cropped image is still useful; a wrong
guess at cell boundaries from unstructured text would not be.

First real run tells us which case we're in, which no amount of further
searching would have — check a few parsed tables' raw_data specifically
before assuming either way.
"""

from __future__ import annotations

import re
from typing import Optional

from lxml import etree

from temporal_graphrag.core.entities import BoundingBox, Table
from temporal_graphrag.ingestion.figure_extractor import crop_page_region

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_NS_ID = "{http://www.w3.org/XML/1998/namespace}id"
_WHITESPACE_RE = re.compile(r"\s+")


def extract_tables(tei_tree: etree._Element, paper_id: str) -> list[Table]:
    """Extract Table entities from a paper's TEI tree, in document order.

    Skips any <figure type="table"> that yields neither a table_number
    nor a caption — nothing usable to record about it, same standard
    figure_extractor.py applies.
    """
    body = tei_tree.find(".//tei:text/tei:body", namespaces=TEI_NS)
    if body is None:
        return []

    tables: list[Table] = []
    index = 0

    for fig_el in body.findall(".//tei:figure", namespaces=TEI_NS):
        if fig_el.get("type") != "table":
            continue

        head_el = fig_el.find("tei:head", namespaces=TEI_NS)
        table_number = _clean_text(head_el) or None

        desc_el = fig_el.find("tei:figDesc", namespaces=TEI_NS)
        caption = _clean_text(desc_el) or None

        if not table_number and not caption:
            continue

        graphic_el = fig_el.find("tei:graphic", namespaces=TEI_NS)
        bounding_box = _parse_bounding_box(graphic_el)
        raw_data = _extract_grid(fig_el)

        tables.append(
            Table(
                id=Table.build_id(paper_id, index),
                paper_id=paper_id,
                table_number=table_number,
                caption=caption,
                page_number=bounding_box.page if bounding_box else None,
                bounding_box=bounding_box,
                grobid_ref_id=fig_el.get(XML_NS_ID),
                raw_data=raw_data,
            )
        )
        index += 1

    return tables


def crop_table_image(table: Table, pdf_path: str, output_dir: str, dpi: int = 150) -> Optional[str]:
    """Crops one table's region out of the original PDF, same coordinate
    math figure_extractor.py's crop_figure_image() uses (crop_page_region
    is shared between the two, not duplicated) — a useful fallback
    regardless of whether raw_data came back structured or None."""
    if table.bounding_box is None:
        return None
    return crop_page_region(table.bounding_box, table.id, pdf_path, output_dir, dpi=dpi)


def _extract_grid(fig_el: etree._Element) -> Optional[list[list[str]]]:
    """Looks for <row>/<cell> anywhere inside the table figure — see
    module docstring for why this is a "try it, don't assume it's there"
    search rather than a fixed path. Returns None if no rows are found,
    or if every row that IS found has zero cells (equally uninformative)."""
    row_els = fig_el.findall(".//tei:row", namespaces=TEI_NS)
    if not row_els:
        return None

    grid = [
        [_clean_text(cell_el) for cell_el in row_el.findall("tei:cell", namespaces=TEI_NS)]
        for row_el in row_els
    ]
    if all(len(row) == 0 for row in grid):
        return None
    return grid


def _parse_bounding_box(graphic_el: Optional[etree._Element]) -> Optional[BoundingBox]:
    """Identical to figure_extractor.py's helper of the same purpose —
    duplicated rather than imported for the usual reason (these files
    stay independent), except crop_page_region itself, which was worth
    sharing since getting that coordinate math right once matters more
    than keeping every function independent."""
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
    """Same flattening as the other ingestion files' helper of the same
    name."""
    if element is None:
        return ""
    raw = "".join(element.itertext())
    return _WHITESPACE_RE.sub(" ", raw).strip()
