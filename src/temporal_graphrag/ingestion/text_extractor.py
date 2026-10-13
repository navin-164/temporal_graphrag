"""
ingestion/text_extractor.py

Turns the TEI XML tree returned by pdf_parser.GrobidClient.
process_fulltext_document() into Section/Paragraph entities — the actual
"structured JSON per paper" content for this week's deliverable.

Scope boundary: paragraphs and section headings only
--------------------------------------------------------
Deliberately NOT extracting sentence-level structure here, even though
the timeline mentions "sentence segmentation." Reason: core/entities.py
has no Sentence class, because nothing downstream needs one yet — no
retrieval logic (Week 11) has been designed that would consume
sentence-level chunks specifically. GROBID CAN produce this (pass
segmentSentences=1 to processFulltextDocument, then read <s> elements
inside each <p>), so adding it later is a small, contained change to
pdf_parser.py's request plus this file's paragraph-building loop, not a
redesign. Building it now, before anything consumes it, would mean
guessing at a shape we don't actually know we need yet.

No hierarchical subsections
------------------------------
GROBID sometimes nests <div> elements for subsections (3 -> 3.1 -> 3.1.2).
Section has no parent_section_id field, so nested divs become separate,
sibling Sections in reading order (via `order`), not a tree. If we later
want a real table-of-contents view, that's a Section schema change, not
a change to the extraction logic below.

Guarding against references/acknowledgments leaking into <body>
--------------------------------------------------------------------
References normally live in <back>, not <body>, so a well-formed GROBID
parse shouldn't produce them here at all — this file only ever looks
inside <body>. The heading-text filter below is a cheap defensive guard
for PDF layout quirks that occasionally put a references or
acknowledgments div inside <body> anyway. Not something observed in this
specific pilot corpus yet — untested against real PDFs, same limitation
as pdf_parser.py — just cheap enough to include preemptively.
"""

from __future__ import annotations

import re
from typing import Optional

from lxml import etree

from temporal_graphrag.core.entities import Paragraph, Section

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_WHITESPACE_RE = re.compile(r"\s+")
_ADMINISTRATIVE_HEADING_PATTERNS = ("reference", "bibliograph", "acknowledg")


def extract_sections(tei_tree: etree._Element, paper_id: str) -> list[Section]:
    """Extract Section (with nested Paragraph) entities from a paper's
    parsed TEI tree, in reading order.

    Skips: <div> elements with no <head> (nothing sensible to call them),
    <div> elements whose heading looks like a reference list or
    acknowledgments section (see module docstring), and any <div> whose
    every paragraph is empty after cleaning (a heading with no real
    content isn't a useful Section).
    """
    body = tei_tree.find(".//tei:text/tei:body", namespaces=TEI_NS)
    if body is None:
        return []

    sections: list[Section] = []
    order = 0

    for div in body.findall(".//tei:div", namespaces=TEI_NS):
        head_el = div.find("tei:head", namespaces=TEI_NS)
        heading = _clean_text(head_el)
        if not heading or _looks_administrative(heading):
            continue

        section_id = Section.build_id(paper_id, order)
        paragraphs = _extract_paragraphs(div, section_id)
        if not paragraphs:
            continue

        sections.append(
            Section(
                id=section_id,
                paper_id=paper_id,
                heading=heading,
                order=order,
                paragraphs=paragraphs,
            )
        )
        order += 1

    return sections


def _extract_paragraphs(div: etree._Element, section_id: str) -> list[Paragraph]:
    """Direct-child <p> elements only — deliberately NOT `.//tei:p`, which
    would also pull in paragraphs belonging to a nested subsection <div>
    and double-count them once that subsection is processed as its own
    Section by the loop in extract_sections()."""
    paragraphs: list[Paragraph] = []
    kept_index = 0
    for p_el in div.findall("tei:p", namespaces=TEI_NS):
        text = _clean_text(p_el)
        if not text:
            continue
        paragraphs.append(
            Paragraph(
                id=Paragraph.build_id(section_id, kept_index),
                section_id=section_id,
                text=text,
                paragraph_index=kept_index,
            )
        )
        kept_index += 1
    return paragraphs


def _clean_text(element: Optional[etree._Element]) -> str:
    """Flattens an element's full text content — including nested inline
    markup like <ref> (citation markers), <hi> (emphasis), inline
    <formula> — into one clean string. None of those inline element types
    are special-cased here; citation_extractor.py handles <ref> markers on
    their own terms later, working from the same TEI tree."""
    if element is None:
        return ""
    raw = "".join(element.itertext())
    return _WHITESPACE_RE.sub(" ", raw).strip()


def _looks_administrative(heading: str) -> bool:
    """References/acknowledgments normally live in <back>, not <body>, so
    a well-formed parse shouldn't hit this at all. Cheap guard for the PDF
    layout quirks that occasionally put them in <body> anyway."""
    lowered = heading.lower()
    return any(pattern in lowered for pattern in _ADMINISTRATIVE_HEADING_PATTERNS)
