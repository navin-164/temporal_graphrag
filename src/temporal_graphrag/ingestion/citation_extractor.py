"""
ingestion/citation_extractor.py

Turns the bibliography portion of the same TEI tree pdf_parser.py and
text_extractor.py already work from into Citation entities — the
"references" half of Week 3's "sections, references" scope.

TEI's <analytic>/<monogr> split, and why both get checked
--------------------------------------------------------------
GROBID encodes each bibliography entry as a <biblStruct>. For a journal
article, the article's own title/authors sit in <analytic>, and the
containing journal's name sits in <monogr>. But for a standalone work
(a book, a thesis, a report — anything with no separate "container"),
GROBID emits no <analytic> at all: the title and authors sit directly in
<monogr> instead. So every lookup here tries <analytic> first and falls
back to <monogr> — skipping the fallback would silently drop every
book/report reference's title and authors.

Real-world messiness, handled defensively
----------------------------------------------
GROBID's own documentation examples show inconsistent output: a <date
when="0782">1997 10.782</date> where the attribute clearly isn't a real
year, `<idno>` elements that sometimes lack a `type` attribute, and a
noted tendency to occasionally over-include trailing document content as
if it were still part of the bibliography. None of that is specific to
this pilot corpus — it just means citation parsing is inherently noisier
than section/paragraph parsing, so failures here are handled per-entry
(skip that one field, or that one biblStruct) rather than letting one
malformed reference take down extraction for the other 30+ in the same
paper.

Same testing situation as the last two ingestion files: lxml isn't
installable in the sandbox this was written in, so this was verified by
running the actual extraction logic (via the same import-swap approach
used for text_extractor.py) against a hand-built TEI sample covering the
analytic/monogr fallback, a malformed date, and a missing DOI — not
against a real GROBID response.
"""

from __future__ import annotations

import re
from typing import Optional

from lxml import etree

from temporal_graphrag.core.entities import Citation

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_NS_ID = "{http://www.w3.org/XML/1998/namespace}id"
_WHITESPACE_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"(\d{4})")


def extract_citations(tei_tree: etree._Element, paper_id: str) -> list[Citation]:
    """Extract one Citation per <biblStruct> found anywhere in the tree
    (normally under <back>/<div type="references">/<listBibl>, but this
    searches the whole tree rather than assuming that exact path, since
    <listBibl> placement isn't always identical across GROBID versions).

    Skips any <biblStruct> that yields neither a title, any authors, nor
    raw text — i.e. one GROBID couldn't extract anything useful from,
    which does happen (see module docstring).
    """
    bibl_structs = tei_tree.findall(".//tei:listBibl/tei:biblStruct", namespaces=TEI_NS)

    citations: list[Citation] = []
    kept_index = 0
    for bibl in bibl_structs:
        citation = _parse_bibl_struct(bibl, paper_id=paper_id, index=kept_index)
        if citation is None:
            continue
        citations.append(citation)
        kept_index += 1

    return citations


def _parse_bibl_struct(
    bibl: etree._Element, *, paper_id: str, index: int
) -> Optional[Citation]:
    analytic = bibl.find("tei:analytic", namespaces=TEI_NS)
    monogr = bibl.find("tei:monogr", namespaces=TEI_NS)

    # Title: <analytic> first (the cited work's own title), <monogr> as
    # fallback (books/reports have no <analytic> at all — see module docstring).
    title_el = analytic.find("tei:title", namespaces=TEI_NS) if analytic is not None else None
    if title_el is None and monogr is not None:
        title_el = monogr.find("tei:title", namespaces=TEI_NS)
    cited_title = _clean_text(title_el) or None

    # Authors: same analytic-then-monogr fallback, for the same reason.
    author_container = analytic if analytic is not None else monogr
    cited_authors = (
        _extract_author_names(author_container) if author_container is not None else []
    )

    # Year lives in <imprint><date when="YYYY"/></imprint> under <monogr> —
    # in the `when` ATTRIBUTE, not the element's text content.
    cited_year = None
    if monogr is not None:
        date_el = monogr.find("tei:imprint/tei:date", namespaces=TEI_NS)
        if date_el is not None:
            cited_year = _extract_year(date_el.get("when"))

    doi_el = bibl.find(".//tei:idno[@type='DOI']", namespaces=TEI_NS)
    cited_doi = _clean_text(doi_el) or None

    arxiv_el = bibl.find(".//tei:idno[@type='arXiv']", namespaces=TEI_NS)
    cited_arxiv_id = _clean_text(arxiv_el) or None

    raw_note = bibl.find("tei:note[@type='raw_reference']", namespaces=TEI_NS)
    raw_text = _clean_text(raw_note) or None

    if not cited_title and not cited_authors and not raw_text:
        # Nothing usable came out of this entry at all — GROBID does
        # occasionally produce these (see module docstring). Not worth
        # keeping a Citation with every field empty.
        return None

    grobid_ref_id = bibl.get(XML_NS_ID)

    return Citation(
        id=Citation.build_id(paper_id, index),
        paper_id=paper_id,
        raw_text=raw_text,
        cited_title=cited_title,
        cited_authors=cited_authors,
        cited_year=cited_year,
        cited_arxiv_id=cited_arxiv_id,
        cited_doi=cited_doi,
        grobid_ref_id=grobid_ref_id,
    )


def _extract_author_names(container: etree._Element) -> list[str]:
    """Combines forename(s) + surname for each <author>/<persName> inside
    the given <analytic> or <monogr> element. Only direct <author>
    children — <editor> uses the identical <persName> shape but isn't an
    author, so a broader search would risk mislabeling editors as authors."""
    names: list[str] = []
    for author_el in container.findall("tei:author", namespaces=TEI_NS):
        pers_name = author_el.find("tei:persName", namespaces=TEI_NS)
        if pers_name is None:
            continue
        forenames = [
            _clean_text(f) for f in pers_name.findall("tei:forename", namespaces=TEI_NS)
        ]
        surname = _clean_text(pers_name.find("tei:surname", namespaces=TEI_NS))
        full_name = " ".join(part for part in [*forenames, surname] if part)
        if full_name:
            names.append(full_name)
    return names


def _extract_year(when_value: Optional[str]) -> Optional[int]:
    """Parses the `when` attribute of a <date> element into a year. Real
    GROBID output is sometimes messy here (see module docstring) — fails
    soft to None rather than raising, since one malformed date shouldn't
    break extraction for the other references in the same paper."""
    if not when_value:
        return None
    match = _YEAR_RE.match(when_value)
    if not match:
        return None
    year = int(match.group(1))
    if not (1000 <= year <= 2100):  # sanity bound against garbage matches
        return None
    return year


def _clean_text(element: Optional[etree._Element]) -> str:
    """Same flattening as text_extractor.py's helper of the same name —
    duplicated rather than imported from there, since these two files are
    meant to stay independent (a change to one shouldn't require touching
    the other) and this is four lines, not worth a shared-utils import for."""
    if element is None:
        return ""
    raw = "".join(element.itertext())
    return _WHITESPACE_RE.sub(" ", raw).strip()
