"""
core/entities.py

The domain model: pure data structures representing a scientific paper and
its parts (authors, sections, figures, tables, equations, citations). This
is the innermost layer of the architecture — no I/O, no Neo4j, no FastAPI,
no knowledge of GROBID or arXiv's JSON format. Every other layer (ingestion,
embeddings, knowledge_graph, api) imports FROM here; this module imports
from nothing else in the project.

Why Pydantic instead of @dataclass
-----------------------------------
Two concrete reasons, not just "because settings.py already uses it":
  1. Week 3's actual deliverable is "structured JSON per paper" — Pydantic's
     `.model_dump_json()` gives that for free, correctly handling nested
     entities and datetimes, instead of a hand-rolled JSON encoder.
  2. Free validation. `validate_assignment=True` below means if code three
     weeks from now does `figure.embedding = <wrong type>`, it fails right
     at that line — not silently, and not three files later inside a
     Neo4j write where it'd be much harder to trace back.

Deliberately NOT here yet
--------------------------
No `Concept`, `Method`, `Dataset`, or `Topic` entities. Those are Layer 4
graph nodes *derived* from paper content by relation extraction — they
don't exist until that extraction logic is designed in Weeks 7-8. Adding
placeholder classes for them now would mean guessing at fields we can't
justify yet; they'll be added here when relation_extractor.py is built.

Also not here: any Neo4j-specific serialization (e.g. `to_cypher_props()`).
That mapping is knowledge_graph/schema.py's job (Week 7-8), once we've
actually settled on node/property naming conventions.

ID convention
-------------
`Paper.arxiv_id` is the one natural identifier in this model — arXiv IDs
are already globally unique, so there's no synthetic UUID for Paper. Every
other entity (Section, Paragraph, Figure, Table, Equation, Citation) uses a
DETERMINISTIC composite id built from its parent's id plus its position,
via each class's `build_id()` staticmethod, e.g.
`Section.build_id("2003.08964", 2)` -> `"2003.08964::section::2"`.
Deterministic on purpose: re-running the parser on the same paper produces
the SAME ids, which is what lets Week 7-8's graph writes use Neo4j's MERGE
(upsert) instead of creating duplicate nodes on every re-run.

Usage
-----
    from temporal_graphrag.core.entities import Paper, Author, PaperVersion
    from datetime import datetime

    paper = Paper(
        arxiv_id="2003.08964",
        title="...",
        authors=[Author(name="Jane Doe")],
        abstract="...",
        categories=["cs.CL"],
        versions=[PaperVersion(version="v1", created=datetime(2020, 3, 19))],
    )
    section_id = Section.build_id(paper.arxiv_id, order=0)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Author",
    "Venue",
    "PaperVersion",
    "Paragraph",
    "Section",
    "BoundingBox",
    "Figure",
    "Table",
    "Equation",
    "Citation",
    "Paper",
]


class _BaseEntity(BaseModel):
    """Shared config for every entity below. Not exported — internal only."""

    model_config = ConfigDict(
        str_strip_whitespace=True,  # trims stray leading/trailing whitespace,
                                     # which PDF/XML extraction tends to leave behind
        validate_assignment=True,   # re-validates on later attribute writes,
                                     # e.g. `figure.embedding = ...` in Week 6
    )


class Author(_BaseEntity):
    """One author of a paper. Deliberately lightweight — no cross-paper
    author identity resolution (that needs ORCID matching or similar,
    out of scope for a pilot corpus)."""

    name: str = Field(..., min_length=1)
    affiliation: Optional[str] = None
    orcid: Optional[str] = None


class Venue(_BaseEntity):
    """A publication venue (conference/journal). Many arXiv preprints have
    none yet — leave `venue=None` on Paper rather than inventing one."""

    name: str = Field(..., min_length=1)
    venue_type: Optional[str] = None  # e.g. "conference", "journal", "workshop"


class PaperVersion(_BaseEntity):
    """One entry in a paper's arXiv revision history. `created` is a real
    datetime, not the raw arXiv string ("Mon, 2 Jan 2020 18:30:00 GMT") —
    parsing that string is the ingestion layer's job (it already exists,
    see 01_build_pilot_corpus.py's parse_first_version_date), so by the
    time it reaches this model it's already a datetime."""

    version: str = Field(..., min_length=1)  # e.g. "v1", "v2"
    created: datetime


class Paragraph(_BaseEntity):
    """One paragraph of body text within a Section."""

    id: str
    section_id: str
    text: str = Field(..., min_length=1)
    paragraph_index: int = Field(..., ge=0)

    @staticmethod
    def build_id(section_id: str, paragraph_index: int) -> str:
        return f"{section_id}::para::{paragraph_index}"


class Section(_BaseEntity):
    """A titled section of a paper (Introduction, Related Work, ...) plus
    its paragraphs, in reading order."""

    id: str
    paper_id: str
    heading: str = Field(..., min_length=1)
    # Not an Enum yet, on purpose — we don't know the real distribution of
    # heading text GROBID will hand back until 03_parse_documents.py runs
    # against actual papers. Once we do, this can tighten into a Literal
    # (introduction / related_work / method / results / ...) without
    # touching anything else in this file.
    section_type: Optional[str] = None
    order: int = Field(..., ge=0)
    paragraphs: list[Paragraph] = Field(default_factory=list)

    @staticmethod
    def build_id(paper_id: str, order: int) -> str:
        return f"{paper_id}::section::{order}"


class BoundingBox(_BaseEntity):
    """A page region in GROBID's own coordinate convention: PDF points,
    origin at the page's upper-left corner (confirmed in GROBID's own
    "Coordinates in PDF" documentation) — the same convention PyMuPDF's
    Page-level API uses by default, so figure_extractor.py's cropping
    needs no axis flip. One real caveat carried over from that same
    documentation, not something guessed at here: GROBID measures against
    the PDF's CropBox, falling back to MediaBox only if no CropBox is set
    — if a crop ever looks consistently offset by a fixed amount, that
    mismatch is the first thing to check."""

    page: int = Field(..., ge=1)  # 1-indexed, matching GROBID's own convention
    x: float
    y: float
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class Figure(_BaseEntity):
    """A figure. `caption_understanding` / `embedding` stay unpopulated
    until Week 6 — present now so downstream code has a stable field to
    eventually fill in. `image_path` IS populated by this week's
    figure_extractor.py, via `bounding_box` — extracting the cropped
    image is an extraction-layer task; captioning and embedding it are
    embeddings-layer tasks that happen once the image already exists."""

    id: str
    paper_id: str
    figure_number: Optional[str] = None  # e.g. "Figure 1", "Fig. 2a"
    caption: Optional[str] = None
    page_number: Optional[int] = Field(default=None, ge=1)
    bounding_box: Optional[BoundingBox] = None
    grobid_ref_id: Optional[str] = None   # GROBID's own xml:id — what in-text
                                           # "see Figure 1" callouts point at,
                                           # same rationale as Citation.grobid_ref_id
    image_path: Optional[str] = None              # Week 4, this file
    caption_understanding: Optional[str] = None    # Week 6, BLIP-2
    embedding: Optional[list[float]] = None         # Week 6, CLIP

    @staticmethod
    def build_id(paper_id: str, index: int) -> str:
        return f"{paper_id}::figure::{index}"


class Table(_BaseEntity):
    """A table. `raw_data` / `embedding` stay unpopulated until Weeks 4-6."""

    id: str
    paper_id: str
    table_number: Optional[str] = None
    caption: Optional[str] = None
    page_number: Optional[int] = Field(default=None, ge=1)
    bounding_box: Optional[BoundingBox] = None
    grobid_ref_id: Optional[str] = None    # same rationale as Figure.grobid_ref_id
    raw_data: Optional[list[list[str]]] = None  # row-major grid, IF GROBID's cascaded
                                                  # table model provided cell structure —
                                                  # may be None even for a successfully
                                                  # extracted table; see table_extractor.py
    image_path: Optional[str] = None       # cropped fallback — useful even when raw_data
                                            # isn't, and always available since it only
                                            # needs bounding_box, not cell structure
    embedding: Optional[list[float]] = None      # Week 6, TAPAS

    @staticmethod
    def build_id(paper_id: str, index: int) -> str:
        return f"{paper_id}::table::{index}"


class Equation(_BaseEntity):
    """A math equation. `latex` is the Partial-scope LaTeX-detection
    fallback we agreed on, not a full Nougat/MathPix extraction."""

    id: str
    paper_id: str
    latex: Optional[str] = None  # best-effort: GROBID's own <formula> content is Unicode
                                  # math glyphs from the PDF's text layer, not true LaTeX —
                                  # equation_extractor.py maps common symbols (Greek letters,
                                  # operators) to LaTeX commands, but structure like fractions
                                  # and superscripts is already lost by the time it gets here
    equation_number: Optional[str] = None  # GROBID's <label>, e.g. "10" for "(10)"
    grobid_ref_id: Optional[str] = None
    inline: bool = False  # always False for anything extracted from GROBID's <formula> —
                           # GROBID only tags standalone/display equations this way; inline
                           # formulas aren't marked at all, they're indistinguishable from
                           # surrounding paragraph text. Field kept for schema completeness
                           # (a future inline-equation detector could still set it True) —
                           # equation_extractor.py itself never produces True here.
    page_number: Optional[int] = Field(default=None, ge=1)  # stays None under current
                                                              # scope — populating this needs
                                                              # teiCoordinates=formula, which
                                                              # pdf_parser.py doesn't request
                                                              # (no image-cropping use for it,
                                                              # unlike figures/tables)

    @staticmethod
    def build_id(paper_id: str, index: int) -> str:
        return f"{paper_id}::equation::{index}"


class Citation(_BaseEntity):
    """One entry in a paper's bibliography — a reference this paper
    (`paper_id`, the CITING paper) makes to some other work. Outgoing
    direction only: Semantic Scholar's *incoming* citation list (who cites
    this paper) is corpus-level data, not a property of any single paper,
    so it stays in the raw semantic_scholar enrichment data rather than
    becoming Citation objects here."""

    id: str
    paper_id: str
    raw_text: Optional[str] = None  # the bibliography entry as GROBID sees it
    cited_title: Optional[str] = None
    cited_authors: list[str] = Field(default_factory=list)
    cited_year: Optional[int] = None
    cited_arxiv_id: Optional[str] = None  # set if this resolves to another paper in our corpus
    cited_doi: Optional[str] = None
    external_id: Optional[str] = None     # Semantic Scholar ID, for enrichment matching
    grobid_ref_id: Optional[str] = None   # GROBID's own xml:id for this entry (e.g. "b0") —
                                           # what in-text <ref target="#b0"> markers point at.
                                           # text_extractor.py flattens those markers into plain
                                           # text, so this is the only remaining thread back to
                                           # them; nothing reads it yet, same status as the field below.
    in_text_context: Optional[str] = None  # the sentence(s) invoking this citation —
                                            # populated later, once the Verification Agent
                                            # needs to check a claim against its source

    @staticmethod
    def build_id(paper_id: str, index: int) -> str:
        return f"{paper_id}::citation::{index}"


class Paper(_BaseEntity):
    """The aggregate root — one scientific paper and everything we know
    about it. `arxiv_id` is the only natural identifier in this model; see
    the module docstring for why every other entity gets a synthetic one.

    Fields split into two groups on purpose:
      - metadata fields (title, authors, abstract, categories, versions)
        are required and validated non-empty, because they're all
        available immediately from pilot_corpus_metadata.json — there's
        no legitimate state where a Paper exists without them.
      - content fields (sections, figures, tables, equations, citations)
        default to empty lists, because they're populated incrementally
        by the slower PDF-parsing pipeline (Weeks 3-5), and some papers
        genuinely have zero figures or equations — that's valid data,
        not a sign parsing failed.
    """

    # -- Identity & metadata (required — from arXiv metadata directly) --
    arxiv_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    authors: list[Author] = Field(..., min_length=1)
    abstract: str = Field(..., min_length=1)
    categories: list[str] = Field(..., min_length=1)
    versions: list[PaperVersion] = Field(..., min_length=1)

    # -- Optional metadata --
    doi: Optional[str] = None
    venue: Optional[Venue] = None

    # -- Semantic Scholar enrichment (Optional — 02_enrich_citation_graph.py
    #    already documents that some pilot papers won't be found on S2) --
    semantic_scholar_id: Optional[str] = None
    publication_date: Optional[date] = None  # S2's formal pub date — can differ
                                              # from the arXiv v1 date above;
                                              # kept as a separate field on purpose
    citation_count: Optional[int] = Field(default=None, ge=0)
    reference_count: Optional[int] = Field(default=None, ge=0)
    fields_of_study: list[str] = Field(default_factory=list)

    # -- Content, populated by the parsing pipeline (Weeks 3-5) --
    sections: list[Section] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    # -- Bookkeeping --
    last_updated: Optional[datetime] = None  # when WE last (re)processed this
                                              # paper — not a property of the
                                              # paper itself

    # -- Convenience properties --------------------------------------------
    # Small, read-only, directly serve "track how research evolves" — none
    # of these do file I/O or touch the network.

    @property
    def first_version_date(self) -> datetime:
        """Submission date of v1. List order matches arXiv's own
        chronological ordering, so no re-sorting is needed here."""
        return self.versions[0].created

    @property
    def latest_version_date(self) -> datetime:
        return self.versions[-1].created

    @property
    def version_count(self) -> int:
        return len(self.versions)
