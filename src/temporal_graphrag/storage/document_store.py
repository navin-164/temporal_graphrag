"""
storage/document_store.py

Original-content lookup — the sibling to vector_store.py. A FAISS search
hit is just an id and a similarity score; this class turns that id back
into the actual Paper, Section, Paragraph, Figure, Table, Equation, or
Citation it came from, so retrieval (Week 11) and the agents (Weeks
12-14) have real content to reason over and show the user, not just
numbers. Matches the architecture diagram's "5.3 Document Store
(Original Chunks & Metadata)" box.

Why this wraps parsed/*.json, not a database
--------------------------------------------------
parsed/*.json (03_parse_documents.py's output) already IS the original,
fully-structured content — every Paper, with all its sections, figures,
tables, equations, and citations, exactly as extracted. At pilot scale
(40 papers), loading every one of those files into memory once at
startup is simpler and faster than standing up a database just to serve
lookups this small — reading 40 JSON files takes a fraction of a second,
and there is no benefit to querying disk repeatedly for data that fits
comfortably in memory. Revisit this if the corpus grows well past pilot
size; not a decision worth making now (same reasoning vector_store.py's
docstring gives for choosing an exact FAISS index over an approximate one
at this scale).

The id convention this file depends on
-------------------------------------------
core/entities.py's module docstring documents this already, repeated
here because this file leans on it directly: every entity below Paper
uses a DETERMINISTIC composite id of the form
`{parent_id}::{type}::{index}` (Paragraph nests one level deeper:
`{paper_id}::section::{n}::para::{m}`). Because the composite always
starts with the owning paper's arxiv_id followed by "::", splitting any
composite id on the FIRST "::" recovers the parent arxiv_id regardless of
nesting depth — `_parent_arxiv_id()` below relies on exactly this, once.

resolve(): one entry point when the caller doesn't know the type upfront
------------------------------------------------------------------------
vector_store.py's search() is called per-modality ("text", "figure",
"table", ...), so a caller usually already knows what kind of id it's
holding. But Week 11's hybrid_scorer.py fuses results across modalities
into one ranked list, at which point re-deriving "is this a figure id or
a table id" from the id string itself (rather than threading a separate
type tag through the whole fusion pipeline) is the simpler shape.
resolve() does that sniffing in one place so nothing else has to.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

from temporal_graphrag.core.entities import (
    Citation,
    Equation,
    Figure,
    Paper,
    Paragraph,
    Section,
    Table,
)
from temporal_graphrag.core.exceptions import NotFoundError

# Plain stdlib logging, not structlog, even though structlog is in
# requirements.txt — same reasoning as ingestion/pdf_parser.py:
# utils/logging_config.py (not built yet) is what decides how structlog
# gets wired up; using it directly here would just be guessing at that.
logger = logging.getLogger(__name__)

__all__ = ["DocumentStore"]

ResolvedEntity = Union[Paper, Section, Paragraph, Figure, Table, Equation, Citation]


class DocumentStore:
    """In-memory lookup from an id (paper, or any composite sub-entity id)
    back to the original, fully-structured content it refers to.

    Usage:
        store = DocumentStore.from_directory("parsed")
        paper = store.get_paper("2003.08964")
        figure = store.get_figure("2003.08964::figure::0")

        # or, when the caller doesn't know the type upfront (e.g. after
        # hybrid_scorer.py fuses results across modalities):
        entity = store.resolve(some_id)
    """

    def __init__(self, papers: dict[str, Paper]) -> None:
        self._papers = papers

    # -- Construction ------------------------------------------------------

    @classmethod
    def from_directory(cls, parsed_dir: str = "parsed") -> "DocumentStore":
        """Loads every parsed/*.json into memory, keyed by arxiv_id. A
        single corrupted or incomplete file is logged and skipped, not
        fatal to the rest — same "one bad item shouldn't stop everything
        else" reasoning this project's build scripts already use for the
        exact same load_papers() pattern (03_parse_documents.py,
        06_build_knowledge_graph.py, 05_build_embeddings.py)."""
        directory = Path(parsed_dir)
        if not directory.exists():
            raise FileNotFoundError(f"'{parsed_dir}/' doesn't exist. Run 03_parse_documents.py first.")

        papers: dict[str, Paper] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                paper = Paper.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("couldn't load %s, skipping: %s", path.name, e)
                continue
            papers[paper.arxiv_id] = paper

        return cls(papers)

    # -- Direct lookups ------------------------------------------------------

    def get_paper(self, arxiv_id: str) -> Paper:
        try:
            return self._papers[arxiv_id]
        except KeyError:
            raise NotFoundError("Paper", arxiv_id) from None

    def get_section(self, section_id: str) -> Section:
        paper = self.get_paper(self._parent_arxiv_id(section_id))
        for section in paper.sections:
            if section.id == section_id:
                return section
        raise NotFoundError("Section", section_id)

    def get_paragraph(self, paragraph_id: str) -> Paragraph:
        paper = self.get_paper(self._parent_arxiv_id(paragraph_id))
        for section in paper.sections:
            for paragraph in section.paragraphs:
                if paragraph.id == paragraph_id:
                    return paragraph
        raise NotFoundError("Paragraph", paragraph_id)

    def get_figure(self, figure_id: str) -> Figure:
        paper = self.get_paper(self._parent_arxiv_id(figure_id))
        for figure in paper.figures:
            if figure.id == figure_id:
                return figure
        raise NotFoundError("Figure", figure_id)

    def get_table(self, table_id: str) -> Table:
        paper = self.get_paper(self._parent_arxiv_id(table_id))
        for table in paper.tables:
            if table.id == table_id:
                return table
        raise NotFoundError("Table", table_id)

    def get_equation(self, equation_id: str) -> Equation:
        paper = self.get_paper(self._parent_arxiv_id(equation_id))
        for equation in paper.equations:
            if equation.id == equation_id:
                return equation
        raise NotFoundError("Equation", equation_id)

    def get_citation(self, citation_id: str) -> Citation:
        paper = self.get_paper(self._parent_arxiv_id(citation_id))
        for citation in paper.citations:
            if citation.id == citation_id:
                return citation
        raise NotFoundError("Citation", citation_id)

    # -- Type-sniffing entry point -------------------------------------------

    def resolve(self, id_: str) -> ResolvedEntity:
        """Returns whatever entity `id_` refers to, inferring the type from
        the id's own shape (see module docstring). Checked most-specific
        first — "::para::" before "::section::" — since a paragraph id
        contains "::section::" too (it nests under its parent section)."""
        if "::para::" in id_:
            return self.get_paragraph(id_)
        if "::section::" in id_:
            return self.get_section(id_)
        if "::figure::" in id_:
            return self.get_figure(id_)
        if "::table::" in id_:
            return self.get_table(id_)
        if "::equation::" in id_:
            return self.get_equation(id_)
        if "::citation::" in id_:
            return self.get_citation(id_)
        return self.get_paper(id_)  # no "::" at all -> a bare arxiv_id

    # -- Introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._papers)

    def __contains__(self, arxiv_id: str) -> bool:
        return arxiv_id in self._papers

    def all_arxiv_ids(self) -> list[str]:
        return list(self._papers.keys())

    # -- Internal ------------------------------------------------------------

    @staticmethod
    def _parent_arxiv_id(composite_id: str) -> str:
        """`"2003.08964::figure::0"` -> `"2003.08964"`. Works for every
        composite id in this project, nested or not, because they all
        start with the owning paper's arxiv_id followed by "::" — see
        module docstring."""
        return composite_id.split("::", 1)[0]