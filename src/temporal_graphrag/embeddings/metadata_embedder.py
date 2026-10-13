"""
embeddings/metadata_embedder.py

Structured-metadata embeddings — authors, venue, year, categories, fields
of study — via text serialization, reusing text_embedder.py's SPECTER2
pipeline. This is the fourth and final embedding modality from the
architecture diagram's "3.5 Metadata Embedding" box, and it closes out
Week 6.

Why serialize-and-reuse, not a dedicated model
-----------------------------------------------
Same reasoning table_embedder.py already lays out for tables: there's no
well-established "author/venue/year encoder" the way SPECTER2 exists for
(title, abstract) or CLIP exists for images. Structured metadata turned
into a short, templated text string and pushed through a general-purpose
text encoder is a common, defensible pattern — and reusing the SPECTER2
pipeline already loaded for TextEmbedder means zero new model weights in
memory, consistent with this project's "lightweight fallbacks" goal.

Why this is a SEPARATE vector from TextEmbedder's (title, abstract)
embedding, not folded into it
---------------------------------------------------------------------
TextEmbedder answers "is this paper *about* the same thing" (semantic
content). This file answers a different question: "does this paper come
from the same author / venue / era / subfield". Keeping them as two
vectors (rather than concatenating metadata onto the title+abstract
string before embedding once) means Week 11's hybrid_scorer.py can weight
them independently — e.g. a query like "recent work by this author on
this topic" wants BOTH signals, but weighted differently than a pure
topic search does. Collapsing them into one vector now would foreclose
that option later for no benefit today.

Why no numeric fields (citation_count, reference_count) in the text
---------------------------------------------------------------------
Embedding raw numbers as text tokens ("42 citations") doesn't give a
transformer encoder a meaningful notion of magnitude or ordering — two
papers with 42 and 4200 citations aren't reliably "close" in embedding
space just because both mention numbers. Fields like citation_count are
better used directly as numeric filters/boosts in hybrid_scorer.py than
smuggled into a semantic embedding where they add noise more than signal.
Same reasoning excludes doi and semantic_scholar_id — identifiers, not
semantic content.

Year, not full publication_date
--------------------------------
Day-level precision doesn't carry semantic meaning for "was this recent
research" the way year does, and collapsing to year keeps the serialized
text short and consistent. Prefers `publication_date` (Semantic Scholar's
formal date) when available, falling back to `first_version_date` (the
arXiv v1 submission date, always present) — matching Paper's own
docstring note that these two dates can legitimately differ.

Never returns None
-------------------
Unlike TableEmbedder.embed_tables() (which can return None per-table if a
table serializes to nothing usable), Paper's authors, categories, and at
least one version are all REQUIRED fields (see core/entities.py) — every
valid Paper has enough to produce a non-empty metadata string. No
Optional in this file's return types, on purpose.
"""

from __future__ import annotations

from typing import Optional

from temporal_graphrag.core.entities import Paper
from temporal_graphrag.embeddings.text_embedder import TextEmbedder


def serialize_metadata(paper: Paper) -> str:
    """Flattens a Paper's structured metadata into clean, templated text
    for embedding through a general-purpose text encoder.

    Field order is deliberate: authors first (the strongest, most
    distinctive signal for "same research group"), then venue/year
    (publication context), then categories/fields_of_study (subfield).
    """
    parts: list[str] = []

    parts.append("Authors: " + ", ".join(author.name for author in paper.authors))

    if paper.venue and paper.venue.name:
        parts.append(f"Venue: {paper.venue.name}")

    year = (paper.publication_date or paper.first_version_date).year
    parts.append(f"Year: {year}")

    if paper.categories:
        parts.append("Categories: " + ", ".join(paper.categories))

    if paper.fields_of_study:
        parts.append("Fields of study: " + ", ".join(paper.fields_of_study))

    return "\n".join(parts)


class MetadataEmbedder:
    """Embeds a Paper's structured metadata (authors, venue, year,
    categories, fields of study) via text serialization + text_embedder.
    py's SPECTER2 pipeline.

    Accepts an existing TextEmbedder rather than always creating its own
    — same rationale as TableEmbedder: a caller processing a paper's
    title/abstract AND its metadata AND its tables should share ONE
    loaded model between all three, not load the same ~440MB of weights
    two or three times over.

    Usage:
        text_embedder = TextEmbedder()                        # load once
        metadata_embedder = MetadataEmbedder(text_embedder)     # reuses it
        vectors = metadata_embedder.embed_papers(papers)
        # vectors[i] corresponds to papers[i] — always the same length,
        # never None (see module docstring)
    """

    def __init__(self, text_embedder: Optional[TextEmbedder] = None) -> None:
        self.text_embedder = text_embedder or TextEmbedder()

    def embed_papers(self, papers: list[Paper]) -> list[list[float]]:
        """Embeds each paper's serialized metadata. Returns a list the
        same length as `papers`, in the same order."""
        if not papers:
            return []
        texts = [serialize_metadata(paper) for paper in papers]
        return self.text_embedder.embed_texts(texts)

    def embed_paper(self, paper: Paper) -> list[float]:
        """Single-paper convenience wrapper around embed_papers()."""
        return self.embed_papers([paper])[0]