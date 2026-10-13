"""
retrieval/vector_retriever.py

Week 11's first Hybrid GraphRAG piece: wraps vector_store.py's raw FAISS
search with query embedding, so a caller passes a plain-text question
("what datasets does this survey compare?") instead of a pre-computed
vector. Matches the architecture diagram's "6.1 Vector Retrieval
(Semantic Search)" box — the first of the four retrieval strategies that
Week 11's hybrid_scorer.py will eventually fuse together.

Two embedding spaces, not one
----------------------------------
vector_index/ holds four modalities, but they only live in TWO distinct
vector spaces, and a query has to be embedded into whichever space it's
about to be compared against:
  - "text", "metadata", "table" -> all embedded via TextEmbedder
    (SPECTER2). This isn't a coincidence: metadata_embedder.py and
    table_embedder.py both explicitly reuse TextEmbedder's pipeline
    rather than loading separate models (see their own docstrings) —
    so a query string embedded through that same TextEmbedder lands in
    the identical space each of those three indexes was built in.
  - "figure" -> embedded via FigureEmbedder.embed_text_query(), which
    uses CLIP's TEXT encoder — a different model entirely, but one
    specifically trained so text and image embeddings share one space.
    Trying to search "figure" with a SPECTER2-embedded query (or
    vice versa) would silently return meaningless results: same
    vector shape, wrong space, no error from FAISS to catch it.
MODALITY_SPACES below is the one place that mapping lives, so adding a
fifth modality later means updating one dict, not hunting through every
method that touches modalities.

RetrievalHit lives here, not in core/entities.py
-------------------------------------------------------
core/entities.py's own docstring scopes it to the paper DOMAIN model —
what a paper factually contains. A search result (which id, what score,
from which modality) isn't part of a paper; it's a retrieval-layer
value object that exists only because a query happened. This is the
first retrieval/ file built, so RetrievalHit is defined here; graph_
retriever.py and temporal_retriever.py (built next) import it from here
rather than each defining their own incompatible version, and hybrid_
scorer.py fuses lists of exactly this type regardless of which
retriever produced them.

Plain @dataclass, not Pydantic
------------------------------------
Every entity in core/entities.py is Pydantic on purpose (validation,
JSON serialization — see that file's own docstring). RetrievalHit needs
neither: it's constructed internally, in a hot loop (every hit from
every search call), and never deserialized from external JSON. Frozen
because a hit's id/score/modality shouldn't be mutated after the fact —
if reranker.py wants to adjust a score, it should return a NEW RetrievalHit,
not mutate one that other code might already be holding a reference to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from temporal_graphrag.embeddings import FigureEmbedder, TextEmbedder
from temporal_graphrag.storage.vector_store import VectorStore

__all__ = ["RetrievalHit", "VectorRetriever"]

# Which embedding space each modality's vectors live in. "text" is used as
# the TextEmbedder(SPECTER2) marker for readability; "figure" is the only
# CLIP-space modality today. See module docstring for why this matters.
_SPECTER2_MODALITIES = {"text", "metadata", "table"}
_CLIP_MODALITIES = {"figure"}


@dataclass(frozen=True)
class RetrievalHit:
    """One search result: which entity, how similar it was to the query,
    and which modality/index it came from (needed downstream — e.g.
    hybrid_scorer.py may weight "figure" hits differently than "text"
    hits, and document_store.py's resolve() needs the id, not the
    modality, but callers often want both together)."""

    id: str
    score: float
    modality: str


class VectorRetriever:
    """Query-time wrapper around a loaded VectorStore: embeds a plain-text
    query into the right space for a given modality, searches that
    modality's FAISS index, and returns typed RetrievalHit results
    instead of raw (id, score) tuples.

    Usage:
        store = VectorStore.load("vector_index")
        retriever = VectorRetriever(store)

        hits = retriever.search("text", "temporal knowledge graphs", k=5)
        # -> [RetrievalHit(id="2003.08964", score=0.83, modality="text"), ...]

        all_hits = retriever.search_all("transformer architecture diagram", k=3)
        # -> {"text": [...], "metadata": [...], "table": [...], "figure": [...]}
        # (only for modalities actually present in the loaded store)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        text_embedder: Optional[TextEmbedder] = None,
        figure_embedder: Optional[FigureEmbedder] = None,
    ) -> None:
        self.store = vector_store
        # Lazily-shared, not eagerly loaded: a caller only searching "text"
        # shouldn't pay CLIP's load cost, and vice versa. Each embedder is
        # still loaded at most once per VectorRetriever instance, the
        # first time a search actually needs it — not once per call.
        self._text_embedder = text_embedder
        self._figure_embedder = figure_embedder

    @property
    def text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder()
        return self._text_embedder

    @property
    def figure_embedder(self) -> FigureEmbedder:
        if self._figure_embedder is None:
            self._figure_embedder = FigureEmbedder()
        return self._figure_embedder

    # -- Searching -----------------------------------------------------

    def search(self, modality: str, query: str, k: int = 5) -> list[RetrievalHit]:
        """Embeds `query` into whichever space `modality` lives in, then
        searches that modality's index. Raises ValueError for a modality
        this retriever doesn't know how to embed queries for — a
        programmer error (typo, or a fifth modality added to vector_
        store.py without updating MODALITY_SPACES above), not a normal
        "no results" case, so it's raised rather than returning []."""
        if modality in _SPECTER2_MODALITIES:
            query_vector = self.text_embedder.embed_texts([query])[0]
        elif modality in _CLIP_MODALITIES:
            query_vector = self.figure_embedder.embed_text_query(query)
        else:
            known = sorted(_SPECTER2_MODALITIES | _CLIP_MODALITIES)
            raise ValueError(f"Unknown modality '{modality}' — expected one of {known}.")

        raw_hits = self.store.search(modality, query_vector, k=k)
        return [RetrievalHit(id=hit_id, score=score, modality=modality) for hit_id, score in raw_hits]

    def search_all(self, query: str, k: int = 5) -> dict[str, list[RetrievalHit]]:
        """Searches every modality actually present in the loaded store
        (via store.modalities — not a hardcoded list, so this doesn't
        break if a modality happens to be empty for a small/edge-case
        corpus) and returns one RetrievalHit list per modality."""
        return {modality: self.search(modality, query, k=k) for modality in self.store.modalities}

    # -- Convenience per-modality wrappers -----------------------------
    # Thin, but worth having: callers doing "just give me the top papers"
    # shouldn't need to remember the string "text" is the right modality
    # name for that.

    def search_papers(self, query: str, k: int = 5) -> list[RetrievalHit]:
        """Semantic search over paper (title, abstract) content."""
        return self.search("text", query, k=k)

    def search_paper_metadata(self, query: str, k: int = 5) -> list[RetrievalHit]:
        """Search over authors/venue/year/categories — e.g. "recent work
        by this research group" rather than topical content."""
        return self.search("metadata", query, k=k)

    def search_tables(self, query: str, k: int = 5) -> list[RetrievalHit]:
        return self.search("table", query, k=k)

    def search_figures(self, query: str, k: int = 5) -> list[RetrievalHit]:
        """Cross-modal: a text query against image embeddings, e.g.
        "architecture diagram" matching a figure with no caption keyword
        overlap at all — see figure_embedder.py's own docstring."""
        return self.search("figure", query, k=k)