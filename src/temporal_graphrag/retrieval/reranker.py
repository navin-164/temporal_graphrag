"""
retrieval/reranker.py

Week 11's final retrieval piece: cross-encoder reranking over a SMALL
candidate list (typically hybrid_scorer.py's fused top-k), matching the
architecture diagram's "6.5 Reranking" box — Cross-Encoder Reranker /
BERT-based Reranking.

Bi-encoder vs. cross-encoder — why this is a separate, final stage
--------------------------------------------------------------------------
Everything upstream (vector_retriever.py, and metadata/table_embedder.py
before it) is bi-encoder style: query and candidate are each embedded
SEPARATELY, once, then compared by cosine similarity — fast enough to
search thousands of candidates, but the model never actually sees the
query and candidate together. A cross-encoder feeds (query, candidate
text) through one model TOGETHER as a single input, letting it attend
across both — measurably more accurate, but it means one full forward
pass PER CANDIDATE, with no way to precompute anything in advance the
way FAISS's index lets vector search do. That trade-off is exactly why
this only ever runs on a short list (tens of candidates, already
narrowed down by cheaper retrieval), never the whole corpus — reranking
IS the "spend more compute on fewer, already-promising candidates" stage
by design, not a replacement for vector/graph/hybrid retrieval.

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2
------------------------------------------------------------
No scientific-paper-specific cross-encoder exists that's as
well-established as SPECTER2 is for bi-encoder embeddings (text_embedder.
py's own docstring covers that gap for the bi-encoder side). MS
MARCO-trained cross-encoders are the standard general-purpose choice for
this role, and the MiniLM-L-6 variant is the small/fast end of that
family (~80MB) — consistent with this project's "lightweight fallbacks"
stance elsewhere. Swap in ms-marco-MiniLM-L-12-v2 (roughly 2x slower, a
bit more accurate) if quality matters more than latency for your use
case; nothing else in this file assumes L-6 specifically.

Why scores are passed through a sigmoid before becoming RetrievalHit.score
---------------------------------------------------------------------------
sentence-transformers' CrossEncoder.predict() for MS MARCO models returns
a raw logit (roughly -10 to +10, unbounded), not a bounded similarity
score the way cosine similarity is. Every other score in this project's
RetrievalHit convention is bounded and "higher is better" in a roughly
comparable range — squashing the logit through a sigmoid keeps that
convention intact (scores land in (0, 1)) without pretending the result
is a calibrated probability, which it isn't.

Text per candidate: reusing table_embedder.py's serialize_table()
for Table hits, and picking the obvious field for everything else
---------------------------------------------------------------------------
A RetrievalHit only has an id — reranking needs actual text, resolved via
DocumentStore.resolve() (built for exactly this "id -> real content"
purpose) and then mapped to whichever field represents that entity type's
text: Paper -> title + abstract, Section -> its paragraphs' text, Table
-> table_embedder.py's own serialize_table() (already solves "how do you
turn a table into text" once — no reason to solve it twice), etc.

Unresolvable hits are DROPPED, not scored, unlike hybrid_scorer.py's
0-default for missing signals
---------------------------------------------------------------------------
hybrid_scorer.py can meaningfully default a missing temporal/citation
signal to 0 because those are additive terms alongside others. Here,
text is the ENTIRE input to the model — there's no defensible score to
assign a candidate the model literally cannot see. Dropping it is the
only option that doesn't fabricate a number.
"""

from __future__ import annotations

import math
from typing import Optional

from temporal_graphrag.core.entities import Citation, Equation, Figure, Paper, Paragraph, Section, Table
from temporal_graphrag.embeddings.table_embedder import serialize_table
from temporal_graphrag.retrieval.vector_retriever import RetrievalHit
from temporal_graphrag.storage.document_store import DocumentStore

__all__ = ["Reranker", "text_for_entity"]

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def text_for_entity(entity: object) -> Optional[str]:
    """Maps a resolved entity (whatever DocumentStore.resolve() returned)
    to the text a cross-encoder should actually read. Returns None for an
    entity with genuinely no usable text (e.g. a Figure with no caption)
    — see module docstring on why that means the hit gets dropped, not
    scored with a fallback."""
    if isinstance(entity, Paper):
        return f"{entity.title}. {entity.abstract}"
    if isinstance(entity, Section):
        paragraph_text = " ".join(p.text for p in entity.paragraphs)
        return paragraph_text or entity.heading
    if isinstance(entity, Paragraph):
        return entity.text
    if isinstance(entity, Figure):
        return entity.caption
    if isinstance(entity, Table):
        return serialize_table(entity) or entity.caption
    if isinstance(entity, Equation):
        return entity.latex
    if isinstance(entity, Citation):
        return entity.raw_text or entity.cited_title
    return None


class Reranker:
    """Cross-encoder reranking over a short RetrievalHit list.

    Usage:
        document_store = DocumentStore.from_directory("parsed")
        reranker = Reranker(document_store)

        fused = hybrid_scorer.fuse(vector_hits=vector_hits, graph_hits=graph_hits, k=20)
        final = reranker.rerank("temporal knowledge graphs", fused, top_k=5)
    """

    def __init__(self, document_store: DocumentStore, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.document_store = document_store
        self.model_name = model_name
        self._model = None  # lazy — see the `model` property below

    @property
    def model(self):
        """Loaded on first actual use, not at construction — same lazy
        pattern VectorRetriever uses for its embedders, so building a
        Reranker doesn't cost a ~80MB model load until rerank() is
        actually called."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_k: Optional[int] = None,
    ) -> list[RetrievalHit]:
        """Rescoring pass over `hits` using the cross-encoder, replacing
        each survivor's score entirely (whatever score it arrived with —
        cosine similarity, hop-based, or hybrid-fused — is superseded,
        not blended) and returning them sorted by the new score,
        descending. Hits with no resolvable text are dropped — see
        module docstring. modality is set to "reranked" on every
        surviving hit, so downstream code can tell a cross-encoder score
        apart from whatever score it started as.
        """
        resolvable: list[tuple[RetrievalHit, str]] = []
        for hit in hits:
            try:
                entity = self.document_store.resolve(hit.id)
            except Exception:
                continue
            text = text_for_entity(entity)
            if text:
                resolvable.append((hit, text))

        if not resolvable:
            return []

        pairs = [(query, text) for _, text in resolvable]
        raw_scores = self.model.predict(pairs)

        reranked = [
            RetrievalHit(id=hit.id, score=_sigmoid(float(raw_score)), modality="reranked")
            for (hit, _), raw_score in zip(resolvable, raw_scores)
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked


def _sigmoid(x: float) -> float:
    """Maps an unbounded cross-encoder logit to (0, 1) — see module
    docstring's note on why this isn't the same as a calibrated
    probability, just a bounded, "higher is better" score consistent with
    every other RetrievalHit in this project."""
    return 1.0 / (1.0 + math.exp(-x))