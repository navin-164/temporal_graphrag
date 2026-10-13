"""
retrieval/temporal_retriever.py

Week 11's third Hybrid GraphRAG piece: time-based filtering and decay
over an existing list of RetrievalHit results (from vector_retriever.py
or graph_retriever.py), matching the architecture diagram's "6.3 Temporal
Retrieval" box — Before/After/Between filtering, time decay weighting,
and Latest First / Importance + Recency ranking.

Why this operates on RetrievalHit lists, not as a fifth "search" method
------------------------------------------------------------------------
vector_retriever.py and graph_retriever.py both answer "what's relevant
to this query" from their own signal (semantic similarity, citation
structure). Temporal relevance isn't a THIRD independent signal in that
same sense — there's no "temporal embedding space" to search. It's a
lens applied to results another retriever already found: filter them to
a date range, or reweight them by recency. That's why every method here
takes hits in and returns hits out, rather than taking a query string —
hybrid_scorer.py (next) is where a query's actual temporal intent
("recent work on X") gets turned into a call to one of these methods
alongside vector/graph retrieval, not a fourth thing this file does on
its own.

Why this needs a DocumentStore
------------------------------------
A RetrievalHit only carries (id, score, modality) — vector_store.py and
graph_retriever.py were both deliberately kept that thin. Every date
comparison here needs the actual Paper a hit belongs to, which means
resolving id -> Paper -> date, exactly what document_store.py exists
for.

Which date, when a hit isn't itself a Paper id
-------------------------------------------------
A "figure" or "table" modality hit's id is a composite
(paper_id::figure::N), not a paper id — figures don't have their own
publication date, they inherit their parent paper's. Same id-splitting
logic document_store.py's own module docstring already establishes (a
composite id always starts with the owning paper's arxiv_id followed by
"::") is reused here directly, rather than importing document_store's
private _parent_arxiv_id() — a one-line split is cheap enough to repeat
that borrowing a private method isn't worth the coupling (same
"duplication beats an import" call this project makes elsewhere, e.g.
metadata_embedder.py's serialize_metadata vs table_embedder.py's
serialize_table).

Which date field: publication_date, or first_version_date
-------------------------------------------------------------
Same choice, same reasoning, metadata_embedder.py already made:
prefer `publication_date` (Semantic Scholar's formal date) when
available, fall back to `first_version_date` (the arXiv v1 submission
date — always present, per core/entities.py's own guarantee that a Paper
always has at least one version). Kept consistent with that file rather
than inventing a second convention for "which date represents a paper."

Missing/unresolvable dates are dropped, not fatal
--------------------------------------------------------
A hit whose parent paper can't be found in the DocumentStore (shouldn't
normally happen — hits only ever come from retrievers built against
the same corpus — but isn't assumed impossible either) is silently
excluded from filter/decay results rather than raising. A single
unresolvable hit shouldn't crash filtering the other 20.

apply_time_decay() returns NEW RetrievalHit objects, re-sorted
--------------------------------------------------------------------
RetrievalHit is frozen (see vector_retriever.py's own docstring for why:
scores shouldn't be mutated in place once other code may be holding a
reference). Decay-weighting a score means constructing a new hit with
the adjusted score, not editing the old one — and since the whole point
of decay-weighting is to change the ranking, results come back re-sorted
by the NEW score, not in whatever order they arrived in.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from typing import Callable, Optional

from temporal_graphrag.core.entities import Paper
from temporal_graphrag.retrieval.vector_retriever import RetrievalHit
from temporal_graphrag.storage.document_store import DocumentStore

__all__ = ["TemporalRetriever", "paper_date"]


def paper_date(paper: Paper) -> date:
    """The single date this project treats as "when a paper happened" —
    S2's publication_date when available, else the arXiv v1 submission
    date. Same preference order as metadata_embedder.py's
    serialize_metadata(), kept consistent rather than reinvented here."""
    return paper.publication_date or paper.first_version_date.date()


class TemporalRetriever:
    """Applies time-based filtering and decay to RetrievalHit lists
    already produced by another retriever.

    Usage:
        document_store = DocumentStore.from_directory("parsed")
        temporal = TemporalRetriever(document_store)

        hits = vector_retriever.search_papers("temporal knowledge graphs", k=20)

        recent_only = temporal.filter_after(hits, date(2023, 1, 1))
        decayed = temporal.apply_time_decay(hits, half_life_days=365)
        newest_first = temporal.rank_latest_first(hits)
    """

    def __init__(self, document_store: DocumentStore) -> None:
        self.document_store = document_store

    def _date_for_hit(self, hit: RetrievalHit) -> Optional[date]:
        """Resolves a hit back to its owning paper's date. Returns None
        if that paper can't be found — see module docstring on why this
        is dropped rather than raised."""
        parent_arxiv_id = hit.id.split("::", 1)[0]
        try:
            paper = self.document_store.get_paper(parent_arxiv_id)
        except Exception:
            return None
        return paper_date(paper)

    # -- Before / After / Between filtering -------------------------------

    def filter_before(self, hits: list[RetrievalHit], cutoff: date) -> list[RetrievalHit]:
        """Keeps hits whose paper date is strictly before `cutoff`."""
        return self._filter(hits, lambda d: d < cutoff)

    def filter_after(self, hits: list[RetrievalHit], cutoff: date) -> list[RetrievalHit]:
        """Keeps hits whose paper date is strictly after `cutoff`."""
        return self._filter(hits, lambda d: d > cutoff)

    def filter_between(self, hits: list[RetrievalHit], start: date, end: date) -> list[RetrievalHit]:
        """Keeps hits whose paper date falls in [start, end], inclusive
        on both ends."""
        return self._filter(hits, lambda d: start <= d <= end)

    def _filter(self, hits: list[RetrievalHit], predicate: Callable[[date], bool]) -> list[RetrievalHit]:
        kept = []
        for hit in hits:
            hit_date = self._date_for_hit(hit)
            if hit_date is not None and predicate(hit_date):
                kept.append(hit)
        return kept

    # -- Decay weighting and recency ranking -------------------------------

    def apply_time_decay(
        self,
        hits: list[RetrievalHit],
        half_life_days: float = 365,
        reference_date: Optional[date] = None,
    ) -> list[RetrievalHit]:
        """Reweights each hit's score by exponential decay based on the
        paper's age relative to `reference_date` (defaults to today —
        "how relevant is this paper RIGHT NOW", the usual framing for a
        live query, not relative to the corpus's own newest paper).
        A paper exactly `half_life_days` old has its score cut in half; a
        paper twice that old, cut to a quarter; and so on. Returns a NEW
        list, re-sorted by the decayed score, descending — this is
        "Importance + Recency" from the architecture diagram: the
        original score already captures importance (semantic similarity,
        hop distance, ...), and decay folds recency into the same number
        rather than needing a second, separately-tuned recency score.

        Hits whose date can't be resolved keep their original score
        un-decayed (rather than being dropped, unlike the filter methods
        above) — an unknown date isn't evidence the hit is old, and
        dropping potentially-relevant results from a ranked list is a
        worse failure mode here than a filter silently excluding one.
        """
        reference_date = reference_date or date.today()
        decayed: list[RetrievalHit] = []

        for hit in hits:
            hit_date = self._date_for_hit(hit)
            if hit_date is None:
                decayed.append(hit)
                continue
            age_days = max(0, (reference_date - hit_date).days)
            decay_factor = 0.5 ** (age_days / half_life_days)
            decayed.append(replace(hit, score=hit.score * decay_factor))

        return sorted(decayed, key=lambda h: h.score, reverse=True)

    def rank_latest_first(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Pure recency ranking, ignoring the original score entirely —
        "Latest First" from the architecture diagram, for queries where
        recency alone is what's being asked for ("what's the newest work
        on X"), not a blend with relevance. Hits whose date can't be
        resolved sort last (oldest possible date used as a genuinely
        arbitrary but self-consistent fallback), rather than being
        dropped — a "latest first" list should still show everything
        available, just least-confidently placed."""

        def sort_key(hit: RetrievalHit) -> date:
            return self._date_for_hit(hit) or date.min

        return sorted(hits, key=sort_key, reverse=True)