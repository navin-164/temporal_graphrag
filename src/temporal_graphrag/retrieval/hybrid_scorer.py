"""
retrieval/hybrid_scorer.py

Week 11's fusion step: combines vector_retriever.py and graph_retriever.py
results into one ranked list, plus two signals no single retriever
produces on its own — recency and citation importance — matching the
architecture diagram's "6.4 Hybrid Scoring (Fusion)" box:
Score = alpha*Vector + beta*Graph + gamma*Temporal + delta*Citation Importance.

Why this takes RetrievalHit LISTS as input, not queries
--------------------------------------------------------------
GraphRetriever is async (Neo4jClient's own design); VectorRetriever is
sync. If this class orchestrated both retrievers itself, it would be
forced to be async too, purely to accommodate one dependency — a
consequence graph_retriever.py's own docstring predicted. Taking
already-fetched hit lists as plain arguments avoids that entirely:
HybridScorer.fuse() is pure, synchronous, and easily testable with hand-
built RetrievalHit lists, and whoever calls it (an agent, later) owns the
one line of `await` that graph retrieval actually needs.

Why there's no separate "temporal_hits" input
--------------------------------------------------
temporal_retriever.py's own docstring already establishes this: temporal
relevance isn't a THIRD independent way to DISCOVER candidates the way
vector similarity or graph structure are — there's no "temporal
embedding space" to search. It's a property of a candidate paper (how
old is it), computable for any candidate regardless of which retriever
found it. So here, "Temporal Score" is computed per-candidate directly
(reusing temporal_retriever.py's own decay math and paper_date()
preference order), not fed in as a fourth candidate-discovery source.

Citation Importance: a lightweight PageRank substitute, not PageRank
--------------------------------------------------------------------------
graph_retriever.py's own docstring explains why real Personalized
PageRank isn't built yet: it needs the Neo4j GDS plugin, unconfirmed as
installed. Citation Importance here is a deliberately simpler,
dependency-free stand-in for the same underlying idea ("some papers
matter more, independent of this specific query") using Paper.
citation_count (Semantic Scholar's GLOBAL citation count — not just
citations within this small pilot corpus, which graph_retriever.py's own
struggles with a near-empty RESOLVES_TO graph already showed is too
sparse a signal on its own).

Why citation_count is saturated (count / (count + K)), not min-max
normalized like vector/graph scores are
---------------------------------------------------------------------------
Global citation counts are extremely skewed — a landmark paper like BERT
can have tens of thousands, while most papers in a random sample have
single digits. Min-max normalizing THAT within a small candidate set
would let one outlier paper claim the entire [0,1] range and crush every
other candidate's citation score toward 0, regardless of whether they're
also reasonably well-cited. A saturating transform (count / (count + K),
same shape as term-frequency saturation in BM25) gives diminishing
returns instead: going from 0 to 50 citations matters a lot, going from
5,000 to 10,000 barely moves the score — which matches the actual
intuition ("is this paper established at all") better than a linear or
outlier-dominated scale.

Why vector/graph scores ARE min-max normalized per query, but temporal/
citation scores are NOT
---------------------------------------------------------------------------
Vector similarity and graph hop-distance are inherently QUERY-RELATIVE —
"0.93 cosine similarity" only means something relative to this query's
other candidates, so rescaling them to [0,1] within the current
candidate set is appropriate and standard. Temporal recency and citation
importance are QUERY-INDEPENDENT properties of a paper — a paper
published in 2017 is equally old no matter what other papers happen to
show up for a given query, and renormalizing it per-query would make the
same paper's temporal/citation score inconsistently drift depending on
what else was retrieved alongside it. Both are already naturally bounded
((0, 1] for decay, [0, 1) for citation saturation) without needing
per-query rescaling.

Unresolvable dates score 0 here, not "keep original score" like
temporal_retriever.py's apply_time_decay()
---------------------------------------------------------------------------
apply_time_decay() multiplicatively rescales an existing score, so
"can't verify the date" defaulting to "don't rescale it" is the safer
choice there (see that file's own docstring). Here, Temporal Score is one
of four ADDITIVE terms being summed from scratch — there's no existing
score to fall back to, and contributing exactly 0 to an unknown signal is
the more defensible neutral choice than guessing a nonzero value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from temporal_graphrag.retrieval.temporal_retriever import paper_date
from temporal_graphrag.retrieval.vector_retriever import RetrievalHit
from temporal_graphrag.storage.document_store import DocumentStore

__all__ = ["HybridScorer", "HybridWeights"]

# See module docstring's citation-saturation note: 50 means a paper needs
# roughly 50 citations to reach a citation score of 0.5. Not empirically
# tuned against a real evaluation set (evaluation/metrics.py doesn't
# exist yet — Week 15) — a reasonable starting point, not a final answer.
CITATION_SATURATION_K = 50


@dataclass(frozen=True)
class HybridWeights:
    """alpha/beta/gamma/delta from the architecture diagram's fusion
    formula. Defaults sum to 1.0 for interpretability (each weight reads
    as "share of the final score"), but nothing enforces that — pass your
    own if you want to emphasize one signal over another. Not empirically
    tuned, same caveat as CITATION_SATURATION_K above: reasonable
    starting defaults (vector similarity as the dominant signal, graph
    structure second, temporal/citation as smaller adjustments), pending
    real evaluation once Week 15's metrics exist.
    """

    alpha: float = 0.40  # vector (semantic similarity)
    beta: float = 0.30  # graph (citation-structure relatedness)
    gamma: float = 0.15  # temporal (recency)
    delta: float = 0.15  # citation importance (global authority)


class HybridScorer:
    """Fuses vector and graph retrieval results into one ranked list,
    adding temporal recency and citation importance as two more scoring
    terms computed directly from each candidate's Paper.

    Usage:
        document_store = DocumentStore.from_directory("parsed")
        scorer = HybridScorer(document_store)

        vector_hits = vector_retriever.search_papers(query, k=20)
        graph_hits = await graph_retriever.k_hop_neighbors(seed_arxiv_id, hops=2, k=20)

        fused = scorer.fuse(vector_hits=vector_hits, graph_hits=graph_hits, k=10)
    """

    def __init__(self, document_store: DocumentStore, weights: Optional[HybridWeights] = None) -> None:
        self.document_store = document_store
        self.weights = weights or HybridWeights()

    def fuse(
        self,
        vector_hits: list[RetrievalHit] = (),
        graph_hits: list[RetrievalHit] = (),
        half_life_days: float = 365,
        reference_date: Optional[date] = None,
        k: Optional[int] = None,
    ) -> list[RetrievalHit]:
        """Combines vector_hits and graph_hits into one fused, re-ranked
        list. A candidate found by only one of the two still gets
        scored — its missing signal simply contributes 0, not excluded
        entirely (see module docstring: graph_hits is routinely empty on
        a sparse citation graph, and that shouldn't zero out every
        candidate's fused score, just that one term of it).

        Returns at most `k` hits (or all of them, if `k` is None),
        sorted by fused score descending, with modality="hybrid" —
        distinct from "text"/"graph"/etc. so downstream code can tell a
        fused result apart from a single-source one.
        """
        vector_by_id = {hit.id: hit.score for hit in vector_hits}
        graph_by_id = {hit.id: hit.score for hit in graph_hits}

        candidate_ids = set(vector_by_id) | set(graph_by_id)
        if not candidate_ids:
            return []

        normalized_vector = _min_max_normalize(vector_by_id)
        normalized_graph = _min_max_normalize(graph_by_id)
        reference_date = reference_date or date.today()

        fused: list[RetrievalHit] = []
        for candidate_id in candidate_ids:
            vector_score = normalized_vector.get(candidate_id, 0.0)
            graph_score = normalized_graph.get(candidate_id, 0.0)
            temporal_score = self._temporal_score(candidate_id, half_life_days, reference_date)
            citation_score = self._citation_score(candidate_id)

            fused_score = (
                self.weights.alpha * vector_score
                + self.weights.beta * graph_score
                + self.weights.gamma * temporal_score
                + self.weights.delta * citation_score
            )
            fused.append(RetrievalHit(id=candidate_id, score=fused_score, modality="hybrid"))

        fused.sort(key=lambda hit: hit.score, reverse=True)
        return fused[:k] if k is not None else fused

    def _temporal_score(self, candidate_id: str, half_life_days: float, reference_date: date) -> float:
        """Exponential decay factor, same formula temporal_retriever.py's
        apply_time_decay() uses — duplicated here rather than imported as
        a shared "decay factor" helper because it's one line, and this
        file's docstring already explains why it's used differently here
        (additive term needing a 0 default) than there (multiplicative
        rescale needing an unchanged-score default)."""
        parent_arxiv_id = candidate_id.split("::", 1)[0]  # see temporal_retriever.py's identical convention
        try:
            paper = self.document_store.get_paper(parent_arxiv_id)
        except Exception:
            return 0.0
        age_days = max(0, (reference_date - paper_date(paper)).days)
        return 0.5 ** (age_days / half_life_days)

    def _citation_score(self, candidate_id: str) -> float:
        """Saturating transform of Paper.citation_count — see module
        docstring for why saturation, not min-max, and why K=50."""
        parent_arxiv_id = candidate_id.split("::", 1)[0]
        try:
            paper = self.document_store.get_paper(parent_arxiv_id)
        except Exception:
            return 0.0
        if paper.citation_count is None:
            return 0.0
        return paper.citation_count / (paper.citation_count + CITATION_SATURATION_K)


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Rescales a {id: score} dict to [0, 1]. Empty input returns {}
    (nothing to normalize). If every score is identical (including the
    single-item case), every id gets 1.0 rather than dividing by zero —
    "all equally good, by whatever this signal measured" is the correct
    reading of a flat distribution, not an error."""
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        return {id_: 1.0 for id_ in scores}
    return {id_: (score - lo) / (hi - lo) for id_, score in scores.items()}