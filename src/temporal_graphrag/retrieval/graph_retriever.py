"""
retrieval/graph_retriever.py

Week 11's second Hybrid GraphRAG piece: queries the Neo4j knowledge graph
built by 06_build_knowledge_graph.py for citation-based relatedness —
"what does this paper cite," "what cites this paper," and general k-hop
neighborhood search. Matches the architecture diagram's "6.2 Graph
Retrieval (Structure Search)" box.

Scope: citation-graph traversal, not yet Personalized PageRank
--------------------------------------------------------------------
The architecture diagram lists three graph-retrieval techniques: Graph
Traversal (BFS/DFS), Personalized PageRank, and k-hop Neighborhood
Search. This file builds the first and third. Personalized PageRank is
deliberately NOT here: it needs the Neo4j Graph Data Science (GDS)
plugin's `gds.pageRank` procedures, and nothing in this project's setup
(neo4j_client.py, schema.py, or your Neo4j Desktop screenshots) confirms
GDS is actually installed alongside Community Edition — assuming it and
writing a call that fails at runtime on a plugin nobody verified is
present would be worse than not building it yet. Plain Cypher path
traversal has no such dependency and covers real, useful relatedness
today. Add PageRank later as its own addition once GDS is confirmed
installed — not guessed at now (same "build what we can concretely
support" stance schema.py's own docstring already takes with Concept/
Method/Dataset nodes).

Async, unlike vector_retriever.py
----------------------------------------
Neo4jClient is async-only by its own design (see that file's docstring:
execute_read/execute_write, not sync alternatives) — every method here
follows suit. This means hybrid_scorer.py (Week 11's fusion step, next)
will be combining one sync retriever (VectorRetriever) with one async
retriever (this one), and will itself need to be async — either awaiting
this directly from an async context, or wrapping it with asyncio.run() at
whatever top-level entry point calls it. That's a direct, deliberate
consequence of Neo4jClient's own architecture, not something to route
around by making THIS file falsely synchronous.

Why "papers this cites" and "papers citing this" are always exactly
1 hop, but k_hop_neighbors() reports variable hop counts
---------------------------------------------------------------------
Per schema.py: a citation relationship is really Paper -[:CITES]->
Citation -[:RESOLVES_TO]-> Paper — TWO raw graph edges for what's
conceptually "one hop" between papers. citing_papers()/cited_by_papers()
fix that raw distance at exactly 2 edges (one conceptual hop).
k_hop_neighbors() generalizes to N conceptual hops, which means 2*N raw
edges — see _paper_hops_to_raw_range() below for exactly how that
conversion is applied, and why it has to be done in Python rather than
Cypher (see the note on query parameterization further down).

Why hop distance becomes a (0, 1] score, closest first
-----------------------------------------------------------
RetrievalHit.score is meant to be comparable, even if not identical,
across retrievers — vector_retriever.py's scores are cosine similarities
in (0, 1] where higher means "more relevant". Graph traversal has no
native notion of a similarity score, only a hop distance, so this file
maps hop distance N to a score of 1/N: a direct (1-hop) neighbor scores
1.0, a 2-hop neighbor scores 0.5, and so on — same direction (higher is
closer/more relevant), even though the underlying signal is completely
different. hybrid_scorer.py is where these get reconciled with actual
fusion weights; this file's job is just to not hand back something
backwards.

Missing papers return [], they don't raise
------------------------------------------------
Same stance as vector_store.py's search() (an empty result is a
legitimate answer, not an error) rather than document_store.py's
get_paper() (which raises NotFoundError). The difference: document_store
assumes you're looking up something that should exist in YOUR OWN
corpus. Here, "what does paper X cite" for an arxiv_id genuinely outside
the graph (or a citation that never resolved to an in-corpus Paper — see
schema.py's note that most citation targets aren't in this 40-paper
pilot) is an expected, non-exceptional outcome, not a bug.

On Cypher parameters and variable-length relationship ranges
-------------------------------------------------------------------
Cypher does not support parameterizing the hop-count bounds of a
variable-length relationship pattern (`*2..$n` is not valid Cypher —
only literal integers are allowed there). The raw hop range is computed
in Python from the validated `hops` argument and inlined into the query
string via an f-string before being sent — safe here specifically
because it's an internally-computed, bounds-checked integer, never
user-supplied text, unlike every actual data value in these queries
(arxiv_id, k), which stay proper `$parameters` throughout.
"""

from __future__ import annotations

from temporal_graphrag.knowledge_graph.neo4j_client import Neo4jClient
from temporal_graphrag.retrieval.vector_retriever import RetrievalHit

__all__ = ["GraphRetriever"]

# Traversing past this many conceptual (paper-to-paper) hops gets
# expensive fast in a citation graph (each hop can fan out to many
# papers) and stops being "related," practically speaking, well before
# that — a hard cap here is a guardrail against an accidental
# k_hop_neighbors(hops=50) call, not a tuned research parameter.
MAX_PAPER_HOPS = 4


class GraphRetriever:
    """Query-time wrapper around a Neo4jClient for citation-graph-based
    relatedness: direct citations in either direction, and general k-hop
    neighborhood search.

    Usage:
        async with Neo4jClient() as client:
            retriever = GraphRetriever(client)

            cites = await retriever.citing_papers("2003.08964", k=10)
            cited_by = await retriever.cited_by_papers("2003.08964", k=10)
            neighborhood = await retriever.k_hop_neighbors("2003.08964", hops=2, k=10)
    """

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    async def citing_papers(self, arxiv_id: str, k: int = 10) -> list[RetrievalHit]:
        """Papers `arxiv_id` cites (that resolved to a Paper actually in
        this corpus) — exactly one conceptual hop, via Paper-[:CITES]->
        Citation-[:RESOLVES_TO]->Paper."""
        rows = await self.client.run_read(
            """
            MATCH (:Paper {arxiv_id: $arxiv_id})-[:CITES]->(:Citation)-[:RESOLVES_TO]->(cited:Paper)
            RETURN DISTINCT cited.arxiv_id AS arxiv_id
            LIMIT $k
            """,
            arxiv_id=arxiv_id,
            k=k,
        )
        # All direct citations are equally "1 hop" — no finer-grained
        # relevance signal exists at this level, so every hit scores the
        # same (see module docstring's hop-distance-to-score mapping: 1/1 = 1.0).
        return [RetrievalHit(id=row["arxiv_id"], score=1.0, modality="graph") for row in rows]

    async def cited_by_papers(self, arxiv_id: str, k: int = 10) -> list[RetrievalHit]:
        """Papers that cite `arxiv_id` — the reverse direction of
        citing_papers(), same one-hop reasoning."""
        rows = await self.client.run_read(
            """
            MATCH (citing:Paper)-[:CITES]->(:Citation)-[:RESOLVES_TO]->(:Paper {arxiv_id: $arxiv_id})
            RETURN DISTINCT citing.arxiv_id AS arxiv_id
            LIMIT $k
            """,
            arxiv_id=arxiv_id,
            k=k,
        )
        return [RetrievalHit(id=row["arxiv_id"], score=1.0, modality="graph") for row in rows]

    async def k_hop_neighbors(self, arxiv_id: str, hops: int = 2, k: int = 10) -> list[RetrievalHit]:
        """Papers reachable from `arxiv_id` within `hops` conceptual
        citation-hops, in EITHER direction (cites or is cited by),
        ranked closest-first. `hops` is capped at MAX_PAPER_HOPS
        regardless of what's passed in — see that constant's own comment.
        """
        hops = max(1, min(hops, MAX_PAPER_HOPS))
        min_raw_hops, max_raw_hops = self._paper_hops_to_raw_range(hops)

        # min_raw_hops/max_raw_hops are inlined, not parameters — see
        # module docstring's note on why Cypher won't accept a parameter
        # there. arxiv_id and k remain real query parameters below.
        rows = await self.client.run_read(
            f"""
            MATCH (start:Paper {{arxiv_id: $arxiv_id}})
            MATCH path = (start)-[:CITES|RESOLVES_TO*{min_raw_hops}..{max_raw_hops}]-(neighbor:Paper)
            WHERE neighbor.arxiv_id <> $arxiv_id
            WITH neighbor, min(length(path)) AS raw_length
            RETURN neighbor.arxiv_id AS arxiv_id, raw_length
            ORDER BY raw_length ASC
            LIMIT $k
            """,
            arxiv_id=arxiv_id,
            k=k,
        )

        hits: list[RetrievalHit] = []
        for row in rows:
            paper_hops = row["raw_length"] // 2  # see module docstring: 2 raw edges = 1 paper hop
            hits.append(RetrievalHit(id=row["arxiv_id"], score=1.0 / paper_hops, modality="graph"))
        return hits

    @staticmethod
    def _paper_hops_to_raw_range(hops: int) -> tuple[int, int]:
        """Conceptual paper-hops -> raw graph-edge range. 1 paper hop is
        always exactly 2 raw edges (Paper->Citation->Paper), so N paper
        hops spans raw edges [2, 2*N] — the minimum is always 2 (there's
        no such thing as a 1-raw-edge path between two Paper nodes in
        this schema), not 2*hops for both bounds."""
        return 2, hops * 2