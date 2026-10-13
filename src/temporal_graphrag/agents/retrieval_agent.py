"""
agents/retrieval_agent.py

Week 12's second agent node: wires every retrieval/ component together
into one LangGraph node, matching the architecture diagram's
"7.2 Retrieval Agent (Hybrid GraphRAG)" box — Perform Hybrid Retrieval,
Fetch Relevant Chunks, Retrieve Graph Context, Return Evidence.

Pipeline this node runs, per call
--------------------------------------
1. Vector search per subtask (state["subtasks"], falling back to the
   whole user_query if the Planner produced none) — merged and deduped
   by id, keeping each candidate's BEST score across subtasks.
2. Graph expansion seeded from the top few vector hits — k_hop_neighbors
   from graph_retriever.py, not a fresh search from scratch (graph
   retrieval only knows how to expand FROM a paper, not search hitting
   text — see that file's own docstring).
3. Hybrid fusion (hybrid_scorer.py) combining both into one ranked list.
4. Cross-encoder reranking (reranker.py) against the ORIGINAL user_query
   (not per-subtask) for the final ordering — reranking should judge
   relevance to the actual question asked, not an intermediate subtask.
5. Resolve each surviving hit's text via DocumentStore, producing
   Evidence objects (state.py's own value type) for downstream agents.

Why per-subtask vector search, not just the raw user_query
-------------------------------------------------------------------
planner_agent.py already did the work of breaking a compound query (e.g.
"how has X evolved AND what do the benchmark tables show") into separate
subtasks. Searching only the original combined query risks one part of a
compound question dominating the embedding and starving evidence for the
other part; searching each subtask separately and merging gives better
coverage for exactly the kind of multi-part query the Planner exists to
decompose in the first place.

Why graph expansion is seeded from vector hits, not run independently
---------------------------------------------------------------------------
graph_retriever.py's k_hop_neighbors() needs a starting arxiv_id — there's
no text-query entry point into the citation graph. Seeding from the top
vector hits means graph expansion explores citation-neighborhood context
AROUND what's already semantically relevant, not a disconnected second
search.

Why this only searches "text" (paper-level), leaving figures/tables to
the Multimodal Understanding agent
---------------------------------------------------------------------------
vector_retriever.py can search figure/table modalities directly, but this
agent's job (per the diagram) is finding the right PAPERS via hybrid
GraphRAG signals — deciding what a specific figure or table actually
shows, once a paper is already known to be relevant, is squarely
multimodal_agent.py's job (built next). Splitting it this way keeps each
agent's responsibility matched to what the diagram actually assigns it,
rather than Retrieval trying to do Multimodal's job too.

Why this is a class holding long-lived resources, not a bare function
---------------------------------------------------------------------------
Unlike planner_agent.py (stateless — just calls an LLM), this node needs
a loaded VectorStore/DocumentStore (fast, but real I/O) and an open
Neo4jClient (a live network connection). neo4j_client.py's own __init__
already opens the driver immediately, not lazily in __aenter__ — so a
RetrievalAgent instance built once via create() and reused across every
query in a session avoids reopening a Neo4j connection (and reloading the
FAISS index and parsed corpus) on every single call. __call__ is what
actually gets registered as the LangGraph node — LangGraph supports a
bound method/callable object as a node, not just a plain function.
"""

from __future__ import annotations

from temporal_graphrag.agents.state import AgentState, Evidence
from temporal_graphrag.core.exceptions import ServiceUnavailableError
from temporal_graphrag.knowledge_graph.neo4j_client import Neo4jClient
from temporal_graphrag.retrieval.graph_retriever import GraphRetriever
from temporal_graphrag.retrieval.hybrid_scorer import HybridScorer
from temporal_graphrag.retrieval.reranker import Reranker, text_for_entity
from temporal_graphrag.retrieval.vector_retriever import RetrievalHit, VectorRetriever
from temporal_graphrag.storage.document_store import DocumentStore
from temporal_graphrag.storage.vector_store import VectorStore

__all__ = ["RetrievalAgent"]

GRAPH_SEED_COUNT = 3
GRAPH_EXPANSION_HOPS = 2
CANDIDATES_PER_STAGE = 15
FINAL_EVIDENCE_COUNT = 8


class RetrievalAgent:
    """Holds the long-lived resources (FAISS index, parsed corpus, open
    Neo4j connection) every retrieval call needs, and exposes __call__ as
    the actual LangGraph node function — see module docstring for why
    this is a class, not a bare function, unlike planner_agent.py.

    Usage:
        agent = await RetrievalAgent.create()
        update = await agent(state)
        ...
        await agent.aclose()   # once, when the whole app/session ends
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        hybrid_scorer: HybridScorer,
        reranker: Reranker,
        document_store: DocumentStore,
        neo4j_client: Neo4jClient,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.hybrid_scorer = hybrid_scorer
        self.reranker = reranker
        self.document_store = document_store
        self._neo4j_client = neo4j_client

    @classmethod
    async def create(
        cls,
        parsed_dir: str = "parsed",
        vector_index_dir: str = "vector_index",
    ) -> "RetrievalAgent":
        """Async factory, not __init__ directly: loading the FAISS index
        and parsed papers is fast but real I/O, and Neo4j connectivity
        should be checked once up front (verify_connectivity()) rather
        than surfacing as a confusing failure on the very first query."""
        document_store = DocumentStore.from_directory(parsed_dir)
        vector_store = VectorStore.load(vector_index_dir)
        vector_retriever = VectorRetriever(vector_store)

        neo4j_client = Neo4jClient()
        if not await neo4j_client.verify_connectivity():
            raise ServiceUnavailableError(
                "Neo4j",
                "not reachable — is the DBMS started in Neo4j Desktop? "
                "RetrievalAgent needs it for graph-based retrieval.",
            )
        graph_retriever = GraphRetriever(neo4j_client)

        hybrid_scorer = HybridScorer(document_store)
        reranker = Reranker(document_store)

        return cls(vector_retriever, graph_retriever, hybrid_scorer, reranker, document_store, neo4j_client)

    async def aclose(self) -> None:
        """Closes the Neo4j connection — call once, when the whole
        session/app is shutting down, not after every query (see module
        docstring on why this class holds a long-lived connection at
        all)."""
        await self._neo4j_client.__aexit__(None, None, None)

    async def __call__(self, state: AgentState) -> dict:
        """The actual LangGraph node. Reads state["user_query"] and
        state["subtasks"], returns a partial update with "evidence"
        populated."""
        user_query = state["user_query"]
        subtasks = state.get("subtasks") or [user_query]

        vector_hits = self._search_subtasks(subtasks)
        graph_hits = await self._expand_graph(vector_hits)
        fused = self.hybrid_scorer.fuse(vector_hits=vector_hits, graph_hits=graph_hits, k=CANDIDATES_PER_STAGE)
        reranked = self.reranker.rerank(user_query, fused, top_k=FINAL_EVIDENCE_COUNT)

        evidence = self._to_evidence(reranked)

        return {
            "evidence": evidence,
            "messages": [{"role": "assistant", "content": f"[Retrieval] {len(evidence)} evidence item(s)"}],
        }

    def _search_subtasks(self, subtasks: list[str]) -> list[RetrievalHit]:
        """Vector search per subtask, merged and deduped by id, keeping
        each id's BEST score across every subtask it matched — see module
        docstring on why per-subtask search beats one combined-query
        search for compound questions."""
        best_by_id: dict[str, RetrievalHit] = {}
        for subtask in subtasks:
            for hit in self.vector_retriever.search_papers(subtask, k=CANDIDATES_PER_STAGE):
                existing = best_by_id.get(hit.id)
                if existing is None or hit.score > existing.score:
                    best_by_id[hit.id] = hit
        return list(best_by_id.values())

    async def _expand_graph(self, vector_hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Graph expansion seeded from the top few vector hits — see
        module docstring on why graph retrieval can't run independently
        of a starting point. Gracefully returns [] if vector_hits is
        empty (nothing to seed from) rather than erroring."""
        if not vector_hits:
            return []

        top_hits = sorted(vector_hits, key=lambda h: h.score, reverse=True)[:GRAPH_SEED_COUNT]
        best_by_id: dict[str, RetrievalHit] = {}
        for seed in top_hits:
            neighbors = await self.graph_retriever.k_hop_neighbors(
                seed.id, hops=GRAPH_EXPANSION_HOPS, k=CANDIDATES_PER_STAGE
            )
            for hit in neighbors:
                existing = best_by_id.get(hit.id)
                if existing is None or hit.score > existing.score:
                    best_by_id[hit.id] = hit
        return list(best_by_id.values())

    def _to_evidence(self, hits: list[RetrievalHit]) -> list[Evidence]:
        """Resolves each surviving hit's text via DocumentStore ONCE here
        — see state.py's own docstring on why Evidence carries resolved
        text rather than making every downstream agent re-resolve it.
        Reuses reranker.py's text_for_entity() rather than reimplementing
        the same id-to-text mapping a third time."""
        evidence: list[Evidence] = []
        for hit in hits:
            try:
                entity = self.document_store.resolve(hit.id)
            except Exception:
                continue
            text = text_for_entity(entity)
            if not text:
                continue
            source_arxiv_id = hit.id.split("::", 1)[0]
            evidence.append(Evidence(hit=hit, text=text, source_arxiv_id=source_arxiv_id))
        return evidence