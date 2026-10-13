"""
api/routers/query.py

The /query endpoint — the one HTTP entry point into agents/graph.py's
full pipeline (Planner -> Retrieval -> Multimodal -> Reasoning ->
Verification -> Summarization -> Trend -> Evaluation).

Why the compiled graph lives on app.state, not rebuilt per request
--------------------------------------------------------------------
agents/graph.py's own docstring is explicit: build_agent_graph() opens a
real Neo4j connection and loads the FAISS index, and is meant to be
called ONCE per app/session, not once per query. main.py's lifespan
context manager builds it once at startup and stores it on app.state;
this router just reads it from there per request, never constructing its
own graph.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from temporal_graphrag.agents.state import create_initial_state
from api.schemas.query import EvidenceSource, QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    graph = http_request.app.state.agent_graph
    state = create_initial_state(request.query)
    result = await graph.ainvoke(state)

    sources = [
        EvidenceSource(arxiv_id=ev.source_arxiv_id, score=ev.hit.score) for ev in (result.get("evidence") or [])
    ]

    return QueryResponse(
        summary=result.get("summary", ""),
        confidence_score=result.get("confidence_score"),
        verified=result.get("verified", False),
        contradictions=result.get("contradictions") or [],
        trend_analysis=result.get("trend_analysis"),
        sources=sources,
    )