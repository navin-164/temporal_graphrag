"""
api/main.py

FastAPI app — the HTTP surface over agents/graph.py's pipeline.

Why the agent graph is built in a lifespan context manager, not at
import time or per-request
--------------------------------------------------------------------------
agents/graph.py's build_agent_graph() is explicitly a "call once per
session" function (opens a live Neo4j connection, loads the FAISS index)
— FastAPI's lifespan context manager is the standard place for exactly
this kind of "set up once at startup, tear down once at shutdown"
resource, matching build_agent_graph()'s own (graph, aclose) contract.
Building it at plain import time would run async setup code outside an
event loop; building it per-request would reopen a Neo4j connection and
reload the FAISS index on every single call — both wrong for the same
reason retrieval_agent.py's own docstring already gives.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from temporal_graphrag.agents.graph import build_agent_graph
from api.routers.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph, aclose = await build_agent_graph()
    app.state.agent_graph = graph
    try:
        yield
    finally:
        await aclose()


app = FastAPI(
    title="Temporal Hybrid GraphRAG API",
    description="Multi-agent, temporal-aware GraphRAG over a corpus of scientific papers.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(query_router, prefix="/api", tags=["query"])


@app.get("/health")
async def health() -> dict:
    """Plain liveness check — deliberately does NOT touch agent_graph, so
    it stays useful even if pipeline setup is slow or mid-startup."""
    return {"status": "ok"}