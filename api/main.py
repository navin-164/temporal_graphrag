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

def _unpack_graph(built: object):
    if hasattr(built, "ainvoke") or hasattr(built, "invoke"):
        return built, None
    if isinstance(built, (tuple, list)) and len(built) >= 1:
        graph = built[0]
        cleanup = built[1] if len(built) > 1 and callable(built[1]) else None
        if hasattr(graph, "ainvoke") or hasattr(graph, "invoke"):
            return graph, cleanup
    raise TypeError(f"Unexpected return from build_agent_graph(): {type(built).__name__}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    built = await build_agent_graph()
    graph, cleanup = _unpack_graph(built)
    app.state.agent_graph = graph
    try:
        yield
    finally:
        if cleanup is not None:
            res = cleanup()
            if hasattr(res, "__await__"):
                await res


app = FastAPI(
    title="Temporal Hybrid GraphRAG API",
    description="Multi-agent, temporal-aware GraphRAG over a corpus of scientific papers.",
    version="0.1.0",
    lifespan=lifespan,
)

# After creating `app = FastAPI(...)` and lifespan:
app.include_router(query_router)
app.include_router(query_router, prefix="/api", tags=["query"])


@app.get("/health")
async def health() -> dict:
    """Plain liveness check — deliberately does NOT touch agent_graph, so
    it stays useful even if pipeline setup is slow or mid-startup."""
    return {"status": "ok"}