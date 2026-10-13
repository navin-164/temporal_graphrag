"""
api/schemas/query.py

Pydantic request/response models for the /query endpoint — the HTTP-
facing contract over agents/graph.py's pipeline.

Why a separate EvidenceSource, not the raw Evidence/RetrievalHit types
--------------------------------------------------------------------------
agents/state.py's Evidence and retrieval/vector_retriever.py's
RetrievalHit are both internal value objects (frozen dataclasses, not
Pydantic) built for the pipeline's own use — see their own docstrings for
why. An HTTP response needs a JSON-serializable, intentionally-thin
contract instead: just enough (arxiv_id, score) for a client to show
"which papers this answer drew on," not every internal field those
dataclasses carry.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The research question to answer.")


class EvidenceSource(BaseModel):
    arxiv_id: str
    score: float


class QueryResponse(BaseModel):
    summary: str
    confidence_score: Optional[float] = None
    verified: bool
    contradictions: list[str] = Field(default_factory=list)
    trend_analysis: Optional[str] = None
    sources: list[EvidenceSource] = Field(default_factory=list)