"""
agents/state.py

Shared state schema for the multi-agent pipeline (Weeks 12-14),
implemented as a LangGraph StateGraph state — the TypedDict every agent
node reads from and returns partial updates to, matching the
architecture diagram's 7.1-7.8 agent pipeline (Planner -> Retrieval ->
Multimodal -> Reasoning -> Verification -> Summarization -> Trend ->
Evaluation).

Why LangGraph, and why a TypedDict rather than a Pydantic model
--------------------------------------------------------------------
requirements.txt already settled this: langgraph + langchain-core (for
message primitives), not AutoGen or a hand-rolled orchestrator.
LangGraph's own convention is a TypedDict whose fields each graph node
returns a PARTIAL update to — LangGraph merges those updates into the
running state between node calls. A Pydantic model would add validation
overhead on every single node transition for state that's already been
validated at its original source (RetrievalHit from retrieval/,
LLMResponse from llm/) — redundant here, same reasoning
retrieval/vector_retriever.py's RetrievalHit itself gives for choosing a
dataclass over Pydantic for hot-path internal objects.

Why `messages` uses LangGraph's add_messages reducer
------------------------------------------------------------
Most fields below are "the last agent's output overwrites this slot" —
LangGraph's default merge behavior for a plain field. `messages` is
different: it's an append-only trace of what happened (useful for
debugging the Planner's ReAct-style reasoning specifically, per the
diagram's own "Method: ReAct / Plan-and-Solve" note on that agent) —
add_messages is LangGraph's built-in reducer for exactly that "append,
don't overwrite" behavior.

Evidence, not raw RetrievalHit, flows through the pipeline
------------------------------------------------------------------
retrieval/vector_retriever.py's RetrievalHit deliberately stays thin
(id, score, modality only) — every retrieval file's own docstring already
explains why. But Reasoning, Verification, and Summarization all need the
ACTUAL resolved text a hit points to, not just its id — re-resolving via
DocumentStore in every single downstream agent would mean repeating the
same lookup three times per piece of evidence. Evidence (below) pairs a
RetrievalHit with its resolved text ONCE, right after retrieval, so every
later agent just reads .text directly.

Scope note: this is a linear pipeline's state, not a general agent
framework
-------------------------------------------------------------------------
The 8 agents this state supports run in a largely fixed order (per the
diagram's own left-to-right arrows) — this file doesn't attempt to
support arbitrary agent-to-agent branching, cycles, or dynamic replanning
beyond what the Planner's subtask list already provides. That's a
deliberate scope match to what's actually being built (a pilot RAG
pipeline), not a limitation of LangGraph itself, which supports far more
complex graphs than this project currently needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

from temporal_graphrag.retrieval.vector_retriever import RetrievalHit

__all__ = ["AgentState", "Evidence", "create_initial_state"]


@dataclass(frozen=True)
class Evidence:
    """A retrieval hit paired with its resolved text — see module
    docstring for why this exists instead of passing raw RetrievalHit
    objects downstream. `source_arxiv_id` is always the owning paper
    (even for a figure/table/section-level hit), for citation display
    without re-deriving it from `hit.id` in every agent that wants to
    cite a source."""

    hit: RetrievalHit
    text: str
    source_arxiv_id: str


class AgentState(TypedDict, total=False):
    """The state object flowing through every node in agents/graph.py's
    StateGraph. `total=False` because a freshly-created state (see
    create_initial_state()) only has a few fields meaningfully populated
    at first — most get filled in as the pipeline progresses.
    """

    # -- Input --------------------------------------------------------
    user_query: str

    # -- 7.1 Planner Agent output --------------------------------------
    subtasks: list[str]
    required_agents: list[str]

    # -- 7.2 Retrieval Agent output --------------------------------------
    evidence: list[Evidence]

    # -- 7.3 Multimodal Understanding Agent output -----------------------
    multimodal_notes: list[str]

    # -- 7.4 Reasoning Agent output --------------------------------------
    reasoning: str
    key_concepts: list[str]

    # -- 7.5 Verification Agent output ------------------------------------
    verified: bool
    contradictions: list[str]

    # -- 7.6 Summarization Agent output -----------------------------------
    summary: str

    # -- 7.7 Trend & Evolution Agent output --------------------------------
    trend_analysis: Optional[str]

    # -- 7.8 Evaluation Agent output ----------------------------------------
    confidence_score: Optional[float]

    # -- Trace, append-only across every node (see module docstring) -------
    messages: Annotated[list, add_messages]


def create_initial_state(user_query: str) -> AgentState:
    """Builds a clean starting state for one pipeline run — every list/
    dict field starts genuinely empty rather than omitted, so downstream
    agent code can do `state["evidence"]` without a KeyError before the
    Retrieval Agent has actually run yet."""
    return AgentState(
        user_query=user_query,
        subtasks=[],
        required_agents=[],
        evidence=[],
        multimodal_notes=[],
        reasoning="",
        key_concepts=[],
        verified=False,
        contradictions=[],
        summary="",
        trend_analysis=None,
        confidence_score=None,
        messages=[],
    )