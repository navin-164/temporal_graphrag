"""
agents/graph.py

Week 14's final piece: the actual LangGraph StateGraph wiring every agent
node together into one pipeline, matching the architecture diagram's
full 7.1-7.8 agent sequence (Planner -> Retrieval -> Multimodal ->
Reasoning -> Verification -> Summarization -> Trend -> Evaluation).

Which agents are conditionally skippable, and why only these two
--------------------------------------------------------------------------
state.py's own docstring already committed to this exact scope: "a
bounded, single-hop routing decision... e.g. a query with no figures/
tables... no reason to run Multimodal... no temporal angle... no reason
to run Trend." Those are the two — and only the two — agents this graph
conditionally skips based on planner_agent.py's required_agents. Every
other agent (Retrieval, Reasoning, Verification, Summarization,
Evaluation) always runs: they form the backbone that actually produces a
trustworthy final answer, and skipping any of them would mean either no
answer at all (no Summarization) or an unchecked one (no Verification) —
outcomes worse than the modest cost of running an agent whose output
might end up mattering less for a given query. planner_agent.py's own
_parse_plan() already enforces "retrieval" can never be excluded, for
the same reason.

Why each agent needing a DocumentStore loads its own, rather than
sharing one instance across RetrievalAgent/MultimodalAgent/TrendAgent
--------------------------------------------------------------------------
document_store.py's own docstring already makes this exact case for why
it's not a database: "loading every JSON file into memory once at
startup is simpler... than [added complexity], and there is no benefit
to querying disk repeatedly for data that fits comfortably in memory."
The same reasoning extends here — three DocumentStore.from_directory()
calls for a pilot-sized corpus cost a small fraction of a second
combined, cheaper than threading one shared instance through three
different constructors just to avoid it. A production system at a much
larger corpus size might reconsider; this pilot's scale doesn't need to.

Why build_agent_graph() also hands back a way to close resources
--------------------------------------------------------------------------
retrieval_agent.py's own docstring explains why RetrievalAgent holds a
long-lived Neo4j connection rather than reopening one per call — but that
means SOMETHING has to close it when the app/session actually ends. This
function returns (compiled_graph, aclose) rather than just the graph, so
a caller can do `await aclose()` on shutdown without needing to reach
into the compiled graph's internals to find the one node that happens to
hold a network connection.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from temporal_graphrag.agents.evaluation_agent import evaluation_node
from temporal_graphrag.agents.multimodal_agent import MultimodalAgent
from temporal_graphrag.agents.planner_agent import planner_node
from temporal_graphrag.agents.reasoning_agent import reasoning_node
from temporal_graphrag.agents.retrieval_agent import RetrievalAgent
from temporal_graphrag.agents.state import AgentState
from temporal_graphrag.agents.summarization_agent import summarization_node
from temporal_graphrag.agents.trend_agent import TrendAgent
from temporal_graphrag.agents.verification_agent import verification_node
from temporal_graphrag.storage.document_store import DocumentStore

__all__ = ["build_agent_graph"]


def _route_after_retrieval(state: AgentState) -> str:
    """Routes to "multimodal" only if the Planner selected it — see
    module docstring on why this is one of only two conditional routing
    points in the whole graph."""
    required_agents = state.get("required_agents") or []
    return "multimodal" if "multimodal" in required_agents else "reasoning"


def _route_after_summarization(state: AgentState) -> str:
    """Routes to "trend" only if the Planner selected it — the other of
    the two conditional routing points."""
    required_agents = state.get("required_agents") or []
    return "trend" if "trend" in required_agents else "evaluation"


async def build_agent_graph(
    parsed_dir: str = "parsed",
    vector_index_dir: str = "vector_index",
) -> tuple[object, Callable[[], Awaitable[None]]]:
    """Builds every agent that needs long-lived resources, wires the full
    StateGraph together, and returns (compiled_graph, aclose).

    Call this ONCE per app/session — not once per query, since
    RetrievalAgent.create() opens a real Neo4j connection and loads the
    FAISS index (see retrieval_agent.py's own docstring on why that's
    meant to be reused, not rebuilt per call).

    Usage:
        graph, aclose = await build_agent_graph()
        try:
            result = await graph.ainvoke(create_initial_state("your question"))
            print(result["summary"])
        finally:
            await aclose()
    """
    retrieval_agent = await RetrievalAgent.create(parsed_dir=parsed_dir, vector_index_dir=vector_index_dir)
    multimodal_agent = MultimodalAgent(DocumentStore.from_directory(parsed_dir))
    trend_agent = TrendAgent(DocumentStore.from_directory(parsed_dir))

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("multimodal", multimodal_agent)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("verification", verification_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("trend", trend_agent)
    graph.add_node("evaluation", evaluation_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieval")
    graph.add_conditional_edges(
        "retrieval", _route_after_retrieval, {"multimodal": "multimodal", "reasoning": "reasoning"}
    )
    graph.add_edge("multimodal", "reasoning")
    graph.add_edge("reasoning", "verification")
    graph.add_edge("verification", "summarization")
    graph.add_conditional_edges(
        "summarization", _route_after_summarization, {"trend": "trend", "evaluation": "evaluation"}
    )
    graph.add_edge("trend", "evaluation")
    graph.add_edge("evaluation", END)

    return graph.compile(), retrieval_agent.aclose