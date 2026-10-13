"""
agents/reasoning_agent.py

Week 13's first agent node: analyzes and synthesizes the evidence
retrieval_agent.py and multimodal_agent.py gathered, matching the
architecture diagram's "7.4 Reasoning Agent (Analysis & Synthesis)" box —
Analyze Retrieved Info, Identify Key Concepts, Discover Relationships,
Synthesize Insights.

Scope: LLM reasoning (Chain-of-Thought) only, not Graph Reasoning (GNN/GAT)
--------------------------------------------------------------------------
The architecture diagram lists two methods for this agent: LLM Reasoning
(CoT) and Graph Reasoning (GNN/GAT). This project has no trained GNN/GAT
— building or fine-tuning one is well outside a pilot project's scope,
and nothing in requirements.txt anticipates one (unlike langgraph/
sentence-transformers, which are actually installed). "Graph reasoning"
here instead means the graph SIGNAL is already baked into the evidence
this agent receives — hybrid_scorer.py already folded graph-hop-distance
into each evidence item's fused score before reranking, and
retrieval_agent.py's graph expansion already surfaced citation-
neighborhood papers as evidence candidates in the first place. This
agent's LLM reasoning pass operates over evidence that graph structure
already shaped, rather than running a second, separate graph-neural
computation — same "build what's concretely supportable, don't assume
infrastructure we don't have" stance as graph_retriever.py's own
Personalized PageRank deferral.

Why evidence AND multimodal_notes are combined into one context
--------------------------------------------------------------------
A figure's interpretation ("GPT-3's compute scales to 3,600 petaflop/s-
days") is exactly the kind of concrete detail that should inform
synthesis alongside the paper text it came from — treating multimodal
notes as a separate, unconnected pool the reasoning pass never sees would
throw away exactly the content multimodal_agent.py was built to surface.

Same structured-JSON-in-prompt pattern as planner_agent.py, different
fallback philosophy
--------------------------------------------------------------------------
Same reasoning as that file: llm/base.py's provider-agnostic interface is
text-only, so structured output means asking for JSON and parsing it, not
a function-calling schema. Here, though, a parsing failure falls back to
using the ENTIRE raw response as the reasoning text with an empty
key_concepts list, rather than discarding the LLM's actual synthesis work
— unlike the Planner's fallback (which safely defaults to "run
everything"), throwing away a real analysis because of one formatting
slip would lose actual content summarization_agent.py needs next.
"""

from __future__ import annotations

import json
import re

from temporal_graphrag.agents.state import AgentState
from temporal_graphrag.llm.base import LLMMessage
from temporal_graphrag.llm.factory import llm_factory

__all__ = ["reasoning_node"]

_SYSTEM_PROMPT = """You are the reasoning component of a research-paper \
question-answering system. You're given a user's question, evidence \
passages retrieved from a corpus of papers (each labeled with its \
source arxiv id), and notes on relevant figures/tables from those \
papers. Your job:

1. Analyze the evidence in relation to the question.
2. Identify the key concepts/methods/findings involved.
3. Discover relationships between different pieces of evidence — where \
they agree, extend each other, or disagree.
4. Synthesize these into a coherent analysis that directly addresses \
the question, citing source arxiv ids for specific claims.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"reasoning": "...", "key_concepts": ["...", "..."]}
"""


async def reasoning_node(state: AgentState) -> dict:
    """LangGraph node: reads state["user_query"], state["evidence"], and
    state["multimodal_notes"]; returns a partial update with "reasoning"
    and "key_concepts" populated."""
    user_query = state["user_query"]
    evidence = state.get("evidence") or []
    multimodal_notes = state.get("multimodal_notes") or []

    context = _build_context(evidence, multimodal_notes)
    user_message = f'Question: "{user_query}"\n\n{context}'

    provider = llm_factory.get_provider(
        [LLMMessage(role="system", content=_SYSTEM_PROMPT), LLMMessage(role="user", content=user_message)]
    )
    response_text = await provider.generate_text(user_message, system=_SYSTEM_PROMPT, temperature=0.3)

    reasoning, key_concepts = _parse_reasoning(response_text)

    return {
        "reasoning": reasoning,
        "key_concepts": key_concepts,
        "messages": [{"role": "assistant", "content": f"[Reasoning] {len(key_concepts)} key concept(s) identified"}],
    }


def _build_context(evidence: list, multimodal_notes: list[str]) -> str:
    """Formats evidence + multimodal notes into one labeled context
    block — see module docstring on why both are combined, not kept
    separate."""
    parts: list[str] = []

    if evidence:
        parts.append("Evidence passages:")
        for i, ev in enumerate(evidence, 1):
            parts.append(f"[{i}] ({ev.source_arxiv_id}) {ev.text}")

    if multimodal_notes:
        parts.append("\nFigure/table notes:")
        parts.extend(multimodal_notes)

    return "\n".join(parts) if parts else "(no evidence retrieved)"


def _parse_reasoning(response_text: str) -> tuple[str, list[str]]:
    """Parses the LLM's JSON response. Falls back to using the WHOLE raw
    response as the reasoning text on any parsing failure — see module
    docstring on why this differs from planner_agent.py's fallback
    philosophy (discarding real synthesis work would be worse here than
    the Planner's safe "run everything" default)."""
    try:
        json_text = _extract_json(response_text)
        parsed = json.loads(json_text)
        reasoning = str(parsed.get("reasoning", "")).strip()
        key_concepts = [str(c) for c in parsed.get("key_concepts", [])]
        if not reasoning:
            raise ValueError("empty reasoning")
        return reasoning, key_concepts
    except Exception:
        return response_text.strip(), []


def _extract_json(text: str) -> str:
    """Same markdown-fence-stripping logic as planner_agent.py's
    _extract_json() — small enough to duplicate rather than import, same
    reasoning this project uses elsewhere for similarly tiny, shared-
    shaped helpers (e.g. temporal_retriever.py/hybrid_scorer.py's
    duplicated decay-factor math)."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text