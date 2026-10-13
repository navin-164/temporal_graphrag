"""
agents/trend_agent.py

Week 13's fourth agent node: analyzes how the retrieved evidence's
topic/methods have changed over time, matching the architecture
diagram's "7.7 Trend & Evolution Agent" box — Detect Research Trends,
Topic Evolution, Influence Analysis, Key Milestones.

Scope: LLM-based timeline analysis, not TGAT/BERTopic
--------------------------------------------------------------
The architecture diagram lists Temporal GNN (TGAT) and Topic Modeling
(BERTopic) for this agent. Both are trained models this project has no
reason to add just for a pilot corpus of a few dozen papers — BERTopic
in particular expects hundreds-to-thousands of documents to produce
meaningful topic clusters, well beyond this project's scale. Instead,
this agent builds an explicit, DATED timeline from the evidence
retrieval_agent.py already gathered (using temporal_retriever.py's own
paper_date() — same date-preference logic used throughout this project)
and asks the LLM to reason over that chronological structure directly.
Same "lightweight fallback over infrastructure this project's scale
doesn't justify" stance as reasoning_agent.py's own Graph-Reasoning
scope note.

Why this needs a DocumentStore, unlike reasoning/verification/
summarization
--------------------------------------------------------------------------
Evidence (state.py's own value type) carries resolved TEXT but not a
paper's publication date — those other three agents never needed dates,
only content. This agent's entire purpose is chronological analysis, so
it resolves each evidence item's owning Paper via DocumentStore (shared
with RetrievalAgent/MultimodalAgent, not reloaded — same DI pattern
multimodal_agent.py already established) specifically to get
paper_date().

Why "influence analysis" here means citation counts, not a real
influence-propagation model
--------------------------------------------------------------------------
A rigorous citation-influence model is its own research project. This
agent gives the LLM each evidence paper's citation_count (Semantic
Scholar's global count — the same field hybrid_scorer.py's Citation
Importance term already uses) alongside its date, letting the model
reason about "was this an influential milestone" from a real number, not
inventing an influence score from nothing.
"""

from __future__ import annotations

from temporal_graphrag.agents.state import AgentState
from temporal_graphrag.llm.base import LLMMessage
from temporal_graphrag.llm.factory import llm_factory
from temporal_graphrag.retrieval.temporal_retriever import paper_date
from temporal_graphrag.storage.document_store import DocumentStore

__all__ = ["TrendAgent"]

_SYSTEM_PROMPT = """You are the trend-and-evolution component of a \
research-paper question-answering system. You're given a user's \
question and a chronologically-ordered list of relevant papers, each \
with its publication date, (approximate) global citation count, and a \
snippet of its content.

Analyze how the research on this topic has evolved over time. Cover, as \
relevant to the question:
- The overall trend or trajectory (what changed from earliest to latest).
- Key milestone papers — ones that appear to mark a meaningful shift, \
using citation count as one signal of influence (not the only one).
- Whether the pace of work in this area appears to be accelerating.

Cite arxiv ids inline in parentheses for specific claims, same as the \
rest of this system. If the evidence doesn't span enough time to say \
anything meaningful about evolution (e.g. everything is from the same \
year), say so plainly rather than inventing a trend.

Respond with the analysis text only — no preamble, no JSON.
"""


class TrendAgent:
    """Holds a DocumentStore (shared with other agents, not reloaded) and
    exposes __call__ as the LangGraph node.

    Usage:
        agent = TrendAgent(document_store)
        update = await agent(state)
    """

    def __init__(self, document_store: DocumentStore) -> None:
        self.document_store = document_store

    async def __call__(self, state: AgentState) -> dict:
        user_query = state["user_query"]
        evidence = state.get("evidence") or []

        timeline = self._build_timeline(evidence)
        if not timeline:
            return {
                "trend_analysis": None,
                "messages": [{"role": "assistant", "content": "[Trend] skipped — no dated evidence available"}],
            }

        timeline_block = "\n".join(
            f"- {date} | {arxiv_id} | ~{citations if citations is not None else 'unknown'} citations | {text[:150]}"
            for date, arxiv_id, citations, text in timeline
        )
        user_message = f'Question: "{user_query}"\n\nPapers, oldest first:\n{timeline_block}'

        provider = llm_factory.get_provider(
            [LLMMessage(role="system", content=_SYSTEM_PROMPT), LLMMessage(role="user", content=user_message)]
        )
        trend_analysis = await provider.generate_text(user_message, system=_SYSTEM_PROMPT, temperature=0.3)

        return {
            "trend_analysis": trend_analysis.strip(),
            "messages": [{"role": "assistant", "content": "[Trend] analysis generated"}],
        }

    def _build_timeline(self, evidence: list) -> list[tuple]:
        """Resolves each evidence item's paper date + citation count,
        deduplicates by source paper (one timeline entry per paper, not
        per evidence snippet), and sorts oldest-first."""
        seen_papers: dict[str, tuple] = {}
        for ev in evidence:
            if ev.source_arxiv_id in seen_papers:
                continue
            try:
                paper = self.document_store.get_paper(ev.source_arxiv_id)
            except Exception:
                continue
            seen_papers[ev.source_arxiv_id] = (paper_date(paper), ev.source_arxiv_id, paper.citation_count, ev.text)

        return sorted(seen_papers.values(), key=lambda row: row[0])