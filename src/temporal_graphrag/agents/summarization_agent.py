"""
agents/summarization_agent.py

Week 13's third agent node: generates the final answer to the user's
query, matching the architecture diagram's "7.6 Summarization Agent
(Generation)" box — Generate Summary, Section-wise Summary, Timeline
Summary, Use Evidence Links.

Scope: one coherent summary, not separate section-wise/timeline outputs
--------------------------------------------------------------------------
state.py's schema has a single `summary: str` field, not separate slots
for a section-wise breakdown and a timeline view. Rather than extending
the schema for a distinction the rest of the pipeline doesn't otherwise
need, this agent asks the LLM to naturally structure the summary (with
brief headers/paragraphs where the content calls for it) within that one
field — a timeline-style answer for an evolution question will naturally
read differently from a comparison-style answer for a "how does X relate
to Y" question, without needing a rigid separate output slot for each.
trend_agent.py (built after this one, per the diagram's own 7.6-then-7.7
ordering) is where a genuinely dedicated timeline/evolution view lives.

Why this doesn't assume state["trend_analysis"] is populated
--------------------------------------------------------------------
The diagram places Summarization (7.6) BEFORE Trend & Evolution (7.7) —
so trend_analysis genuinely isn't available yet when this agent runs.
This agent reads it defensively (falls back to None) rather than assuming
it's there, matching the actual pipeline order rather than an idealized
one.

Why Verification's findings actively CONSTRAIN the summary, not just
sit unused alongside it
--------------------------------------------------------------------------
verification_agent.py's whole purpose ("Reduce Hallucination") is wasted
if its output is computed but never actually changes what the user sees.
This agent's prompt explicitly lists any flagged unsupported claims/
contradictions and instructs the model to exclude or caveat them, rather
than summarizing the Reasoning Agent's output as if verification hadn't
run at all. If verification failed closed (verified=False with a
"[Verification failed]" entry — see verification_agent.py's own
docstring), that shows up as an instruction to be more conservative and
caveat that the answer is unverified, not silently ignored.

Why Evidence Links means literal (arxiv_id) citations in the output text
--------------------------------------------------------------------------
"Use Evidence Links" is interpreted literally here: the prompt requires
every substantive claim to carry its source arxiv_id inline, e.g.
"(2005.14165)" — reasoning_agent.py already established this citation
style, so the summary continues it rather than introducing a different
citation convention at the last step of the pipeline.
"""

from __future__ import annotations

from temporal_graphrag.agents.state import AgentState
from temporal_graphrag.llm.base import LLMMessage
from temporal_graphrag.llm.factory import llm_factory

__all__ = ["summarization_node"]

_SYSTEM_PROMPT = """You are the summarization component of a research-\
paper question-answering system. You're given a user's question, an \
analysis of the relevant evidence, notes on relevant figures/tables, and \
a verification report flagging any unsupported claims or contradictions \
found in the analysis.

Write a clear, well-organized answer to the user's question, using \
short paragraphs or brief headers if that helps organize a multi-part \
answer. Requirements:

- Base the answer on the analysis and evidence provided — do not \
introduce facts that aren't in them.
- Cite the source arxiv id inline for every substantive factual claim, \
in parentheses, e.g. "GPT-3 has 175 billion parameters (2005.14165)."
- If the verification report flags any unsupported claims or \
contradictions, DO NOT include those specific claims in your answer as \
stated facts — either omit them or, if they're central to the question, \
explicitly note that this point is uncertain/unverified.
- If verification itself failed to run, briefly note that the answer \
below has not been independently verified.
- Relevant figure/table notes should be woven in naturally where they \
support the answer, not appended as an afterthought.

Respond with the answer text only — no preamble like "Here is the \
summary," no JSON, just the answer itself.
"""


async def summarization_node(state: AgentState) -> dict:
    """LangGraph node: reads state["reasoning"], state["evidence"],
    state["multimodal_notes"], state["verified"], and
    state["contradictions"]; returns a partial update with "summary"
    populated."""
    user_query = state["user_query"]
    reasoning = state.get("reasoning", "")
    multimodal_notes = state.get("multimodal_notes") or []
    verified = state.get("verified", False)
    contradictions = state.get("contradictions") or []

    verification_block = _build_verification_block(verified, contradictions)
    notes_block = "\n".join(multimodal_notes) if multimodal_notes else "(none)"

    user_message = (
        f'Question: "{user_query}"\n\n'
        f"Analysis:\n{reasoning or '(no analysis was produced)'}\n\n"
        f"Figure/table notes:\n{notes_block}\n\n"
        f"Verification report:\n{verification_block}"
    )

    provider = llm_factory.get_provider(
        [LLMMessage(role="system", content=_SYSTEM_PROMPT), LLMMessage(role="user", content=user_message)]
    )
    summary = await provider.generate_text(user_message, system=_SYSTEM_PROMPT, temperature=0.4)

    return {
        "summary": summary.strip(),
        "messages": [{"role": "assistant", "content": "[Summarization] answer generated"}],
    }


def _build_verification_block(verified: bool, contradictions: list[str]) -> str:
    if verified and not contradictions:
        return "No issues found — the analysis is fully supported by the evidence."
    if not contradictions:
        return "Verification did not complete successfully; treat the analysis as unverified."
    return "Issues found (exclude or caveat these specific points):\n" + "\n".join(f"- {c}" for c in contradictions)