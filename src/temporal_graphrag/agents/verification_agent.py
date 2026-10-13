"""
agents/verification_agent.py

Week 13's second agent node: checks the Reasoning Agent's output against
the actual retrieved evidence, matching the architecture diagram's
"7.5 Verification Agent (Consistency Check)" box — Fact Verification,
Check Contradictions, Cross-check Citations, Reduce Hallucination.

Scope: LLM-based verification, not a dedicated NLI model (DeBERTa)
--------------------------------------------------------------------------
The architecture diagram suggests DeBERTa-based NLI (natural language
inference — classifying a claim as entailed/contradicted/neutral given a
premise) for this agent. Adding a dedicated NLI model would mean a fifth
model download in this project's stack, purely for a task an LLM prompted
directly to compare claims against evidence can already do reasonably
well, with the added benefit of producing a human-readable explanation
of WHY something is unsupported, not just an entailment label. Same
"lightweight fallback over an exotic model dependency" reasoning
multimodal_agent.py already gives for substituting Gemini's native vision
for BLIP-2/TAPAS.

What "contradictions" actually captures here
--------------------------------------------------
state.py's schema has one list field for this (`contradictions`), so this
agent folds TWO related but distinct failure modes into it, each entry
prefixed to say which: "[Unsupported claim]" (something the Reasoning
Agent's synthesis asserted that the retrieved evidence doesn't actually
back up — the literal "Reduce Hallucination" bullet from the diagram) and
"[Contradiction]" (two evidence passages that disagree with each other,
independent of what the Reasoning Agent concluded). Both matter for
"should I trust this answer," which is what `verified` ultimately reports
— extending state.py's schema with a second field for this distinction
wasn't worth it for a difference a string prefix already conveys clearly.

Fallback philosophy: UNLIKE reasoning_agent.py's "preserve content on
parse failure," this agent fails CLOSED
--------------------------------------------------------------------------
reasoning_agent.py's parsing fallback preserves the LLM's raw output
rather than discarding real synthesis work — reasonable there, because a
formatting slip doesn't make the reasoning itself wrong. Here, a parsing
failure means verification genuinely didn't complete, and defaulting to
`verified=True` in that case would be actively dangerous — it would
silently present unverified content as if it had passed a check that
never actually ran. This agent always fails closed: any parsing problem
sets verified=False with a contradictions entry explaining verification
itself failed, so a caller checking `state["verified"]` before trusting
the summary can't be fooled by a format hiccup into treating unchecked
output as checked.
"""

from __future__ import annotations

import json
import re

from temporal_graphrag.agents.state import AgentState
from temporal_graphrag.llm.base import LLMMessage
from temporal_graphrag.llm.factory import llm_factory

__all__ = ["verification_node"]

_SYSTEM_PROMPT = """You are the verification component of a research-\
paper question-answering system. You're given an analysis someone wrote \
(the "reasoning") and the actual evidence passages it was supposedly \
based on. Your job:

1. Check every factual claim in the reasoning against the evidence — \
flag any claim the evidence does NOT actually support as an unsupported \
claim.
2. Check whether the evidence passages contradict each other on any \
point, independent of what the reasoning concluded.

Be strict: a claim needs to be genuinely stated or clearly implied by \
the evidence, not just plausible or generally true, to count as \
supported.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"unsupported_claims": ["...", "..."], "evidence_contradictions": ["...", "..."]}
Use empty lists if there are none of either.
"""


async def verification_node(state: AgentState) -> dict:
    """LangGraph node: reads state["reasoning"] and state["evidence"];
    returns a partial update with "verified" and "contradictions"
    populated."""
    reasoning = state.get("reasoning", "")
    evidence = state.get("evidence") or []

    if not reasoning:
        # Nothing to verify — not a failure, just nothing was produced
        # upstream for this agent to check.
        return {
            "verified": False,
            "contradictions": ["[Unsupported claim] No reasoning was produced to verify."],
            "messages": [{"role": "assistant", "content": "[Verification] skipped — no reasoning to check"}],
        }

    evidence_block = "\n".join(f"[{i}] ({ev.source_arxiv_id}) {ev.text}" for i, ev in enumerate(evidence, 1))
    user_message = f"Reasoning to verify:\n{reasoning}\n\nEvidence:\n{evidence_block or '(none)'}"

    provider = llm_factory.get_provider(
        [LLMMessage(role="system", content=_SYSTEM_PROMPT), LLMMessage(role="user", content=user_message)]
    )
    response_text = await provider.generate_text(user_message, system=_SYSTEM_PROMPT, temperature=0.1)

    verified, contradictions = _parse_verification(response_text)

    return {
        "verified": verified,
        "contradictions": contradictions,
        "messages": [
            {"role": "assistant", "content": f"[Verification] verified={verified}, {len(contradictions)} issue(s)"}
        ],
    }


def _parse_verification(response_text: str) -> tuple[bool, list[str]]:
    """Parses the LLM's JSON response into (verified, contradictions).
    Fails CLOSED on any parsing problem — see module docstring for why
    this is deliberately the opposite default from reasoning_agent.py's
    fallback."""
    try:
        json_text = _extract_json(response_text)
        parsed = json.loads(json_text)
        unsupported = [f"[Unsupported claim] {c}" for c in parsed.get("unsupported_claims", [])]
        contradicted = [f"[Contradiction] {c}" for c in parsed.get("evidence_contradictions", [])]
        contradictions = unsupported + contradicted
        return len(contradictions) == 0, contradictions
    except Exception:
        return False, ["[Verification failed] Could not parse the verification model's response."]


def _extract_json(text: str) -> str:
    """Same markdown-fence-stripping logic as planner_agent.py's
    _extract_json() — duplicated rather than imported, same reasoning
    used throughout this project's other small shared-shaped helpers."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text