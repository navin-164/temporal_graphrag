"""
agents/evaluation_agent.py

Week 14's final individual agent node: computes a confidence score for
the generated summary, matching the architecture diagram's
"7.8 Evaluation Agent (Scoring)" box — Evaluate Summary, Compute Metrics,
Provide Confidence Score.

Scope: why ROUGE/BERTScore/BLEURT/FactScore do NOT belong here
--------------------------------------------------------------------------
The architecture diagram lists these four metrics for this agent. All
four (FactScore partially excepted) fundamentally need a REFERENCE
summary to compare against — ROUGE and BLEURT are reference-based
similarity metrics, BERTScore compares generated text to a reference
embedding-wise. A live user query has no gold-standard reference summary
to compare against; those metrics belong to Week 15's evaluation/
metrics.py, which will run them OFFLINE against a labeled benchmark
dataset with known correct answers — a fundamentally different task from
scoring one live, unlabeled query. FactScore IS reference-free (it
decomposes a summary into atomic facts and checks each against a
knowledge source) — but verification_agent.py already does exactly this
job (checking claims against retrieved evidence), so this agent doesn't
duplicate that check, it USES its result.

Why this is a deterministic formula, not another LLM call asking "how
confident are you"
--------------------------------------------------------------------------
LLMs are well-documented to be poorly calibrated when asked to self-
report confidence — a model can be equally "confident-sounding" whether
it's right or fabricating. This agent instead computes confidence from
concrete signals already produced upstream: whether verification passed
(and how many issues it found, from verification_agent.py) and how
strong the underlying evidence was (from the reranked scores retrieval_
agent.py already computed). A deterministic formula over real signals is
more interpretable and reproducible than an LLM's self-assessment, and
costs no extra API call.

Why verification dominates the score (60/40 weighting)
------------------------------------------------------------
A well-retrieved but unverified summary is a bigger risk than a
weakly-retrieved but fully-verified one — an unsupported or contradicted
claim is a concrete, specific problem with the actual output the user
will read, whereas a modest evidence-quality score just means the
retrieved passages were a so-so match, not that the summary is wrong.
Weighting verification higher reflects that asymmetry.
"""

from __future__ import annotations

from temporal_graphrag.agents.state import AgentState

__all__ = ["evaluation_node"]

# See module docstring: verification issues are a bigger red flag than
# so-so evidence quality, hence the 60/40 split rather than an even one.
VERIFICATION_WEIGHT = 0.6
EVIDENCE_WEIGHT = 0.4

# A single flagged issue meaningfully hurts confidence, but one issue in
# an otherwise-clean answer shouldn't crater it to near-zero — this rate
# and floor keep the penalty graduated rather than all-or-nothing.
PENALTY_PER_ISSUE = 0.25
MIN_VERIFICATION_SCORE = 0.1


async def evaluation_node(state: AgentState) -> dict:
    """LangGraph node: reads state["verified"], state["contradictions"],
    and state["evidence"]; returns a partial update with
    "confidence_score" populated. No LLM call — see module docstring."""
    verified = state.get("verified", False)
    contradictions = state.get("contradictions") or []
    evidence = state.get("evidence") or []

    verification_score = _verification_score(verified, contradictions)
    evidence_score = _evidence_score(evidence)

    confidence_score = round(VERIFICATION_WEIGHT * verification_score + EVIDENCE_WEIGHT * evidence_score, 3)

    return {
        "confidence_score": confidence_score,
        "messages": [{"role": "assistant", "content": f"[Evaluation] confidence_score={confidence_score}"}],
    }


def _verification_score(verified: bool, contradictions: list[str]) -> float:
    if verified and not contradictions:
        return 1.0
    return max(MIN_VERIFICATION_SCORE, 1.0 - PENALTY_PER_ISSUE * len(contradictions))


def _evidence_score(evidence: list) -> float:
    """Average of the evidence's own (already-scored) relevance —
    reranker.py's sigmoid-squashed cross-encoder scores if reranking ran,
    or whatever score the evidence otherwise carries. No evidence at all
    means no basis for confidence, so this returns 0.0, not some assumed
    middle value."""
    if not evidence:
        return 0.0
    return sum(ev.hit.score for ev in evidence) / len(evidence)