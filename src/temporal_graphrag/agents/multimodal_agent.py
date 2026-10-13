"""
agents/multimodal_agent.py

Week 12's third agent node: interprets figures and tables from the
papers retrieval_agent.py found relevant, matching the architecture
diagram's "7.3 Multimodal Understanding Agent" box — Interpret Figures,
Understand Tables, Parse Equations, Link with Text.

Why this re-fetches figures/tables per-paper, not from state["evidence"]
--------------------------------------------------------------------------
retrieval_agent.py's own docstring already explains the division of
labor: it searches paper-level text only, leaving "what does this
paper's figure/table actually show" to this agent. By the time evidence
reaches this node, hybrid_scorer.py and reranker.py have both already
overwritten each hit's modality ("hybrid", then "reranked") — so evidence
itself carries no reliable signal about which items were originally
figures vs tables vs paper text even if it had included them. This agent
instead looks up each relevant PAPER's own figures/tables directly via
DocumentStore, which is the actually-correct source for "what visual
content does this paper contain," not a reconstruction from hit ids.

Substituting Gemini's native vision for CLIP/BLIP-2, and skipping TAPAS
-------------------------------------------------------------------------
The architecture diagram lists CLIP/BLIP-2/LLaVA for figures and TAPAS
for tables. This project already uses CLIP for figure EMBEDDING
(figure_embedder.py) — but embedding and interpretation are different
jobs (see that file's own docstring: embeddings support similarity
search, not "describe what this shows"). Rather than adding a second
heavy vision-language model (BLIP-2/LLaVA) or a dedicated table model
(TAPAS) just for interpretation, this agent uses Gemini's already-
available native multimodal capability for figures, and its plain text
capability for tables (which are already fully serialized structured
data via table_embedder.py's serialize_table() — no vision needed to
read a table that's already text). Same "lightweight fallback over an
exotic model dependency" reasoning this project uses throughout (citation
resolution via LLM prompting instead of dedicated NER, Personalized
PageRank deferred rather than assumed).

Query-aware interpretation, not a generic pre-computed caption
--------------------------------------------------------------------------
core/entities.py's Figure.caption_understanding field is explicitly
scoped for a future generic, pre-computed-once caption ("Week 6,
BLIP-2"). This agent does something different and arguably better for a
QA system: it asks Gemini to interpret each figure/table SPECIFICALLY IN
THE CONTEXT of the current user query, computed fresh per query rather
than once at ingestion time. That means caption_understanding stays
unpopulated by this file (still a future TODO, not this agent's job) —
results go into state["multimodal_notes"] instead, transient per query
run, not persisted back to parsed/*.json.

Why equations aren't sent through the LLM here
-----------------------------------------------------
Equation.latex is already directly-readable text (see that field's own
docstring) — there's no "vision" or "understanding" gap to close the way
there is for a raster figure image or a table needing structure recovery.
Downstream agents (Reasoning, Summarization) can read equation LaTeX
directly from evidence text if relevant; spending an extra LLM call to
"interpret" already-textual content would add cost for no real benefit.

Caps on how much gets interpreted per query
--------------------------------------------------
Only the top PAPERS_TO_INSPECT evidence papers, and only
MAX_FIGURES_PER_PAPER/MAX_TABLES_PER_PAPER items from each, are actually
sent to Gemini — controlling both latency and Gemini's free-tier RPM
limit (core/exceptions.py's own ServiceRateLimitError docstring already
flags how tight that is), rather than interpreting every figure in every
retrieved paper regardless of how many that turns out to be.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from temporal_graphrag.agents.state import AgentState
from temporal_graphrag.core.entities import Figure, Table
from temporal_graphrag.embeddings.table_embedder import serialize_table
from temporal_graphrag.llm.gemini_provider import GeminiProvider
from temporal_graphrag.storage.document_store import DocumentStore

__all__ = ["MultimodalAgent"]

PAPERS_TO_INSPECT = 3
MAX_FIGURES_PER_PAPER = 2
MAX_TABLES_PER_PAPER = 2

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

_SYSTEM_PROMPT = (
    "You are the multimodal-understanding component of a research-paper "
    "assistant. Given a figure or table from a paper and a user's "
    "question, explain in 2-3 sentences what it shows and how (if at "
    "all) it's relevant to the question. Be concrete about numbers/"
    "trends when present. If it isn't relevant to the question, say so "
    "briefly rather than describing it at length."
)


class MultimodalAgent:
    """Holds a DocumentStore (shared with RetrievalAgent, not reloaded)
    and a GeminiProvider, and exposes __call__ as the LangGraph node.

    Usage:
        agent = MultimodalAgent(document_store, GeminiProvider())
        update = await agent(state)
    """

    def __init__(self, document_store: DocumentStore, gemini_provider: Optional[GeminiProvider] = None) -> None:
        self.document_store = document_store
        self.gemini_provider = gemini_provider or GeminiProvider()

    async def __call__(self, state: AgentState) -> dict:
        user_query = state["user_query"]
        evidence = state.get("evidence") or []

        seen_arxiv_ids: list[str] = []
        for ev in sorted(evidence, key=lambda e: e.hit.score, reverse=True):
            if ev.source_arxiv_id not in seen_arxiv_ids:
                seen_arxiv_ids.append(ev.source_arxiv_id)
            if len(seen_arxiv_ids) >= PAPERS_TO_INSPECT:
                break

        notes: list[str] = []
        for arxiv_id in seen_arxiv_ids:
            try:
                paper = self.document_store.get_paper(arxiv_id)
            except Exception:
                continue

            for figure in paper.figures[:MAX_FIGURES_PER_PAPER]:
                note = await self._interpret_figure(user_query, arxiv_id, figure)
                if note:
                    notes.append(note)

            for table in paper.tables[:MAX_TABLES_PER_PAPER]:
                note = await self._interpret_table(user_query, arxiv_id, table)
                if note:
                    notes.append(note)

        return {
            "multimodal_notes": notes,
            "messages": [{"role": "assistant", "content": f"[Multimodal] {len(notes)} note(s)"}],
        }

    async def _interpret_figure(self, user_query: str, arxiv_id: str, figure: Figure) -> Optional[str]:
        if not figure.image_path:
            return None
        path = Path(figure.image_path)
        if not path.exists():
            return None

        try:
            image_bytes = path.read_bytes()
        except Exception:
            return None

        mime_type = _MIME_TYPES.get(path.suffix.lower(), "image/png")
        prompt = f'User question: "{user_query}"\n\nCaption: {figure.caption or "(no caption)"}'

        try:
            interpretation = await self.gemini_provider.generate_with_image(
                prompt, image_bytes, mime_type, system=_SYSTEM_PROMPT
            )
        except Exception:
            return None

        return f"[{arxiv_id}, Figure] {interpretation}"

    async def _interpret_table(self, user_query: str, arxiv_id: str, table: Table) -> Optional[str]:
        serialized = serialize_table(table)
        if not serialized:
            return None

        prompt = f'User question: "{user_query}"\n\nCaption: {table.caption or "(no caption)"}\n\nTable:\n{serialized}'

        try:
            interpretation = await self.gemini_provider.generate_text(prompt, system=_SYSTEM_PROMPT, temperature=0.4)
        except Exception:
            return None

        return f"[{arxiv_id}, Table] {interpretation}"