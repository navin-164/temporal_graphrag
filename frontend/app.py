"""
frontend/app.py

Week 14's Streamlit dashboard — the interactive UI in front of
api/main.py's FastAPI backend, matching the architecture diagram's
Output Layer boxes: A. Summaries, B. Knowledge Graph Visualization,
C./D. Temporal Evolution Timeline & Research Trend Analysis,
E. Explainable Summary (evidence links, confidence, verification status).

Why this calls the FastAPI backend over HTTP, not agents/graph.py directly
--------------------------------------------------------------------------
Streamlit and FastAPI are two separate, independently-run processes
(`streamlit run frontend/app.py` and `uvicorn api.main:app` are two
different commands, two different ports) — same separation of concerns
api/main.py's own docstring establishes: the agent graph's expensive
startup (Neo4j connection, FAISS index) happens ONCE, in the FastAPI
process, not duplicated in a second process every time someone opens the
Streamlit page. This file is a thin client, not a second copy of the
pipeline.

Why httpx's sync Client, not requests or an async client
--------------------------------------------------------------------
requirements.txt already includes httpx (not requests) for "general
external requests." Streamlit runs a script top-to-bottom synchronously
on every interaction — there's no running event loop to hand an
AsyncClient's coroutines to without extra plumbing, so the plain sync
httpx.Client() is the natural fit here, not a mismatch with the rest of
this project's async-first backend code (which lives in a completely
separate process anyway).

Why the knowledge-graph visualization queries Neo4j directly, not
through the API
--------------------------------------------------------------------------
api/routers/query.py's response is deliberately a thin contract (summary,
sources, scores — see api/schemas/query.py's own docstring) matching what
the PIPELINE needs to return, not every possible presentational extra a
UI might eventually want. "Which of these specific source papers cite
each other" is a presentation-only need — worth a small, separate,
read-only Neo4j query from the frontend itself (using asyncio.run() to
bridge into Neo4jClient's async interface from Streamlit's synchronous
execution model) rather than growing the API's response contract for a
feature only this one UI panel uses.

Scope note: no separate visual timeline chart
---------------------------------------------------
The architecture diagram's "C. Temporal Evolution Timeline" box is
covered here through trend_analysis's narrative text (already
chronologically structured — see trend_agent.py), not a dedicated bar/
line chart. api/schemas/query.py's EvidenceSource intentionally only
carries (arxiv_id, score), not publication dates, so a real timeline
chart would need the API contract extended first — a reasonable future
addition, not built now to avoid growing that contract for a single UI
panel before it's clear the text-based trend view isn't already enough.
"""

from __future__ import annotations

import asyncio

import httpx
import pandas as pd
import streamlit as st
from pyvis.network import Network

from temporal_graphrag.knowledge_graph.neo4j_client import Neo4jClient

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Temporal Hybrid GraphRAG", layout="wide")


def call_query_api(query: str) -> dict:
    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{API_BASE_URL}/api/query", json={"query": query})
        response.raise_for_status()
        return response.json()


async def _fetch_citation_edges(arxiv_ids: list[str]) -> list[tuple[str, str]]:
    """Read-only lookup: which pairs among these specific papers have a
    direct citation link, per the knowledge graph built in
    06_build_knowledge_graph.py. A short-lived Neo4jClient, opened and
    closed within this one call — a presentation-only, infrequent lookup
    (once per query the user runs), not a hot path justifying a
    long-lived connection the way retrieval_agent.py's does."""
    if len(arxiv_ids) < 2:
        return []
    async with Neo4jClient() as client:
        rows = await client.run_read(
            """
            MATCH (a:Paper)-[:CITES]->(:Citation)-[:RESOLVES_TO]->(b:Paper)
            WHERE a.arxiv_id IN $ids AND b.arxiv_id IN $ids
            RETURN DISTINCT a.arxiv_id AS citing, b.arxiv_id AS cited
            """,
            ids=arxiv_ids,
        )
    return [(row["citing"], row["cited"]) for row in rows]


def render_knowledge_graph(sources: list[dict]) -> None:
    arxiv_ids = [s["arxiv_id"] for s in sources]
    edges = asyncio.run(_fetch_citation_edges(arxiv_ids))

    net = Network(height="500px", width="100%", directed=True, notebook=False, bgcolor="#0e1117", font_color="white")
    for source in sources:
        # Node size reflects relevance score — the most useful single
        # visual signal to encode without cluttering the graph further.
        net.add_node(source["arxiv_id"], label=source["arxiv_id"], value=source["score"] * 50 + 10)
    for citing, cited in edges:
        net.add_edge(citing, cited)

    net.save_graph("kg_view.html")
    with open("kg_view.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=520)

    if not edges:
        st.caption("No direct citation links found among these specific sources — nodes shown without edges.")


def render_sources_chart(sources: list[dict]) -> None:
    if not sources:
        return
    df = pd.DataFrame(sources).set_index("arxiv_id")
    st.bar_chart(df["score"])


def main() -> None:
    st.title("Temporal Hybrid GraphRAG")
    st.caption("Multi-agent, temporal-aware question answering over a corpus of scientific papers.")

    query = st.text_input(
        "Ask a research question:",
        placeholder="e.g. How has GPT-3's approach to few-shot learning influenced later models?",
    )
    submitted = st.button("Ask", type="primary")

    if not submitted or not query.strip():
        return

    with st.spinner("Running the multi-agent pipeline (retrieval, reasoning, verification, summarization)..."):
        try:
            result = call_query_api(query)
        except httpx.HTTPError as e:
            st.error(f"Couldn't reach the API — is `uvicorn api.main:app` running? ({e})")
            return

    # -- A. Summary + E. Explainable Summary (confidence, verification) --
    st.subheader("Answer")
    st.write(result["summary"])

    col1, col2 = st.columns(2)
    with col1:
        confidence = result.get("confidence_score")
        st.metric("Confidence", f"{confidence:.0%}" if confidence is not None else "N/A")
    with col2:
        st.metric("Verified", "Yes" if result["verified"] else "No")

    if result["contradictions"]:
        with st.expander(f"Issues flagged by verification ({len(result['contradictions'])})"):
            for issue in result["contradictions"]:
                st.write(f"- {issue}")

    # -- D. Research Trend Analysis (and C., narratively — see module docstring) --
    if result.get("trend_analysis"):
        st.subheader("Trend & Evolution")
        st.write(result["trend_analysis"])

    # -- E. Evidence links, as a relevance chart --
    sources = result.get("sources") or []
    if sources:
        st.subheader("Sources")
        render_sources_chart(sources)

        # -- B. Knowledge Graph Visualization --
        st.subheader("Knowledge Graph")
        render_knowledge_graph(sources)


if __name__ == "__main__":
    main()