"""
scripts/09_evaluate_pipeline.py

The ablation study this project's evaluation/metrics.py was built for:
runs every question in evaluation/benchmark.json through three retrieval
configurations — vector-only, hybrid-fused (vector+graph), and hybrid+
reranked — scores each against the benchmark's known-correct sources
using evaluation/metrics.py's IR metrics, then runs the FULL agent
pipeline once per question and scores the generated summary against your
hand-written reference via ROUGE. Produces one comparison table for each,
saved as CSV for your report.

Why three retrieval configurations, not just "the final pipeline"
--------------------------------------------------------------------------
"It works" isn't a result — "hybrid retrieval improves nDCG@5 over
vector-only by X" is. Each configuration isolates one stage this
project's retrieval/ package added on top of plain vector search:
  - vector_only: retrieval/vector_retriever.py alone.
  - hybrid_fused: + retrieval/graph_retriever.py's citation-neighborhood
    expansion, combined via retrieval/hybrid_scorer.py.
  - hybrid_reranked: + retrieval/reranker.py's cross-encoder pass.
Comparing these three directly answers "did each added stage actually
help," rather than asserting it did.

Why summarization is scored separately, and only once per question
--------------------------------------------------------------------------
The full agent pipeline (agents/graph.py) makes several real LLM calls
per question — unlike the retrieval-only comparison above, there's no
cheap way to run 3 separate "configurations" of the whole pipeline
without tripling API usage. This runs the pipeline exactly as it exists
today, once per benchmark question, and reports ROUGE against your
reference answers as a headline "how good are the final summaries"
number, distinct from the retrieval ablation above.

Why this reuses retrieval/graph_retriever.py's actual seeding logic,
not RetrievalAgent directly
--------------------------------------------------------------------------
RetrievalAgent.__call__() bundles all three retrieval stages into one
method by design (see that file's own docstring) — appropriate for the
live pipeline, but this script needs to inspect each stage's output
SEPARATELY to score them independently. The graph-seeding constants
(GRAPH_SEED_COUNT, GRAPH_EXPANSION_HOPS) are duplicated from retrieval_
agent.py here rather than imported, so this script's ablation reflects
the same configuration the live system actually uses.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd

from temporal_graphrag.agents import build_agent_graph, create_initial_state
from temporal_graphrag.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_scores,
)
from temporal_graphrag.knowledge_graph.neo4j_client import Neo4jClient
from temporal_graphrag.retrieval.graph_retriever import GraphRetriever
from temporal_graphrag.retrieval.hybrid_scorer import HybridScorer
from temporal_graphrag.retrieval.reranker import Reranker
from temporal_graphrag.retrieval.vector_retriever import RetrievalHit, VectorRetriever
from temporal_graphrag.storage.document_store import DocumentStore
from temporal_graphrag.storage.vector_store import VectorStore

# ---------------- CONFIG — edit these ----------------
BENCHMARK_PATH = "evaluation/benchmark.json"
RESULTS_DIR = "evaluation/results"
TOP_K = 5  # cutoff for precision@k / recall@k / ndcg@k, and candidates kept per stage
GRAPH_SEED_COUNT = 3       # matches retrieval_agent.py's own constant
GRAPH_EXPANSION_HOPS = 2   # matches retrieval_agent.py's own constant
# -------------------------------------------------------


def load_benchmark() -> list[dict[str, Any]]:
    path = Path(BENCHMARK_PATH)
    if not path.exists():
        raise FileNotFoundError(f"'{BENCHMARK_PATH}' doesn't exist yet — see the guide for its format.")
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(hits: list[RetrievalHit]) -> list[str]:
    """Extracts just the ids, in rank order — what every IR metric here
    actually needs."""
    return [hit.id for hit in hits]


def _score_ir_metrics(retrieved_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    return {
        f"precision@{TOP_K}": precision_at_k(retrieved_ids, relevant_ids, TOP_K),
        f"recall@{TOP_K}": recall_at_k(retrieved_ids, relevant_ids, TOP_K),
        f"ndcg@{TOP_K}": ndcg_at_k(retrieved_ids, relevant_ids, TOP_K),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
    }


async def evaluate_retrieval(
    benchmark: list[dict[str, Any]],
    vector_retriever: VectorRetriever,
    graph_retriever: GraphRetriever,
    hybrid_scorer: HybridScorer,
    reranker: Reranker,
) -> pd.DataFrame:
    """Runs all three retrieval configurations for every benchmark
    question, returning one row per (question, configuration)."""
    rows: list[dict[str, Any]] = []

    for item in benchmark:
        query = item["query"]
        relevant_ids = set(item["relevant_arxiv_ids"])

        vector_hits = vector_retriever.search_papers(query, k=TOP_K * 2)

        top_seeds = sorted(vector_hits, key=lambda h: h.score, reverse=True)[:GRAPH_SEED_COUNT]
        graph_hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for seed in top_seeds:
            for hit in await graph_retriever.k_hop_neighbors(seed.id, hops=GRAPH_EXPANSION_HOPS, k=TOP_K * 2):
                if hit.id not in seen:
                    seen.add(hit.id)
                    graph_hits.append(hit)

        fused = hybrid_scorer.fuse(vector_hits=vector_hits, graph_hits=graph_hits, k=TOP_K * 2)
        reranked = reranker.rerank(query, fused, top_k=TOP_K * 2)

        configs = {
            "vector_only": _ids(vector_hits),
            "hybrid_fused": _ids(fused),
            "hybrid_reranked": _ids(reranked),
        }

        for config_name, retrieved_ids in configs.items():
            row = {"query": query, "configuration": config_name}
            row.update(_score_ir_metrics(retrieved_ids, relevant_ids))
            rows.append(row)

        print(f"  scored retrieval for: {query[:60]}...")

    return pd.DataFrame(rows)


async def evaluate_summarization(benchmark: list[dict[str, Any]]) -> pd.DataFrame:
    """Runs the FULL agent pipeline once per question and scores the
    generated summary against the hand-written reference via ROUGE.
    Saves incrementally, and skips any query already present in a
    previous run's saved CSV — same resume-on-rerun pattern this
    project's build scripts (03, 05) already use, needed here because
    Groq's free-tier daily token quota is a real, easily-hit limit for a
    benchmark of any real size (see the rate limit you already hit)."""
    output_path = Path(RESULTS_DIR) / "summarization_scores.csv"

    already_done: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        already_done = {row["query"]: row.to_dict() for _, row in existing_df.iterrows()}
        print(f"  {len(already_done)} question(s) already scored — skipping those.")

    rows: list[dict[str, Any]] = list(already_done.values())
    pending = [item for item in benchmark if item["query"] not in already_done]

    if not pending:
        print("  Nothing left to score.")
        return pd.DataFrame(rows)

    graph, aclose = await build_agent_graph()
    try:
        for item in pending:
            query = item["query"]
            reference = item["reference_answer"]

            try:
                state = create_initial_state(query)
                result = await graph.ainvoke(state)
                summary = result.get("summary", "")
                scores = rouge_scores(prediction=summary, reference=reference)
                rows.append(
                    {
                        "query": query,
                        "verified": result.get("verified", False),
                        "confidence_score": result.get("confidence_score"),
                        **scores,
                    }
                )
                print(f"  scored summary for: {query[:60]}...")
            except Exception as e:
                print(f"  FAILED on: {query[:60]}... ({e})")
                print(f"  Saving {len(rows)} completed result(s) so far before stopping.")
                break

            pd.DataFrame(rows).to_csv(output_path, index=False)
    finally:
        await aclose()

    return pd.DataFrame(rows)


async def main() -> None:
    benchmark = load_benchmark()
    print(f"Loaded {len(benchmark)} benchmark question(s) from {BENCHMARK_PATH}\n")

    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    document_store = DocumentStore.from_directory("parsed")
    vector_store = VectorStore.load("vector_index")
    vector_retriever = VectorRetriever(vector_store)
    hybrid_scorer = HybridScorer(document_store)
    reranker = Reranker(document_store)

    print("=== Retrieval ablation ===")
    async with Neo4jClient() as client:
        graph_retriever = GraphRetriever(client)
        retrieval_df = await evaluate_retrieval(
            benchmark, vector_retriever, graph_retriever, hybrid_scorer, reranker
        )

    retrieval_df.to_csv(output_dir / "retrieval_ablation.csv", index=False)

    summary_metrics = ["precision@5", "recall@5", "ndcg@5", "mrr"]
    comparison = retrieval_df.groupby("configuration")[summary_metrics].mean().round(3)
    print("\nAverage retrieval metrics by configuration:")
    print(comparison.to_string())
    comparison.to_csv(output_dir / "retrieval_ablation_summary.csv")

    print("\n=== Full-pipeline summarization ===")
    summarization_df = await evaluate_summarization(benchmark)
    summarization_df.to_csv(output_dir / "summarization_scores.csv", index=False)

    print("\nAverage summarization scores:")
    print(summarization_df[["rouge1", "rouge2", "rougeL"]].mean().round(3).to_string())
    print(f"\nVerification pass rate: {summarization_df['verified'].mean():.1%}")
    print(f"Average confidence score: {summarization_df['confidence_score'].mean():.3f}")

    print(f"\nAll results saved to {RESULTS_DIR}/ — use these tables directly in your report.")


if __name__ == "__main__":
    asyncio.run(main())