"""
evaluation/metrics.py

Week 15's offline evaluation toolkit — the metrics from the architecture
diagram's "10. Evaluation Framework" box, run against a labeled benchmark
(a set of (query, reference_summary, relevant_arxiv_ids) examples you
construct), NOT part of the live per-query agent pipeline.
evaluation_agent.py already computes a live, per-query confidence score
from signals available at runtime; this file is for reporting how well
the SYSTEM AS A WHOLE performs once you have (or hand-write) some ground
truth to compare against — the natural next step for a project write-up
or defense.

Summarization Metrics: ROUGE + BERTScore, not BLEURT or FactScore
--------------------------------------------------------------------------
requirements.txt already anticipates rouge-score and bert-score
specifically — both are implemented here directly. BLEURT is
deliberately NOT: it needs a specific, separately-downloaded TensorFlow
checkpoint (not a pip-installable model the way rouge-score/bert-score
are), adding a whole second deep learning framework to this project's
stack for one metric — the same "don't add an exotic dependency for a
single feature" stance this project has taken everywhere else (BLIP-2/
TAPAS in multimodal_agent.py, DeBERTa NLI in verification_agent.py,
GNN/TGAT/BERTopic in reasoning_agent.py/trend_agent.py). FactScore is
reference-free in principle, but its actual job — decomposing a summary
into atomic claims and checking each against evidence — is EXACTLY what
verification_agent.py already does live, per query. reference_free_fact_
score() below reuses that same approach (LLM-based claim checking against
evidence), packaged to run in aggregate across a benchmark set, rather
than pulling in the separate `factscore` package and its own LLM-calling
pipeline to duplicate a check this project already has working code for.

Retrieval Metrics: plain Python, no dependencies
------------------------------------------------------
Precision@k, Recall@k, MRR, and nDCG@k are all simple enough over a
(retrieved ids, relevant ids) pair that a dedicated IR-metrics library
would be overkill — implemented directly, each independently testable
against a hand-computed example.

Graph Quality Metrics: computed on the largest connected component
--------------------------------------------------------------------------
This project's own earlier debugging already established that this
corpus's citation graph is SPARSE — many papers have no resolved citation
link to another paper in the corpus at all (see 06_build_knowledge_
graph.py's own "Resolved N citation links" output, and the fuzzy-title-
matching fix added to graph_builder.py after finding 0 exact matches
initially). average_shortest_path_length is undefined on a disconnected
graph, and average_clustering behaves oddly when isolated nodes dominate
— computing these against the LARGEST connected component (with how much
of the graph that actually covers reported alongside, not hidden) gives
an honest, actually-computable picture instead of crashing or silently
producing a meaningless number.

Temporal Validity: a genuinely meaningful check for a TEMPORAL graph
--------------------------------------------------------------------------
The diagram lists "Temporal Validity" as a graph-quality metric without
specifying what it means. The most concrete, actually-useful
interpretation for a project centered on temporal reasoning: a citing
paper's date should never be EARLIER than a paper it cites — that would
be a chronological impossibility, and catching it validates that date
extraction (01_build_pilot_corpus.py's first_version_date) and citation
resolution (graph_builder.py) are both working correctly together.
temporal_validity() checks exactly this across every resolved citation
edge.
"""

from __future__ import annotations

import math

import networkx as nx
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

from temporal_graphrag.knowledge_graph.neo4j_client import Neo4jClient
from temporal_graphrag.retrieval.temporal_retriever import paper_date
from temporal_graphrag.storage.document_store import DocumentStore

__all__ = [
    "rouge_scores",
    "bertscore",
    "reference_free_fact_score",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "graph_quality_metrics",
    "temporal_validity",
]


# ======================================================================
# Summarization Metrics
# ======================================================================

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L F-measure between a generated summary and a reference
    one. Returns F-measure only (not precision/recall separately) — the
    standard single number reported for ROUGE in most summarization
    papers, and enough for a project write-up's headline metric."""
    scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {rouge_type: scores[rouge_type].fmeasure for rouge_type in _ROUGE_TYPES}


def bertscore(predictions: list[str], references: list[str], lang: str = "en") -> dict[str, float]:
    """BERTScore precision/recall/F1, averaged across the whole batch.
    Takes lists (not a single pair) because bert-score's own model-
    loading cost is amortized across a batch — computing it one summary
    at a time in a loop would reload/re-run the underlying model far
    less efficiently for a benchmark of any real size."""
    precision, recall, f1 = bert_score_fn(predictions, references, lang=lang, verbose=False)
    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }


async def reference_free_fact_score(summary: str, evidence_texts: list[str], llm_provider) -> float:
    """A FactScore-style, reference-free check: decomposes `summary` into
    atomic factual claims and checks what fraction are actually supported
    by `evidence_texts` — same underlying approach verification_agent.py
    already uses live, per query (see module docstring for why this
    reuses that idea rather than the separate `factscore` package).
    Returns a float in [0, 1] — the fraction of claims judged supported.

    `llm_provider` is any LLMProvider (pass llm.factory.llm_factory.
    get_provider(...) or construct one directly) — not imported as a
    default here, so this function stays importable/testable without
    needing real API credentials just to import evaluation/metrics.py.
    """
    evidence_block = "\n".join(f"[{i}] {text}" for i, text in enumerate(evidence_texts, 1))
    prompt = (
        f"Summary to check:\n{summary}\n\nEvidence:\n{evidence_block or '(none)'}\n\n"
        "List each distinct factual claim in the summary, and for each say SUPPORTED or "
        "UNSUPPORTED based on whether the evidence actually backs it up. Respond as one "
        "line per claim: 'SUPPORTED: <claim>' or 'UNSUPPORTED: <claim>'. No other text."
    )
    response_text = await llm_provider.generate_text(prompt, temperature=0.0)

    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    supported = sum(1 for line in lines if line.upper().startswith("SUPPORTED"))
    return supported / len(lines)


# ======================================================================
# Retrieval Metrics
# ======================================================================


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for id_ in top_k if id_ in relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for id_ in top_k if id_ in relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Despite the name, this computes the RECIPROCAL RANK for ONE query
    (1 / rank of the first relevant hit, or 0 if none found) — "mean" in
    MRR refers to averaging this across many queries, which the CALLER
    does across a benchmark set, not this function itself."""
    for rank, id_ in enumerate(retrieved_ids, 1):
        if id_ in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (each relevant id contributes gain 1, each
    irrelevant one contributes 0) — this project has no graded relevance
    labels (e.g. a 0-3 scale), only "is this paper relevant to the query
    or not," so binary gain is the honest choice rather than inventing
    graded relevance from nothing."""
    top_k = retrieved_ids[:k]

    dcg = sum(1.0 / math.log2(rank + 1) for rank, id_ in enumerate(top_k, 1) if id_ in relevant_ids)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ======================================================================
# Graph Quality Metrics
# ======================================================================


async def graph_quality_metrics(client: Neo4jClient) -> dict[str, float]:
    """Density, clustering coefficient, and average shortest path length
    over the citation graph — the latter two computed on the LARGEST
    CONNECTED COMPONENT, with how much of the graph that actually covers
    reported alongside. See module docstring on why: this project's own
    citation graph is known to be sparse, and these two metrics are
    either undefined or misleading on a disconnected graph."""
    rows = await client.run_read(
        """
        MATCH (a:Paper)-[:CITES]->(:Citation)-[:RESOLVES_TO]->(b:Paper)
        RETURN DISTINCT a.arxiv_id AS citing, b.arxiv_id AS cited
        """
    )
    all_papers = await client.run_read("MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id")

    graph = nx.DiGraph()
    graph.add_nodes_from(row["arxiv_id"] for row in all_papers)
    graph.add_edges_from((row["citing"], row["cited"]) for row in rows)

    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))
    largest_component = max(components, key=len) if components else set()
    subgraph = undirected.subgraph(largest_component)

    node_count = graph.number_of_nodes()
    return {
        "node_count": node_count,
        "edge_count": graph.number_of_edges(),
        "density": nx.density(graph),
        "largest_component_size": len(largest_component),
        "largest_component_fraction": (len(largest_component) / node_count) if node_count else 0.0,
        "average_clustering_largest_component": nx.average_clustering(subgraph) if len(subgraph) > 0 else 0.0,
        "average_shortest_path_length_largest_component": (
            nx.average_shortest_path_length(subgraph) if len(subgraph) > 1 else 0.0
        ),
    }


async def temporal_validity(client: Neo4jClient, document_store: DocumentStore) -> float:
    """Fraction of resolved citation edges where the citing paper's date
    is NOT earlier than the cited paper's date — see module docstring on
    why this is the concrete, checkable interpretation of "Temporal
    Validity" used here. 1.0 means every resolved citation is
    chronologically sound; anything less flags a genuine date-extraction
    or citation-resolution problem worth investigating."""
    rows = await client.run_read(
        """
        MATCH (a:Paper)-[:CITES]->(:Citation)-[:RESOLVES_TO]->(b:Paper)
        RETURN DISTINCT a.arxiv_id AS citing, b.arxiv_id AS cited
        """
    )
    if not rows:
        return 1.0  # vacuously valid — nothing to violate

    valid = 0
    for row in rows:
        try:
            citing_paper = document_store.get_paper(row["citing"])
            cited_paper = document_store.get_paper(row["cited"])
        except Exception:
            continue
        if paper_date(citing_paper) >= paper_date(cited_paper):
            valid += 1

    return valid / len(rows)