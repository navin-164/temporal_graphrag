"""
tests/unit/test_evaluation_metrics.py

Covers evaluation/metrics.py's retrieval metrics (precision@k, recall@k,
MRR, nDCG@k) and ROUGE — all pure-Python or CPU-only-and-fast, computed
against hand-worked examples so the expected values are independently
verifiable, not just "whatever the code happens to produce."

Why bert_score isn't tested here
--------------------------------------
bertscore() downloads a BERT checkpoint on first use (same as the
reranker's cross-encoder) — appropriate for a manual sanity check, not
for a unit suite that should run fast and offline. rouge_scores() needs
no such download, so it's covered directly.
"""

from __future__ import annotations

from pytest import approx

from temporal_graphrag.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_scores,
)


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}

    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # a hit, b miss
    assert precision_at_k(retrieved, relevant, k=4) == 0.5  # a, c hit out of 4


def test_precision_at_k_empty_retrieval():
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c", "z"}  # 3 relevant total, 2 appear in top 3

    assert recall_at_k(retrieved, relevant, k=3) == approx(2 / 3)


def test_recall_at_k_no_relevant_items():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_mean_reciprocal_rank_first_hit_at_rank_2():
    retrieved = ["x", "a", "y"]
    relevant = {"a"}

    assert mean_reciprocal_rank(retrieved, relevant) == 0.5  # 1/rank, rank=2


def test_mean_reciprocal_rank_no_hit():
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}  # both relevant items ranked first -> ideal ordering

    assert ndcg_at_k(retrieved, relevant, k=3) == approx(1.0)


def test_ndcg_at_k_worst_ranking_is_less_than_one():
    retrieved = ["c", "a", "b"]  # relevant items pushed to the back
    relevant = {"a", "b"}

    score = ndcg_at_k(retrieved, relevant, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_no_relevant_items_is_zero():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


def test_rouge_scores_identical_text_is_near_one():
    text = "GPT-3 has 175 billion parameters."
    scores = rouge_scores(prediction=text, reference=text)

    assert scores["rouge1"] == approx(1.0)
    assert scores["rougeL"] == approx(1.0)


def test_rouge_scores_unrelated_text_is_low():
    scores = rouge_scores(
        prediction="The weather today is sunny and warm.",
        reference="GPT-3 has 175 billion parameters.",
    )
    assert scores["rouge1"] < 0.3