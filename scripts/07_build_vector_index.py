"""
scripts/07_build_vector_index.py

Week 10's other deliverable: reads every embeddings/*.json (produced by
05_build_embeddings.py) and loads all four vector modalities — text,
metadata, figure, table — into a VectorStore, then saves it to disk as a
searchable FAISS index. This is the file that turns "we've computed a lot
of vectors" into "we have an index we can actually query."

BEFORE RUNNING:
1. `pip install -e .` from the project root, if you haven't yet.
2. 05_build_embeddings.py must have already produced embeddings/*.json.

No skip-if-already-loaded, same reasoning as 06_build_knowledge_graph.py
------------------------------------------------------------------------
06_build_knowledge_graph.py's own docstring explains why it always
processes every paper rather than skipping ones already in Neo4j:
MERGE-based writes are idempotent, so there's no real cost to redoing
them. This script is the opposite case for a related reason: FAISS's
IndexFlatIP.add() is NOT idempotent — vector_store.py's own docstring is
explicit that add() has no dedup, so calling it twice for the same id
produces two rows for that id, silently corrupting search results (two
hits for one paper, splitting whatever ranking weight it should have
gotten). The correct way to avoid that isn't a skip-check, though — it's
what this script actually does: build the ENTIRE VectorStore fresh from
embeddings/*.json every run, then overwrite the saved index directory.
Loading ~40 papers' worth of already-computed vectors and adding them to
a flat index is a few hundred numpy operations, not a slow model
inference pass — there's no meaningful cost to redoing it every time, so
"just rebuild it" is both simpler AND correct here, where a resume/skip
approach would actively risk duplicate rows.

Why this script doesn't need DocumentStore
-----------------------------------------------
Building a searchable index only needs ids + vectors, both of which
embeddings/*.json already has. DocumentStore's job — resolving an id back
to its original content — is what happens AFTER a search returns a hit,
which is Week 11's retrieval layer's job, not this script's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from temporal_graphrag.storage.vector_store import VectorStore

# ---------------- CONFIG — edit these ----------------
EMBEDDINGS_DIR = "embeddings"
OUTPUT_DIR = "vector_index"
# -------------------------------------------------------


def load_embedding_records() -> list[dict[str, Any]]:
    """Loads every embeddings/*.json written by 05_build_embeddings.py. A
    single corrupted file is logged and skipped, not fatal to the rest —
    same reasoning as every other load_*() in this project."""
    embeddings_dir = Path(EMBEDDINGS_DIR)
    if not embeddings_dir.exists():
        raise FileNotFoundError(
            f"'{EMBEDDINGS_DIR}/' doesn't exist. Run 05_build_embeddings.py first."
        )

    paths = sorted(embeddings_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"'{EMBEDDINGS_DIR}/' exists but has no .json files. Run 05_build_embeddings.py first."
        )

    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  WARNING: couldn't load {path.name}, skipping: {e}")
    return records


def build_index() -> None:
    records = load_embedding_records()
    print(f"Loaded {len(records)} embedding record(s) from {EMBEDDINGS_DIR}/")

    store = VectorStore()

    for record in records:
        arxiv_id = record["arxiv_id"]

        # -- Paper-level vectors: one each, always present -------------
        store.add("text", ids=[arxiv_id], vectors=[record["text_embedding"]])
        store.add("metadata", ids=[arxiv_id], vectors=[record["metadata_embedding"]])

        # -- Figure/table vectors: zero or more, some may be null -------
        # (a figure/table can fail to embed — see figure_embedder.py /
        # table_embedder.py's own None-on-failure conventions — so these
        # are filtered here rather than assumed present.)
        figure_ids = [f["id"] for f in record["figures"] if f["embedding"] is not None]
        figure_vectors = [f["embedding"] for f in record["figures"] if f["embedding"] is not None]
        if figure_ids:
            store.add("figure", ids=figure_ids, vectors=figure_vectors)

        table_ids = [t["id"] for t in record["tables"] if t["embedding"] is not None]
        table_vectors = [t["embedding"] for t in record["tables"] if t["embedding"] is not None]
        if table_ids:
            store.add("table", ids=table_ids, vectors=table_vectors)

    print("\nIndex built:")
    for modality in store.modalities:
        print(f"  {modality}: {store.count(modality)} vector(s)")

    store.save(OUTPUT_DIR)
    print(f"\nSaved to {OUTPUT_DIR}/ — one .index + one _ids.json file per modality.")


if __name__ == "__main__":
    build_index()
    print(f"\nThe FAISS index is ready in {OUTPUT_DIR}/. Load it back with:")
    print("  from temporal_graphrag.storage.vector_store import VectorStore")
    print(f'  store = VectorStore.load("{OUTPUT_DIR}")')
    print("\nNext: Week 11's retrieval/ builds vector_retriever.py (queries this index),")
    print("graph_retriever.py (queries the Neo4j graph from script 06), and the")
    print("temporal + hybrid fusion logic that combines both with the knowledge graph.")