"""
scripts/05_build_embeddings.py

Week 6's deliverable: reads every parsed/*.json (produced by
03_parse_documents.py — already has sections, citations, figures, tables,
equations) and computes all four embedding modalities from the
architecture diagram — text (SPECTER2), metadata (SPECTER2, serialized),
figures (CLIP), tables (SPECTER2, serialized) — saving one JSON file per
paper to embeddings/.

BEFORE RUNNING:
1. `pip install -e .` from the project root, if you haven't yet.
2. 03_parse_documents.py must have already produced parsed/*.json.
3. No Docker, no Neo4j, no API keys needed — every model here runs
   locally via transformers/adapters. The only network use is
   HuggingFace downloading model weights the FIRST time each model
   loads (~1.5GB total across SPECTER2 + CLIP); every run after that is
   fully offline.

Why no async, unlike scripts 03 and 06
-----------------------------------------
Those scripts are I/O-bound around a slow external service (GROBID,
Neo4j) — async lets the event loop do other useful work while waiting on
a network response. This script is CPU/GPU-bound local model inference
with no network call to wait on (post-download), so asyncio would add
complexity for zero benefit here.

Why text + metadata are batched across ALL pending papers, but figures/
tables are handled per-paper
---------------------------------------------------------------------
TextEmbedder.embed_papers() and MetadataEmbedder.embed_papers() already
batch internally (see text_embedder.py's DEFAULT_BATCH_SIZE) — calling
each once across the full pending list lets that batching actually help,
rather than forcing single-item batches by looping. Figures and tables
belong to a specific paper's own lists (paper.figures, paper.tables), so
those naturally stay in the per-paper loop below.

Output format: one JSON per paper, not one big file
---------------------------------------------------------
Matches parsed/*.json's own convention (per-paper, not one 40-paper
blob) — same benefits apply: a single corrupted file doesn't threaten the
other 39, and re-running after a partial failure only needs to touch
files that don't exist yet (see skip logic below).

Why embeddings/ (not embeddings_output/ or similar)
------------------------------------------------------
Consistent with this project's existing data folders (parsed/, metadata/,
citations/, extracted_images/) — plain nouns naming what's inside. This
does share a name with src/temporal_graphrag/embeddings/, the CODE
package, but the two are never ambiguous in practice: nothing in this
project does a bare `import embeddings`, only
`from temporal_graphrag.embeddings import ...`, which resolves through
the installed package regardless of what's sitting in the current
working directory.

Resume support
-------------------
Same reasoning as 03_parse_documents.py: computing embeddings for 40
papers (plus their figures/tables) takes real time, so a paper whose
output file already exists is skipped rather than recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from temporal_graphrag.core.entities import Paper
from temporal_graphrag.embeddings import FigureEmbedder, MetadataEmbedder, TableEmbedder, TextEmbedder

# ---------------- CONFIG — edit these ----------------
PARSED_DIR = "parsed"
OUTPUT_DIR = "embeddings"
# -------------------------------------------------------


def load_papers() -> list[Paper]:
    """Loads every parsed/*.json into a Paper. A single corrupted or
    incomplete file is logged and skipped, not fatal to the other 39 —
    same reasoning as 06_build_knowledge_graph.py's load_papers()."""
    parsed_dir = Path(PARSED_DIR)
    if not parsed_dir.exists():
        raise FileNotFoundError(f"'{PARSED_DIR}/' doesn't exist. Run 03_parse_documents.py first.")

    paths = sorted(parsed_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"'{PARSED_DIR}/' exists but has no .json files. Run 03_parse_documents.py first."
        )

    papers: list[Paper] = []
    for path in paths:
        try:
            papers.append(Paper.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  WARNING: couldn't load {path.name}, skipping: {e}")
    return papers


def build_embeddings() -> None:
    papers = load_papers()
    print(f"Loaded {len(papers)} papers from {PARSED_DIR}/")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # -- Resume support: skip papers already fully embedded ---------------
    pending: list[Paper] = []
    already_done = 0
    for paper in papers:
        out_path = output_dir / f"{paper.arxiv_id.replace('/', '_')}.json"
        if out_path.exists():
            already_done += 1
        else:
            pending.append(paper)

    if already_done:
        print(f"  {already_done} paper(s) already embedded, skipping.")
    if not pending:
        print("Nothing left to embed.")
        return

    print(f"Embedding {len(pending)} paper(s)...")

    # -- Load every model ONCE, up front -----------------------------------
    # text_embedder is shared with metadata_embedder and table_embedder
    # (both reuse the same loaded SPECTER2 weights, not separate copies —
    # see metadata_embedder.py / table_embedder.py docstrings for why).
    print("Loading embedding models (first run downloads weights, ~1.5GB total)...")
    try:
        text_embedder = TextEmbedder()
        metadata_embedder = MetadataEmbedder(text_embedder)
        table_embedder = TableEmbedder(text_embedder)
        figure_embedder = FigureEmbedder()  # CLIP — a separate model, separate embedding space
    except Exception as e:
        print(f"\nERROR: couldn't load embedding models: {e}")
        print("Check your internet connection (first run needs to download from HuggingFace)")
        print("and that enough disk/memory is available.")
        return

    # -- Batch the two paper-level embeddings across ALL pending papers ----
    print("Computing paper-level text + metadata embeddings...")
    text_vectors = text_embedder.embed_papers(pending)
    metadata_vectors = metadata_embedder.embed_papers(pending)

    succeeded = 0
    failed: list[tuple[str, str]] = []

    for i, (paper, text_vec, meta_vec) in enumerate(zip(pending, text_vectors, metadata_vectors), 1):
        out_path = output_dir / f"{paper.arxiv_id.replace('/', '_')}.json"
        try:
            figure_vecs = figure_embedder.embed_figures(paper.figures)
            table_vecs = table_embedder.embed_tables(paper.tables)

            record: dict[str, Any] = {
                "arxiv_id": paper.arxiv_id,
                "text_embedding": text_vec,
                "metadata_embedding": meta_vec,
                "figures": [
                    {"id": fig.id, "embedding": vec} for fig, vec in zip(paper.figures, figure_vecs)
                ],
                "tables": [
                    {"id": tbl.id, "embedding": vec} for tbl, vec in zip(paper.tables, table_vecs)
                ],
            }
        except Exception as e:
            # A bug specific to this paper's figures/tables shouldn't lose
            # the paper-level vectors we already computed — same "one bad
            # item, not the whole batch" reasoning used throughout this
            # project's other scripts.
            print(f"  [{i}/{len(pending)}] FAILED {paper.arxiv_id}: {e}")
            failed.append((paper.arxiv_id, str(e)))
            continue

        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        figures_embedded = sum(1 for f in record["figures"] if f["embedding"] is not None)
        tables_embedded = sum(1 for t in record["tables"] if t["embedding"] is not None)
        print(
            f"  [{i}/{len(pending)}] embedded {paper.arxiv_id}: "
            f"text + metadata vectors, {figures_embedded}/{len(paper.figures)} figures, "
            f"{tables_embedded}/{len(paper.tables)} tables"
        )
        succeeded += 1

    print(f"\nDone. {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed papers:")
        for arxiv_id, reason in failed:
            print(f"  {arxiv_id}: {reason}")


if __name__ == "__main__":
    build_embeddings()
    print(f"\nEmbeddings are saved to {OUTPUT_DIR}/*.json — one file per paper, holding")
    print("its text, metadata, figure, and table vectors.")
    print("\nNext: storage/vector_store.py (FAISS) and storage/document_store.py, then")
    print("scripts/07_build_vector_index.py loads these files into a searchable index.")