"""
scripts/06_build_knowledge_graph.py

Week 7-8's actual deliverable: reads every parsed/*.json (produced by
03_parse_documents.py) and writes it into Neo4j via GraphBuilder,
applying schema.py's constraints first and resolving citation links last.

BEFORE RUNNING:
1. `pip install -e .` from the project root, if you haven't yet.
2. Neo4j Desktop: DBMS started, a database created, NEO4J_PASSWORD set in
   .env (see the note this project's config/settings.py already has
   about this).
3. 03_parse_documents.py must have already produced parsed/*.json — this
   script reads THAT output, not the raw metadata/citations files.

No skip-if-already-loaded, unlike 03_parse_documents.py — deliberate,
not an oversight
------------------------------------------------------------------------------
03_parse_documents.py skips already-parsed papers because each GROBID
call is genuinely slow (tens of seconds per paper, confirmed in your own
earlier run). Writing one paper's graph to Neo4j isn't slow — a handful
of MERGE statements against a local database — and GraphBuilder's writes
are idempotent by design (MERGE, not CREATE, everywhere; see graph_
builder.py's own docstring). Re-running this whole script after a partial
failure just re-confirms whatever already made it in and continues from
where it actually stopped. There's no meaningful waste here to justify
the extra bookkeeping a skip-check would add.

WHAT THIS SCRIPT DOES:
- Applies schema.py's constraints/indexes once, up front.
- Loads each parsed/*.json into a Paper, and writes it into the graph via
  GraphBuilder.build_paper() — one atomic transaction per paper.
- Runs resolve_citations() once, after every paper has been attempted —
  this is what turns a Citation node whose cited_arxiv_id happens to
  match another paper in THIS corpus into an actual graph edge between
  the two papers, regardless of what order they were processed in.
- Stops the whole run (rather than grinding through the rest identically)
  if Neo4j itself becomes unreachable mid-batch — the same distinction
  03_parse_documents.py already makes for GROBID, same reasoning.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from temporal_graphrag.core.entities import Paper
from temporal_graphrag.core.exceptions import ServiceUnavailableError, TemporalGraphRAGError
from temporal_graphrag.knowledge_graph import GraphBuilder, Neo4jClient, apply_schema

# ---------------- CONFIG — edit these ----------------
PARSED_DIR = "parsed"
# -------------------------------------------------------


def load_papers() -> list[Paper]:
    """Loads every parsed/*.json into a Paper. A single corrupted or
    incomplete file is logged and skipped, not fatal to the other 39 —
    the same "one bad item shouldn't stop the batch" reasoning used
    throughout this project's ingestion scripts."""
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


async def build_graph() -> None:
    papers = load_papers()
    print(f"Loaded {len(papers)} papers from {PARSED_DIR}/")

    async with Neo4jClient() as client:
        if not await client.verify_connectivity():
            print("\nERROR: Neo4j isn't reachable.")
            print("Check: DBMS started in Neo4j Desktop, a database created,")
            print("and NEO4J_PASSWORD set correctly in .env.")
            return

        print("Applying schema (constraints + indexes)...")
        await apply_schema(client)

        builder = GraphBuilder(client)
        succeeded = 0
        failed: list[tuple[str, str]] = []

        for i, paper in enumerate(papers, 1):
            try:
                await builder.build_paper(paper)
            except ServiceUnavailableError as e:
                print(f"\nSTOPPING: Neo4j became unreachable while writing {paper.arxiv_id}: {e}")
                print("Fix it and re-run — MERGE-based writes are idempotent, so this")
                print("just re-confirms what's already there and continues from here.")
                break
            except TemporalGraphRAGError as e:
                print(f"  [{i}/{len(papers)}] FAILED {paper.arxiv_id}: {e}")
                failed.append((paper.arxiv_id, str(e)))
                continue

            print(f"  [{i}/{len(papers)}] wrote {paper.arxiv_id}")
            succeeded += 1

        print(f"\n{succeeded} papers written, {len(failed)} failed.")
        if failed:
            print("Failed papers:")
            for arxiv_id, reason in failed:
                print(f"  {arxiv_id}: {reason}")

        print("\nResolving citation links (Citation -> Paper, where cited_arxiv_id matches)...")
        resolved = await builder.resolve_citations()
        print(f"Resolved {resolved} citation links within this corpus.")

        print("Resolving additional citation links via fuzzy title matching...")
        resolved_by_title = await builder.resolve_citations_by_title(papers)
        print(f"Resolved {resolved_by_title} additional citation link(s) via title matching.")


if __name__ == "__main__":
    asyncio.run(build_graph())
    print("\nThe knowledge graph is built. Open Neo4j Browser (or Bloom) against")
    print("neo4j://127.0.0.1:7687 to explore it directly — e.g. MATCH (p:Paper)")
    print("-[:CITES]->(c:Citation)-[:RESOLVES_TO]->(p2:Paper) RETURN p, c, p2")
    print("to see which papers in this corpus cite each other.")
    print("\nNext: Week 10-11's storage/ and retrieval/ build on top of this")
    print("graph plus the FAISS vector index still to come.")
