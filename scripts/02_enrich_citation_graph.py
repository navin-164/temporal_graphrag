# ==============================================================================
# Temporal Hybrid GraphRAG — Step 02: Enrich Citation Graph
# Target: Python 3.11.x
# ==============================================================================

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "metadata" / "pilot_corpus_metadata.json"
CITATIONS_PATH = PROJECT_ROOT / "citations" / "pilot_corpus_citations.json"
OUTPUT_PATH = PROJECT_ROOT / "citations" / "enriched_citation_graph.json"


def _load_json(path: Path, default: Any) -> Any:
    """Safely load JSON data from disk if the file exists."""
    if not path.exists():
        logger.warning("json_file_missing", path=str(path))
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_papers(raw_meta: Any) -> dict[str, dict[str, Any]]:
    """Normalize corpus metadata into a dictionary keyed by paper/arXiv ID."""
    papers: dict[str, dict[str, Any]] = {}
    if isinstance(raw_meta, dict):
        items = raw_meta.items()
    elif isinstance(raw_meta, list):
        items = [
            (
                str(entry.get("arxiv_id") or entry.get("paper_id") or entry.get("id") or idx),
                entry,
            )
            for idx, entry in enumerate(raw_meta)
            if isinstance(entry, dict)
        ]
    else:
        return papers

    for paper_id, info in items:
        pid = str(paper_id)
        if isinstance(info, dict):
            papers[pid] = {
                "paper_id": pid,
                "title": info.get("title", ""),
                "year": info.get("year"),
                "published_date": info.get("published_date") or info.get("date"),
                "authors": info.get("authors", []),
            }
        else:
            papers[pid] = {"paper_id": pid, "title": str(info)}
    return papers


def enrich_citation_graph(
    metadata_path: Path = METADATA_PATH,
    citations_path: Path = CITATIONS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    """Build directed citation edges, in/out-degree stats, and temporal citation links."""
    raw_meta = _load_json(metadata_path, default={})
    raw_citations = _load_json(citations_path, default={})

    papers = _normalize_papers(raw_meta)
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)

    # Support either {source_id: [target_id, ...]} or [{"source": ..., "target": ...}]
    if isinstance(raw_citations, dict):
        for src, refs in raw_citations.items():
            src_id = str(src)
            papers.setdefault(src_id, {"paper_id": src_id})
            if isinstance(refs, list):
                for ref in refs:
                    ref_id = str(ref.get("target") or ref.get("paper_id") or ref.get("id") or ref) if isinstance(ref, dict) else str(ref)
                    if ref_id and ref_id != src_id:
                        outgoing[src_id].add(ref_id)
                        incoming[ref_id].add(src_id)
                        papers.setdefault(ref_id, {"paper_id": ref_id})
    elif isinstance(raw_citations, list):
        for edge in raw_citations:
            if not isinstance(edge, dict):
                continue
            src_id = str(edge.get("source") or edge.get("citing_paper_id") or edge.get("from") or "")
            tgt_id = str(edge.get("target") or edge.get("cited_paper_id") or edge.get("to") or "")
            if src_id and tgt_id and src_id != tgt_id:
                outgoing[src_id].add(tgt_id)
                incoming[tgt_id].add(src_id)
                papers.setdefault(src_id, {"paper_id": src_id})
                papers.setdefault(tgt_id, {"paper_id": tgt_id})

    edges: list[dict[str, Any]] = []
    for src_id, targets in sorted(outgoing.items()):
        src_year = papers.get(src_id, {}).get("year")
        for tgt_id in sorted(targets):
            tgt_year = papers.get(tgt_id, {}).get("year")
            temporal_direction = "unknown"
            if isinstance(src_year, int) and isinstance(tgt_year, int):
                if src_year >= tgt_year:
                    temporal_direction = "cites_earlier_or_same_year"
                else:
                    temporal_direction = "cites_later_year_anomaly"
            edges.append(
                {
                    "source": src_id,
                    "target": tgt_id,
                    "source_year": src_year,
                    "target_year": tgt_year,
                    "temporal_direction": temporal_direction,
                }
            )

    nodes: list[dict[str, Any]] = []
    for pid, meta in sorted(papers.items()):
        out_list = sorted(outgoing.get(pid, set()))
        in_list = sorted(incoming.get(pid, set()))
        nodes.append(
            {
                **meta,
                "out_degree": len(out_list),
                "in_degree": len(in_list),
                "references": out_list,
                "cited_by": in_list,
            }
        )

    enriched = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    logger.info(
        "citation_graph_enriched",
        nodes=len(nodes),
        edges=len(edges),
        output_path=str(output_path),
    )
    return enriched


if __name__ == "__main__":
    enrich_citation_graph()