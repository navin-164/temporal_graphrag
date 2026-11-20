# ==============================================================================
# Temporal Hybrid GraphRAG — Step 04: Extract Multimodal Elements
# Target: Python 3.11.x
# ==============================================================================

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import structlog

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDFS_DIR = PROJECT_ROOT / "pdfs"
PARSED_DIR = PROJECT_ROOT / "parsed"
EXTRACTED_IMAGES_DIR = PROJECT_ROOT / "extracted_images"
#MANIFEST_PATH = PARSED_DIR / "multimodal_manifest.json"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "multimodal_manifest.json"

FIGURE_FILENAME_RE = re.compile(r"^(?P<paper_id>.+?)_figure_(?P<idx>\d+)\.(?P<ext>png|jpg|jpeg|webp)$", re.IGNORECASE)


def _load_parsed_records(parsed_dir: Path) -> dict[str, dict[str, Any]]:
    """Load existing parsed JSON files keyed by paper ID."""
    records: dict[str, dict[str, Any]] = {}
    if not parsed_dir.exists():
        return records

    for json_path in sorted(parsed_dir.glob("*.json")):
        if json_path.name == MANIFEST_PATH.name:
            continue
        paper_id = json_path.stem
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                records[paper_id] = data
        except Exception as exc:
            logger.warning("failed_reading_parsed_json", path=str(json_path), error=str(exc))
    return records


def _extract_pdf_images_if_missing(pdfs_dir: Path, images_dir: Path) -> int:
    """Extract embedded images from PDFs if no images for that paper exist yet."""
    if not pdfs_dir.exists():
        return 0

    images_dir.mkdir(parents=True, exist_ok=True)
    existing_papers = {
        m.group("paper_id")
        for img in images_dir.iterdir()
        if img.is_file() and (m := FIGURE_FILENAME_RE.match(img.name))
    }

    created_count = 0
    for pdf_path in sorted(pdfs_dir.glob("*.pdf")):
        paper_id = pdf_path.stem
        if paper_id in existing_papers:
            continue

        try:
            with fitz.open(pdf_path) as doc:
                fig_idx = 0
                for page_num, page in enumerate(doc):
                    for img_info in page.get_images(full=True):
                        xref = img_info[0]
                        base_img = doc.extract_image(xref)
                        img_bytes = base_img.get("image")
                        ext = base_img.get("ext", "png").lower()
                        if not img_bytes:
                            continue
                        out_file = images_dir / f"{paper_id}_figure_{fig_idx}.{ext}"
                        out_file.write_bytes(img_bytes)
                        fig_idx += 1
                        created_count += 1
        except Exception as exc:
            logger.warning("pdf_image_extraction_failed", pdf=str(pdf_path), error=str(exc))

    return created_count


def _extract_tables_from_parsed(parsed_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect table entries if present inside a parsed document dictionary."""
    tables: list[dict[str, Any]] = []
    raw_tables = parsed_data.get("tables")
    if isinstance(raw_tables, list):
        for idx, tbl in enumerate(raw_tables):
            if isinstance(tbl, dict):
                tables.append({"table_index": idx, **tbl})
            else:
                tables.append({"table_index": idx, "content": tbl})
    return tables


def build_multimodal_manifest(
    pdfs_dir: Path = PDFS_DIR,
    parsed_dir: Path = PARSED_DIR,
    images_dir: Path = EXTRACTED_IMAGES_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Consolidate extracted figures and parsed tables into a structured manifest."""
    newly_extracted = _extract_pdf_images_if_missing(pdfs_dir, images_dir)
    parsed_records = _load_parsed_records(parsed_dir)

    figures_by_paper: dict[str, list[dict[str, Any]]] = {}
    if images_dir.exists():
        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file():
                continue
            match = FIGURE_FILENAME_RE.match(img_path.name)
            if not match:
                continue
            paper_id = match.group("paper_id")
            rel_path = str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            figures_by_paper.setdefault(paper_id, []).append(
                {
                    "figure_index": int(match.group("idx")),
                    "path": rel_path,
                    "size_bytes": img_path.stat().st_size,
                }
            )

    all_paper_ids = sorted(set(parsed_records.keys()) | set(figures_by_paper.keys()))
    papers_summary: dict[str, dict[str, Any]] = {}
    total_figures = 0
    total_tables = 0

    for paper_id in all_paper_ids:
        figs = sorted(figures_by_paper.get(paper_id, []), key=lambda x: x["figure_index"])
        tbls = _extract_tables_from_parsed(parsed_records.get(paper_id, {}))
        total_figures += len(figs)
        total_tables += len(tbls)
        papers_summary[paper_id] = {
            "figure_count": len(figs),
            "table_count": len(tbls),
            "figures": figs,
            "tables": tbls,
        }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "newly_extracted_images": newly_extracted,
        "paper_count": len(all_paper_ids),
        "total_figures": total_figures,
        "total_tables": total_tables,
        "papers": papers_summary,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "multimodal_manifest_built",
        papers=len(all_paper_ids),
        figures=total_figures,
        tables=total_tables,
        output_path=str(manifest_path),
    )
    return manifest


if __name__ == "__main__":
    build_multimodal_manifest()