# ==============================================================================
# Temporal Hybrid GraphRAG — Master Pipeline Runner
# Target: Python 3.11.x
# ==============================================================================

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

PIPELINE_STEPS: list[tuple[int, str]] = [
    (1, "01_build_pilot_corpus.py"),
    (2, "02_enrich_citation_graph.py"),
    (3, "03_parse_documents.py"),
    (4, "04_extract_multimodal.py"),
    (5, "05_build_embeddings.py"),
    (6, "06_build_knowledge_graph.py"),
    (7, "07_build_vector_index.py"),
    (8, "08_add_connector_papers.py"),
    (9, "09_evaluate_pipeline.py"),
]


def run_step(step_num: int, script_name: str, python_exe: str = sys.executable) -> float:
    """Execute a single pipeline script and return elapsed runtime in seconds."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    start = time.perf_counter()
    logger.info("pipeline_step_started", step=step_num, script=script_name)

    completed = subprocess.run(
        [python_exe, str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    elapsed = round(time.perf_counter() - start, 2)

    if completed.returncode != 0:
        logger.error(
            "pipeline_step_failed",
            step=step_num,
            script=script_name,
            returncode=completed.returncode,
            elapsed_sec=elapsed,
        )
        raise RuntimeError(
            f"Step {step_num} ({script_name}) exited with code {completed.returncode}"
        )

    logger.info("pipeline_step_completed", step=step_num, script=script_name, elapsed_sec=elapsed)
    return elapsed


def run_pipeline(
    from_step: int = 1,
    to_step: int = 9,
    skip_steps: set[int] | None = None,
    python_exe: str = sys.executable,
) -> None:
    """Run selected pipeline stages in sequential order."""
    skipped = skip_steps or set()
    total_start = time.perf_counter()

    for step_num, script_name in PIPELINE_STEPS:
        if step_num < from_step or step_num > to_step:
            continue
        if step_num in skipped:
            logger.info("pipeline_step_skipped", step=step_num, script=script_name)
            continue

        script_path = SCRIPTS_DIR / script_name
        if script_path.stat().st_size == 0:
            logger.warning("pipeline_script_empty", step=step_num, script=script_name)

        run_step(step_num, script_name, python_exe=python_exe)

    total_elapsed = round(time.perf_counter() - total_start, 2)
    logger.info("pipeline_finished", total_elapsed_sec=total_elapsed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Temporal Hybrid GraphRAG pipeline steps.")
    parser.add_argument("--from-step", type=int, default=1, help="First step number to run (1-9).")
    parser.add_argument("--to-step", type=int, default=9, help="Last step number to run (1-9).")
    parser.add_argument(
        "--skip-steps",
        type=int,
        nargs="*",
        default=[],
        help="Optional step numbers to skip (e.g., --skip-steps 1 2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        from_step=args.from_step,
        to_step=args.to_step,
        skip_steps=set(args.skip_steps),
    )