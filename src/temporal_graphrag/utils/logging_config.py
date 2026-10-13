"""
utils/logging_config.py

Structured logging setup — the file every plain `logging.getLogger(__name__)`
call across this project (pdf_parser.py, document_store.py, graph_builder.py,
...) was written anticipating, per each of their own comments: "utils/
logging_config.py (not built yet) is what decides how structlog is wired
up." Call configure_logging() once, at the start of any entry point
(a script's `if __name__ == "__main__":` block, api/main.py's lifespan,
frontend/app.py) — every existing stdlib logger elsewhere in this project
then automatically routes through it, with zero changes needed to those
files.

Why stdlib logging elsewhere routes THROUGH structlog, rather than every
file calling structlog directly
--------------------------------------------------------------------------
Those other files' own comments already explain why they used plain
stdlib logging instead of importing structlog directly: this file is what
decides the actual wiring, and guessing at that wiring from inside e.g.
pdf_parser.py would risk it not matching whatever this file eventually
sets up. structlog.stdlib.ProcessorFormatter is specifically designed for
this: it lets normal `logging.getLogger(__name__).warning(...)` calls
everywhere else in the codebase pass through the SAME structlog
processors (timestamps, log level, JSON or console rendering) as any log
call structlog itself would produce, without touching any of those files.

Console vs JSON output
---------------------------
`json_output=False` (the default) gives colorized, human-readable console
output — what you want while developing and watching a script's terminal
output directly (which is how every sanity test in this project has been
run so far). `json_output=True` gives newline-delimited JSON — the format
you'd actually want if this were shipped and its logs collected by
something else (e.g. piped to a file for later analysis, or a real
logging platform) — worth having available now even though nothing in
this pilot project currently consumes JSON logs downstream.

A caveat worth being upfront about
---------------------------------------
Same category of caveat as gemini_provider.py's and agents/graph.py's:
this is structlog's own documented recipe for stdlib integration, not
verified against a live install in this session. If configure_logging()
raises on import or at call time, that's the one thing to check against
your installed `structlog` version.
"""

from __future__ import annotations

import logging
import sys

import structlog

__all__ = ["configure_logging"]


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Call this ONCE, as early as possible in any entry point. Safe to
    call multiple times (e.g. accidentally, from an imported module) —
    each call simply reconfigures the same root logger/structlog setup,
    it doesn't stack duplicate handlers."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)