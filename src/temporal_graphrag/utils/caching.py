"""
utils/caching.py

Disk-backed caching — avoids recomputing embeddings and re-spending
free-tier LLM quota when re-running the same script/query repeatedly
during development, per requirements.txt's own stated purpose for
diskcache.

Why diskcache, not an in-memory dict or functools.lru_cache
------------------------------------------------------------------
functools.lru_cache only persists for the life of one Python process —
worthless for this project's actual pain point (re-running
05_build_embeddings.py or a query multiple times across separate `python`
invocations during development). diskcache persists across runs, on
local disk, needing no external service (unlike Redis/memcached) — fits
this project's "free, local, no extra infrastructure" constraints.

Why a separate async_cached decorator, not diskcache's built-in memoize()
--------------------------------------------------------------------------
diskcache.Cache.memoize() is sync-only. Most of the functions worth
caching here — LLM calls in particular (llm/base.py's generate_text()) —
are async. async_cached() wraps the same underlying Cache with manual
async-aware get/set logic, so it can decorate an `async def` directly
rather than forcing every cached call through asyncio.to_thread() just to
use a sync-only cache.

Why cache keys hash the arguments, not use them directly
------------------------------------------------------------------
A key built from a function's qualified name + a stable hash of its
arguments keeps keys a fixed, short size regardless of how large an
argument (e.g. a long prompt string) happens to be — easier to log/debug
than a multi-KB raw key.

This is opt-in, not wired into any existing agent/provider automatically
-------------------------------------------------------------------------
Nothing in llm/, agents/, or retrieval/ has been modified to use these
decorators. Caching an LLM call by default would silently make repeated
"identical" queries return a stale, non-fresh answer — fine for
deliberately replaying a call during development, but not a decision to
make silently on existing, already-tested code. Apply @cached/
@async_cached yourself to whichever specific function you're iterating
on, e.g. while tuning a prompt without re-spending Groq/Gemini quota on
every re-run.

What this deliberately does NOT cache
------------------------------------------
Neo4j reads (the graph changes as papers are added — see
08_add_connector_papers.py) or FAISS searches (the index changes on
rebuild) aren't good caching targets — both depend on mutable state this
cache has no way to invalidate on. Embeddings and LLM calls are the two
costs requirements.txt's own comment names, and the two genuinely
expensive-to-redo operations in this project.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, TypeVar

import diskcache

__all__ = ["cached", "async_cached", "get_cache"]

DEFAULT_CACHE_DIR = ".cache/temporal_graphrag"

_cache: Optional[diskcache.Cache] = None

T = TypeVar("T")


def get_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> diskcache.Cache:
    """Returns the shared, process-wide Cache instance, creating it (and
    its on-disk directory) on first use — one shared instance, not one
    per decorated function, since diskcache's Cache is itself lightweight
    to share and its sqlite backing handles concurrent access fine."""
    global _cache
    if _cache is None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(cache_dir)
    return _cache


def _make_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """A short, stable key from a function's name + its arguments — see
    module docstring on why a hash, not the raw arguments themselves."""
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{func_name}:{digest}"


def cached(cache_dir: str = DEFAULT_CACHE_DIR) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for a plain (sync) function, e.g. an embedder call
    during iterative script development."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = get_cache(cache_dir)
            key = _make_key(func.__qualname__, args, kwargs)
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return decorator


def async_cached(
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """Decorator for an `async def` function, e.g. LLMProvider.
    generate_text(), so re-running the same query during development
    doesn't re-spend free-tier LLM quota on an identical call. See module
    docstring on why this exists separately from diskcache's sync-only
    memoize()."""

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = get_cache(cache_dir)
            key = _make_key(func.__qualname__, args, kwargs)
            if key in cache:
                return cache[key]
            result = await func(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return decorator