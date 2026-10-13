"""
config/settings.py

Centralized application configuration, loaded once from environment
variables (and, in local development, from a `.env` file in the project
root — see `.env.example` for the full list of variables and what each
one is for).

Design notes
------------
- Built on Pydantic v2's `BaseSettings` (from the separate `pydantic-settings`
  package — it moved out of core Pydantic in v2).
- Every external-service credential (Neo4j password, Groq/Gemini API keys)
  is Optional with a `None` default, on purpose: this module needs to
  import and instantiate cleanly this week, when only S2_API_KEY and
  GROBID_URL are actually filled in. Making every field required would
  make `Settings()` raise a validation error on import for anyone who
  hasn't set up every credential yet — which would break unrelated code
  that only needed, say, the GROBID URL.
  Fail LATE and LOUD instead: whichever module actually needs a given
  key (llm/groq_provider.py, knowledge_graph/neo4j_client.py, ...) is
  responsible for checking it's set and raising a clear error at the
  point of use — not here.
- `get_settings()` is wrapped in `functools.lru_cache` so `.env` is
  parsed and validated exactly once per process, no matter how many
  modules import it.

Usage
-----
    from temporal_graphrag.config import settings
    print(settings.grobid_url)

For tests, force a fresh (uncached) read after patching env vars:
    from temporal_graphrag.config.settings import get_settings
    get_settings.cache_clear()
    fresh = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application configuration in one place.

    Field names are lowercase versions of the environment variable names
    in `.env.example` (e.g. `S2_API_KEY` -> `s2_api_key`) — pydantic-settings
    matches env vars to fields case-insensitively by default, so no manual
    aliasing is needed for the simple cases below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # unrelated vars sitting in .env won't break this
    )

    # -- Needed now: Week 2-3 --------------------------------------------
    s2_api_key: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key, used by 02_enrich_citation_graph.py.",
    )
    grobid_url: str = Field(
        default="http://localhost:8070",
        description="Base URL of the running GROBID Docker container.",
    )
    environment: Literal["development", "production"] = Field(
        default="development",
        description="Gates debug behavior — verbose errors, auto-reload, etc.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # -- Needed soon: Weeks 7-8 (knowledge graph) ------------------------
    neo4j_uri: str = Field(default="neo4j://127.0.0.1:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: Optional[str] = Field(
        default=None,
        description="No safe default — set this once you've created a database in Neo4j Desktop.",
    )
    neo4j_database: str = Field(default="neo4j")

    # -- Needed for agents: Weeks 12-14 ----------------------------------
    groq_api_key: Optional[str] = Field(default=None)
    groq_default_model: str = Field(default="openai/gpt-oss-120b")
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_default_model: str = Field(default="gemini-3.6-flash")
    llm_groq_token_overflow_threshold: int = Field(
        default=5000,
        gt=0,
        description="Prompts above this estimated token count route to Gemini instead of Groq.",
    )

    # -- Needed later: Week 14+ (API + frontend) -------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, gt=0, lt=65536)
    streamlit_api_base_url: str = Field(default="http://localhost:8000")

    # -- Convenience properties, used by other modules as we build them --

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def neo4j_auth(self) -> tuple[str, Optional[str]]:
        """(username, password) tuple, the shape neo4j.GraphDatabase.driver() expects."""
        return (self.neo4j_username, self.neo4j_password)

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @field_validator("grobid_url", "streamlit_api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        # So callers can always safely do f"{settings.grobid_url}/api/..."
        # without worrying whether the .env value already ends in "/".
        return v.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    """Return the cached, validated Settings instance.

    Cached with lru_cache (rather than a bare module-level `Settings()`
    call) so `.env` is read and validated exactly once per process, and
    so tests can force a re-read with `get_settings.cache_clear()` after
    monkeypatching environment variables.
    """
    return Settings()


# Convenience singleton for the common case — most call sites just want
# `from temporal_graphrag.config import settings`, not to think about caching.
settings = get_settings()


if __name__ == "__main__":
    # Quick manual sanity check: `python -m temporal_graphrag.config.settings`
    # Never prints actual secret values — only whether they're set.
    s = get_settings()
    print(f"environment           = {s.environment}")
    print(f"log_level              = {s.log_level}")
    print(f"grobid_url             = {s.grobid_url}")
    print(f"s2_api_key             = {'set' if s.s2_api_key else 'NOT set'}")
    print(f"neo4j_uri              = {s.neo4j_uri}")
    print(f"neo4j_password         = {'set' if s.neo4j_password else 'NOT set'}")
    print(f"groq_configured        = {s.groq_configured}")
    print(f"gemini_configured      = {s.gemini_configured}")
