"""
llm/base.py

Provider-agnostic LLM interface — every agent (Weeks 12-14) talks to
LLMProvider, never to Groq's or Gemini's SDK directly, so swapping or
adding a provider later doesn't touch agent code.

Why async
------------
Every external-service call elsewhere in this project (Neo4jClient, GROBID
via pdf_parser.py) is async — HTTP/network calls to Groq and Gemini fit
the same "I/O-bound, worth not blocking the event loop on" category, and
Week 12-14's multi-agent system will likely want to run independent agent
calls concurrently (asyncio.gather), which needs every provider to be
async-native rather than sync-wrapped.

Why a shared retry policy lives here, not duplicated per provider
--------------------------------------------------------------------------
core/exceptions.py's own docstring for ServiceRateLimitError flags this
directly: "the tenacity retry decorators we'll add around Groq/Gemini
calls" — anticipating exactly this. Groq and Gemini fail in the same
CATEGORIES (rate limited, momentarily unavailable) even though the raw
exceptions each SDK raises are completely different; each provider's own
file is responsible for catching its SDK's native exception and
re-raising the right ExternalServiceError subclass from core/exceptions.py,
then this one shared retry decorator handles backoff identically for
both, rather than two near-duplicate tenacity configurations drifting
apart over time.

Why LLMMessage/LLMResponse are plain dataclasses, not Pydantic
---------------------------------------------------------------------
Same reasoning retrieval/vector_retriever.py's RetrievalHit already gives:
these are internal, hot-path value objects, never deserialized from
external JSON directly (each provider's own SDK response gets mapped
into an LLMResponse by hand, deliberately, not auto-parsed) — Pydantic's
validation overhead buys nothing here.

generate_text() as a base-class convenience, not reimplemented per
provider
-------------------------------------------------------------------------
The overwhelming majority of agent calls are "one system prompt + one
user prompt, give me a string back" — forcing every call site to build a
list[LLMMessage] and unpack response.content would be needless
boilerplate repeated across 8 different agents. generate_text() does that
packing/unpacking once, here, on top of whatever generate() a subclass
implements — subclasses only ever need to implement generate() itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from temporal_graphrag.core.exceptions import ServiceRateLimitError, ServiceUnavailableError

__all__ = ["LLMMessage", "LLMResponse", "LLMProvider", "with_llm_retries"]

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """A provider's answer, normalized to one shape regardless of which
    SDK produced it. Token counts are Optional because not every provider
    response reliably includes them (or a mocked/test provider might
    not) — callers that need them (e.g. a future cost-tracking feature)
    should handle None rather than assume they're always present."""

    content: str
    model: str
    provider: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


# Shared retry policy for every provider's raw API call: up to 4 attempts
# total, exponential backoff starting at 2s, capped at 30s between tries.
# Retries ONLY on the two ExternalServiceError subclasses that mean "this
# might work if we just wait and try again" — NOT on
# ServiceAuthenticationError (a bad API key won't fix itself by retrying)
# or any bug in our own code, which should surface immediately rather
# than be retried into a confusing delayed failure.
with_llm_retries = retry(
    retry=retry_if_exception_type((ServiceRateLimitError, ServiceUnavailableError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, max=30),
    reraise=True,
)


class LLMProvider(ABC):
    """Every concrete provider (GroqProvider, GeminiProvider) implements
    generate(); generate_text() is free once that's done."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Sends `messages` to the underlying model and returns a
        normalized LLMResponse. Implementations are responsible for
        applying `with_llm_retries` around their own raw API call, and
        for translating their SDK's native exceptions into the
        ExternalServiceError hierarchy (see core/exceptions.py) BEFORE
        that retry decorator ever sees them — with_llm_retries only knows
        how to recognize OUR exception types, not Groq's or Gemini's own.
        """
        raise NotImplementedError

    async def generate_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Convenience wrapper for the common "one prompt in, one string
        out" case — see module docstring for why this lives here once,
        rather than in every agent's own code."""
        messages: list[LLMMessage] = []
        if system:
            messages.append(LLMMessage(role="system", content=system))
        messages.append(LLMMessage(role="user", content=prompt))

        response = await self.generate(messages, temperature=temperature, max_tokens=max_tokens)
        return response.content