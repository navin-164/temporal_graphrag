"""
llm/groq_provider.py

Groq implementation of LLMProvider — the FAST, primary provider for most
agent calls (Planner, Retrieval, Reasoning, Verification, Summarization,
Trend, Evaluation — per the architecture diagram's own agent-to-provider
assignments), using Llama models on Groq's LPU inference hardware.

Why Groq is "primary" and Gemini is "routed" (see llm/factory.py, next)
--------------------------------------------------------------------------
Groq's free tier is fast with generous rate limits for Llama models, but
a real per-request context window ceiling. Gemini's free tier has a much
larger context window but, per core/exceptions.py's own
ServiceRateLimitError docstring, "how tight Gemini's free-tier RPM turned
out to be" — a noticeably tighter requests-per-minute limit. factory.py's
job (built next) is choosing between the two per-request based on
estimated prompt size; this file's job is just "how do you actually call
Groq" once that choice is made.

Exception translation: groq's SDK exceptions -> this project's own
--------------------------------------------------------------------------
The groq Python SDK mirrors OpenAI's SDK shape (Groq's API is OpenAI-
compatible) — groq.RateLimitError, groq.APIConnectionError,
groq.APITimeoutError, groq.AuthenticationError, groq.InternalServerError,
etc. base.py's with_llm_retries only recognizes THIS project's own
ServiceRateLimitError/ServiceUnavailableError, so every groq.* exception
is caught here and re-raised as the matching core/exceptions.py type
BEFORE with_llm_retries ever sees it. Skipping this translation would
mean retries silently never trigger (retry_if_exception_type wouldn't
recognize a raw groq.RateLimitError at all), and callers catching
ExternalServiceError to handle "any external service problem" wouldn't
catch a bare groq exception either.

Why retry_after is read from the exception when Groq provides it
--------------------------------------------------------------------------
Groq's RateLimitError carries a response with a `retry-after` header when
the API sends one. ServiceRateLimitError.retry_after_seconds exists
specifically to carry this through (see its own docstring) — read it when
present rather than guessing, since tenacity's own exponential-backoff
schedule isn't necessarily what the service is actually asking for.

Why ConfigurationError fires in __init__, not lazily in generate()
--------------------------------------------------------------------------
config/settings.py's own policy is "fail late and loud, at the point of
use" — and constructing a GroqProvider already IS the point of use for
GROQ_API_KEY. Deferring the check to the first generate() call would just
delay the same error to a more confusing place (mid-agent-run instead of
at setup), for no benefit.
"""

from __future__ import annotations

from typing import Optional

import groq

from temporal_graphrag.config import settings
from temporal_graphrag.core.exceptions import (
    ConfigurationError,
    ServiceAuthenticationError,
    ServiceRateLimitError,
    ServiceUnavailableError,
)
from temporal_graphrag.llm.base import LLMMessage, LLMProvider, LLMResponse, with_llm_retries

__all__ = ["GroqProvider"]


class GroqProvider(LLMProvider):
    """Groq-backed LLMProvider. Raises ConfigurationError immediately if
    GROQ_API_KEY isn't set — see module docstring."""

    def __init__(self, model: Optional[str] = None) -> None:
        if not settings.groq_configured:
            raise ConfigurationError(
                "groq_api_key",
                hint="Set GROQ_API_KEY in .env — get one free at https://console.groq.com/keys.",
            )
        self.model = model or settings.groq_default_model
        self._client = groq.AsyncGroq(api_key=settings.groq_api_key)

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        return await self._call(messages, temperature=temperature, max_tokens=max_tokens)

    @with_llm_retries
    async def _call(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """The actual API call, kept separate from generate() only so
        with_llm_retries wraps just this method — generate() itself keeps
        the stable, retry-agnostic signature every LLMProvider promises."""
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except groq.RateLimitError as e:
            raise ServiceRateLimitError("Groq", str(e), retry_after_seconds=_extract_retry_after(e)) from e
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as e:
            raise ServiceUnavailableError("Groq", str(e)) from e
        except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
            raise ServiceAuthenticationError("Groq", str(e)) from e

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider="groq",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


def _extract_retry_after(error: "groq.RateLimitError") -> Optional[float]:
    """Best-effort read of the `retry-after` header Groq sometimes
    includes on a 429 — None if it's absent, malformed, or the SDK's
    response object shape ever changes; a missing hint isn't worth
    crashing over, tenacity's own backoff still applies regardless."""
    try:
        header_value = error.response.headers.get("retry-after")
        return float(header_value) if header_value is not None else None
    except Exception:
        return None