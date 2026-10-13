"""
llm/gemini_provider.py

Gemini implementation of LLMProvider — routed to for large-context calls
that would overflow Groq's window, and for the Multimodal Understanding
agent per the architecture diagram's own provider assignment (Google's
multimodal capabilities are why that agent specifically routes here, not
to Groq's text-only Llama/GPT-OSS models).

Which SDK, and why
----------------------
`google-genai` (imported as `from google import genai`), NOT the older
`google-generativeai` package — requirements.txt's own comment already
flags this: the old package is deprecated (EOL Nov 2025), and installing
it instead, even by following an older tutorial, would work today but
stop working once Google actually pulls it.

Async via client.aio, not a sync-wrapped call
--------------------------------------------------
Same reasoning as groq_provider.py and base.py's own docstring: every
external-service call in this project is async-native. google-genai's
Client exposes an `.aio` namespace specifically for this — using the sync
`client.models.generate_content(...)` and wrapping it in
asyncio.to_thread() would work, but .aio is the SDK's own intended async
path and avoids spinning up a thread pool just to fake async.

Exception translation, same reasoning as groq_provider.py
------------------------------------------------------------------
google.genai.errors.ClientError/ServerError carry an HTTP status code
(`.code`) rather than one dedicated exception class per failure kind the
way groq's SDK does — so the mapping here branches on that status code
directly rather than on distinct exception class names. 429 -> rate
limit, 401/403 -> auth, everything else -> service unavailable. Same
destination types either way (core/exceptions.py's ServiceRateLimitError
/ ServiceAuthenticationError / ServiceUnavailableError), so
with_llm_retries (base.py) treats a rate-limited Gemini call exactly like
a rate-limited Groq call, and factory.py (next) doesn't need to know or
care which SDK actually raised what.

No true "system" role in Gemini's chat turns
--------------------------------------------------
Gemini's API takes a system prompt via a separate `system_instruction`
config field, not as a turn inside `contents` the way OpenAI/Groq treat a
"system" message. generate() below pulls any system-role LLMMessage out
and passes it through system_instruction instead of trying to force it
into a chat turn — translated once here, so callers can keep writing the
same LLMMessage(role="system", ...) regardless of which provider ends up
handling the call.

A caveat worth being upfront about
---------------------------------------
Unlike files checked interactively against actual output this session
(graph_builder.py, groq_provider.py), this file's exact SDK shape
(attribute names on ClientError/ServerError, whether `.code` is really
where the HTTP status lives) is written from documented SDK behavior, not
verified against a live install. If an AttributeError surfaces on first
real use, that's the one part of this file to send back for a fix against
your actually-installed SDK version — same situation groq_provider.py's
model name just turned out to be.
"""

from __future__ import annotations

from typing import Optional

from google import genai
from google.genai import errors as genai_errors

from temporal_graphrag.config import settings
from temporal_graphrag.core.exceptions import (
    ConfigurationError,
    ServiceAuthenticationError,
    ServiceRateLimitError,
    ServiceUnavailableError,
)
from temporal_graphrag.llm.base import LLMMessage, LLMProvider, LLMResponse, with_llm_retries

__all__ = ["GeminiProvider"]

# Gemini calls the assistant's own turns "model", not "assistant" the way
# OpenAI/Groq (and this project's LLMMessage) do — translated once here.
# "system" isn't mapped: it's pulled out separately, see generate() below.
_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiProvider(LLMProvider):
    """Gemini-backed LLMProvider. Raises ConfigurationError immediately if
    GEMINI_API_KEY isn't set — same "fail at the point of use" policy
    groq_provider.py follows."""

    def __init__(self, model: Optional[str] = None) -> None:
        if not settings.gemini_configured:
            raise ConfigurationError(
                "gemini_api_key",
                hint="Set GEMINI_API_KEY in .env — get one free at https://aistudio.google.com/apikey.",
            )
        self.model = model or settings.gemini_default_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

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
        """Kept separate from generate() so with_llm_retries wraps just
        this method — same structure as groq_provider.py's _call()."""
        system_instruction = next((m.content for m in messages if m.role == "system"), None)
        contents = [
            {"role": _ROLE_MAP[m.role], "parts": [{"text": m.content}]}
            for m in messages
            if m.role in _ROLE_MAP
        ]

        config: dict[str, object] = {"temperature": temperature}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as e:
            status = getattr(e, "code", None)
            if status == 429:
                raise ServiceRateLimitError("Gemini", str(e)) from e
            if status in (401, 403):
                raise ServiceAuthenticationError("Gemini", str(e)) from e
            raise ServiceUnavailableError("Gemini", str(e)) from e
        except genai_errors.ServerError as e:
            raise ServiceUnavailableError("Gemini", str(e)) from e

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            content=response.text or "",
            model=self.model,
            provider="gemini",
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        )
        
    async def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        *,
        system: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Vision-capable generation — deliberately NOT part of the
        LLMProvider interface (base.py's own docstring: that contract is
        text-only, since Groq's current models here have no vision
        capability, and forcing an image parameter onto an interface only
        one implementation can honor would break the "every provider
        implements the same contract" promise). This exists because
        agents/multimodal_agent.py has a HARD requirement for Gemini's
        vision specifically, not a soft routing preference."""
        response = await self._call_with_image(
            prompt, image_bytes, mime_type, system=system, temperature=temperature, max_tokens=max_tokens
        )
        return response.content

    @with_llm_retries
    async def _call_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        *,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Mirrors _call()'s structure exactly (same error translation,
        same retry wrapping) — the only difference is an image part
        added to `contents`."""
        contents = [
            {
                "role": "user",
                "parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": image_bytes}}],
            }
        ]
        config: dict[str, object] = {"temperature": temperature}
        if system:
            config["system_instruction"] = system
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except genai_errors.ClientError as e:
            status = getattr(e, "code", None)
            if status == 429:
                raise ServiceRateLimitError("Gemini", str(e)) from e
            if status in (401, 403):
                raise ServiceAuthenticationError("Gemini", str(e)) from e
            raise ServiceUnavailableError("Gemini", str(e)) from e
        except genai_errors.ServerError as e:
            raise ServiceUnavailableError("Gemini", str(e)) from e

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            content=response.text or "",
            model=self.model,
            provider="gemini",
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        )