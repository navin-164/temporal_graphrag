"""
llm/factory.py

Routes an LLM call to Groq or Gemini based on estimated prompt size —
the piece settings.py's own llm_groq_token_overflow_threshold field
anticipated ("Prompts above this estimated token count route to Gemini
instead of Groq"). The last file in llm/, closing out the provider layer
Week 12's agents build on.

Why a character-count heuristic, not a real tokenizer
------------------------------------------------------------
Groq serves Llama/GPT-OSS models and Gemini serves its own family — two
different tokenizers, neither of which this project has a reason to
depend on just to make a ROUTING decision (not a billing-accuracy one).
tiktoken (OpenAI's tokenizer) isn't even the right vocabulary for either
model actually being called here, and isn't in requirements.txt for
exactly that reason. The widely-used "~4 characters per token" rule of
thumb for English text is dependency-free, fast, and precise enough for
its one job: deciding which side of a 5,000-token threshold a prompt
roughly falls on, not billing anyone by the token.

Why routing looks at every message, not just the latest prompt
--------------------------------------------------------------------
A multi-turn conversation's total size is what actually risks overflowing
Groq's context window, not just the newest message — so estimate_tokens()
sums every LLMMessage's content, matching what actually gets sent over
the wire in generate().

Why `prefer` exists as an override, not just automatic routing
--------------------------------------------------------------------
The architecture diagram assigns the Multimodal Understanding agent to
Gemini specifically (Google's multimodal support, not Groq's text-only
models) — that's a routing decision based on WHAT the agent needs, not
how big its prompt happens to be. `prefer` lets a caller like that skip
size-based routing entirely and just say which provider it needs, while
every other agent can leave `prefer` unset and get automatic
threshold-based routing.

Provider instances are cached, not rebuilt per call
--------------------------------------------------------
Constructing a GroqProvider/GeminiProvider creates a real SDK client
underneath (groq.AsyncGroq / genai.Client) — reusing one instance per
provider across every call this factory ever routes is strictly better
than paying that construction cost (and re-validating the API key) on
every single generate() call. Same lazy-singleton shape VectorRetriever
already uses for its embedders.

Falling back when only one provider is configured
--------------------------------------------------------
A prompt over threshold with only GROQ_API_KEY set shouldn't hard-fail
just because Gemini would have been preferred — it should fall back to
whichever provider IS actually usable. ConfigurationError only fires when
NEITHER is configured, since at that point there's genuinely nothing this
factory can route to.
"""

from __future__ import annotations

from typing import Optional

from temporal_graphrag.config import settings
from temporal_graphrag.core.exceptions import ConfigurationError
from temporal_graphrag.llm.base import LLMMessage, LLMProvider
from temporal_graphrag.llm.gemini_provider import GeminiProvider
from temporal_graphrag.llm.groq_provider import GroqProvider

__all__ = ["LLMFactory", "estimate_tokens", "llm_factory"]

# ~4 characters per token for English text — see module docstring for why
# this heuristic, not a real tokenizer, is the right tool for a routing
# decision rather than a billing-accuracy one.
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(messages: list[LLMMessage]) -> int:
    """Rough token count across every message's content combined — see
    module docstring for why this is a heuristic, and why it sums the
    whole conversation rather than just the latest turn."""
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // CHARS_PER_TOKEN_ESTIMATE


class LLMFactory:
    """Chooses between Groq and Gemini per call, and caches whichever
    provider instances actually get used so repeated calls don't pay SDK-
    client construction cost twice.

    Usage:
        factory = LLMFactory()

        provider = factory.get_provider(messages)                   # automatic, size-based routing
        provider = factory.get_provider(messages, prefer="gemini")  # explicit override
        text = await provider.generate_text("...")
    """

    def __init__(self) -> None:
        self._groq: Optional[GroqProvider] = None
        self._gemini: Optional[GeminiProvider] = None

    def get_provider(
        self,
        messages: list[LLMMessage],
        *,
        prefer: Optional[str] = None,
    ) -> LLMProvider:
        """Returns a GroqProvider or GeminiProvider for this call.

        `prefer`, if given, must be "groq" or "gemini" and skips
        size-based routing entirely (see module docstring — this is for
        agents with a genuine reason to need a specific provider, like
        Multimodal Understanding needing Gemini regardless of prompt
        size). Without `prefer`, routing is automatic: prompts at or
        under settings.llm_groq_token_overflow_threshold go to Groq,
        larger ones go to Gemini — falling back to whichever provider is
        actually configured if the "wanted" one for a given call isn't
        available.
        """
        if prefer is not None:
            if prefer not in ("groq", "gemini"):
                raise ValueError(f"prefer must be 'groq', 'gemini', or None — got {prefer!r}")
            wants_groq = prefer == "groq"
        else:
            estimated = estimate_tokens(messages)
            wants_groq = estimated <= settings.llm_groq_token_overflow_threshold

        if wants_groq and settings.groq_configured:
            return self._get_groq()
        if not wants_groq and settings.gemini_configured:
            return self._get_gemini()

        # The "wanted" provider for this call isn't configured — fall
        # back to whichever one IS, rather than failing a request that
        # could still succeed elsewhere (see module docstring).
        if settings.groq_configured:
            return self._get_groq()
        if settings.gemini_configured:
            return self._get_gemini()

        raise ConfigurationError(
            "groq_api_key / gemini_api_key",
            hint="Set at least one of GROQ_API_KEY or GEMINI_API_KEY in .env — both are free.",
        )

    def _get_groq(self) -> GroqProvider:
        if self._groq is None:
            self._groq = GroqProvider()
        return self._groq

    def _get_gemini(self) -> GeminiProvider:
        if self._gemini is None:
            self._gemini = GeminiProvider()
        return self._gemini


# Module-level default instance — same convenience-singleton pattern
# config/settings.py already uses for `settings`, so agents (Week 12+)
# can `from temporal_graphrag.llm.factory import llm_factory` without
# constructing their own LLMFactory each time.
llm_factory = LLMFactory()