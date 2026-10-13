"""
core/exceptions.py

Custom exception hierarchy for the whole project. Every exception raised
on purpose inherits from TemporalGraphRAGError, so callers choose how
narrow to catch:

    except ServiceRateLimitError:   # just rate limits
    except ExternalServiceError:    # any external service problem
    except TemporalGraphRAGError:   # anything from our own code

Preserve the original error when wrapping one — the `from e` below isn't
decorative, it keeps the real traceback attached as __cause__:

    try:
        resp = httpx.post(f"{settings.grobid_url}/api/processFulltextDocument", ...)
    except httpx.ConnectError as e:
        raise ServiceUnavailableError(
            "GROBID", "connection refused — is the Docker container running?"
        ) from e

Deliberately NOT here yet: agent-specific errors (verification failures,
planning failures) and graph-construction-specific errors. Both are Weeks
7-14 territory — they'll be added here once those modules exist and it's
clear what actually needs its own exception type, rather than guessed at
now.

This file has zero dependencies outside the standard library, on purpose —
core/ shouldn't need pydantic, httpx, or anything else installed just to
define what can go wrong.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "TemporalGraphRAGError",
    "ConfigurationError",
    "NotFoundError",
    "DomainValidationError",
    "ParsingError",
    "ExternalServiceError",
    "ServiceUnavailableError",
    "ServiceAuthenticationError",
    "ServiceRateLimitError",
]


class TemporalGraphRAGError(Exception):
    """Base class for every exception this project raises on purpose.

    Catching this — rather than bare Exception — at API/agent boundaries
    means "something in OUR domain went wrong," distinct from unexpected
    bugs, third-party library internals, or the process being killed.
    Those warrant different handling than a bad arxiv_id does.
    """


class ConfigurationError(TemporalGraphRAGError):
    """A required setting isn't configured. This is the other half of the
    promise made in config/settings.py: every API key there defaults to
    None so the app can start without every credential filled in — THIS
    is what fires the moment something actually tries to use one that's
    still missing, with a clear message instead of a raw AttributeError
    or a silent None flowing three calls deep before failing confusingly.
    """

    def __init__(self, setting_name: str, *, hint: Optional[str] = None):
        self.setting_name = setting_name
        self.hint = hint
        message = f"Required setting '{setting_name}' is not configured."
        if hint:
            message += f" {hint}"
        super().__init__(message)


class NotFoundError(TemporalGraphRAGError):
    """A specific entity was looked up and doesn't exist — not for "the
    whole corpus is empty" type problems, which aren't really exceptional."""

    def __init__(self, entity_type: str, entity_id: str):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} not found: {entity_id!r}")


class DomainValidationError(TemporalGraphRAGError):
    """A business-rule violation, distinct from pydantic.ValidationError.
    Pydantic's fires for TYPE errors — wrong shape of data. This is for
    rules that hold even when the data IS well-typed, e.g. "a paper's v2
    date can't be earlier than its v1 date." Named distinctly (not just
    ValidationError) so it never shadows, or gets shadowed by, pydantic's
    own exception of the same short name in a file that imports both.
    """


class ParsingError(TemporalGraphRAGError):
    """A document, or part of one, couldn't be parsed into our domain
    model — a malformed PDF, GROBID returning something unexpected, empty
    output where content was expected. Carries arxiv_id so a batch run
    over all 40 pilot papers can log which one failed without the whole
    run stopping."""

    def __init__(self, arxiv_id: str, message: str):
        self.arxiv_id = arxiv_id
        super().__init__(f"[{arxiv_id}] {message}")


class ExternalServiceError(TemporalGraphRAGError):
    """Base for anything that goes wrong talking to a service we don't
    control: GROBID, Neo4j, Semantic Scholar, Groq, Gemini. Catch this
    specifically around external calls — not the bare TemporalGraphRAGError
    base — so a genuine bug in our own logic isn't silently treated the
    same as "the network hiccuped."""

    def __init__(self, service: str, message: str):
        self.service = service
        super().__init__(f"[{service}] {message}")


class ServiceUnavailableError(ExternalServiceError):
    """The service didn't respond at all: connection refused, timeout, DNS
    failure. Usually means GROBID's Docker container isn't running, Neo4j
    Desktop's DBMS is stopped, or there's no network."""


class ServiceAuthenticationError(ExternalServiceError):
    """The service responded but rejected our credentials (401/403) — not
    a connectivity problem. Distinct from ConfigurationError: that one
    fires when a key is missing entirely; this fires when a key IS set
    but the service won't accept it (rotated, revoked, wrong project —
    like the Semantic Scholar key earlier in this project)."""


class ServiceRateLimitError(ExternalServiceError):
    """The service rejected the call for rate limiting (HTTP 429). Carries
    `retry_after_seconds` when the service tells us how long to wait, so
    callers — and the tenacity retry decorators we'll add around Groq/
    Gemini calls — can back off intelligently instead of guessing. Directly
    relevant given how tight Gemini's free-tier RPM turned out to be."""

    def __init__(
        self,
        service: str,
        message: str,
        *,
        retry_after_seconds: Optional[float] = None,
    ):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(service, message)
