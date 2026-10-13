"""Core domain layer — pure entities and the exception hierarchy, no I/O.

Note on imports: only exceptions are re-exported here, not entities.
`temporal_graphrag.core.exceptions` has zero third-party dependencies by
design (see that file's docstring) — if this __init__.py also eagerly
imported entities.py (which needs pydantic), that guarantee would quietly
break, since importing ANY submodule of a package always runs the
package's __init__.py first regardless of which submodule you asked for.
So: `from temporal_graphrag.core import TemporalGraphRAGError` works with
nothing but the standard library. Entities need one level more explicit:
`from temporal_graphrag.core.entities import Paper`.
"""

from .exceptions import (
    ConfigurationError,
    DomainValidationError,
    ExternalServiceError,
    NotFoundError,
    ParsingError,
    ServiceAuthenticationError,
    ServiceRateLimitError,
    ServiceUnavailableError,
    TemporalGraphRAGError,
)

__all__ = [
    "ConfigurationError",
    "DomainValidationError",
    "ExternalServiceError",
    "NotFoundError",
    "ParsingError",
    "ServiceAuthenticationError",
    "ServiceRateLimitError",
    "ServiceUnavailableError",
    "TemporalGraphRAGError",
]
