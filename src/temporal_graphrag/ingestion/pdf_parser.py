"""
ingestion/pdf_parser.py

Async client for a running GROBID service — sends a PDF, gets back GROBID's
structural parse as a TEI XML tree. This is Layer 2.1 in the architecture
diagram, and its ONLY job is "talk to GROBID and hand back a parsed tree."
Turning that tree into Section/Paragraph/Citation entities is
text_extractor.py and citation_extractor.py's job (next couple of files) —
kept separate so a change to, say, how we detect section headings doesn't
touch the HTTP/retry logic in here, and vice versa.

Why we don't extract title/authors/abstract from GROBID's output
------------------------------------------------------------------
We already have all three, cleanly, from pilot_corpus_metadata.json
(Week 2) — arXiv's own metadata is more reliable than a PDF-layout guess
for exactly these fields. GROBID's header-parsing exists for corpora that
DON'T already have trustworthy metadata; ours does.

Consolidation is off by default, on purpose
----------------------------------------------
GROBID can call out to CrossRef/biblio-glutton (`consolidateHeader`,
`consolidateCitations`) to improve metadata accuracy. We don't enable it:
GROBID's own docs recommend raising the timeout to 200-600s when
consolidation is on, and warn that citation consolidation caps throughput
around one document per second because of CrossRef's own rate limit.
Semantic Scholar already gives us citation data (02_enrich_citation_
graph.py), so consolidation would cost real time for no benefit here.
Left as constructor parameters, not deleted, in case a specific later use
case genuinely needs it.

GROBID's 503 is not a hard failure
-------------------------------------
GROBID's own docs: a 503 usually just means all of GROBID's worker
threads are currently busy, and recommend waiting 5-10 seconds and
resending — not "something is broken." Modeled as ServiceRateLimitError
(not ServiceUnavailableError) specifically so it carries
retry_after_seconds, and wired to tenacity (already in requirements.txt)
so retrying is a decorator, not hand-rolled recursion. A connection
failure — GROBID not running at all — is NOT retried the same way: that
won't fix itself in 10 seconds, so it fails fast as
ServiceUnavailableError instead of silently stalling for half a minute.

Usage
-----
    async with GrobidClient() as client:
        if not await client.is_alive():
            raise RuntimeError("GROBID isn't reachable — start the Docker container first")
        for pdf_path in pdf_paths:
            tei_tree = await client.process_fulltext_document(pdf_path=pdf_path)
            # tei_tree is a parsed lxml element, ready for text_extractor.py etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx
import tenacity
from lxml import etree

from temporal_graphrag.config import settings
from temporal_graphrag.core.exceptions import (
    ParsingError,
    ServiceRateLimitError,
    ServiceUnavailableError,
)

# Plain stdlib logging, not structlog, even though structlog is in
# requirements.txt: utils/logging_config.py (not built yet) is what decides
# how structlog is wired up, and a well-configured structlog setup usually
# routes stdlib logging through it anyway. Using structlog directly here
# would bake in an assumption about a module that doesn't exist yet.
logger = logging.getLogger(__name__)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# GROBID's own docs recommend this specific wait for processFulltextDocument
# on a 503 (worker threads busy) -- not an arbitrary guess.
GROBID_BUSY_RETRY_SECONDS = 10.0


class GrobidClient:
    """Async client for a single running GROBID instance.

    Holds one shared httpx.AsyncClient (connection pool) for the lifetime
    of the `async with` block, rather than opening a new connection per
    PDF — matters once 03_parse_documents.py loops over all 40 pilot papers
    through the same client instance.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_seconds: float = 120.0,
        consolidate_header: bool = False,
        consolidate_citations: bool = False,
    ) -> None:
        self.base_url = (base_url or settings.grobid_url).rstrip("/")
        # GROBID's API takes these as the strings "0"/"1", not booleans.
        self._consolidate_header = "1" if consolidate_header else "0"
        self._consolidate_citations = "1" if consolidate_citations else "0"
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def __aenter__(self) -> "GrobidClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.aclose()

    async def is_alive(self) -> bool:
        """Hits GROBID's /api/isalive endpoint. Use this once before looping
        over the pilot corpus, so a stopped Docker container fails fast and
        clearly instead of surfacing as 40 confusing timeouts."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/isalive", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(ServiceRateLimitError),
        wait=tenacity.wait_fixed(GROBID_BUSY_RETRY_SECONDS),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,  # on final failure, raise OUR ServiceRateLimitError,
                        # not tenacity's wrapper RetryError
    )
    async def process_fulltext_document(
        self,
        *,
        pdf_path: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        arxiv_id: Optional[str] = None,
    ) -> etree._Element:
        """Send one PDF to GROBID and return the parsed TEI XML tree.

        Pass exactly one of `pdf_path` / `pdf_bytes`. `arxiv_id` is optional
        and only used to make error messages traceable during a batch run —
        it's never sent to GROBID itself.
        """
        if (pdf_path is None) == (pdf_bytes is None):
            raise ValueError("Pass exactly one of pdf_path or pdf_bytes.")

        label = arxiv_id or (pdf_path if pdf_path else "<in-memory PDF>")

        if pdf_path is not None:
            # Deliberately NOT caught/wrapped: FileNotFoundError is already
            # exactly as clear as a custom exception would be here, and this
            # is a caller-usage error, not an external-service or parsing one.
            pdf_bytes = Path(pdf_path).read_bytes()

        # Everything goes through `files=` as a list of tuples — including
        # the plain (non-file) fields, via httpx's documented (None, value)
        # convention for "this is a form field, not a real file upload".
        # This is NOT the original design: an earlier version put the plain
        # fields in a separate `data=` list of tuples, which threw a
        # confusing "Attempted to send a sync request with an AsyncClient
        # instance" error — misleading, since it's not actually a sync/async
        # problem at all. An httpx maintainer confirms on GitHub that this
        # exact message is a generic fallback whenever httpx can't encode
        # what was passed to it; `data=` as a list of tuples combined with
        # `files=` simply isn't the tested code path — a dict is. Routing
        # everything through `files=` instead (which DOES officially support
        # both a list of tuples for repeated keys, and (None, value) for
        # plain fields) avoids that path entirely. GROBID's teiCoordinates
        # parameter needs repeating once per structure type (per its own
        # curl examples: --form teiCoordinates=figure --form teiCoordinates=
        # table) — figure_extractor.py and table_extractor.py need these
        # coordinates to know WHERE on the page each figure/table sits.
        files = [
            ("input", ("document.pdf", pdf_bytes, "application/pdf")),
            ("consolidateHeader", (None, self._consolidate_header)),
            ("consolidateCitations", (None, self._consolidate_citations)),
            ("teiCoordinates", (None, "figure")),
            ("teiCoordinates", (None, "table")),
        ]

        try:
            response = await self._client.post(
                f"{self.base_url}/api/processFulltextDocument",
                files=files,
            )
        except httpx.ConnectError as e:
            raise ServiceUnavailableError(
                "GROBID",
                f"connection refused at {self.base_url} — is the Docker container "
                f"running? (docker run --rm --init -p 8070:8070 grobid/grobid:<tag>-crf)",
            ) from e
        except httpx.TimeoutException as e:
            raise ServiceUnavailableError(
                "GROBID", f"timed out processing {label}"
            ) from e

        if response.status_code == 503:
            # Caught by the @tenacity.retry decorator above; after 3 total
            # attempts this becomes the error the caller actually sees.
            raise ServiceRateLimitError(
                "GROBID",
                f"still busy processing {label}",
                retry_after_seconds=GROBID_BUSY_RETRY_SECONDS,
            )

        if response.status_code != 200:
            raise ParsingError(
                str(label),
                f"GROBID returned HTTP {response.status_code}: {response.text[:300]}",
            )

        if not response.content.strip():
            raise ParsingError(str(label), "GROBID returned a 200 but an empty response body")

        try:
            tree = etree.fromstring(response.content)
        except etree.XMLSyntaxError as e:
            raise ParsingError(str(label), f"GROBID's response wasn't valid XML: {e}") from e

        body_paragraphs = tree.findall(".//tei:text/tei:body//tei:p", namespaces=TEI_NS)
        if not body_paragraphs:
            # A 200 with essentially no body text usually means GROBID
            # couldn't make sense of this specific PDF (scanned image,
            # corrupted file, unusual layout) — a parsing problem, not a
            # connectivity one, so ParsingError rather than a service error.
            raise ParsingError(
                str(label),
                "GROBID returned 200 but the parsed body has no paragraphs — likely "
                "a scanned/image-only PDF or a layout GROBID couldn't segment.",
            )

        logger.info("GROBID parsed %s: %d paragraphs found", label, len(body_paragraphs))
        return tree
