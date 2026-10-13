"""
knowledge_graph/neo4j_client.py

Async client for Neo4j — the connection/query layer, playing the same
role for the knowledge graph that GrobidClient plays for ingestion: talk
to the external service correctly first, let content-specific logic
(graph_builder.py, next) build on top of it. This file has no knowledge
of Paper/Section/Citation or any Cypher schema — just "run this query,
get plain dicts back," the same way GrobidClient just gets a TEI tree.

Verified against the driver actually installed, not assumed
------------------------------------------------------------------
pip install put neo4j 6.2.0 on your machine — a major-version jump from
the >=5.22 floor originally set in requirements.txt. Before writing
async connection code, checked whether v6 renamed or removed anything
this file depends on: it hasn't. `execute_read`/`execute_write` remain
the current, documented pattern for v6.2 specifically (not superseded by
something else), and the driver's own upgrade notes state "driver
upgrades within a major version will never contain breaking API changes"
for the 6.x series — 6.0 to 6.2 is exactly that kind of upgrade. The
`neo4j-driver` package name deprecation mentioned throughout Neo4j's own
6.0 docs doesn't apply here either — we're already on the correct `neo4j`
package name, not the old one.

Why execute_read/execute_write, not the newer execute_query()
--------------------------------------------------------------------
The driver also offers a simpler one-shot `driver.execute_query(...)`
that skips manual session handling — genuinely simpler for a single throw-
away query. Not used here on purpose: it loads a query's entire result
into memory before returning (Neo4j's own docs contrast this as "eager"
against execute_read's "lazy" streaming), which is the wrong default for
a knowledge graph that Week 7-8 intends to make CENTRAL, per this
project's own Scope Map — some future query here (fetching a large
neighborhood, or the whole citation graph) could plausibly return enough
rows that eager-loading everything at once isn't the right default to
have picked. Sticking to one query pattern throughout, rather than mixing
both APIs depending on the query, keeps this file simpler to reason about
too.

Errors mapped to this project's own exception hierarchy
--------------------------------------------------------------
neo4j.exceptions.ServiceUnavailable and neo4j.exceptions.AuthError (the
driver's own documented exception types for "can't reach it" and "bad
credentials") get wrapped into this project's ServiceUnavailableError and
ServiceAuthenticationError, the same pattern pdf_parser.py already
established for GROBID — so callers catching our own exception hierarchy
don't also need to know Neo4j's specific exception names.
"""

from __future__ import annotations

from typing import Any, Optional

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from temporal_graphrag.config import settings
from temporal_graphrag.core.exceptions import (
    ConfigurationError,
    ServiceAuthenticationError,
    ServiceUnavailableError,
)


async def _run_and_collect(tx: Any, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """The transaction function execute_read()/execute_write() actually
    calls. `.data()` converts Neo4j's own Node/Relationship/Path types
    into plain dicts/lists — callers of this file work with ordinary
    Python data, never Neo4j-specific types."""
    result = await tx.run(query, params)
    return [record.data() async for record in result]


async def _run_many(tx: Any, statements: list[tuple[str, dict[str, Any]]]) -> None:
    """The transaction function run_write_many() uses — runs every
    statement against the SAME transaction object `tx`, so they all
    commit or roll back together rather than each getting its own."""
    for query, params in statements:
        await tx.run(query, params)


class Neo4jClient:
    """Async client for a single Neo4j database.

    Usage:
        async with Neo4jClient() as client:
            if not await client.verify_connectivity():
                raise RuntimeError("Neo4j isn't reachable — is the DBMS started in Neo4j Desktop?")
            rows = await client.run_write(
                "MERGE (p:Paper {arxiv_id: $arxiv_id}) RETURN p", arxiv_id="2003.08964"
            )
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database

        if not self.password:
            # The "fail late and loud" half of config/settings.py's design —
            # NEO4J_PASSWORD defaults to None there specifically so the app
            # can start without it; THIS is the point of use that needs it.
            raise ConfigurationError(
                "NEO4J_PASSWORD",
                hint="Start the DBMS in Neo4j Desktop, create a database if you haven't, "
                "then put its password in .env.",
            )

        self._driver = AsyncGraphDatabase.driver(self.uri, auth=(self.username, self.password))

    async def __aenter__(self) -> "Neo4jClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._driver.close()

    async def verify_connectivity(self) -> bool:
        """Confirms the database is actually reachable — use this once
        before a batch of writes, so a stopped DBMS fails fast and
        clearly instead of surfacing as a confusing error partway through
        building the graph."""
        try:
            await self._driver.verify_connectivity()
            return True
        except (ServiceUnavailable, AuthError):
            return False

    async def run_read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Runs a read-only Cypher query, returning each result row as a
        plain dict. Keyword arguments become Cypher query parameters
        (`$name` in the query maps to `name=...` here)."""
        return await self._run(read=True, query=query, params=params)

    async def run_write(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Runs a write Cypher query (CREATE/MERGE/SET/DELETE), returning
        each result row as a plain dict. Same parameter convention as
        run_read()."""
        return await self._run(read=False, query=query, params=params)

    async def run_write_many(self, statements: list[tuple[str, dict[str, Any]]]) -> None:
        """Runs several write statements in ONE transaction — all commit
        together, or none do. graph_builder.py needs this: writing one
        paper's full subgraph is naturally several statements (the Paper
        node, its authors, its sections, its figures...), and without a
        shared transaction, a failure partway through would leave that
        paper's data in the graph incomplete — a Paper node with some but
        not all of its Sections attached, say — rather than either fully
        present or fully absent. `statements` is a list of (query, params)
        pairs, executed in the given order.
        """
        async with self._driver.session(database=self.database) as session:
            try:
                await session.execute_write(_run_many, statements)
            except ServiceUnavailable as e:
                raise ServiceUnavailableError(
                    "Neo4j", f"couldn't reach {self.uri} — is the DBMS started in Neo4j Desktop?"
                ) from e
            except AuthError as e:
                raise ServiceAuthenticationError(
                    "Neo4j", f"authentication rejected for user '{self.username}': {e}"
                ) from e

    async def _run(self, *, read: bool, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self._driver.session(database=self.database) as session:
            execute = session.execute_read if read else session.execute_write
            try:
                return await execute(_run_and_collect, query, params)
            except ServiceUnavailable as e:
                raise ServiceUnavailableError(
                    "Neo4j", f"couldn't reach {self.uri} — is the DBMS started in Neo4j Desktop?"
                ) from e
            except AuthError as e:
                raise ServiceAuthenticationError(
                    "Neo4j", f"authentication rejected for user '{self.username}': {e}"
                ) from e
