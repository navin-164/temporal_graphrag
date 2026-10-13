"""
knowledge_graph/schema.py

Node labels, relationship types, and constraints for the temporal
knowledge graph — the file the Scope Map called "CORE — CENTRAL, protect
this time above all else." Constraint syntax verified against the
official current Cypher Manual before writing anything here, not assumed
from memory: `CREATE CONSTRAINT name IF NOT EXISTS FOR (n:Label) REQUIRE
n.property IS UNIQUE` is confirmed current for the 5.x/6.x series your
2026.06.0 instance is part of.

Scope: the structural graph, not yet the semantic layer
------------------------------------------------------------
This defines nodes for everything core/entities.py already models —
Paper, Author, Venue, Section, Paragraph, Figure, Table, Equation,
Citation — plus a new Version node (see below). It deliberately does NOT
define Concept, Method, Dataset, or Topic nodes, or the semantic edges
between them (explains/uses/improves/extends/contradicts/similar_to)
from the original diagram. Those require relation extraction — reading
papers and inferring "this method improves on that one" — which is NLP
work core/entities.py already deferred for the same reason, back when it
was written: there's no relation_extractor.py yet to populate them, and
guessing at that schema before the extraction logic exists would mean
designing blind. This file covers what we can build FROM data we
already, concretely have. The semantic layer is a real, separate
addition once relation_extractor.py exists — not abandoned, just not
guessed at now.

Why Version is its own node, not a Paper property
--------------------------------------------------------
Neo4j property values must be primitives or flat arrays of primitives —
not arrays of objects. Paper.versions is a list of {version, created}
pairs, which can't be stored as a single property value. Modeling each
version as its own :Version node, linked via HAS_VERSION, is also the
more useful choice for a project whose entire premise is temporal: it
lets Cypher directly query version history ("papers with 3+ revisions,"
"the version closest to a given date") rather than requiring every such
query to parse a flattened string first.

Version's uniqueness constraint is on a single `id` string
(f"{arxiv_id}::version::{version_label}", built the same way graph_
builder.py builds it), not a composite key on (paper_arxiv_id, version).
An earlier version of this file had no constraint on Version at all —
composite uniqueness constraints exist in Neo4j but are Enterprise-only,
and this project has no guarantee of running against anything but the
free Community Edition (Neo4j Desktop, in this case), so depending on one
would risk the constraint silently failing to apply. A single-property id
string sidesteps the question entirely, and matches the exact pattern
every other synthetic-id entity here already uses.

Why Citation stays a node, with an optional resolved link
----------------------------------------------------------------
A citation's target is usually NOT in this 40-paper pilot corpus — most
references point outside it. Modeling Citation as its own node (CITES
from the citing Paper) preserves every citation's metadata (title,
authors, year) regardless of whether the target is resolvable. When
cited_arxiv_id DOES match a Paper actually in the graph, a separate
RESOLVES_TO edge links them — graph traversal can then hop citing paper
-> citation -> cited paper for the subset where that's possible, without
requiring it for the rest.

Why Author has no uniqueness constraint
--------------------------------------------
Two different real people can share a rendered name ("J. Smith" is not
enough to assume one person). Enforcing uniqueness on Author.name would
bake in a wrong assumption before any real name-canonicalization logic
exists (ORCID matching or similar) — that's graph_builder.py's problem to
solve deliberately, not this file's to assume away. Author still gets a
plain (non-unique) index, since it'll be looked up by name often.

Table.raw_data isn't stored as graph structure
------------------------------------------------------
Same "no nested arrays" limitation as Version — raw_data is a grid
(list of lists), which Neo4j property values can't represent directly,
and modeling every cell as its own node would be real structural
overkill for data that's already fully preserved, exactly as extracted,
in parsed/*.json. This graph stores a JSON-serialized string of it
(raw_data_json) for reference/display, not a queryable cell-by-cell
structure — nothing here currently needs to query into individual cells.
"""

from __future__ import annotations

import logging

from temporal_graphrag.knowledge_graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class NodeLabels:
    """Namespaced string constants — avoids magic strings scattered
    across graph_builder.py and future retrieval/query code."""

    PAPER = "Paper"
    AUTHOR = "Author"
    VENUE = "Venue"
    VERSION = "Version"
    SECTION = "Section"
    PARAGRAPH = "Paragraph"
    FIGURE = "Figure"
    TABLE = "Table"
    EQUATION = "Equation"
    CITATION = "Citation"


class RelationshipTypes:
    HAS_VERSION = "HAS_VERSION"
    AUTHORED_BY = "AUTHORED_BY"
    PUBLISHED_IN = "PUBLISHED_IN"
    HAS_SECTION = "HAS_SECTION"
    HAS_PARAGRAPH = "HAS_PARAGRAPH"
    HAS_FIGURE = "HAS_FIGURE"
    HAS_TABLE = "HAS_TABLE"
    HAS_EQUATION = "HAS_EQUATION"
    CITES = "CITES"
    RESOLVES_TO = "RESOLVES_TO"  # Citation -> Paper, only when cited_arxiv_id
                                   # matches a Paper actually in the graph


# Every entity with a synthetic composite id (everything except Paper,
# Author, Venue) gets a uniqueness constraint on that id — these ids are
# already designed to be deterministic (Section.build_id() and friends,
# from core/entities.py), so re-running graph_builder.py's MERGE-based
# writes stays idempotent rather than creating duplicate nodes.
_UNIQUE_CONSTRAINTS: list[tuple[str, str, str]] = [
    # (constraint_name, label, property)
    ("paper_arxiv_id_unique", NodeLabels.PAPER, "arxiv_id"),
    ("venue_name_unique", NodeLabels.VENUE, "name"),
    ("version_id_unique", NodeLabels.VERSION, "id"),
    ("section_id_unique", NodeLabels.SECTION, "id"),
    ("paragraph_id_unique", NodeLabels.PARAGRAPH, "id"),
    ("figure_id_unique", NodeLabels.FIGURE, "id"),
    ("table_id_unique", NodeLabels.TABLE, "id"),
    ("equation_id_unique", NodeLabels.EQUATION, "id"),
    ("citation_id_unique", NodeLabels.CITATION, "id"),
]

# Author is deliberately absent above (see module docstring) — indexed,
# not constrained, since it's looked up by name often but must allow
# duplicates until real canonicalization exists.
_INDEXES: list[tuple[str, str, str]] = [
    # (index_name, label, property)
    ("author_name_index", NodeLabels.AUTHOR, "name"),
    ("paper_publication_date_index", NodeLabels.PAPER, "publication_date"),
    ("version_created_index", NodeLabels.VERSION, "created"),
]


def constraint_statements() -> list[str]:
    """The CREATE CONSTRAINT statements, as plain Cypher strings —
    IF NOT EXISTS makes every one safe to re-run without erroring on
    a constraint that's already there."""
    return [
        f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        for name, label, prop in _UNIQUE_CONSTRAINTS
    ]


def index_statements() -> list[str]:
    """The CREATE INDEX statements, as plain Cypher strings."""
    return [
        f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
        for name, label, prop in _INDEXES
    ]


async def apply_schema(client: Neo4jClient) -> None:
    """Runs every constraint and index statement against a real database.
    Safe to call repeatedly (IF NOT EXISTS on every statement) — this is
    meant to run once at the start of graph_builder.py's batch, not to be
    guarded behind a "has this run before" check of its own.
    """
    statements = constraint_statements() + index_statements()
    logger.info("Applying %d schema statements (constraints + indexes)...", len(statements))
    for statement in statements:
        await client.run_write(statement)
    logger.info("Schema applied.")
