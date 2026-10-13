"""
knowledge_graph/graph_builder.py

Where schema.py's node labels and constraints actually get used: takes a
Paper (as loaded from parsed/*.json) and writes it — and everything
connected to it — into Neo4j.

One paper, one transaction
-------------------------------
build_paper() collects every statement needed for one paper's full
subgraph (the Paper node itself, its authors, venue, versions, sections,
paragraphs, figures, tables, equations, citations) and runs them all
through Neo4jClient.run_write_many() — one shared transaction, not one
per statement. A paper with 15 sections and 40 citations still only
touches the database as ONE atomic unit: either all of it lands, or none
of it does. Without that, a failure partway through would leave a Paper
node in the graph with some but not all of its Sections attached, which
is a genuinely confusing state to debug later.

Sections and paragraphs are two statements, not one nested UNWIND
------------------------------------------------------------------------
Cypher supports nested UNWIND (iterate a list, and for each item iterate
a sub-list), which would let sections+paragraphs be written in a single
statement. Deliberately not done that way here: this project has no live
Neo4j instance to test Cypher against directly, and a single-level UNWIND
per statement is much easier to reason about correctly without one than
a nested version would be. Paragraphs are flattened in Python instead
(each carrying its own section_id), and written as a second statement
that runs immediately after sections within the SAME transaction — order
is preserved by run_write_many(), so paragraphs' MATCH on their parent
Section is guaranteed to find it already there.

No embedding vectors written here
--------------------------------------
Text/figure/table embeddings exist in parsed data by the time this runs
(if Week 6's embedders have been run), but they aren't written into
Neo4j. The project's own architecture keeps these as two separate stores
for a reason — Neo4j for structure, FAISS (Week 10) for vector similarity
search — and duplicating potentially-hundreds-of-floats arrays as node
properties in a store that isn't optimized for vector search serves
neither store well. FAISS will look entities up by the same id/arxiv_id
already used here as the join key.

Dates passed as native Python objects, not ISO strings
-------------------------------------------------------------
paper.publication_date (a date) and version.created (a datetime) are
passed directly as query parameters rather than converted to strings
first. The Neo4j driver converts Python date/datetime objects to Neo4j's
own native temporal types automatically — storing them as strings would
work too, but would lose the ability to do actual date arithmetic/range
queries in Cypher (`WHERE p.publication_date > date(...)`) without
parsing them back out of a string first, which defeats a fair amount of
the point for a project whose whole premise is temporal.

Author fields use coalesce() to avoid overwriting good data
------------------------------------------------------------------
Two different papers can share an author (MERGE on Author.name — see
schema.py for why that's a known, deliberate simplification, not
canonicalization). If paper A supplies an affiliation and paper B (a
later write, same author name) does not, a plain SET would overwrite A's
real affiliation with null. `coalesce(new_value, existing_value)` keeps
whichever isn't null, so processing papers in any order doesn't
regress previously-good data on a shared node.

Citation resolution is a separate pass, not part of build_paper()
------------------------------------------------------------------------
A citation's target (cited_arxiv_id) might belong to a paper this
project hasn't processed yet, depending on what order the 40 papers get
built in. resolve_citations() runs once, after all papers are loaded (or
any time after — it's a MERGE, safe to re-run), and links any Citation
node whose cited_arxiv_id matches a Paper node that now exists.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any, Optional

from temporal_graphrag.core.entities import Paper
from temporal_graphrag.knowledge_graph.neo4j_client import Neo4jClient
from temporal_graphrag.knowledge_graph.schema import NodeLabels, RelationshipTypes

logger = logging.getLogger(__name__)

Statement = tuple[str, dict[str, Any]]


def _normalize_title(title: Optional[str]) -> str:
    """Lowercase, strip punctuation, collapse whitespace — enough to make
    "BERT: Pre-training of Deep..." and "bert pre training of deep..."
    compare as near-identical, without a full NLP pipeline for what's
    fundamentally a formatting-noise problem (see resolve_citations_by_
    title()'s own docstring for why this is needed at all)."""
    if not title:
        return ""
    text = title.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class GraphBuilder:
    """Writes Paper entities into Neo4j, using schema.py's labels and
    relationship types.

    Usage:
        async with Neo4jClient() as client:
            builder = GraphBuilder(client)
            await builder.build_paper(paper)   # once per paper
            ...
            resolved = await builder.resolve_citations()  # once, after all papers
    """

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    async def build_paper(self, paper: Paper) -> None:
        """Writes one paper's full subgraph as a single transaction."""
        statements = self._build_statements(paper)
        await self.client.run_write_many(statements)
        logger.info(
            "Wrote %s: %d authors, %d sections, %d paragraphs, %d figures, "
            "%d tables, %d equations, %d citations",
            paper.arxiv_id,
            len(paper.authors),
            len(paper.sections),
            sum(len(s.paragraphs) for s in paper.sections),
            len(paper.figures),
            len(paper.tables),
            len(paper.equations),
            len(paper.citations),
        )

    async def resolve_citations(self) -> int:
        """Links Citation nodes to actual Paper nodes where
        cited_arxiv_id matches one now in the graph. Safe to re-run any
        time — e.g. after adding more papers to the corpus later, to pick
        up citations that couldn't resolve before those papers existed."""
        query = f"""
        MATCH (c:{NodeLabels.CITATION}) WHERE c.cited_arxiv_id IS NOT NULL
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: c.cited_arxiv_id}})
        MERGE (c)-[:{RelationshipTypes.RESOLVES_TO}]->(p)
        RETURN count(*) AS resolved
        """
        rows = await self.client.run_write(query)
        resolved = rows[0]["resolved"] if rows else 0
        logger.info("Resolved %d citations to papers already in the graph", resolved)
        return resolved
    
    async def resolve_citations_by_title(self, papers: list[Paper], threshold: float = 0.85) -> int:
        """A second resolution pass, alongside resolve_citations()'s exact
        cited_arxiv_id match: fuzzy-matches a Citation's cited_title
        against every Paper's title in the corpus, for citations that
        never got a GROBID-tagged arXiv id in the first place.

        Why this needs to exist at all, separate from the exact-id pass
        -----------------------------------------------------------------
        cited_arxiv_id is only ever populated when GROBID recognizes a
        literal "arXiv:XXXX.XXXXX" string in the citing paper's own
        bibliography text (see citation_extractor.py). Authors citing an
        already-published paper overwhelmingly cite the conference/
        journal version instead — this project's own corpus confirmed
        real, unresolved citations to well-known papers (BERT,
        "Attention Is All You Need", etc.) with no arXiv id in the
        bibliography entry at all, even though the cited paper is
        plainly present in this project's own corpus. Exact-id matching
        alone systematically misses this whole category, regardless of
        how well-connected the corpus actually is.

        Why title similarity, not exact match
        --------------------------------------------
        Even normalized-exact title comparison found ZERO matches on
        this project's own corpus, while difflib.SequenceMatcher ratio
        >= 0.85 found real, correct matches at ratio 1.0 once case and
        punctuation differences were normalized away. GROBID's own title
        extraction introduces small formatting differences (line-wrap
        artifacts from the arXiv source, capitalization drift, stray
        whitespace) that exact matching is too brittle to survive even
        for the identical paper — a well-known entity-resolution
        problem, not a bug unique to this corpus.

        Why this runs entirely in Python, not Cypher
        ------------------------------------------------------
        Cypher has no built-in fuzzy string similarity without the APOC
        plugin — same reasoning graph_retriever.py already gives for why
        Personalized PageRank needs GDS and isn't built yet: nothing
        confirms APOC is installed here. difflib is Python's own
        standard library and needs no plugin; the corpus is small enough
        (tens of papers, thousands of citations) that an O(citations x
        papers) comparison in Python is genuinely fast.

        Only ONE write query, not one per match
        ----------------------------------------------
        Every (citation_id, arxiv_id) pair found by the Python-side
        matching is sent as a single UNWIND-based batch write — same
        reasoning build_paper() already gives for batching a paper's
        statements into one transaction rather than one round-trip per
        statement.
        """
        corpus_titles = [(_normalize_title(p.title), p.arxiv_id) for p in papers if p.title]

        matches: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for paper in papers:
            for citation in paper.citations:
                if citation.cited_arxiv_id:
                    continue  # already handled by resolve_citations()'s exact match
                normalized_cited = _normalize_title(citation.cited_title)
                if not normalized_cited or len(normalized_cited) < 8:
                    continue  # too short to compare meaningfully — avoids noisy false positives
                for normalized_corpus_title, arxiv_id in corpus_titles:
                    if not normalized_corpus_title or arxiv_id == paper.arxiv_id:
                        continue  # a paper can't cite itself
                    ratio = difflib.SequenceMatcher(None, normalized_cited, normalized_corpus_title).ratio()
                    if ratio >= threshold:
                        pair = (citation.id, arxiv_id)
                        if pair not in seen:
                            seen.add(pair)
                            matches.append({"citation_id": citation.id, "arxiv_id": arxiv_id})
                        break  # one match per citation is enough

        if not matches:
            logger.info("No additional title-based citation matches found")
            return 0

        query = f"""
        UNWIND $matches AS match
        MATCH (c:{NodeLabels.CITATION} {{id: match.citation_id}})
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: match.arxiv_id}})
        MERGE (c)-[:{RelationshipTypes.RESOLVES_TO}]->(p)
        RETURN count(*) AS resolved
        """
        rows = await self.client.run_write(query, matches=matches)
        resolved = rows[0]["resolved"] if rows else 0
        logger.info("Resolved %d additional citations via fuzzy title matching", resolved)
        return resolved

    # ---- Statement builders — each is pure: Paper in, (query, params) out.
    # Testable without a database, which is exactly how they're tested. ----

    def _build_statements(self, paper: Paper) -> list[Statement]:
        statements = [self._paper_statement(paper)]
        if paper.authors:
            statements.append(self._authors_statement(paper))
        if paper.venue:
            statements.append(self._venue_statement(paper))
        if paper.versions:
            statements.append(self._versions_statement(paper))
        if paper.sections:
            statements.append(self._sections_statement(paper))
            paragraphs_stmt = self._paragraphs_statement(paper)
            if paragraphs_stmt:
                statements.append(paragraphs_stmt)
        if paper.figures:
            statements.append(self._figures_statement(paper))
        if paper.tables:
            statements.append(self._tables_statement(paper))
        if paper.equations:
            statements.append(self._equations_statement(paper))
        if paper.citations:
            statements.append(self._citations_statement(paper))
        return statements

    def _paper_statement(self, paper: Paper) -> Statement:
        query = f"""
        MERGE (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        SET p.title = $title,
            p.abstract = $abstract,
            p.categories = $categories,
            p.doi = $doi,
            p.publication_date = $publication_date,
            p.citation_count = $citation_count,
            p.reference_count = $reference_count,
            p.fields_of_study = $fields_of_study,
            p.first_version_date = $first_version_date,
            p.latest_version_date = $latest_version_date
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "categories": paper.categories,
            "doi": paper.doi,
            "publication_date": paper.publication_date,
            "citation_count": paper.citation_count,
            "reference_count": paper.reference_count,
            "fields_of_study": paper.fields_of_study,
            "first_version_date": paper.first_version_date,
            "latest_version_date": paper.latest_version_date,
        }
        return query, params

    def _authors_statement(self, paper: Paper) -> Statement:
        query = f"""
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        UNWIND $authors AS author
        MERGE (a:{NodeLabels.AUTHOR} {{name: author.name}})
        SET a.affiliation = coalesce(author.affiliation, a.affiliation),
            a.orcid = coalesce(author.orcid, a.orcid)
        MERGE (p)-[:{RelationshipTypes.AUTHORED_BY}]->(a)
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "authors": [
                {"name": a.name, "affiliation": a.affiliation, "orcid": a.orcid}
                for a in paper.authors
            ],
        }
        return query, params

    def _venue_statement(self, paper: Paper) -> Statement:
        assert paper.venue is not None  # caller (_build_statements) already checked
        query = f"""
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        MERGE (v:{NodeLabels.VENUE} {{name: $venue_name}})
        SET v.venue_type = coalesce($venue_type, v.venue_type)
        MERGE (p)-[:{RelationshipTypes.PUBLISHED_IN}]->(v)
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "venue_name": paper.venue.name,
            "venue_type": paper.venue.venue_type,
        }
        return query, params

    def _versions_statement(self, paper: Paper) -> Statement:
        query = f"""
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        UNWIND $versions AS version
        MERGE (v:{NodeLabels.VERSION} {{id: version.id}})
        SET v.version = version.version, v.created = version.created
        MERGE (p)-[:{RelationshipTypes.HAS_VERSION}]->(v)
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "versions": [
                {
                    "id": f"{paper.arxiv_id}::version::{v.version}",
                    "version": v.version,
                    "created": v.created,
                }
                for v in paper.versions
            ],
        }
        return query, params

    def _sections_statement(self, paper: Paper) -> Statement:
        query = f"""
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        UNWIND $sections AS section
        MERGE (s:{NodeLabels.SECTION} {{id: section.id}})
        SET s.heading = section.heading,
            s.section_type = section.section_type,
            s.order = section.order
        MERGE (p)-[:{RelationshipTypes.HAS_SECTION}]->(s)
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "sections": [
                {
                    "id": s.id,
                    "heading": s.heading,
                    "section_type": s.section_type,
                    "order": s.order,
                }
                for s in paper.sections
            ],
        }
        return query, params

    def _paragraphs_statement(self, paper: Paper) -> Optional[Statement]:
        # Flattened here in Python (each paragraph carries its own
        # section_id) rather than nested UNWIND in Cypher — see module
        # docstring for why.
        flat = [
            {
                "id": para.id,
                "section_id": section.id,
                "text": para.text,
                "paragraph_index": para.paragraph_index,
            }
            for section in paper.sections
            for para in section.paragraphs
        ]
        if not flat:
            return None
        query = f"""
        UNWIND $paragraphs AS para
        MATCH (s:{NodeLabels.SECTION} {{id: para.section_id}})
        MERGE (pn:{NodeLabels.PARAGRAPH} {{id: para.id}})
        SET pn.text = para.text, pn.paragraph_index = para.paragraph_index
        MERGE (s)-[:{RelationshipTypes.HAS_PARAGRAPH}]->(pn)
        """
        return query, {"paragraphs": flat}

    def _figures_statement(self, paper: Paper) -> Statement:
        query = f"""
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        UNWIND $figures AS fig
        MERGE (f:{NodeLabels.FIGURE} {{id: fig.id}})
        SET f.figure_number = fig.figure_number,
            f.caption = fig.caption,
            f.page_number = fig.page_number,
            f.grobid_ref_id = fig.grobid_ref_id,
            f.image_path = fig.image_path
        MERGE (p)-[:{RelationshipTypes.HAS_FIGURE}]->(f)
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "figures": [
                {
                    "id": f.id,
                    "figure_number": f.figure_number,
                    "caption": f.caption,
                    "page_number": f.page_number,
                    "grobid_ref_id": f.grobid_ref_id,
                    "image_path": f.image_path,
                }
                for f in paper.figures
            ],
        }
        return query, params

    def _tables_statement(self, paper: Paper) -> Statement:
        query = f"""
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        UNWIND $tables AS tbl
        MERGE (t:{NodeLabels.TABLE} {{id: tbl.id}})
        SET t.table_number = tbl.table_number,
            t.caption = tbl.caption,
            t.page_number = tbl.page_number,
            t.grobid_ref_id = tbl.grobid_ref_id,
            t.image_path = tbl.image_path,
            t.raw_data_json = tbl.raw_data_json
        MERGE (p)-[:{RelationshipTypes.HAS_TABLE}]->(t)
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "tables": [
                {
                    "id": t.id,
                    "table_number": t.table_number,
                    "caption": t.caption,
                    "page_number": t.page_number,
                    "grobid_ref_id": t.grobid_ref_id,
                    "image_path": t.image_path,
                    "raw_data_json": json.dumps(t.raw_data) if t.raw_data else None,
                }
                for t in paper.tables
            ],
        }
        return query, params

    def _equations_statement(self, paper: Paper) -> Statement:
        query = f"""
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        UNWIND $equations AS eq
        MERGE (e:{NodeLabels.EQUATION} {{id: eq.id}})
        SET e.latex = eq.latex,
            e.equation_number = eq.equation_number,
            e.inline = eq.inline,
            e.grobid_ref_id = eq.grobid_ref_id
        MERGE (p)-[:{RelationshipTypes.HAS_EQUATION}]->(e)
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "equations": [
                {
                    "id": e.id,
                    "latex": e.latex,
                    "equation_number": e.equation_number,
                    "inline": e.inline,
                    "grobid_ref_id": e.grobid_ref_id,
                }
                for e in paper.equations
            ],
        }
        return query, params

    def _citations_statement(self, paper: Paper) -> Statement:
        query = f"""
        MATCH (p:{NodeLabels.PAPER} {{arxiv_id: $arxiv_id}})
        UNWIND $citations AS cit
        MERGE (c:{NodeLabels.CITATION} {{id: cit.id}})
        SET c.raw_text = cit.raw_text,
            c.cited_title = cit.cited_title,
            c.cited_authors = cit.cited_authors,
            c.cited_year = cit.cited_year,
            c.cited_arxiv_id = cit.cited_arxiv_id,
            c.cited_doi = cit.cited_doi,
            c.grobid_ref_id = cit.grobid_ref_id
        MERGE (p)-[:{RelationshipTypes.CITES}]->(c)
        """
        params = {
            "arxiv_id": paper.arxiv_id,
            "citations": [
                {
                    "id": c.id,
                    "raw_text": c.raw_text,
                    "cited_title": c.cited_title,
                    "cited_authors": c.cited_authors,
                    "cited_year": c.cited_year,
                    "cited_arxiv_id": c.cited_arxiv_id,
                    "cited_doi": c.cited_doi,
                    "grobid_ref_id": c.grobid_ref_id,
                }
                for c in paper.citations
            ],
        }
        return query, params
