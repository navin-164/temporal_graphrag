"""
embeddings/table_embedder.py

Table embeddings via text serialization, reusing text_embedder.py's
SPECTER2 pipeline — not TAPAS, despite the original architecture diagram
naming it. Worth explaining why, since this deviates from that plan.

Why not TAPAS
------------------
TAPAS is real and well-documented, but its entire design — training,
tokenizer, position/token-type embeddings — is built around jointly
encoding a table WITH a natural-language question ("How many movies has
George Clooney played in?"). Every usage example found, official docs
included, follows that shape. Its base model does technically expose a
pooler_output that could serve as an embedding, so using it without a
question isn't impossible — but there's no confirmed evidence anyone
does that meaningfully, which is a deeper kind of uncertain than an
implementation-syntax question. Pulling in a whole new BERT-sized model
for a use case its own training never covered is a bad trade against
this project's own "lightweight fallbacks for heavy models" goal,
especially when a simpler, well-established alternative exists.

Text serialization + a general-purpose text encoder is a common pattern
for tables in RAG systems generally. Reusing text_embedder.py's SPECTER2
pipeline specifically (rather than loading a separate model) means one
fewer model in memory and one fewer thing to independently verify.

Falls back to caption alone if raw_data is empty
--------------------------------------------------------
table_extractor.py is itself honest that raw_data may or may not be
populated, depending on whether GROBID's cascaded table model actually
ran (see that file's docstring). serialize_table() below produces
something embeddable either way — full grid when available, caption
alone when it isn't — rather than nothing.
"""

from __future__ import annotations

from typing import Optional

from temporal_graphrag.core.entities import Table
from temporal_graphrag.embeddings.text_embedder import TextEmbedder


def serialize_table(table: Table) -> str:
    """Flattens a Table into clean text for embedding through a
    general-purpose text encoder, rather than a table-specific model.

    Assumes raw_data's first row is a header (matching how table_
    extractor.py's <row>/<cell> extraction naturally orders GROBID's
    output) — not independently verified here, since it inherits
    whatever uncertainty that file already documents about its own input.
    Empty rows (every cell blank after stripping) are skipped rather than
    contributing a blank line.
    """
    parts: list[str] = []
    if table.table_number:
        parts.append(table.table_number)
    if table.caption:
        parts.append(table.caption)

    if table.raw_data:
        header, *rows = table.raw_data
        if header and any(cell.strip() for cell in header):
            parts.append("Columns: " + ", ".join(header))
        for row in rows:
            if any(cell.strip() for cell in row):
                parts.append(" | ".join(row))

    return "\n".join(parts).strip()


class TableEmbedder:
    """Embeds tables via text serialization + text_embedder.py's SPECTER2
    pipeline.

    Accepts an existing TextEmbedder rather than always creating its own
    — a caller processing both papers and tables should share ONE loaded
    model between them; loading the same transformer weights twice wastes
    real memory and startup time for no benefit. Creates one internally
    only as a convenience for standalone use.

    Usage:
        text_embedder = TextEmbedder()                  # load once
        table_embedder = TableEmbedder(text_embedder)     # reuses it
        embeddings = table_embedder.embed_tables(paper.tables)
        for table, embedding in zip(paper.tables, embeddings):
            table.embedding = embedding  # None for any table that
                                           # serialized to nothing usable
    """

    def __init__(self, text_embedder: Optional[TextEmbedder] = None) -> None:
        self.text_embedder = text_embedder or TextEmbedder()

    def embed_tables(self, tables: list[Table]) -> list[Optional[list[float]]]:
        """Embeds each table's serialized text. Returns a list the same
        length as `tables`, in the same order — None at any position
        whose table serialized to an empty string (no caption AND no
        usable raw_data), so callers can always do
        `zip(tables, embed_tables(tables))` without the lists drifting
        out of alignment."""
        texts_to_embed: list[str] = []
        positions: list[int] = []
        for i, table in enumerate(tables):
            text = serialize_table(table)
            if text:
                texts_to_embed.append(text)
                positions.append(i)

        results: list[Optional[list[float]]] = [None] * len(tables)
        if texts_to_embed:
            embeddings = self.text_embedder.embed_texts(texts_to_embed)
            for position, embedding in zip(positions, embeddings):
                results[position] = embedding
        return results

    def embed_table(self, table: Table) -> Optional[list[float]]:
        """Single-table convenience wrapper around embed_tables()."""
        return self.embed_tables([table])[0]
