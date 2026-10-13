"""Local embedding models — no API calls, no rate limits. Everything here
loads once and runs on your own machine via transformers/adapters."""

from .figure_embedder import FigureEmbedder
from .metadata_embedder import MetadataEmbedder, serialize_metadata
from .table_embedder import TableEmbedder, serialize_table
from .text_embedder import TextEmbedder

__all__ = [
    "FigureEmbedder",
    "MetadataEmbedder",
    "TableEmbedder",
    "TextEmbedder",
    "serialize_metadata",
    "serialize_table",
]