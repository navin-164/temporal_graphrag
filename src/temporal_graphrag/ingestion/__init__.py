"""Document ingestion layer — GROBID client and the text/figure/table/
equation/citation extractors that turn its TEI output into entities."""

from .citation_extractor import extract_citations
from .equation_extractor import extract_equations
from .figure_extractor import crop_figure_image, crop_page_region, extract_figures
from .pdf_parser import GrobidClient
from .table_extractor import crop_table_image, extract_tables
from .text_extractor import extract_sections

__all__ = [
    "GrobidClient",
    "crop_figure_image",
    "crop_page_region",
    "crop_table_image",
    "extract_citations",
    "extract_equations",
    "extract_figures",
    "extract_sections",
    "extract_tables",
]
