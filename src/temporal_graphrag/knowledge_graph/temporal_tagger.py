# ==============================================================================
# Temporal Hybrid GraphRAG — Knowledge Graph Temporal Tagger
# Target: Python 3.11.x
# ==============================================================================

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

ARXIV_YYMM_RE = re.compile(r"\b(?P<yy>\d{2})(?P<mm>\d{2})\.\d{4,5}(?:v\d+)?\b")
YEAR_RE = re.compile(r"\b(?:19\d{2}|20\d{2})\b")
ISO_DATE_RE = re.compile(r"\b(?P<year>19\d{2}|20\d{2})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])\b")

TEMPORAL_CUES: dict[str, tuple[str, ...]] = {
    "historical_or_superseded": (
        "previously",
        "formerly",
        "earlier work",
        "outdated",
        "deprecated",
        "superseded",
        "limitation of",
        "prior to",
    ),
    "current_or_emerging": (
        "currently",
        "state-of-the-art",
        "sota",
        "recent",
        "novel",
        "emerging",
        "as of",
        "new baseline",
    ),
    "evolutionary_transition": (
        "evolved into",
        "replaced by",
        "extended to",
        "transitioned from",
        "over time",
        "chronological",
    ),
}


@dataclass(slots=True)
class TemporalTag:
    """Temporal anchor and evolution cues extracted from a document or passage."""

    paper_id: str | None = None
    inferred_year: int | None = None
    inferred_month: int | None = None
    explicit_dates: list[str] | None = None
    referenced_years: list[int] | None = None
    temporal_cues: list[str] | None = None
    valid_from: str | None = None
    valid_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalTagger:
    """Infers publication timestamps, date mentions, and temporal evolution signals."""

    def infer_from_paper_id(self, paper_id: str | None) -> tuple[int | None, int | None]:
        """Extract `(year, month)` from arXiv YYMM IDs or leading 4-digit years."""
        if not paper_id:
            return None, None

        match = ARXIV_YYMM_RE.search(paper_id)
        if match:
            yy = int(match.group("yy"))
            mm = int(match.group("mm"))
            year = 2000 + yy if yy < 90 else 1900 + yy
            month = mm if 1 <= mm <= 12 else None
            return year, month

        year_match = YEAR_RE.search(paper_id)
        if year_match:
            return int(year_match.group(0)), None

        return None, None

    def tag_text(
        self,
        text: str,
        paper_id: str | None = None,
        metadata_year: int | None = None,
    ) -> TemporalTag:
        """Analyze text and metadata to produce a structured `TemporalTag`."""
        id_year, id_month = self.infer_from_paper_id(paper_id)
        inferred_year = metadata_year if isinstance(metadata_year, int) else id_year

        explicit_dates = sorted(set(m.group(0) for m in ISO_DATE_RE.finditer(text or "")))

        years_found: set[int] = set()
        for m in YEAR_RE.finditer(text or ""):
            yr = int(m.group(0))
            if 1990 <= yr <= 2035:
                years_found.add(yr)
        if isinstance(inferred_year, int):
            years_found.add(inferred_year)

        lower_text = (text or "").lower()
        cues: list[str] = []
        for category, phrases in TEMPORAL_CUES.items():
            for phrase in phrases:
                if phrase in lower_text:
                    cues.append(f"{category}:{phrase}")

        valid_from: str | None = None
        if explicit_dates:
            valid_from = explicit_dates[0]
        elif isinstance(inferred_year, int) and isinstance(id_month, int):
            valid_from = f"{inferred_year:04d}-{id_month:02d}-01"
        elif isinstance(inferred_year, int):
            valid_from = f"{inferred_year:04d}-01-01"

        return TemporalTag(
            paper_id=paper_id,
            inferred_year=inferred_year,
            inferred_month=id_month,
            explicit_dates=explicit_dates,
            referenced_years=sorted(years_found),
            temporal_cues=sorted(set(cues)),
            valid_from=valid_from,
            valid_to=None,
        )


def tag_documents(
    records: list[dict[str, Any]],
    text_field: str = "text",
    id_field: str = "paper_id",
) -> list[dict[str, Any]]:
    """Attach a `temporal_tag` dictionary to each record in a list of dictionaries."""
    tagger = TemporalTagger()
    tagged: list[dict[str, Any]] = []
    for record in records:
        pid = record.get(id_field) or record.get("arxiv_id") or record.get("id")
        meta_year = record.get("year")
        text_val = record.get(text_field) or record.get("abstract") or ""
        tag = tagger.tag_text(
            text=str(text_val),
            paper_id=str(pid) if pid is not None else None,
            metadata_year=meta_year if isinstance(meta_year, int) else None,
        )
        tagged.append({**record, "temporal_tag": tag.to_dict()})
    return tagged