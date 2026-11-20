# ==============================================================================
# Temporal Hybrid GraphRAG — Knowledge Graph Relation Extractor
# Target: Python 3.11.x
# ==============================================================================

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import structlog

logger = structlog.get_logger(__name__)

EVOLUTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "SUPERSEDES",
        re.compile(
            r"\b(?P<subj>[A-Z][A-Za-z0-9\-_]{1,40})\s+(?:supersedes|replaces|outperforms|improves upon)\s+(?P<obj>[A-Z][A-Za-z0-9\-_]{1,40})\b"
        ),
    ),
    (
        "EXTENDS",
        re.compile(
            r"\b(?P<subj>[A-Z][A-Za-z0-9\-_]{1,40})\s+(?:extends|builds on|adapts|generalizes)\s+(?P<obj>[A-Z][A-Za-z0-9\-_]{1,40})\b"
        ),
    ),
    (
        "INTRODUCES",
        re.compile(
            r"\b(?:we\s+(?:introduce|propose|present)|this\s+(?:paper|work)\s+(?:introduces|proposes))\s+(?P<obj>[A-Za-z0-9\-_ ]{2,50}?)(?:[.,;]|$)",
            re.IGNORECASE,
        ),
    ),
]


@dataclass(slots=True)
class ExtractedRelation:
    """Represents a directed relation triple extracted from scientific text."""

    subject: str
    predicate: str
    object: str
    paper_id: str | None = None
    year: int | None = None
    confidence: float = 0.7
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RelationExtractor:
    """Extracts typed scientific relations using rule heuristics and optional LLM factory."""

    def __init__(self, use_llm: bool = False, provider: str | None = None) -> None:
        self.use_llm = use_llm
        self.provider = provider
        self._llm = None

        if self.use_llm:
            try:
                from temporal_graphrag.llm.factory import get_llm  # type: ignore

                self._llm = get_llm(provider=self.provider)
            except Exception as exc:
                logger.warning("llm_factory_unavailable_for_relation_extraction", error=str(exc))
                self.use_llm = False

    def extract_heuristic(
        self,
        text: str,
        paper_id: str | None = None,
        year: int | None = None,
    ) -> list[ExtractedRelation]:
        """Extract candidate subject-predicate-object relations via regex patterns."""
        if not text or not text.strip():
            return []

        relations: list[ExtractedRelation] = []
        seen: set[tuple[str, str, str]] = set()

        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue

            for predicate, pattern in EVOLUTION_PATTERNS:
                for match in pattern.finditer(clean_sentence):
                    groups = match.groupdict()
                    subj = (groups.get("subj") or paper_id or "CurrentWork").strip()
                    obj = (groups.get("obj") or "").strip(" -_.,;:")
                    if not subj or not obj or subj.lower() == obj.lower():
                        continue

                    key = (subj.lower(), predicate, obj.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    relations.append(
                        ExtractedRelation(
                            subject=subj,
                            predicate=predicate,
                            object=obj,
                            paper_id=paper_id,
                            year=year,
                            confidence=0.65,
                            evidence=clean_sentence[:240],
                        )
                    )

        return relations

    def extract_with_llm(
        self,
        text: str,
        paper_id: str | None = None,
        year: int | None = None,
    ) -> list[ExtractedRelation]:
        """Extract JSON-formatted relations using the configured LLM provider."""
        if not self._llm or not text.strip():
            return []

        prompt = (
            "Extract up to 10 scientific entity relations from the text below as a JSON list of "
            'objects with keys "subject", "predicate", "object", "confidence". '
            "Prefer predicates like INTRODUCES, EXTENDS, SUPERSEDES, EVALUATES_ON, USES_METHOD.\n\n"
            f"Text:\n{text[:3000]}"
        )
        try:
            raw_response = (
                self._llm.invoke(prompt)
                if hasattr(self._llm, "invoke")
                else self._llm.generate(prompt)
            )
            content = getattr(raw_response, "content", str(raw_response))
            json_match = re.search(r"\[[\s\S]*\]", content)
            if not json_match:
                return []

            items = json.loads(json_match.group(0))
            results: list[ExtractedRelation] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                subj = str(item.get("subject", "")).strip()
                pred = str(item.get("predicate", "RELATES_TO")).strip().upper().replace(" ", "_")
                obj = str(item.get("object", "")).strip()
                if subj and obj:
                    results.append(
                        ExtractedRelation(
                            subject=subj,
                            predicate=pred,
                            object=obj,
                            paper_id=paper_id,
                            year=year,
                            confidence=float(item.get("confidence", 0.8)),
                            evidence=text[:200],
                        )
                    )
            return results
        except Exception as exc:
            logger.warning("llm_relation_extraction_failed", paper_id=paper_id, error=str(exc))
            return []

    def extract(
        self,
        text: str,
        paper_id: str | None = None,
        year: int | None = None,
    ) -> list[ExtractedRelation]:
        """Run heuristic extraction and optionally merge LLM-extracted relations."""
        relations = self.extract_heuristic(text=text, paper_id=paper_id, year=year)
        if self.use_llm:
            llm_relations = self.extract_with_llm(text=text, paper_id=paper_id, year=year)
            existing = {(r.subject.lower(), r.predicate, r.object.lower()) for r in relations}
            for rel in llm_relations:
                key = (rel.subject.lower(), rel.predicate, rel.object.lower())
                if key not in existing:
                    existing.add(key)
                    relations.append(rel)
        return relations


def extract_relations_from_texts(
    texts: Sequence[tuple[str, str | None, int | None]],
    use_llm: bool = False,
) -> list[dict[str, Any]]:
    """Batch helper accepting `(text, paper_id, year)` tuples and returning serializable dicts."""
    extractor = RelationExtractor(use_llm=use_llm)
    all_relations: list[dict[str, Any]] = []
    for text, paper_id, year in texts:
        for rel in extractor.extract(text=text, paper_id=paper_id, year=year):
            all_relations.append(rel.to_dict())
    return all_relations