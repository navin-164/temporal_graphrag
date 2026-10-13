"""
ingestion/equation_extractor.py

The Partial-scope "LaTeX-detection fallback" for equations agreed on at
the start of this project — not a full Nougat/MathPix-style OCR pipeline.
Turns GROBID's <formula> elements into Equation entities.

What GROBID actually gives us here, and what this file does about it
--------------------------------------------------------------------------
Two things confirmed independently (GROBID's own training documentation,
plus the Nougat paper — a real OCR benchmark that explicitly compared its
output against GROBID's), not assumed:

1. GROBID only tags STANDALONE/display equations as <formula>. Inline
   formulas are "not specifically annotated" (GROBID's own words) — they
   flow into the surrounding paragraph text, already captured by
   text_extractor.py, indistinguishable from regular prose. There is no
   inline-equation data for this file to extract; Equation.inline stays
   False for everything produced here (see core/entities.py's comment on
   that field for why it's kept rather than removed).

2. <formula> content isn't LaTeX — the Nougat paper states GROBID "stores
   [formulas] as the Unicode representations embedded in the PDF," and
   that its own authors had to map each Unicode symbol back to a LaTeX
   command themselves to get something LaTeX-like. This file does the
   same small mapping (Greek letters, common operators) for the same
   reason. It is cosmetic, not structural: a fraction, superscript, or
   subscript is already lost by the time GROBID hands us the text — no
   symbol substitution recovers that. The same Nougat comparison notes
   GROBID's formula recognition itself misses a meaningful fraction of
   equations outright; this file inherits that ceiling, it doesn't fix it.

This is why the project scoped equation understanding as Partial rather
than Core: recovering real structured LaTeX needs a dedicated
vision-to-LaTeX model (Nougat, MathPix, pix2tex), which is genuinely a
different, heavier tool than anything else in this ingestion pipeline —
out of scope for a semester timeline, not just this file.
"""

from __future__ import annotations

import re
from typing import Optional

from lxml import etree

from temporal_graphrag.core.entities import Equation

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_NS_ID = "{http://www.w3.org/XML/1998/namespace}id"
_WHITESPACE_RE = re.compile(r"\s+")

# Cosmetic Unicode -> LaTeX-command substitution — see module docstring.
# Deliberately a small, common set (Greek letters + everyday operators
# that show up constantly in ML/CS papers), not an exhaustive Unicode
# math table — this is a fallback, not a math OCR engine.
_UNICODE_TO_LATEX = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
    "Ψ": r"\Psi", "Ω": r"\Omega",
    "×": r"\times", "÷": r"\div", "±": r"\pm", "∓": r"\mp",
    "≤": r"\le", "≥": r"\ge", "≠": r"\neq", "≈": r"\approx",
    "∞": r"\infty", "√": r"\sqrt", "∑": r"\sum", "∏": r"\prod",
    "∫": r"\int", "∂": r"\partial", "∇": r"\nabla",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq",
    "∪": r"\cup", "∩": r"\cap", "→": r"\to", "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow", "∀": r"\forall", "∃": r"\exists",
    "·": r"\cdot",
}


def extract_equations(tei_tree: etree._Element, paper_id: str) -> list[Equation]:
    """Extract Equation entities from a paper's TEI tree, in document
    order. Skips any <formula> that yields no usable text at all — an
    empty formula element isn't worth recording.
    """
    body = tei_tree.find(".//tei:text/tei:body", namespaces=TEI_NS)
    if body is None:
        return []

    equations: list[Equation] = []
    index = 0

    for formula_el in body.findall(".//tei:formula", namespaces=TEI_NS):
        latex = _to_latex_approximation(formula_el)
        if not latex:
            continue

        label_el = formula_el.find("tei:label", namespaces=TEI_NS)
        equation_number = _clean_text(label_el) or None

        equations.append(
            Equation(
                id=Equation.build_id(paper_id, index),
                paper_id=paper_id,
                latex=latex,
                equation_number=equation_number,
                grobid_ref_id=formula_el.get(XML_NS_ID),
                inline=False,  # see module docstring — GROBID never gives us the other kind
            )
        )
        index += 1

    return equations


def _to_latex_approximation(formula_el: etree._Element) -> str:
    """Flattens the formula's text and substitutes common Unicode math
    symbols for their LaTeX command names. Explicitly excludes the
    <label> child's own text (e.g. "10" from "(10)") — equation_number
    already captures that separately, and leaving it mixed into the main
    text produced confusing output like "...+ \\beta 3" with no indication
    the trailing 3 was a label, not part of the formula. Walking children
    manually (rather than the simpler itertext() used elsewhere in this
    file) is what makes skipping just that one child possible."""
    label_tag = f"{{{TEI_NS['tei']}}}label"
    parts = [formula_el.text or ""]
    for child in formula_el:
        if child.tag != label_tag:
            parts.append("".join(child.itertext()))
        parts.append(child.tail or "")  # text after the child's closing tag
                                          # belongs to the formula either way
    raw = "".join(parts)
    cleaned = _WHITESPACE_RE.sub(" ", raw).strip()
    for symbol, latex_command in _UNICODE_TO_LATEX.items():
        cleaned = cleaned.replace(symbol, f" {latex_command} ")
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _clean_text(element: Optional[etree._Element]) -> str:
    """Same flattening as the other ingestion files' helper of the same
    name."""
    if element is None:
        return ""
    raw = "".join(element.itertext())
    return _WHITESPACE_RE.sub(" ", raw).strip()
