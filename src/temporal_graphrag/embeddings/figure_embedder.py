"""
embeddings/figure_embedder.py

CLIP embeddings for the figure images figure_extractor.py already
cropped. CLIP is a dual encoder — image and text share the same
embedding space by design — so this file exposes both sides: embedding
the cropped figures themselves, and embedding arbitrary text (a caption,
or later a user's search query) into that same space for cross-modal
retrieval ("show me the architecture diagram" matching an actual figure
with no caption keyword overlap at all).

Images are embedded one at a time, not batched — here's why
------------------------------------------------------------------
text_embedder.py batches internally for throughput, and that's the right
call there: a batch run processes 40+ papers, so batching meaningfully
helps. This file doesn't do the same for images, on purpose: a documented
HuggingFace transformers issue (#30443) reports CLIPModel.
get_image_features() returning DIFFERENT embeddings for the same image
depending on what else is in the batch with it — a real inconsistency,
not a rumor, though its current status in the specific transformers
version this project ends up using isn't something this environment can
check. Given a typical paper has a handful of figures, not hundreds, the
throughput cost of skipping batching here is small; the correctness risk
of hitting that inconsistency isn't. If this turns out to be fixed in
whatever version actually gets installed, this is a safe default to keep
anyway — simpler code, one clear failure mode per image.

Normalized on purpose
--------------------------
Same reasoning as text_embedder.py: L2-normalizing before returning
doesn't change cosine similarity, but it's what CLIP's own usage examples
do before comparing embeddings, and it keeps this consistent with
text_embedder.py's output for whenever Week 10's FAISS index combines
both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from temporal_graphrag.core.entities import Figure

logger = logging.getLogger(__name__)  # stdlib, not structlog — same reasoning as pdf_parser.py

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"  # smallest standard CLIP checkpoint —
                                                        # matches this project's lightweight-
                                                        # fallback preference, and is the most
                                                        # commonly documented/tested variant


class FigureEmbedder:
    """Wraps CLIP for both figure-image embeddings and text embeddings in
    the same shared space.

    Usage:
        embedder = FigureEmbedder()  # loads once — downloads ~600MB of
                                       # weights the first time
        embeddings = embedder.embed_figures(paper.figures)
        for figure, embedding in zip(paper.figures, embeddings):
            figure.embedding = embedding  # None for any figure whose
                                            # image couldn't be embedded —
                                            # see embed_figures() below

        query_vector = embedder.embed_text_query("architecture diagram")
        # compare query_vector against figure embeddings with cosine
        # similarity (or inner product, since both are L2-normalized) —
        # that comparison is retrieval's job, not this file's
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # inference only — disables dropout, no gradient tracking needed

    def embed_figures(self, figures: list[Figure]) -> list[Optional[list[float]]]:
        """Embeds each figure's cropped image. Returns a list the same
        length as `figures`, in the same order — None at any position
        whose figure has no image_path, or whose image file couldn't be
        opened/embedded — rather than silently dropping that position, so
        callers can always do `zip(figures, embed_figures(figures))`
        without the two lists drifting out of alignment.
        """
        return [self._embed_one_image(figure) for figure in figures]

    def embed_figure(self, figure: Figure) -> Optional[list[float]]:
        """Single-figure convenience wrapper around embed_figures()."""
        return self._embed_one_image(figure)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of plain text strings into CLIP's shared space —
        batched, unlike images, since text_encoder batching isn't the part
        the known inconsistency (see module docstring) was reported against,
        and there's no image-decoding cost pushing toward one-at-a-time here."""
        if not texts:
            return []

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().tolist()

    def embed_text_query(self, text: str) -> list[float]:
        """Single-text convenience wrapper around embed_texts()."""
        return self.embed_texts([text])[0]

    def _embed_one_image(self, figure: Figure) -> Optional[list[float]]:
        if not figure.image_path:
            return None

        try:
            image = Image.open(figure.image_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            logger.warning("couldn't open image for %s (%s): %s", figure.id, figure.image_path, e)
            return None

        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().tolist()[0]
        except Exception as e:
            # Broad catch deliberate here, same reasoning as 03_parse_
            # documents.py's _crop_all_images: a bad image shouldn't take
            # down embedding for every other figure in the batch, and
            # CLIP/PIL can raise several different error types for a
            # malformed image.
            logger.warning("couldn't embed image for %s: %s", figure.id, e)
            return None
