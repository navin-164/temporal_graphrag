"""
embeddings/text_embedder.py

Paper-level embeddings via SPECTER2 — allenai's citation-informed
scientific paper encoder, embedding a paper's (title, abstract) pair into
a single dense vector for similarity search. This is Week 6's first
embedding module, and its input shape (title + abstract) is a direct fit
for what Paper already has — no extraction work needed, unlike the
ingestion modules.

Why the adapter, not just the base model (see requirements.txt for the
same note)
--------------------------------------------------------------------------
allenai/specter2_base's own model card is explicit: "for general
embedding purposes, please use allenai/specter2" — not the base model
alone. The confusing part is the naming: allenai/specter2_base IS the
transformer encoder, but allenai/specter2 isn't a separate full model —
it's an adapter (the retrieval/proximity one, renamed from
allenai/specter2_proximity) that gets loaded ONTO the base model via the
`adapters` library. This corrects an earlier plan in this project's
requirements.txt to skip `adapters` — that would have run without any
error and just silently produced worse embeddings than SPECTER2's own
documentation recommends.

Scope: paper-level only, not paragraphs
--------------------------------------------
SPECTER2 is trained specifically on (title, abstract) pairs via citation
triplets — that's its actual training distribution, and it's exactly what
Paper already has cleanly. Using it on arbitrary paragraph text for
finer-grained chunk retrieval would be a different, off-label use, with
no clear answer yet on whether it's even the right tool for that job.
That's a decision for whenever Week 11's retrieval design makes the
actual chunking granularity concrete — not this file.

Normalized on purpose
--------------------------
Embeddings are L2-normalized before being returned. This changes nothing
for cosine similarity (which normalizes internally regardless), but sets
up Week 10's FAISS index to use inner-product search (IndexFlatIP) —
cheaper to compute than cosine — while still getting correct
cosine-similarity results, since inner product on unit vectors IS cosine
similarity. Costs nothing now, useful either way later.

Loaded once, not per call
------------------------------
Loading a transformer model is expensive (downloading, then initializing
weights). TextEmbedder is built to be instantiated once and reused across
every paper in a batch run, not re-created per paper.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from temporal_graphrag.core.entities import Paper

DEFAULT_MODEL_NAME = "allenai/specter2_base"
DEFAULT_ADAPTER_NAME = "allenai/specter2"  # the renamed proximity/retrieval adapter
MAX_LENGTH = 512  # matches SPECTER2's own usage examples and BERT-family limits
DEFAULT_BATCH_SIZE = 16  # internal chunking so a large paper list doesn't get
                          # tokenized and forwarded through the model all at
                          # once — fine at pilot scale (40 papers) either way,
                          # but cheap insurance for when this project scales
                          # beyond a pilot, per its own README's stated plan


class TextEmbedder:
    """Wraps SPECTER2 (base model + proximity adapter) for paper-level
    (title, abstract) -> embedding vector.

    Usage:
        embedder = TextEmbedder()  # loads once — downloads ~440MB of
                                     # weights the first time, seconds to
                                     # load on every run after that
        vector = embedder.embed_paper(paper)            # one Paper
        vectors = embedder.embed_papers([p1, p2, p3])    # batched — call
                                                            # this over a
                                                            # loop of
                                                            # embed_paper()
                                                            # whenever
                                                            # embedding more
                                                            # than one paper
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        adapter_name: str = DEFAULT_ADAPTER_NAME,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
    
        self.model.load_adapter(adapter_name, source="hf", load_as="proximity", set_active=True)
        # `set_active=True` above SHOULD be enough, but has been observed (see
        # this project's own test run) to not always persist to the actual
        # forward call, depending on the installed `adapters`/`transformers`
        # version combination. Calling set_active_adapters() explicitly, by
        # the LOCAL name ("proximity", from load_as= above — not the hub id
        # in adapter_name), guarantees it regardless. This one line is the
        # difference between silently getting base-model embeddings and
        # getting the actual retrieval-tuned SPECTER2 output.
        self.model.set_active_adapters("proximity")
    
        if not self.model.active_adapters:
            raise RuntimeError(
                f"SPECTER2 adapter '{adapter_name}' failed to activate — embeddings "
                "would silently come from the base model instead of the proximity "
                "adapter. Check that your installed `adapters` package version is "
                "compatible with your installed `transformers` version "
                "(pip show adapters transformers)."
            )
    
        self.model.to(self.device)
        self.model.eval()  # inference only — disables dropout, no gradient tracking needed

    def embed_papers(self, papers: list[Paper], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        """Embeds a batch of papers — meaningfully faster per paper than
        calling embed_paper() in a loop, since each internal chunk of
        `batch_size` papers is processed in one forward pass rather than
        one at a time. Order of the returned list matches `papers`."""
        if not papers:
            return []
        # Title + abstract, joined with the tokenizer's own [SEP] token —
        # SPECTER2's documented input format, not an arbitrary choice.
        text_batch = [
            paper.title + self.tokenizer.sep_token + (paper.abstract or "") for paper in papers
        ]
        return self.embed_texts(text_batch, batch_size=batch_size)

    def embed_paper(self, paper: Paper) -> list[float]:
        """Single-paper convenience wrapper around embed_papers()."""
        return self.embed_papers([paper])[0]

    def embed_texts(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        """Embeds arbitrary text through the same loaded SPECTER2 pipeline
        used for papers. Exposed as a general primitive — not Paper-
        specific — so other embedding modules (table_embedder.py,
        embedding a table's serialized text) can share this one loaded
        model instead of loading a second copy of the same weights."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            all_embeddings.extend(self._embed_batch(chunk))
        return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,  # SPECTER2's own examples explicitly omit these
            max_length=MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():  # inference only — no need to track gradients
            output = self.model(**inputs)

        embeddings = output.last_hidden_state[:, 0, :]  # CLS token, per SPECTER2's own examples
        embeddings = F.normalize(embeddings, p=2, dim=1)  # see module docstring
        return embeddings.cpu().tolist()
