"""
storage/vector_store.py

FAISS-backed multi-modal vector store — one IndexFlatIP per embedding
modality (text, metadata, figure, table), matching 05_build_embeddings.py's
four output vectors and Week 10's storage-layer deliverable in the
architecture diagram ("5.2 Vector Store — FAISS/Milvus/Qdrant").

Why separate indexes per modality, not one combined index
-------------------------------------------------------------
SPECTER2 (text, metadata) produces 768-dim vectors; CLIP (figures)
produces 512-dim vectors for the base patch32 checkpoint figure_embedder.
py uses. FAISS indexes are fixed-dimension — there's no meaningful way to
put both in one index, and even if the dimensions happened to match, text
and image embeddings live in different, unrelated vector spaces (nothing
guarantees a text vector and an image vector that happen to be "close"
are actually related). One index per modality is the only correct option
here, not a stylistic choice.

Why IndexFlatIP, not IndexFlatL2 or an approximate index (IVF/HNSW)
------------------------------------------------------------------------
- IP (inner product), not L2: every embedder in this project
  (text_embedder.py, figure_embedder.py) already L2-normalizes its
  output before returning it, specifically so inner product equals
  cosine similarity — see text_embedder.py's own docstring. Using
  IndexFlatIP here is just claiming the benefit that choice already paid
  for; using IndexFlatL2 instead would silently measure the wrong thing.
- Flat (exact), not approximate: at pilot scale (40 papers, a few
  hundred figures/tables/paragraphs), an approximate index buys nothing
  — exact search over a few hundred vectors is already fast — while
  adding tuning parameters (nlist, nprobe, ...) that only matter at a
  scale this project isn't at yet. Swappable later if the corpus grows
  well past pilot size; not a decision worth making now.

Why a parallel Python list for IDs, not FAISS's IndexIDMap
----------------------------------------------------------------
FAISS's native ID support wants int64 ids. This project's real ids are
strings (arxiv_id for papers, or Figure.build_id()/Table.build_id()'s
composite ids, e.g. "2003.08964::figure::0"). Rather than inventing and
maintaining a string<->int64 mapping table just to use IndexIDMap, this
class keeps a plain Python list per modality where position `i` in the
list corresponds to row `i` in that modality's FAISS index. This works
because vectors are only ever appended, never removed or reordered —
true for this pilot-scale, build-once-from-a-batch-script usage. A
real delete/update path (relevant once Week 8's Dynamic Update Module is
in scope — currently Deferred per the project's own Scope Map) would need
IndexIDMap or a rebuild-on-write strategy; not a gap in THIS file, just
scope this file doesn't claim to cover yet.

Cosine similarity scores, not distances
--------------------------------------------
Because every index is IndexFlatIP over L2-normalized vectors, `search()`
returns scores in [-1, 1] where higher is more similar (cosine
similarity) — the opposite direction from L2 distance, where lower is
better. Callers (Week 11's vector_retriever.py) should treat these as
similarity scores directly, not distances to invert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

__all__ = ["VectorStore"]


class VectorStore:
    """A FAISS-backed vector store holding one exact (IndexFlatIP) index
    per embedding modality, plus a parallel id list per modality so search
    results can be reported by our own domain ids instead of bare FAISS
    row numbers.

    Usage:
        store = VectorStore()
        store.add("text", ids=["2003.08964", "2007.09820"], vectors=[...])
        store.add("figure", ids=["2003.08964::figure::0"], vectors=[...])

        results = store.search("text", query_vector, k=5)
        # -> [("2003.08964", 0.87), ("2007.09820", 0.81), ...]
        # sorted by descending cosine similarity, already resolved back
        # to arxiv_ids — not FAISS row numbers

        store.save("vector_index")          # writes one .index + one
                                              # _ids.json file per modality
        store = VectorStore.load("vector_index")   # and reads them back
    """

    def __init__(self) -> None:
        self._indexes: dict[str, faiss.Index] = {}
        self._id_maps: dict[str, list[str]] = {}
        self._id_sets: dict[str, set[str]] = {}  # mirrors _id_maps, for O(1) has() checks

    # -- Building ------------------------------------------------------------

    def add(self, modality: str, ids: list[str], vectors: list[list[float]]) -> None:
        """Appends vectors (and their ids) to `modality`'s index, creating
        that index on first use with whatever dimension `vectors` turns
        out to have. Every subsequent call for the same modality must use
        vectors of that same dimension — enforced by FAISS itself
        (`index.add()` raises if the shape doesn't match), not re-checked
        redundantly here.

        No dedup against ids already present — see the module docstring's
        note on why delete/update is out of scope for this file. Calling
        add() twice with an overlapping id will produce two rows in the
        index for that id, which is a caller bug (e.g. re-running a build
        script without clearing the output dir first), not something this
        method silently corrects.
        """
        if not vectors:
            return
        if len(ids) != len(vectors):
            raise ValueError(
                f"add('{modality}', ...): got {len(ids)} ids but {len(vectors)} vectors — "
                "these must be the same length and in the same order."
            )

        dim = len(vectors[0])
        if modality not in self._indexes:
            self._indexes[modality] = faiss.IndexFlatIP(dim)
            self._id_maps[modality] = []
            self._id_sets[modality] = set()

        array = np.asarray(vectors, dtype="float32")
        self._indexes[modality].add(array)
        self._id_maps[modality].extend(ids)
        self._id_sets[modality].update(ids)

    def has(self, modality: str, id_: str) -> bool:
        """O(1) membership check — lets a build script (07_build_vector_
        index.py) skip re-embedding-and-adding something already indexed,
        the same resume pattern used by this project's other scripts."""
        return id_ in self._id_sets.get(modality, set())

    # -- Searching -------------------------------------------------------

    def search(self, modality: str, query_vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Returns up to `k` (id, cosine_similarity) pairs for `modality`,
        sorted by descending similarity. Returns [] for a modality that
        doesn't exist yet (e.g. searching "figure" before any figure was
        ever added) rather than raising — an empty result set is a
        legitimate answer to "what's the closest figure," not an error.
        """
        index = self._indexes.get(modality)
        if index is None or index.ntotal == 0:
            return []

        k = min(k, index.ntotal)
        query = np.asarray([query_vector], dtype="float32")
        scores, row_indices = index.search(query, k)

        id_map = self._id_maps[modality]
        results: list[tuple[str, float]] = []
        for score, row in zip(scores[0], row_indices[0]):
            if row == -1:  # FAISS pads with -1 if fewer than k results exist
                continue
            results.append((id_map[row], float(score)))
        return results

    # -- Introspection -----------------------------------------------------

    @property
    def modalities(self) -> list[str]:
        """Every modality currently holding at least one vector."""
        return list(self._indexes.keys())

    def count(self, modality: str) -> int:
        """Number of vectors stored for `modality`, or 0 if it doesn't
        exist yet — matches has()'s "missing modality isn't an error"
        stance."""
        index = self._indexes.get(modality)
        return index.ntotal if index is not None else 0

    # -- Persistence -----------------------------------------------------

    def save(self, directory: str) -> None:
        """Writes one `{modality}.index` (FAISS's own binary format) plus
        one `{modality}_ids.json` (the parallel id list) per modality into
        `directory`, creating it if needed."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        for modality, index in self._indexes.items():
            faiss.write_index(index, str(out_dir / f"{modality}.index"))
            ids_path = out_dir / f"{modality}_ids.json"
            ids_path.write_text(json.dumps(self._id_maps[modality]), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Rebuilds a VectorStore from a directory previously written by
        save(). Every `*.index` file found gets loaded, matched to its
        `{same_stem}_ids.json` sibling — a missing ids file is treated as
        empty rather than fatal, since an id-less index (positions with no
        way to resolve back to a domain id) is still loadable, just not
        very useful until that gap is fixed at the source."""
        in_dir = Path(directory)
        if not in_dir.exists():
            raise FileNotFoundError(f"No vector index directory at '{directory}'.")

        store = cls()
        for index_path in sorted(in_dir.glob("*.index")):
            modality = index_path.stem
            index = faiss.read_index(str(index_path))

            ids_path = in_dir / f"{modality}_ids.json"
            ids: list[str] = (
                json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else []
            )

            store._indexes[modality] = index
            store._id_maps[modality] = ids
            store._id_sets[modality] = set(ids)
        return store