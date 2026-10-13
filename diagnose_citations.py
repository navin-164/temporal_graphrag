"""Diagnoses why 0 citation links resolved: checks how many GROBID-
extracted citations have an arXiv id at all, and whether fuzzy title
matching against your actual 48-paper corpus would recover real
connections that exact arXiv-id matching missed."""
import difflib
import json
import re
from pathlib import Path


def norm(t):
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


meta = json.loads(Path("metadata/pilot_corpus_metadata.json").read_text(encoding="utf-8"))
print(f"Papers in metadata: {len(meta)}")
corpus_titles = [(norm(p["title"]), p["arxiv_id"], p["title"]) for p in meta]

total_citations = 0
arxiv_id_present = 0
fuzzy_hits = 0
seen_pairs = set()
examples = []

for path in sorted(Path("parsed").glob("*.json")):
    paper = json.loads(path.read_text(encoding="utf-8"))
    for c in paper.get("citations", []):
        total_citations += 1
        if c.get("cited_arxiv_id"):
            arxiv_id_present += 1
            continue  # already resolvable, don't also count as a fuzzy candidate

        nt = norm(c.get("cited_title"))
        if not nt or len(nt) < 8:
            continue
        for norm_title, arxiv_id, real_title in corpus_titles:
            if not norm_title:
                continue
            ratio = difflib.SequenceMatcher(None, nt, norm_title).ratio()
            if ratio >= 0.85:
                key = (c.get("cited_title"), arxiv_id)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    fuzzy_hits += 1
                    if len(examples) < 10:
                        examples.append((c.get("cited_title"), real_title, arxiv_id, round(ratio, 2)))
                break

print(f"Parsed papers scanned: {len(list(Path('parsed').glob('*.json')))}")
print(f"Total citation entries: {total_citations}")
print(f"Citations with a GROBID-tagged arXiv id: {arxiv_id_present}")
print(f"Additional fuzzy title matches (ratio>=0.85) against your corpus: {fuzzy_hits}")
print()
for cited, real, aid, ratio in examples:
    print(f"  ratio={ratio}  cited_title={cited!r}")
    print(f"           -> {aid}: {real!r}")