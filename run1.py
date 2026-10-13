import json
data = json.load(open("metadata/pilot_corpus_metadata.json"))
for p in sorted(data, key=lambda x: x["title"]):
    print(f"{p['arxiv_id']}  {p['title']}")