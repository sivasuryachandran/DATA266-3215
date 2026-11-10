"""Stream the first 100 train docs of ccdv/arxiv-summarization ('document' config) -> docs100.json.
Streaming avoids downloading/caching the full multi-GB split."""
import json
from datasets import load_dataset

ds = load_dataset("ccdv/arxiv-summarization", "document", split="train", streaming=True)
docs = []
for i, x in enumerate(ds):
    if i >= 100:
        break
    docs.append({"id": i, "article": x["article"], "abstract": x["abstract"]})
json.dump(docs, open("docs100.json", "w"))
print(len(docs), "documents saved")
