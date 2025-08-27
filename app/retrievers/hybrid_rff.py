from collections import defaultdict
from typing import List

def rrf_fuse(ranklists, k=60, topk=10) -> List:
    scores = defaultdict(float)
    for rl in ranklists:
        for r, hit in enumerate(rl, start=1):
            scores[hit["_id"]] += 1.0 / (k + r)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:topk]]
