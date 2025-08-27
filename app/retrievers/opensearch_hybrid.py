from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from .hybrid_rrf import rrf_fuse
from typing import List

class HybridRetriever:
    def __init__(self, host="localhost", index="docs"):
        self.client = OpenSearch(hosts=[{"host": host, "port": 9200}], http_compress=True)
        self.index = index
        self.embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query, k=20) -> List:
        # BM25
        bm = self.client.search(index=self.index, body={
            "size": k,
            "query": {"match": {"text": {"query": query}}}
        })["hits"]["hits"]
        # kNN
        qv = self.embed.encode(query).tolist()
        knn = self.client.search(index=self.index, body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": qv, "k": k}}},
            "_source": {"excludes": ["embedding"]}
        })["hits"]["hits"]
        # RRF fuse ids
        fused_ids = rrf_fuse([bm, knn], k=60, topk=k)
        # fetch in the fused order
        docs = {h["_id"]: h for h in bm + knn}
        return [docs[i] for i in fused_ids]
