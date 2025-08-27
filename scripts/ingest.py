import os, glob, json, uuid
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("OS_HOST", "localhost")
client = OpenSearch(hosts=[{"host": os.environ["OS_HOST"], "port": 9200}], http_compress=True)

def chunk(text, size=700, overlap=120):
    words = text.split()
    i=0
    while i < len(words):
        yield " ".join(words[i:i+size])
        i += size - overlap

def index_docs():
    for path in glob.glob("data/docs/*.md"):
        title = os.path.basename(path).replace(".md","")
        text = open(path, "r", encoding="utf-8").read()
        for i, ch in enumerate(chunk(text)):
            emb = model.encode(ch).tolist()
            doc = {"doc_id": str(uuid.uuid4()), "title": title, "text": ch, "embedding": emb}
            client.index(index="docs", body=doc, refresh=False)
    client.indices.refresh(index="docs")
    print("Ingested.")

if __name__ == "__main__":
    os.makedirs("data/docs", exist_ok=True)
    if not glob.glob("data/docs/*.md"):
        with open("data/docs/example.md","w") as f:
            f.write("# Example\nOpenSearch hybrid search demo. This file is here so you can test.")
    index_docs()
