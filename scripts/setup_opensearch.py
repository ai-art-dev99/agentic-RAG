import requests, json, time

BASE = "http://localhost:9200"
INDEX = "docs"

def create_index() -> None:
    # delete if exists
    requests.delete(f"{BASE}/{INDEX}")
    mapping = {
        "settings": {
            "index": {"knn": True}
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}
                }
            }
        }
    }
    r = requests.put(f"{BASE}/{INDEX}", data=json.dumps(mapping), headers={"Content-Type":"application/json"})
    r.raise_for_status()
    print("Index created:", r.json())

if __name__ == "__main__":
    for _ in range(30):
        try:
            if requests.get(BASE).ok: break
        except Exception: time.sleep(1)
    create_index()
