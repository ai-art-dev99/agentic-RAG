from fastapi import FastAPI
from pydantic import BaseModel
from app.retrievers.opensearch_hybrid import HybridRetriever

app = FastAPI()
retriever = HybridRetriever()

class QueryIn(BaseModel):
    query: str
    mode: str = "auto"   # auto | local | global

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/query")
def query(q: QueryIn):
    # very simple planner: global if "overview|themes|across" present
    ql = q.query.lower()
    is_global = ("overview" in ql) or ("themes" in ql) or ("across" in ql)
    route = q.mode if q.mode != "auto" else ("global" if is_global else "local")

    if route == "local":
        hits = retriever.search(q.query, k=10)
        contexts = [h["_source"]["text"] for h in hits]
        return {"route": "local", "k": len(hits), "contexts": contexts[:3]}

    # GraphRAG fallback if not configured
    try:
        from graphrag.query import query as gr_query  # type: ignore
        # expects you've run `graphrag index --root ./gr_workspace`
        answer = gr_query(root="./gr_workspace", method="global", query=q.query)
        return {"route": "global", "answer": str(answer)}
    except Exception as e:
        return {"route": "global", "error": f"GraphRAG not configured: {e}"}
