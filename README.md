# Agentic RAG — Hybrid (BM25 + Vector) + GraphRAG with RRF, HyDE & Evals

A production-style Retrieval-Augmented Generation stack that **plans** queries and chooses between:
- **Local path (Hybrid RAG):** BM25 + vector search fused via **Reciprocal Rank Fusion (RRF)**; optional **HyDE** booster.
- **Global path (GraphRAG):** builds a knowledge graph with **community summaries** and answers “overview / themes / across the corpus” questions.

Why these choices?
- **RRF** is a robust, rank-based way to combine heterogeneous result lists (BM25, kNN/neural) without score normalization. It’s now a first-class feature in OpenSearch search pipelines. :contentReference[oaicite:0]{index=0}  
- **GraphRAG** adds a *global* reasoning mode by querying precomputed **community summaries** (great for “what are the main themes?”-type queries). :contentReference[oaicite:1]{index=1}  
- **HyDE** improves recall on tough queries by generating a “hypothetical” answer passage, embedding it, and searching for the closest real docs. :contentReference[oaicite:2]{index=2}  
- **RAG evaluation** is built-in (RAGAS, TruLens “RAG triad”, DeepEval’s pytest-like checks) so you can track faithfulness, answer/context relevance, and regressions over time. :contentReference[oaicite:3]{index=3}

---

## Features

- 🔎 **Hybrid retrieval** (BM25 + vector) with **RRF** fusion  
- 🕸️ **GraphRAG** “global search” path using community summaries  
- 🧪 **Evals**: RAGAS metrics, TruLens “RAG triad”, DeepEval tests  
- ⚙️ **FastAPI** API + Docker Compose for **OpenSearch** and **Neo4j**  
- 🧰 Scripts for ingestion, indexing, graph build, and end-to-end runs  
- 🧯 Optional **HyDE** booster for hard queries  
- 📈 Dataset hooks for **HotpotQA / MuSiQue / 2WikiMultihopQA** (multi-hop QA) :contentReference[oaicite:4]{index=4}

---

## Architecture (high level)

```mermaid
flowchart LR
    U[User Query] --> P[Planner]
    P -->|local| HR[Hybrid Retriever<br/>BM25 + Vector + RRF]
    P -->|global| GR[GraphRAG<br/>Community Summaries]
    HR --> A[Answer + Citations]
    GR --> A
````

* **Hybrid path:** BM25 + vector searches run in parallel; results are fused with **RRF** (rank-based) to get resilient top-k. Optionally trigger **HyDE** when recall looks weak. ([OpenSearch][1], [arXiv][2])
* **GraphRAG path:** pre-compute a graph and **community summaries**; at query time, select relevant communities and synthesize a globally grounded answer. ([Microsoft GitHub][3])

---

## Quickstart

### 0) Requirements

* Python 3.10+
* Docker + Docker Compose

### 1) Install deps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start infra

```bash
docker compose -f docker/compose.yml up -d
```

### 3) Create the OpenSearch index

```bash
python scripts/setup_opensearch.py
```

> **Note:** If you use the `sentence-transformers/all-MiniLM-L6-v2` embedder, set the vector **dimension to 384** in your index mapping (or the setup script). That model outputs 384-dim vectors. ([Hugging Face][4])

### 4) Ingest sample docs

```bash
python scripts/ingest.py
```

### 5) Run the API

```bash
uvicorn app.api:app --reload
```

Open `http://localhost:8000/docs` and try:

* `{"query": "overview of the example file"}` → should route **global** (GraphRAG) if configured
* `{"query": "what does the example mention about hybrid search?"}` → **local** (hybrid + RRF)

---

## How the retrieval works

### Hybrid + RRF (local path)

* Run **BM25** and **vector** queries in parallel and combine via **RRF** (sum of reciprocal ranks). In OpenSearch 2.19+, you can also enable engine-side RRF via a **search pipeline** (`score-ranker-processor`) for hybrid queries. ([OpenSearch Docs][5], [OpenSearch][6])

### GraphRAG (global path)

* Builds a corpus graph and **community summaries**, then answers **global** questions by aggregating the most relevant communities; use **local** graph queries for entity-centric questions. ([Microsoft GitHub][3])

### HyDE (optional booster)

* When recall looks weak, generate a short hypothetical answer, embed it, and re-query; fuse with RRF. ([arXiv][2])

---

## Evaluation

This repo includes three complementary evaluation tools:

* **RAGAS**: faithfulness, answer relevance, context precision/recall, etc. ([Ragas][7])
* **TruLens**: the **RAG triad** — context relevance, groundedness, answer relevance. ([TruLens][8])
* **DeepEval**: pytest-like tests for RAG apps to run in CI. ([GitHub][9])

Run a small example:

```bash
python evals/run_ragas.py
pytest -q
```

---

## Datasets (for multi-hop evals)

Hook up a small dev slice from one or more of:

* **HotpotQA** — \~113k multi-hop QA with supporting facts. ([arXiv][10])
* **MuSiQue** — carefully composed 2–4 hop questions. ([ACL Anthology][11])
* **2WikiMultihopQA** — includes reasoning paths/evidence. ([arXiv][12])

---

## Engine-side RRF (optional, OpenSearch)

You can let OpenSearch do the fusion:

1. Create an RRF search pipeline (rank constant \~60).
2. Use a **hybrid** query (e.g., `match` + neural/kNN) with `?search_pipeline=...`.
   References and examples: official hybrid search guides and RRF pipeline docs. ([OpenSearch Docs][13])

---

## Configuration tips

* **Embedding dimension**

  * `all-MiniLM-L6-v2` → **384** dims; set the index `knn_vector` mapping accordingly. ([Hugging Face][4])
* **Planner**

  * Route to **global** if the query asks for themes/overview/unification across documents; otherwise **local**. (You can also use retrieval-agreement heuristics.)
* **Switching to engine-side RRF**

  * Once you register a neural model in OpenSearch, prefer the built-in RRF search pipeline for performance and simplicity. ([OpenSearch][6])

---

## Roadmap

* [ ] Streamlit demo: planner viz, per-retriever top-k, fused list, citations
* [ ] Add ColBERT-style late interaction retriever ablation
* [ ] Expand evals with larger HotpotQA/MuSiQue/2Wiki dev slices
* [ ] Add latency/cost dashboards

---

## References

* OpenSearch hybrid search & **RRF** (tutorials, pipelines, best practices). ([OpenSearch Docs][5], [OpenSearch][1])
* **GraphRAG** modes (global/local) & community summaries. ([Microsoft GitHub][3])
* **HyDE** (Hypothetical Document Embeddings). ([arXiv][2])
* **RAG evaluation**: RAGAS metrics, TruLens RAG Triad, DeepEval framework. ([Ragas][7], [TruLens][8], [GitHub][9])
* `all-MiniLM-L6-v2` model card (384-dim embeddings). ([Hugging Face][4])

---

## License

MIT


[1]: https://opensearch.org/blog/building-effective-hybrid-search-in-opensearch-techniques-and-best-practices/?utm_source=chatgpt.com "Building effective hybrid search in OpenSearch"
[2]: https://arxiv.org/abs/2212.10496?utm_source=chatgpt.com "Precise Zero-Shot Dense Retrieval without Relevance Labels"
[3]: https://microsoft.github.io/graphrag/?utm_source=chatgpt.com "Welcome - GraphRAG"
[4]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2?utm_source=chatgpt.com "sentence-transformers/all-MiniLM-L6-v2"
[5]: https://docs.opensearch.org/latest/tutorials/vector-search/neural-search-tutorial/?utm_source=chatgpt.com "Getting started with semantic and hybrid search"
[6]: https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/?utm_source=chatgpt.com "Introducing reciprocal rank fusion for hybrid search"
[7]: https://docs.ragas.io/en/v0.1.21/concepts/metrics/?utm_source=chatgpt.com "Metrics"
[8]: https://www.trulens.org/getting_started/core_concepts/rag_triad/?utm_source=chatgpt.com "RAG Triad"
[9]: https://github.com/confident-ai/deepeval?utm_source=chatgpt.com "confident-ai/deepeval: The LLM Evaluation Framework"
[10]: https://arxiv.org/abs/1809.09600?utm_source=chatgpt.com "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"
[11]: https://aclanthology.org/2022.tacl-1.31/?utm_source=chatgpt.com "♫ MuSiQue: Multihop Questions via Single-hop ..."
[12]: https://arxiv.org/abs/2011.01060?utm_source=chatgpt.com "Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps"
[13]: https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/?utm_source=chatgpt.com "Hybrid search"
