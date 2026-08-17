# Data & Index Persistence Audit

**HH Goa 2026 — Task 2 | Production Multilingual Voice RAG System**  
*Verification of Dataset Integrity, Vector Index Persistence, Storage Strategy, and Regeneration Runbook.*

---

## 1. Overview & Artifact Status

The RAG system depends on two primary artifact directories: `data/` and `indexes/`.

| Artifact / Index Path | Contents & Format | Size | Runtime Role | Persistence Strategy |
|---|---|---|---|---|
| `data/processed/msmarco_xi_hi_validation.parquet` | 99,925 Hindi validation passages & metadata | ~47 MB | Golden evaluation corpus | Committed / Local storage |
| `data/processed/sample_hi_validation.jsonl` | 1,000 sampled passage benchmark fixture | ~1.3 MB | Fast test & CI evaluation | Committed |
| `indexes/faiss/adaptive/` | FAISS `IndexFlatIP` (384-d) + `metadata.parquet` | ~42 MB | Dense vector similarity search | Persistent Disk / RAM cache |
| `indexes/bm25/adaptive/` | Inverted postings `bm25_index.pkl` + `metadata.parquet` | ~28 MB | Sparse lexical subword search | Persistent Disk / RAM cache |
| `data/statistics/` | Benchmark JSON & Markdown telemetry reports | <1 MB | Transparent metric visualization | Committed |

---

## 2. Production Storage Strategy

Because `indexes/` contains precomputed vector and BM25 postings, production container deployments can either:
1. **Include pre-built indexes in the container image**: Copy `indexes/` during Docker build (`COPY indexes/ /app/indexes/`).
2. **Mount a persistent volume**: Mount a persistent block volume to `/app/indexes`.
3. **Execute the automated offline builder** at deployment startup if missing:
   ```bash
   python -m ingestion.pipeline --input data/raw/hinval.parquet --strategy adaptive
   python -m retrieval.faiss.builder --strategy adaptive
   python -m retrieval.lexical.builder --strategy adaptive
   ```

---

## 3. In-Memory Startup Caching Architecture

To prevent request-time disk I/O:
- `StrategyVectorSearcher` is cached globally in `retrieval/vector_search.py` (`_SEARCHER_CACHE`).
- `BM25Retriever` is cached globally in `retrieval/lexical/bm25.py` (`_BM25_CACHE`).
- Metadata parquet records are mapped to in-memory Python hash maps on initial load.
- Startup lifespan in `backend/app/main.py` executes a single warm query to pre-load all data structures into RAM prior to opening traffic ports.
