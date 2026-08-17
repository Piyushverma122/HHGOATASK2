# Dead Code, Redundancy & Repository Cleanup Report

**HH Goa 2026 — Task 2 | Production Multilingual Voice RAG System**  
*Systematic Analysis of Unused Modules, Temporary Artifacts, and Safe Removals.*

---

## 1. Executive Summary

A repository-wide search was conducted for unused Python files, abandoned prototypes, redundant components, duplicate implementations, and generated temporary artifacts.

---

## 2. Action Classification Table

| File / Component | Type | Initial Finding | Audit Determination | Action Taken |
|---|---|---|---|---|
| `scratch_model.py` | Root Python File | Accidental UTF-16 temporary scratch script | Zero references, 36KB obsolete file | **SAFE TO DELETE (Deleted)** |
| `frontend/src/components/RetrievalInspectorView.tsx` | React Component | Unrouted component during previous nav cleanup | Required by Task 2 Requirement 19 | **RETAINED & RE-INTEGRATED** |
| `retrieval/reranking/cache.py` | Python Module | SQLite cache for CrossEncoder scores | Active in `CrossEncoderReranker` | **RETAINED** |
| `voice/stt/mock.py` | Python Module | Deterministic offline STT provider | Active in offline tests & fallback | **RETAINED** |
| `generation/cache.py` | Python Module | GenerationCache for identical queries | Active in `RAGHarness` | **RETAINED** |
| `data/statistics/final_latency_debug.json` | JSON Metric Data | 20-query real empirical benchmark data | Required for latency transparency | **RETAINED** |

---

## 3. Duplicate Implementation Audit

| Area | Verified Canonical Implementation | Duplicate / Obsolete Paths | Resolution |
|---|---|---|---|
| **Query Normalization** | [`retrieval/query/normalize.py`](file:///d:/Piyush%20Project/GOA%20TASK%202/retrieval/query/normalize.py) | Ingestion text normalizer | Standardized on `retrieval.query.normalize` |
| **BM25 Retrieval** | [`retrieval/lexical/bm25.py`](file:///d:/Piyush%20Project/GOA%20TASK%202/retrieval/lexical/bm25.py) | None | Added `_BM25_CACHE` singleton |
| **Dense Search** | [`retrieval/faiss/search.py`](file:///d:/Piyush%20Project/GOA%20TASK%202/retrieval/faiss/search.py) | None | Cached via `_SEARCHER_CACHE` |
| **Reranker** | [`retrieval/reranking/model.py`](file:///d:/Piyush%20Project/GOA%20TASK%202/retrieval/reranking/model.py) | Heuristic custom reranker | Retained CrossEncoder as default, custom as legacy ablation |

---

## 4. Retained & Protected Critical Components
All 25 core Task 2 requirements remain strictly preserved:
- MSMARCO-XI Dataset & 7 Chunking Strategies
- Multilingual Embeddings & FAISS + BM25 Concurrent Hybrid Search
- Reciprocal Rank Fusion ($K=60$) & Cross-Encoder Reranking
- Sarvam Saaras v3 STT & Grounded Structured Answer Generation
- Input, Context, and Output Guardrails with Abstention
- Voice Studio, Retrieval Inspector, Guardrail Attack Lab, and Latency Analytics Views
- 127/127 Passing Pytest Suite
