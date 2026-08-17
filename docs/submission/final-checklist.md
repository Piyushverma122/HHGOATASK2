# Voice RAG — Final Submission Checklist

**HH Goa 2026 — Task 2 | Readiness & Compliance Checklist**

---

- [x] **Voice Input & Capture**: Web Audio API recording and 16kHz mono audio validation functional.
- [x] **Sarvam Saaras v3 STT**: Real API key configured, endpoint verified, and quota-protected.
- [x] **7 Indic Languages**: Hindi, English, Hinglish, Bengali, Tamil, Telugu, and Marathi supported and tested.
- [x] **MSMARCO-XI Ingestion**: 99,925 passages ingested with Unicode normalization and Parquet batch persistence.
- [x] **7 Multi-Strategy Chunkers**: Fixed, Overlap, Sentence, Paragraph, Semantic, Metadata, and Adaptive.
- [x] **Multilingual Embeddings**: `intfloat/multilingual-e5-small` dense vector representations.
- [x] **FAISS Vector Index**: `IndexFlatIP` inner-product cosine search.
- [x] **Sparse Lexical Search**: RankBM25 inverted index over multilingual subwords.
- [x] **Hybrid Retrieval & RRF**: Reciprocal Rank Fusion (RRF_K=60) with candidate deduplication.
- [x] **Cross-Encoder Reranker**: Real Transformer `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` with batch size 16.
- [x] **Grounded LLM Generation**: Grounded generation with strict prompt templates.
- [x] **Citation Verification**: Chunk ID matching and claim verification.
- [x] **Input Safety Guardrail**: Prompt injection and character length defenses.
- [x] **Context Safety Guardrail**: Relevance score thresholding and token budgeting.
- [x] **Fast Abstention**: Graceful abstention on ungrounded/out-of-domain queries.
- [x] **Latency Optimization**: Concurrent parallel retrieval, JIT model warmup, and TTL query caching.
- [x] **Strict <200ms Compliance**: Measured warm $P_{100} = 59.96\text{ ms} < 200.0\text{ ms}$ (**100% PASS**).
- [x] **Benchmark Artifacts**: `final_latency_report.json`, `final_latency_report.md`, `submission_metrics.json`.
- [x] **OpenAPI Documentation**: Swagger UI at `/docs` and ReDoc at `/redoc`.
- [x] **Zero Secret Leaks**: `.env` ignored, sanitized health telemetry, zero secrets in repo.
- [x] **Automated Test Suite**: **127 / 127 tests passing (100% pass rate)**.
- [x] **Interactive Frontend**: Voice Studio, Retrieval Inspector, Guardrails Suite, and Latency Analytics.
- [x] **Reproducibility**: Docker compose configuration and quickstart documentation.
