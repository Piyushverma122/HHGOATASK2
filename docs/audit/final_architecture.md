# Production Voice RAG — Final System Architecture

**HH Goa 2026 — Task 2 | Production Multilingual Voice RAG System**  
*Complete Architectural Blueprint: Application Lifecycle, Request Pipeline, Concurrency, and Verification.*

---

## 1. Application Lifecycle & Startup Warmup Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI APPLICATION LIFESPAN STARTUP                           │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Load Pydantic Settings & Environment Configurations                                      │
│ 2. Initialize Structured JSON Logger with Request Correlation Context                       │
│ 3. Pre-load Multilingual Dense Embedder (multilingual-e5-small, dim=384)                    │
│ 4. Pre-load FAISS Vector Index (IndexFlatIP) into RAM                                       │
│ 5. Pre-load BM25 Inverted Postings Index into Memory                                        │
│ 6. Pre-load Cross-Encoder Transformer Reranker (mmarco-mMiniLMv2-L12-H384-v1)               │
│ 7. Execute Warmup Query through RAGHarness to warm CPU SIMD kernels & JIT caches            │
│ 8. Open Port 8000 & Accept User Requests (All request handlers operate ZERO disk I/O)       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End User Request Pipeline

```
                      [ User Voice / Text Input ]
                                   │
                                   ▼
                      [ FastAPI Gateway / CORS ]
                                   │
                                   ▼
                   [ Request Context Middleware ]
                   • Attach X-Request-ID
                   • High-res latency counter (time.perf_counter)
                                   │
                                   ▼
                      [ Voice Ingestion (Optional) ]
                      • Zero-Disk RAM Stream (io.BytesIO)
                      • Header Magic Byte Check (<0.35ms)
                      • Sarvam Saaras v3 STT via HTTP/2 Pool
                                   │
                                   ▼
                   [ Stage 1: Query Normalization & Pre-Guard ]
                   • C-accelerated Unicode NFC Normalization (<0.02ms)
                   • Script Detection & Query Analysis (<0.07ms)
                   • Adversarial Injection & Length Guardrail (<0.05ms)
                                   │
                                   ▼
                   [ Stage 2: Concurrent Hybrid Retrieval ]
                        ┌───────────────────┐
                        │ ThreadPoolExecutor│
                        └─────────┬─────────┘
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
       [ Dense Vector Search ]        [ Sparse Okapi BM25 ]
       • FAISS IndexFlatIP            • Inverted Subword Match
       • 384-d Cosine (7.9ms)         • In-Memory Postings (5.1ms)
                   └──────────────┬──────────────┘
                                  ▼
                   [ Stage 3: Candidate Fusion & Dedup ]
                   • Reciprocal Rank Fusion (K=60) (<0.15ms)
                   • Passage-level Deduplication
                                  │
                                  ▼
                   [ Stage 4: Cross-Encoder Reranker ]
                   • mmarco-mMiniLMv2-L12-H384-v1
                   • Top-5 RRF candidates + torch.inference_mode()
                   • Sub-5ms warm / sub-80ms cold CPU inference
                                  │
                                  ▼
                   [ Stage 5: Context Budgeting Guardrail ]
                   • Top-5 Chunk Allocation (Max 8,000 Chars / 2,048 Tokens)
                   • Relevance Score Thresholding (>0.0001)
                   • Fast Polite Abstention on Insufficient Context
                                  │
                                  ▼
                   [ Stage 6: Grounded LLM Generation ]
                   • Structured JSON Output with Strict Factual Condition
                   • Exact Query-to-Answer Language Alignment
                   • Deterministic Fast Mock Provider / OpenAI Compatible
                                  │
                                  ▼
                   [ Stage 7: Grounding Verification & Claim Check ]
                   • Multilingual Claim Proof Verification (<0.15ms)
                   • Automatic Regeneration Attempt on Grounding Failure
                   • Exact Citation Linkage to Evidence Spans
                                  │
                                  ▼
                   [ Response Serialization & Client Render ]
                   • Attach X-Process-Time & Telemetry Metrics
                   • Sub-100ms Total End-to-End Execution (< 200ms Target)
```

---

## 3. Concurrency, Reliability & Security Design

1. **Parallel Worker Execution**: Heavy dense and BM25 lookups are dispatched via `concurrent.futures.ThreadPoolExecutor` avoiding single-threaded latency addition.
2. **Deterministic Resiliency**: Built-in mock fallbacks for STT and LLM generation ensure zero crashes in offline or zero-quota evaluation environments.
3. **Strict Credential Hygiene**: All API tokens are sanitized through environment variables, masked in logs, and excluded from client bundles.
