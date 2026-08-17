# Voice RAG — System Architecture & Data Flow

**HH Goa 2026 — Task 2 | Production System Architecture**

---

## 1. End-to-End Multimodal Pipeline

```
[ Multimodal User Input ]
   │
   ├─► Voice Recording (Web Audio API / WAV 16kHz Mono / 7 Indic Languages)
   │        │
   │        ▼
   │   [ Sarvam Saaras v3 STT ] ──► (NFC Normalization / Language Classification)
   │                                      │
   └─► Text Query (Hindi, En, Hinglish)───┘
                                          │
                                          ▼
                         [ Stage 1: Input Guardrail Pre-Check ]
                         (Regex Injections, Jailbreak, Length, Empty Defense)
                                          │
                                          ▼
                      [ Stage 2: Parallel Concurrent Retrieval ]
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
         [ Dense Vector Search ]                     [ Sparse BM25 Search ]
         • Multilingual-E5-Small                     • RankBM25 Token Inverted Index
         • FAISS IndexFlatIP Cosine Projection       • Indic Subword Tokenization
         • Top-20 Candidate Pool                     • Top-20 Candidate Pool
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      [ Stage 3: Candidate Fusion & Deduplication ]
                      • Reciprocal Rank Fusion (RRF_K = 60)
                      • Candidate Deduplication by Passage ID
                                          │
                                          ▼
                      [ Stage 4: Cross-Encoder Transformer Reranker ]
                      • mmarco-mMiniLMv2-L12-H384-v1
                      • Joint (Query, Passage) Token Cross-Attention
                      • Batch Size 16 Vectorized Scoring + LRU Caching
                                          │
                                          ▼
                      [ Stage 5: Context Guardrail & Budgeting ]
                      • Top-5 Chunk Allocation (Max 8,000 Chars / 2,048 Tokens)
                      • Relevance Score Thresholding (>0.01)
                      • Fast Abstention on Zero-Relevance Contexts
                                          │
                                          ▼
                      [ Stage 6: Grounded LLM Generation ]
                      • System Instruction: Strictly Factual Grounded Responses
                      • OpenAI-Compatible Provider / Fallback Mock Engine
                      • Structured JSON Response with Exact Citations
                                          │
                                          ▼
                      [ Stage 7: Grounding Verification & Claim Check ]
                      • N-Gram Token Verification against Context Chunks
                      • Citation Proof Validation & Abstention Triggering
                                          │
                                          ▼
                      [ Structured Final Response + Micro-Latency Breakdown ]
```

---

## 2. Component Design & Abstraction Layers

### 2.1 Audio Capture & STT Layer (`voice/`)
- **Speech-to-Text Provider**: `SarvamSTTProvider` implementing `BaseSTTProvider`.
- **Audio Validation**: Validates WAV/WebM/MP3 formats, 16kHz mono conversion, amplitude sanity checks, and length boundaries ($0.2\text{s} \le \text{duration} \le 30.0\text{s}$).
- **Resilience**: Exponential backoff with jitter on network timeouts or HTTP 5xx responses.

### 2.2 Ingestion & Multi-Strategy Chunking (`ingestion/` & `data/chunks/`)
- **Dataset**: `ai4bharat/MSMARCO-XI` (Hindi validation split, 99,925 passages).
- **Multi-Strategy Chunkers**:
  1. *Fixed-Size*: Deterministic 128-token chunks.
  2. *Overlap*: 128-token chunks with 32-token sliding overlap.
  3. *Sentence-Aware*: Punctuation & Devanagari Danda (`।`) boundary chunking.
  4. *Paragraph-Aware*: Double newline boundary splits.
  5. *Semantic Cosine*: Embedding-driven semantic shift detection.
  6. *Metadata-Informed*: Structure-aware chunking preserving title/provenance.
  7. *Adaptive Routing*: Dynamic routing based on query complexity and length.

### 2.3 Hybrid Retrieval & Cross-Encoder Layer (`retrieval/`)
- **Dense FAISS Retriever**: 384-dimensional cosine inner-product indexing using `intfloat/multilingual-e5-small`.
- **Lexical BM25 Retriever**: Inverted index scoring over multilingual subwords.
- **Concurrent Search**: `ThreadPoolExecutor` parallelizes dense and sparse searches.
- **Cross-Encoder Reranker**: Pretrained Transformer model `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` performing full self-attention across joint query-passage pairs.

### 2.4 Grounded Generation & Guardrails Layer (`generation/` & `guardrails/`)
- **LLM Provider**: `OpenAICompatibleProvider` with `MockLLMProvider` deterministic fallback.
- **Input Guardrail**: Evaluates regex patterns, character limits, and prompt injection attacks.
- **Context Guardrail**: Enforces strict token budgets and relevance thresholds.
- **Grounding Verifier**: Computes n-gram claim overlap between generated answers and retrieved text chunks, rejecting hallucinated claims.
