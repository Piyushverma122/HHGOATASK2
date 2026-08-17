# Module 5.1 — Real Multilingual Cross-Encoder Reranker Documentation

**HH Goa 2026 — Task 2 | Production Multilingual Voice RAG System**  
*Technical Architecture, Mathematical Formulation, Empirical Evaluation, and Performance Analysis of the Real Multilingual Cross-Encoder Reranker.*

---

## 1. Previous Custom Reranker & Why It Was Not a True Cross-Encoder

In Module 5, the initial reranking implementation employed a handcrafted heuristic score combining:
- Soft TF-IDF subword character n-gram lexical coverage
- Exact contiguous phrase proximity bonuses
- 384-dimensional cosine similarity from pre-computed independent dense embeddings
- Composite sigmoid activation

### Fundamental Architectural Limitation
In a bi-encoder (dense retrieval), queries and candidate passages are mapped into vectors **independently**:
$$\vec{q} = f_\theta(q), \quad \vec{d} = f_\theta(d), \quad \text{Similarity} = \cos(\vec{q}, \vec{d})$$
No token in the query can attend to any token in the candidate passage during encoding.

A **true cross-encoder** processes the concatenated sequence simultaneously:
$$\text{Input} = [\text{CLS}] \, q_1 \, q_2 \dots q_m \, [\text{SEP}] \, d_1 \, d_2 \dots d_n \, [\text{SEP}]$$
Every attention head in every layer evaluates all-to-all cross-attention between each query token and each passage token:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
This enables the network to capture intricate semantic relations, entity coreferences, and context-dependent negation that cannot be modeled by independent bi-encoder projections.

---

## 2. Model Selection & Rationale

We evaluated candidate multilingual cross-encoders from HuggingFace Hub:

1. **`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`** (Selected as Primary Default):
   - **Architecture**: 12-layer multilingual MiniLMv2, 384 hidden dimension, 12 attention heads.
   - **Parameter Count**: ~117 Million parameters (~450 MB disk footprint).
   - **Training Corpus**: Fine-tuned on mMARCO (multilingual MS MARCO) covering Indic languages including Hindi and Bengali.
   - **Inference Latency**: ~70–100ms for 20 candidates on CPU in batch mode.
   - **Licensing**: Apache 2.0 open-source.

2. **`BAAI/bge-reranker-v2-m3`** (Supported High-Capacity Alternative):
   - **Architecture**: XLM-RoBERTa base, 1024 hidden dimension.
   - **Parameter Count**: ~568 Million parameters (~2.24 GB disk footprint).
   - **Capabilities**: Extreme multilingual capacity across 100+ languages.
   - **Inference Latency on CPU**: ~800–1400ms for 20 candidates without GPU acceleration.
   - **Practical Feasibility**: Available for CUDA GPU environments; `mmarco-mMiniLMv2` is selected for CPU-constrained sub-200ms budgets.

---

## 3. Architecture & Query-Passage Pair Construction

### Pipeline Flow
```
User Query
    │
    ▼
Query Normalization (NFC) & Linguistic Analysis
    │
    ├──► Dense Retrieval (Top 20)
    └──► Lexical BM25 Retrieval (Top 20)
    │
    ▼
Candidate Deduplication (by chunk_id)
    │
    ▼
Reciprocal Rank Fusion (RRF with k=60, Top 20 Pool)
    │
    ▼
Joint Tokenization: [[query, passage_1], [query, passage_2], ..., [query, passage_20]]
    │
    ▼
REAL MULTILINGUAL CROSS-ENCODER (Batch Size = 8/16/32, CPU/CUDA Auto-Detect)
    │
    ▼
Calibrated Sigmoidal Scoring & Sorting
    │
    ▼
Top 8 Final Context Chunks (with Metadata & Provenance)
```

---

## 4. Persistent SQLite Caching Architecture

To prevent redundant model forward passes, `RerankerCache` implements persistent disk storage in `data/cache/reranker_cache.sqlite3`.

### Cache Key Specification
```python
cache_key = SHA256(model_name + "::" + model_version + "::" + query.strip() + "::" + chunk_id.strip())
```
- Different chunks for the same query generate unique cache entries.
- Changing model version or model architecture invalidates previous cache entries automatically.

---

## 5. Multilingual Verification Smoke Test (7 Languages)

Evaluated across 7 languages (1 relevant vs 1 irrelevant passage per language):

| Language | Test Query | Relevant Score | Irrelevant Score | Delta ($\Delta$) | Status | Latency |
|---|---|---|---|---|---|---|
| **Hindi (hi)** | `भारत की राजधानी क्या है?` | **0.9958** | 0.0061 | +0.9897 | **PASSED** | 15.86 ms |
| **English (en)** | `What is the capital of India?` | **0.9992** | 0.0009 | +0.9983 | **PASSED** | 21.63 ms |
| **Hinglish (hi-Latn)** | `India ki capital kya hai?` | **0.5975** | 0.0014 | +0.5961 | **PASSED** | 15.60 ms |
| **Bengali (bn)** | `ভারতের राजधानी কী?` | **0.9884** | 0.0469 | +0.9415 | **PASSED** | 17.21 ms |
| **Tamil (ta)** | `இந்தியாவின் தலைநகரம் எது?` | **0.9409** | 0.0086 | +0.9323 | **PASSED** | 20.31 ms |
| **Telugu (te)** | `భారతదేశ రాజధాని ఏది?` | **0.9959** | 0.0080 | +0.9879 | **PASSED** | 16.20 ms |
| **Marathi (mr)** | `भारताची राजधानी कोणती आहे?` | **0.9955** | 0.0058 | +0.9897 | **PASSED** | 18.87 ms |

---

## 6. Empirical Retrieval Evaluation (100 MSMARCO-XI Hindi Queries)

### Strategy & Component Ablation Matrix

| Strategy | Configuration | Recall@1 | Recall@5 | MRR | Mean Warm Latency |
|---|---|---|---|---|---|
| **Fixed** | Dense Only | 0.110 | 0.370 | 0.224 | **11.66 ms** |
| **Fixed** | BM25 Only | 0.150 | 0.400 | 0.267 | **8.27 ms** |
| **Fixed** | Hybrid (Dense+BM25) | 0.140 | 0.430 | 0.265 | **16.88 ms** |
| **Fixed** | **Hybrid + Real Cross-Encoder** | **0.150** | **0.440** | **0.261** | **340.05 ms** |
| **Sentence** | Dense Only | 0.100 | 0.290 | 0.209 | **7.13 ms** |
| **Sentence** | BM25 Only | 0.150 | 0.390 | 0.259 | **7.52 ms** |
| **Sentence** | Hybrid (Dense+BM25) | 0.120 | 0.380 | 0.241 | **13.91 ms** |
| **Sentence** | **Hybrid + Real Cross-Encoder** | **0.180** | **0.500** | **0.294** | **377.60 ms** |
| **Adaptive** | Dense Only | 0.130 | 0.350 | 0.232 | **17.63 ms** |
| **Adaptive** | BM25 Only | 0.110 | 0.390 | 0.241 | **9.37 ms** |
| **Adaptive** | Hybrid (Dense+BM25) | 0.120 | 0.360 | 0.233 | **18.98 ms** |
| **Adaptive** | **Hybrid + Real Cross-Encoder** | **0.190** | **0.440** | **0.295** | **288.22 ms** |

---

## 7. Head-to-Head: CustomReranker vs Real CrossEncoderReranker

| Metric / Attribute | CustomReranker (Heuristic) | CrossEncoderReranker (Transformer) | Relative Change |
|---|---|---|---|
| **Adaptive Recall@1** | 0.080 | **0.190** | **+137.5%** |
| **Adaptive Recall@5** | 0.420 | **0.440** | **+4.8%** |
| **Adaptive MRR** | 0.221 | **0.295** | **+33.5%** |
| **Sentence Recall@1** | 0.090 | **0.180** | **+100.0%** |
| **Sentence Recall@5** | 0.450 | **0.500** | **+11.1%** |
| **Sentence MRR** | 0.234 | **0.294** | **+25.6%** |
| **Warm Inference Latency** | **~18 ms** | ~70–100 ms (Batch 32) | +50–80 ms CPU cost |
| **Model Footprint** | 0 MB (uses embedder) | 450 MB (mMiniLMv2) | Modular add-on |
| **Joint Self-Attention** | No (Cosine Projection) | **Yes (Full Multi-Head)** | Complete interaction |

---

## 8. Latency Percentiles & Batch Size Scaling

### Batch Size Scaling on CPU (20 Candidate Passages)
- **Batch Size 4**: `102.67 ms`
- **Batch Size 8 (Default)**: `102.53 ms`
- **Batch Size 16**: `84.26 ms`
- **Batch Size 32**: `71.94 ms` (Single batch evaluation)

### End-to-End Latency Percentiles (CPU Execution)
- **Fixed**: P50: 329.96ms, P70: 349.05ms, P90: 379.60ms, P95: 397.49ms, Mean: 340.05ms
- **Sentence**: P50: 376.87ms, P70: 393.80ms, P90: 430.21ms, P95: 440.35ms, Mean: 377.60ms
- **Adaptive**: P50: 285.94ms, P70: 307.01ms, P90: 340.61ms, P95: 366.78ms, Mean: 288.22ms

---

## 9. Engineering Recommendations

1. **Retrieval Precision**: The real cross-encoder delivers major gains in top-rank accuracy (**Recall@1 +137%**, **MRR +33.5%** on Adaptive chunking) compared to handcrafted heuristics.
2. **Chunking Strategy**: **Adaptive Routing** achieves the highest MRR (0.295) and the lowest mean latency (288ms un-cached, <20ms with SQLite cache).
3. **Voice RAG Optimization**: For sub-200ms real-time voice RAG on CPU, configure `RERANK_BATCH_SIZE=32` with a max sequence length of 192 tokens or enable SQLite cache warm hits.
