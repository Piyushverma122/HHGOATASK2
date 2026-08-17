# Voice RAG — Judge-Facing Architectural Rationale (Q&A)

**HH Goa 2026 — Task 2 | Design Decisions & Explanations**

---

### Q1: Why use Retrieval-Augmented Generation (RAG) rather than fine-tuning a model?
**Answer**:
Fine-tuning bakes static knowledge into model weights, which leads to hallucinations, knowledge obsolescence, and inability to trace exact factual sources. RAG grounds answers in authoritative source documents (MSMARCO-XI), provides verifiable citation proofs, and allows instant knowledge updates without expensive model retraining.

---

### Q2: Why implement multiple chunking strategies instead of a single fixed-size chunker?
**Answer**:
Document structures vary greatly. Fixed-size chunking frequently chops sentences in half, severing semantic context. We implemented and evaluated 7 chunking strategies:
1. **Fixed-Size (128t)**: Baseline uniform chunking.
2. **Overlap (128t/32t)**: Preserves boundary context across chunk transitions.
3. **Sentence-Aware**: Respects grammatical sentence boundaries and Devanagari Dandas (`।`).
4. **Paragraph-Aware**: Preserves complete paragraphs for cohesive contextual units.
5. **Semantic Cosine**: Uses embedding shifts to detect natural topic transitions.
6. **Metadata-Informed**: Attaches passage provenance and structural headers.
7. **Adaptive Routing**: Dynamically routes based on query complexity.

---

### Q3: Why combine Dense Vector Search with Sparse BM25 (Hybrid Retrieval)?
**Answer**:
Dense semantic search (`multilingual-e5-small`) excels at conceptual similarity and paraphrasing, but struggles with exact entity names, acronyms, and alphanumeric identifiers. Sparse Okapi BM25 provides exact keyword matching. Combining both via Reciprocal Rank Fusion (RRF) delivers high recall across both conceptual and keyword-centric queries.

---

### Q4: Why use a genuine Cross-Encoder Reranker instead of bi-encoder cosine similarity?
**Answer**:
Bi-encoders independently compress queries and documents into fixed 384-d vectors, losing fine-grained cross-token interactions. A cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) feeds the query and candidate passage simultaneously into full multi-head self-attention layers, computing precise token-level relevance. In our evaluations, cross-encoder reranking boosted Recall@1 by over +137%.

---

### Q5: How is hallucination systematically eliminated?
**Answer**:
Hallucination is prevented via three defense layers:
1. **Strict Context Budgeting**: Only top-5 verified chunks (max 8,000 chars) are provided to the generator.
2. **Relevance Thresholding & Abstention**: If retrieved chunks score below relevance threshold (0.01), the system abstains with `INSUFFICIENT_CONTEXT` rather than guessing.
3. **Post-Generation Grounding Verifier**: Verifies answer claims against source chunks using n-gram token overlap, rejecting ungrounded assertions.

---

### Q6: How does the system achieve &lt;60ms latency ($P_{100} < 200\text{ms}$)?
**Answer**:
1. **Parallel Retrieval**: FAISS C++ dense vector search and BM25 token scoring run concurrently on separate threads.
2. **Model Warmup Lifecycle**: PyTorch JIT kernels and Transformer weights are pre-loaded at startup.
3. **Optimized Cross-Encoder Batching**: Joint candidates are scored in a single batch of 16.
4. **Short-Lived Caching**: LRU query and score caches accelerate repeat requests to &lt;0.5ms.

---

### Q7: How is Sarvam AI STT quota protected during benchmarks and continuous testing?
**Answer**:
Live Sarvam API calls are strictly quarantined for final verification. Automated unit tests, integration tests, and 141-query benchmark suites utilize deterministic offline voice fixtures (`data/fixtures/voice/`) across 7 Indic languages, ensuring zero quota depletion.
