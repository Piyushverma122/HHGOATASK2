# VOICE RAG

**Voice-Enabled Multilingual Retrieval-Augmented Generation**  
*HH Goa 2026 — Task 2 Submission*

---

## 1. Executive Summary

**VOICE RAG** is a production-grade, voice-first Retrieval-Augmented Generation (RAG) platform engineered for high-accuracy, sub-60ms conversational knowledge retrieval across Indic languages. The platform captures spoken queries via the Web Audio API, transcribes them using the **Sarvam Saaras v3** foundation model, executes concurrent hybrid retrieval (384-dimensional dense FAISS vectors + subword Okapi BM25), fuses candidates via **Reciprocal Rank Fusion (RRF)**, applies a true multilingual **Cross-Encoder reranker** (`mmarco-mMiniLMv2`), evaluates multi-stage **security & grounding guardrails**, and synthesizes factual answers with verifiable citation proofs.

> **Key Accomplishments**:
> - **Warm Max Latency (P100)**: **59.955 ms** ($\le 200\text{ms}$ Target PASS — **3.33x faster**).
> - **Accuracy & Retrieval**: **MRR 1.000**, **Recall@1 100%**, **Recall@5 100%**, Recall@1 gain **+137%** via Cross-Encoder.
> - **Test Suite**: **127 / 127 Unit & Integration Tests Passing (100%)**.
> - **High Concurrency**: **70.92 QPS** at 50 virtual users with **0.0% error rate**.
> - **Frontend UI/UX**: Neo-Brutalist design system with Claude-inspired Editorial Serif Typography (`Newsreader`).
> - **Multilingual Coverage**: 7 live demo Indic scripts (Hindi, English, Hinglish, Bengali, Tamil, Telugu, Marathi).

---

## 2. End-to-End System Architecture

```
[ User Microphone Input / 1-Click Fixtures ]
                     │
                     ▼
[ Sarvam Saaras v3 STT + Unicode NFC Normalizer ]
                     │
                     ▼
[ Query Normalizer & Subword Token Extraction ]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
[ FAISS Dense IndexFlatIP ]   [ Okapi BM25 Sparse Search ]
   (384-d E5 Vectors)          (Subword Inverted Index)
         └───────────┬───────────┘
                     │ (Parallel ThreadPoolExecutor)
                     ▼
[ Reciprocal Rank Fusion (K=60) + MD5 Deduplication ]
                     │
                     ▼
[ Multilingual Cross-Encoder Reranker (mmarco-mMiniLMv2) ]
                     │
                     ▼
[ Multi-Stage Guardrails (Injection Scan <0.05ms, Budget, Policy) ]
                     │
                     ▼
[ Grounded LLM Synthesis + N-Gram Claim Alignment Verification ]
                     │
                     ▼
[ Structured Response with Verified Citation Evidence Drawer ]
```

---

## 3. Core Modules & Technical Specifications

| Module | Component | Implementation Details |
|---|---|---|
| **STT Layer** | Sarvam Saaras v3 | 16kHz mono audio validation (0.2s–30s), exponential backoff retries, and quota-protected deterministic fixtures. |
| **Ingestion** | MSMARCO-XI Ingestion | Streaming processor for 99,925 Hindi validation passages with Devanagari Danda (`।`) normalization and Parquet batching. |
| **Chunking** | 7 Strategies | Adaptive routing, Semantic cosine boundary, Sentence-aware, Paragraph-aware, Overlap, Fixed-size, and Metadata-informed chunkers. |
| **Dense Search** | FAISS FlatIP | `intfloat/multilingual-e5-small` (384-d dense vectors) with inner-product cosine distance search. |
| **Sparse Search** | Okapi BM25 | Subword tokenized inverted index for exact keyword, numerical, and named-entity matching. |
| **Fusion & Rank** | Parallel RRF | Concurrent retrieval with `ThreadPoolExecutor (max_workers=4)` fused with Reciprocal Rank Fusion ($K=60$). |
| **Reranker** | Cross-Encoder | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` computing joint query-passage cross-attention in batches of 16. |
| **Guardrails** | Security & Safety | <0.05ms regex injection defense, context token budget limit (<8k chars), and relevance threshold (>0.01). |
| **Generation** | Grounded Synthesis | Strict factual prompt template conditioned exclusively on retrieved chunks with verifiable citation pointers. |
| **Telemetry** | Observability | Granular stage-by-stage millisecond timing headers (`X-Process-Time`, `X-Request-ID`) and structured JSON logs. |

---

## 4. Benchmark & Latency Telemetry

Measured over **141 MSMARCO-XI validation queries** with warm JIT model weights:

| Metric | Measured Warm Latency | Target Threshold | Compliance Status |
|---|---|---|---|
| **P50 (Median)** | **24.897 ms** | $< 200\text{ms}$ | **PASS** |
| **P70** | **31.010 ms** | $< 200\text{ms}$ | **PASS** |
| **P90** | **46.480 ms** | $< 200\text{ms}$ | **PASS** |
| **P95** | **51.042 ms** | $< 200\text{ms}$ | **PASS** |
| **P99** | **56.753 ms** | $< 200\text{ms}$ | **PASS** |
| **P100 (Max Latency)** | **59.955 ms** | $< 200\text{ms}$ | **PASS (3.33x margin)** |
| **Mean Latency** | **23.816 ms** | $< 200\text{ms}$ | **PASS** |

### Granular Stage-by-Stage Breakdown
1. **Query Prep & Normalization**: `0.15 ms`
2. **Input Guardrails & Security Scan**: `0.05 ms`
3. **Concurrent Parallel Retrieval (FAISS + BM25)**: `19.74 ms`
4. **RRF Candidate Fusion & Deduplication**: `0.20 ms`
5. **Cross-Encoder Joint Reranking (Batch 16)**: `4.17 ms`
6. **Context Budgeting & Preparation**: `0.02 ms`
7. **Grounded Answer Generation**: `5.66 ms`
8. **Claim & Citation Verification**: `0.15 ms`
9. **Structured Output Assembly**: `0.05 ms`

---

## 5. Quickstart & Local Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11 & 3.14)
- Node.js 18+ and npm
- (Optional) Docker & Docker Compose

### 1. Clone & Environment Configuration
```bash
git clone https://github.com/your-username/voice-rag.git
cd voice-rag

# Copy environment template
cp .env.example .env
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Run FastAPI backend in development mode
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** to access the Voice RAG workspace.

---

## 6. Docker Compose Deployment

To build and run the entire stack locally with production container settings:

```bash
docker compose up --build
```
- **Frontend Dashboard**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **Swagger Documentation**: `http://localhost:8000/docs`

---

## 7. Running the Automated Test Suite

```bash
# Run full test suite with verbose output
python -m pytest backend/tests -v
```
All **127 tests** will execute covering chunking, embeddings, FAISS, hybrid retrieval, cross-encoders, Sarvam STT, guardrails, generation, optimization, and security hardening.

---

## 8. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Sanitized system health status, version, and provider configurations. |
| `GET` | `/api/v1/voice/info` | Sarvam STT provider metadata and audio format constraints. |
| `GET` | `/api/v1/rag/info` | Grounded RAG configuration, thresholds, and guardrail policies. |
| `POST` | `/api/v1/voice/transcribe` | Transcribe 16kHz mono audio into Indic Unicode text. |
| `POST` | `/api/v1/voice/query` | End-to-end Voice $\to$ STT $\to$ Hybrid Retrieval $\to$ Grounded Answer. |
| `POST` | `/api/v1/voice/text-query` | Text fallback query with full hybrid retrieval and citation proofs. |
| `POST` | `/api/v1/rag/query` | Direct text RAG query with strategy and top-K configuration. |
| `POST` | `/api/v1/rag/inspect` | Inspect intermediate candidate pools across Dense, BM25, RRF, and Cross-Encoder. |

---

## 9. Security & Secret Governance

- All credentials (`SARVAM_API_KEY`, `LLM_API_KEY`) are strictly managed through environment variables.
- `.env` and local indexes are excluded from version control via `.gitignore`.
- Automated secret scanning confirms zero hardcoded API keys in repository tracking.

---

## 10. Submission Evidence & Verification

- **Final Demo Checklist**: [`docs/submission/final-demo-checklist.md`](docs/submission/final-demo-checklist.md)
- **Final Submission Document**: [`docs/submission/final-submission.md`](docs/submission/final-submission.md)
- **Full Walkthrough & Verification Report**: [`walkthrough.md`](walkthrough.md)

---

© 2026 Voice RAG Team • HH Goa 2026 Task 2 Submission
