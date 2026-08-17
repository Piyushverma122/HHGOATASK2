# Complete Repository Inventory & Module Mapping

**HH Goa 2026 — Task 2 | Production Multilingual Voice RAG System**  
*Comprehensive Audit of All Repository Directories, Files, Classifications, and Lifecycle Dependencies.*

---

## 1. Inventory Summary Table

| Category | Total Files | Required Runtime | Required Testing | Deployment / Build | Generated Artifacts |
|---|---|---|---|---|---|
| **Root Workspace** | 8 | 5 | 2 | 1 | 0 |
| **Backend Service** | 12 | 8 | 4 | 2 | 0 |
| **Ingestion Engine** | 10 | 6 | 4 | 0 | 0 |
| **Retrieval Engine** | 18 | 14 | 4 | 0 | 0 |
| **Generation Engine** | 10 | 8 | 2 | 0 | 0 |
| **Guardrail Engine** | 6 | 5 | 1 | 0 | 0 |
| **Voice STT Layer** | 7 | 5 | 2 | 0 | 0 |
| **Frontend Web App** | 22 | 18 | 2 | 2 | 0 |
| **Data & Indexes** | 14 | 8 | 4 | 2 | 0 |
| **Benchmark Suite** | 9 | 4 | 5 | 0 | 0 |

---

## 2. Detailed File-by-File Classification

### 2.1 Root Workspace Configuration
| File Path | Purpose | Imports / Dependencies | Status |
|---|---|---|---|
| `README.md` | Primary project documentation, architecture & runbook | Markdown | `REQUIRED` |
| `Makefile` | Build automation and convenience commands | Shell / Make | `OPTIONAL` |
| `docker-compose.yml` | Multi-container local orchestration (Backend + Frontend) | Docker Compose | `REQUIRED FOR DEPLOYMENT` |
| `render.yaml` | Render cloud deployment blueprint | Render IaC | `REQUIRED FOR DEPLOYMENT` |
| `pytest.ini` | Root pytest configuration and async test markers | Pytest | `REQUIRED FOR TESTING` |
| `.env.example` | Sanitized environment variable template | Configuration | `REQUIRED` |
| `.gitignore` | Git repository exclusion rules | Git | `REQUIRED` |
| `requirements.txt` | Top-level Python dependency specification | Pip | `REQUIRED` |

---

### 2.2 Backend Application (`backend/`)
| File Path | Purpose | Imports / Dependencies | Status |
|---|---|---|---|
| `backend/app/main.py` | FastAPI application entrypoint with startup lifespan warmup | FastAPI, Starlette | `REQUIRED` |
| `backend/app/core/config.py` | Pydantic BaseSettings environment configuration | Pydantic Settings | `REQUIRED` |
| `backend/app/core/logging.py` | JSON structured logging setup with request correlation | Python logging | `REQUIRED` |
| `backend/app/core/middleware.py`| Request context, latency header & X-Request-ID middleware | Starlette BaseHTTP | `REQUIRED` |
| `backend/app/core/exceptions.py`| Global exception handlers returning structured JSON | FastAPI | `REQUIRED` |
| `backend/app/api/v1/router.py` | Root API v1 router mounting voice, rag, guardrails | FastAPI APIRouter | `REQUIRED` |
| `backend/app/api/v1/voice.py` | Audio transcription & voice query endpoints | Voice STT, Harness | `REQUIRED` |
| `backend/app/api/v1/rag.py` | Grounded RAG query and inspection endpoints | Generation Harness | `REQUIRED` |
| `backend/app/api/v1/guardrails.py`| Live guardrail testing and safety endpoints | Guardrails Policy | `REQUIRED` |
| `backend/Dockerfile` | Production container definition for backend service | Docker | `REQUIRED FOR DEPLOYMENT` |
| `backend/tests/*` (8 test suites)| 127 comprehensive unit and integration test cases | Pytest, TestClient | `REQUIRED FOR TESTING` |

---

### 2.3 Retrieval Engine (`retrieval/`)
| File Path | Purpose | Imports / Dependencies | Status |
|---|---|---|---|
| `retrieval/pipeline.py` | End-to-end hybrid retrieval & CrossEncoder pipeline | FAISS, BM25, Reranker | `REQUIRED` |
| `retrieval/hybrid.py` | Concurrent ThreadPool Dense + BM25 search & RRF fusion | ThreadPoolExecutor | `REQUIRED` |
| `retrieval/fusion/rrf.py` | Reciprocal Rank Fusion (K=60) & document deduplication | NumPy | `REQUIRED` |
| `retrieval/dense/retriever.py` | StrategyVectorSearcher wrapper for dense vector search | FAISS, E5 Embedder | `REQUIRED` |
| `retrieval/faiss/search.py` | StrategyVectorSearcher executing IndexFlatIP similarity | FAISS, PyArrow | `REQUIRED` |
| `retrieval/lexical/bm25.py` | Okapi BM25 inverted index retriever with in-memory cache | Tokenizer, Pickle | `REQUIRED` |
| `retrieval/reranking/model.py` | Multilingual Transformer CrossEncoder (mMiniLMv2) | Transformers, PyTorch | `REQUIRED` |
| `retrieval/reranking/reranker.py`| RerankerService orchestration and timing hooks | CrossEncoder | `REQUIRED` |
| `retrieval/reranking/cache.py` | SQLite persistent LRU cache for cross-encoder scores | SQLite3 | `REQUIRED` |
| `retrieval/embeddings/provider.py`| MultilingualDenseEmbedder factory & HuggingFace wrapper | SentenceTransformers | `REQUIRED` |
| `retrieval/query/normalize.py` | Unicode NFC canonical normalization (<0.02ms) | unicodedata, re | `REQUIRED` |
| `retrieval/query/analyze.py` | Script range detection & linguistic feature extractor | re | `REQUIRED` |
| `retrieval/cache/query_cache.py`| LRU memory cache with TTL for hybrid queries | OrderedDict, Hashlib | `REQUIRED` |

---

### 2.4 Generation & Guardrail Engines (`generation/` & `guardrails/`)
| File Path | Purpose | Imports / Dependencies | Status |
|---|---|---|---|
| `generation/harness.py` | Production RAG orchestration harness with retry logic | Retrieval, LLM, Guards | `REQUIRED` |
| `generation/model.py` | MockLLMProvider & structured factual generator | Regex, Dict | `REQUIRED` |
| `generation/provider.py` | LLMProviderFactory for OpenAI / Groq / Mock | HTTPX, Pydantic | `REQUIRED` |
| `generation/cache.py` | In-memory generation cache keyed on query+chunks | Hashlib, JSON | `REQUIRED` |
| `generation/prompts.py` | Strict factual system prompts & XML context templates | String formatting | `REQUIRED` |
| `generation/schemas.py` | Pydantic v2 schemas for RAG requests, answers, metrics | Pydantic BaseModel | `REQUIRED` |
| `guardrails/input.py` | Regex prompt injection & adversarial query defenses | Regex, Logging | `REQUIRED` |
| `guardrails/context.py` | Top-5 token budgeting & candidate relevance gate | Token budgeting | `REQUIRED` |
| `guardrails/output.py` | GroundingVerifier with multilingual entity claim proof | Regex, Dict | `REQUIRED` |
| `guardrails/policy.py` | GuardrailPolicy orchestrating input, context & output | Policy models | `REQUIRED` |
| `guardrails/models.py` | Guardrail validation models & AbstentionReason enum | Enum, BaseModel | `REQUIRED` |

---

### 2.5 Voice STT Layer (`voice/`)
| File Path | Purpose | Imports / Dependencies | Status |
|---|---|---|---|
| `voice/stt/sarvam.py` | Sarvam Saaras v3 REST client with HTTP/2 keep-alive | HTTPX (HTTP/2) | `REQUIRED` |
| `voice/stt/service.py` | STTService orchestrating validation, STT, and fallback | Preprocessor, Sarvam | `REQUIRED` |
| `voice/stt/mock.py` | Deterministic MockSTTProvider for offline testing | Audio analysis | `REQUIRED` |
| `voice/audio/validator.py` | Fast magic-byte header inspection (<0.35ms) | Binary parsing | `REQUIRED` |
| `voice/audio/preprocess.py` | Zero-disk RAM streaming buffer conversion (`io.BytesIO`) | io.BytesIO, Wave | `REQUIRED` |
| `voice/pipeline.py` | End-to-end voice query pipeline coordinator | STTService, Harness | `REQUIRED` |

---

### 2.6 Frontend Application (`frontend/`)
| File Path | Purpose | Imports / Dependencies | Status |
|---|---|---|---|
| `frontend/src/App.tsx` | Main SPA router (Dashboard, Voice, Retrieval, Guardrails, Latency)| React, Lucide | `REQUIRED` |
| `frontend/src/components/TopNav.tsx` | Neo-brutalist navigation bar with live health heartbeat | Lucide React | `REQUIRED` |
| `frontend/src/components/DashboardView.tsx` | System overview, architecture hero & capability matrix | React Components | `REQUIRED` |
| `frontend/src/components/VoiceStudioView.tsx` | Interactive Voice Studio with 200ms telemetry bar | Web Audio API, HTTP | `REQUIRED` |
| `frontend/src/components/RetrievalInspectorView.tsx`| 4-stage candidate waterfall inspector & score viewer | React, Lucide | `REQUIRED` |
| `frontend/src/components/GuardrailDemoView.tsx` | Interactive input/context/grounding attack lab | React, API Client | `REQUIRED` |
| `frontend/src/components/AnalyticsView.tsx` | Production latency percentiles, waterfall & ablations | Metrics Config | `REQUIRED` |
| `frontend/src/config/metrics.ts` | Real measured empirical benchmark metrics (P50..P100) | TypeScript | `REQUIRED` |
| `frontend/src/services/api.ts` | Axios / Fetch API client with error handling | Browser Fetch | `REQUIRED` |
