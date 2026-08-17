# Voice RAG — REST API Documentation

**HH Goa 2026 — Task 2 | FastAPI OpenAPI Specifications**

---

## Base URL
- Local: `http://localhost:8000/api/v1`
- Swagger Interactive UI: `http://localhost:8000/docs`
- ReDoc Interactive UI: `http://localhost:8000/redoc`

---

## 1. Health & Status Endpoints

### `GET /api/v1/health`
Returns sanitized service health, environment mode, and provider configurations without secrets.
- **Request**: No parameters.
- **Response**:
```json
{
  "status": "ok",
  "service": "voice-rag",
  "version": "1.0.0",
  "environment": "development",
  "providers": {
    "sarvam": {
      "configured": true,
      "model": "saaras:v3",
      "mock_mode": false
    },
    "llm": {
      "configured": true,
      "provider": "openai_compatible",
      "model": "meta-llama/llama-3.3-70b-instruct"
    }
  },
  "guardrails": {
    "enabled": true,
    "prompt_injection_defense": true,
    "max_context_chunks": 5
  }
}
```

---

## 2. Voice & STT Endpoints

### `GET /api/v1/voice/info`
Returns STT provider configuration, constraints, and supported audio formats.

### `POST /api/v1/voice/transcribe`
Transcribes audio file to text via Sarvam Saaras v3.
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `file`: Audio file (WAV/WebM/MP3, max 10MB, max 30s).
  - `language` *(optional)*: BCP-47 language code (`hi-IN`, `en-IN`, etc.).

### `POST /api/v1/voice/query`
Executes end-to-end Voice RAG pipeline:
`Audio -> Sarvam STT -> Normalization -> Analysis -> Parallel Retrieval -> Cross-Encoder -> Guardrails -> LLM -> Response`.
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `file`: Audio recording.
  - `strategy`: Chunking strategy (`adaptive`, `semantic`, etc.).
  - `top_k`: Number of final context chunks (default 5).

---

## 3. Grounded RAG & Inspection Endpoints

### `GET /api/v1/rag/info`
Returns active LLM provider configuration and guardrail thresholds.

### `POST /api/v1/rag/query`
Executes Grounded RAG lifecycle for text queries.
- **Content-Type**: `application/json`
- **Payload**:
```json
{
  "query": "भारत की राजधानी क्या है?",
  "strategy": "adaptive",
  "top_k": 5,
  "enable_reranking": true
}
```
- **Response**: Includes grounded answer, confidence score, citation list, abstention status, and latency breakdown.

### `POST /api/v1/rag/inspect`
Transparently returns intermediate candidate pools: Dense FAISS candidates, BM25 candidates, RRF fused rankings, and Cross-Encoder scores.
