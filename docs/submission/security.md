# Voice RAG — Security & Secret Management

**HH Goa 2026 — Task 2 | Security Architecture**

---

## 1. Zero Secret Leakage Policy

The Voice RAG system strictly enforces zero secret exposure across all layers:
- **Git Tracking**: `.env`, `backend/.env`, and any credentials are included in `.gitignore`.
- **Sanitized Telemetry**: Health check (`GET /api/v1/health`) and info endpoints return `configured: bool` without exposing API keys or tokens.
- **Log Sanitization**: Structured logs exclude audio byte payloads, API keys, and authorization headers.
- **Error Obfuscation**: Centralized exception handlers return clean client errors (`{ error: { code, message, request_id } }`) without exposing Python stack traces.

---

## 2. In-Memory Rate Limiting

- **Protection**: Protects public endpoints (`/api/v1/voice/query`, `/api/v1/voice/transcribe`, `/api/v1/rag/query`).
- **Algorithm**: Sliding window token tracker per client IP.
- **Default Limit**: 120 requests / minute per client IP.
- **HTTP 429 Response**: Returns standard `Retry-After: 60` header and structured error message.

---

## 3. Adversarial Input Defenses

- **Prompt Injection Defense**: Evaluates regex patterns targeting system instruction overrides (`ignore all instructions`, `print system prompt`, `reveal secrets`, etc.).
- **Payload Size Restrictions**: Audio files capped at 10MB / 30 seconds; text queries capped at 500 characters.
- **Abstention on Malicious Inputs**: Rejects adversarial queries with `PROMPT_INJECTION` rejection code in $< 0.1\text{ ms}$.
