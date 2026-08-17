# Voice RAG — Guardrails & Safety Defenses

**HH Goa 2026 — Task 2 | Multi-Stage Guardrail Architecture**

---

## 1. Multi-Stage Guardrail Lifecycle

```
User Query ──► [ Input Guardrail ]
                    │ (Checks Injection, Length, Whitespace)
                    ▼
               [ Retrieval & Cross-Encoder ]
                    │
                    ▼
               [ Context Guardrail ]
                    │ (Checks Token Budget & Score Threshold > 0.01)
                    ▼
               [ Grounded LLM Generation ]
                    │
                    ▼
               [ Post-Guardrail & Grounding Verifier ]
                    │ (Checks N-Gram Claim Alignment & Citation Proofs)
                    ▼
               Final Safe & Grounded Answer (or Abstention)
```

---

## 2. Input Guardrail (`guardrails/input.py`)

- **Prompt Injection Defense**: Evaluates regex patterns targeting system instruction overrides (`ignore all instructions`, `print system prompt`, `reveal secrets`, etc.).
- **Character Length Enforcement**: Maximum 500 characters per query payload.
- **Whitespace / Empty Detection**: Rejects empty queries with `EMPTY_QUERY` status.
- **Latency Overhead**: Evaluates in $< 0.05\text{ ms}$.

---

## 3. Context Guardrail (`guardrails/context.py`)

- **Context Budget**: Enforces a strict maximum of 5 chunks (up to 8,000 characters / 2,048 tokens).
- **Relevance Score Threshold**: Cross-encoder relevance must exceed `0.01`. If no candidate meets the threshold, the system abstains with `INSUFFICIENT_CONTEXT`.
- **Latency Overhead**: Evaluates in $< 0.02\text{ ms}$.

---

## 4. Grounding Verifier (`guardrails/verifier.py`)

- **Claim Extraction**: Extracts key assertions and entity claims from the generated answer.
- **N-Gram Overlap Matching**: Validates that extracted claims are supported by text spans in the retrieved chunks.
- **Citation Validation**: Confirms that cited chunk IDs exist in the retrieved pool and contain the claimed facts.
- **Latency Overhead**: Evaluates in $< 0.07\text{ ms}$.
