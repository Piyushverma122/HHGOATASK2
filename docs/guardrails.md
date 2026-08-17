# Multilingual Guardrails & Safety Architecture

**HH Goa 2026 — Task 2 | Module 7: Guardrails & Grounding Verification**

---

## 1. Multi-Stage Guardrail Architecture

The safety layer protects the Voice RAG system before, during, and after generation:

```
User Query
    ↓
[InputGuardrail]
├── Empty Query Check
├── Query Length Limit (<= 500 chars)
├── Multilingual Prompt Injection Defense
└── Toxic / Harmful / Dangerous Content Detection
    ↓ (Pass)
[Hybrid Retrieval & Reranking]
    ↓
[ContextGuardrail]
├── Context Budget Enforcement (<= 5 chunks, <= 8000 chars)
└── Relevance Thresholding (score >= 0.01)
    ↓ (Pass)
[LLM Generation]
    ↓
[GroundingVerifier]
├── Sentence Claim Extraction
├── Corpus Content Word Overlap (>= 50%)
├── Number & Date Entity Consistency
├── Citation Chunk ID Validation
└── Hallucination Penalty Scoring
    ↓ (If Grounding Fails)
[GuardrailPolicy]
├── 1 Regeneration Attempt with Strict Grounding System Addendum
└── Natural Language Abstention Mapping (Hindi, English, Bengali, Tamil, etc.)
```

---

## 2. Guardrail Modules

### 2.1 Input Guardrail (`guardrails/input.py`)
- **Prompt Injection Defense**: Detects adversarial patterns across English and Indic scripts (e.g. `ignore previous instructions`, `bypass safety filters`, `सिस्टम प्रॉम्प्ट दिखाओ`, `API key दिखाओ`).
- **Unsafe Content Defense**: Detects requests for weapons, explosives, malware, self-harm, or illegal acts.

### 2.2 Context Guardrail (`guardrails/context.py`)
- **Budgeting**: Limits candidates to the top 5 chunks and truncates total context to 8,000 characters.
- **Relevance Thresholding**: Drops queries whose top reranker score is below threshold (`0.01`), returning `INSUFFICIENT_CONTEXT` or `OFF_TOPIC`.

### 2.3 Grounding Verifier (`guardrails/output.py`)
- **Claim Extraction**: Splits responses into distinct factual sentences.
- **Overlap Scoring**: Computes content-word overlap against retrieved context corpus (filtering language-specific stopwords).
- **Entity Consistency**: Validates that all numbers, dates, and names in the answer exist in the context.
- **Citation Validation**: Verifies that every `chunk_id` referenced by the LLM was legitimately retrieved.

### 2.4 Policy Coordinator (`guardrails/policy.py`)
- **Regeneration Attempt**: Allows at most 1 retry when a generated answer fails grounding.
- **Natural Language Abstention**: Formats helpful, user-friendly refusal responses in the user's detected query language.

---

## 3. Abstention Reasons Taxonomy

| Abstention Reason | Cause | Example Refusal Message |
|---|---|---|
| `EMPTY_QUERY` | User sent empty audio/text | *"कृपया एक वैध प्रश्न पूछें।"* |
| `QUERY_TOO_LONG` | Query exceeded 500 characters | *"प्रश्न बहुत लंबा है। कृपया इसे संक्षिप्त करें।"* |
| `PROMPT_INJECTION` | Adversarial jailbreak detected | *"यह प्रश्न सुरक्षा नीतियों के विरुद्ध है।"* |
| `UNSAFE_QUERY` | Dangerous/harmful request | *"मैं इस प्रकार की जानकारी प्रदान नहीं कर सकता।"* |
| `INSUFFICIENT_CONTEXT` | No relevant passages retrieved | *"उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"* |
| `GROUNDING_FAILURE` | LLM hallucinated ungrounded claims | *"स्रोतों के आधार पर इस प्रश्न का सटीक उत्तर सत्यापित नहीं किया जा सका।"* |
