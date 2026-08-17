# Voice RAG — 3-Minute Live Submission Demo Script

**HH Goa 2026 — Task 2 | Presentation & Judging Guide**

---

### Time: 0:00 – 0:30 | Introduction & Problem Overview
- **Visual**: Dashboard (`http://localhost:5173`)
- **Script**:
  > "Hello judges. We present **Voice RAG**, a production-grade, voice-enabled Retrieval-Augmented Generation system designed for 7 Indic languages. 
  > Unlike standard chatbots that hallucinate or struggle with latency, our pipeline is engineered to complete end-to-end voice retrieval, genuine cross-encoder reranking, and grounded answer synthesis in **under 60 milliseconds**—well below the 200ms threshold."

---

### Time: 0:30 – 1:15 | Voice Studio & STT Transcription
- **Visual**: Navigate to **Voice Studio** (`/voice`).
- **Action**: Click microphone to record a live query or click the 1-Click Hindi Demo fixture:
  - *"भारत की राजधानी क्या है और यह कहाँ स्थित है?"*
- **Script**:
  > "Our system captures audio and transcribes it via **Sarvam Saaras v3**. Notice how it normalizes the text using Unicode NFC and extracts language metadata in under a millisecond. If no microphone is connected, our 7-language demo fixture fallback allows zero-risk demonstration."

---

### Time: 1:15 – 2:00 | Retrieval Inspector & Cross-Encoder Reranking
- **Visual**: Navigate to **Retrieval Inspector** (`/retrieval`).
- **Action**: Search for *"भारत की राजधानी क्या है?"*.
- **Script**:
  > "Let's look under the hood. Our system doesn't rely on naive vector search alone. It runs **Dense FAISS** and **RankBM25** concurrently in parallel threads, fuses them with **Reciprocal Rank Fusion (RRF)**, and passes the top candidates to a real **Transformer Cross-Encoder** (`mmarco-mMiniLMv2-L12-H384-v1`). Here you can inspect the exact scores at every intermediate step."

---

### Time: 2:00 – 2:30 | Grounded Generation & Guardrail Defenses
- **Visual**: Navigate to **Guardrails** (`/guardrails`).
- **Action**: Click *Prompt Injection Attack* test case.
- **Script**:
  > "Safety is central to our architecture. Watch how our Input Guardrail instantly intercepts prompt injection attempts in under 0.1ms without touching the LLM. When given out-of-domain questions, our Context Guardrail triggers an abstention rather than hallucinating."

---

### Time: 2:30 – 3:00 | Latency Dashboard & Production Metrics
- **Visual**: Navigate to **Latency Analytics** (`/analytics`).
- **Script**:
  > "Across 141 benchmark queries on the MSMARCO-XI dataset:
  > - **P50 (Median)**: **24.9 ms**
  > - **P95 Tail**: **51.0 ms**
  > - **P100 (Max)**: **59.96 ms**
  > Every single percentile is 100% compliant with the &lt;200ms requirement. 
  > 127 automated unit tests pass with a 100% pass rate. 
  > Thank you!"

---
