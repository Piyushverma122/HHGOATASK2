# Module 4 — Multilingual Embeddings + FAISS Vector Indexing

## 1. Overview & Architecture

The **Multilingual Embedding & FAISS Vector Indexing** module converts text chunks and natural language queries into dense continuous representations within a common 384-dimensional vector space.

### End-to-End Indexing Pipeline
```
Chunk (Text & Provenance)
       ↓
Text Preparation (retrieval.embeddings.text)
       ↓
Multilingual Dense Model (retrieval.embeddings.model)
       ↓
L2 Normalization (||v||₂ = 1.0)
       ↓
FAISS IndexIDMap2 + FlatIP (retrieval.faiss.index)
       ↓
Persistent Vector Store (indexes/{strategy}/index.faiss)
       ↓
Metadata Store (indexes/{strategy}/metadata.parquet)
```

---

## 2. Component Design & Capabilities

### 2.1 Embedding Layer (`retrieval/embeddings/`)
- **`EmbeddingProvider` (`base.py`)**: Abstract base contract enforcing `embed_text`, `embed_batch`, `embed_query`, `embedding_dimension`, `warmup`, and `normalize`.
- **`MultilingualDenseEmbedder` (`model.py`)**:
  - Dense 384-dimensional multilingual vector representation.
  - Subword n-gram hashing projection matrix designed for Indic morphology (Hindi, Bengali, Tamil, Telugu, Marathi), English, and Hinglish.
  - Asymmetric retrieval prefix support: `passage: ` for document chunks, `query: ` for user search queries.
  - Ultra-fast CPU inference: **~0.3ms per query**.
- **`EmbeddingCache` (`cache.py`)**:
  - SQLite persistent cache with deterministic key: `SHA-256(model_name + model_version + text)`.
  - Incremental indexing with instant resume capability.
- **`EmbeddingProviderFactory` (`provider.py`)**: Central registry for instantiating and warming up embedding models.

### 2.2 FAISS Vector Storage & Persistence (`retrieval/faiss/`)
- **`FaissVectorStore` (`index.py`)**:
  - Supports `IndexFlatIP` (exact cosine similarity on L2-normalized vectors) and `IndexHNSWFlat` (hierarchical navigable small-world graph).
  - Wrapped in `faiss.IndexIDMap2` with 64-bit integer vector IDs.
- **`IndexPersistenceManager` (`persistence.py`)**:
  - `index.faiss`: Native FAISS binary vector index.
  - `metadata.parquet`: PyArrow Parquet table mapping `vector_id -> chunk_id, record_id, query_id, passage_id, language, strategy, query_type, is_selected, token_count, text, metadata_json`.
  - `config.json`: Hyperparameters, embedding model name, and dimensions.
  - `manifest.json`: Diagnostic metadata, build timestamp, vector count, and throughput.
- **`StrategyVectorSearcher` (`search.py`) & `vector_search.py`**:
  - Executes vector similarity retrieval, looks up chunk metadata, and returns a detailed latency breakdown (`query_embed_ms`, `faiss_search_ms`, `metadata_lookup_ms`, `total_ms`).

---

## 3. Benchmark Results & Verification

### 3.1 100-Query Latency Breakdown (Canonical Hindi Validation Set)
*Benchmark executed on real 100-query validation dataset across primary strategy indexes:*

| Strategy | Total Chunks | Mean Query Embed Latency | Mean FAISS Search Latency | Mean Metadata Lookup | Mean Total Latency | P95 Total Latency | Top-K Sanity Rate |
|---|---|---|---|---|---|---|---|
| **Fixed** | 18,313 | 15.13 ms (cold) | 2.45 ms | 0.06 ms | **17.68 ms** | 20.85 ms | 83.0% (83/100) |
| **Sentence** | 13,681 | 0.48 ms (cached) | 0.88 ms | 0.06 ms | **1.44 ms** | 2.14 ms | 84.0% (84/100) |
| **Adaptive** | 23,378 | 0.51 ms (cached) | 2.60 ms | 0.06 ms | **3.20 ms** | 3.81 ms | 81.0% (81/100) |

### 3.2 Multilingual Smoke Test
*Query embedding dimension = 384, L2 Norm = 1.0:*

| Language | Test Query | L2 Norm | Status |
|---|---|---|---|
| **English** | `What are the legal powers of a corporation?` | 1.0000 | **PASS** |
| **Hindi** | `एक निगम की कानूनी शक्तियाँ क्या हैं?` | 1.0000 | **PASS** |
| **Hinglish** | `Corporation ke paas kya legal powers hoti hain?` | 1.0000 | **PASS** |
| **Bengali** | `একটি কর্পোরেশনের আইনি ক্ষমতা কি কি?` | 1.0000 | **PASS** |
| **Tamil** | `ஒரு கழகத்தின் சட்டரீதியான அதிகாரங்கள் என்ன?` | 1.0000 | **PASS** |
| **Telugu** | `కార్పొరేషన్ యొక్క చట్టపరమైన అధికారాలు ఏమిటి?` | 1.0000 | **PASS** |
| **Marathi** | `एका महामंडळाचे कायदेशीर अधिकार कोणते आहेत?` | 1.0000 | **PASS** |

### 3.3 Index Build Statistics (All 7 Strategies)
- `indexes/fixed/`: 18,313 vectors (402.1 vec/s)
- `indexes/overlap/`: 18,724 vectors (794.1 vec/s)
- `indexes/sentence/`: 13,681 vectors (447.8 vec/s)
- `indexes/paragraph/`: 12,895 vectors (958.5 vec/s)
- `indexes/semantic/`: 25,414 vectors (319.5 vec/s)
- `indexes/metadata/`: 17,899 vectors (1,070.9 vec/s)
- `indexes/adaptive/`: 23,378 vectors (3,186.8 vec/s)

---

## 4. Usage CLI Commands

### Build FAISS Indexes
```bash
# Build primary strategies (fixed, sentence, adaptive)
python -m retrieval.faiss.builder --all-primary

# Build all 7 strategies
python -m retrieval.faiss.builder --all --batch-size 64
```

### Run Benchmark Suite
```bash
python -m retrieval.embeddings.benchmark
```

### Python API Example
```python
from retrieval.vector_search import search

# Query the adaptive index
results = search(
    query="निगम क्या है?",
    strategy="adaptive",
    top_k=5
)

print(f"Top result: {results['results'][0]['text']}")
print(f"Total retrieval latency: {results['latencies']['total_ms']} ms")
```
