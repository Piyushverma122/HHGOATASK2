# MSMARCO-XI Dataset & Ingestion Documentation

## 1. Overview & Dataset Source

The ingestion pipeline ingests the **MSMARCO-XI** (MS MARCO Translated to Indic Languages) benchmark dataset from AI4Bharat:
- **Hugging Face Hub**: [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
- **Primary Focus**: Multilingual Indic information retrieval & voice-enabled conversational search.
- **Default Development Language**: Hindi (`hi` / `hin`).

---

## 2. Discovered Dataset Structure & Configurations

The official repository contains pre-partitioned Parquet files per Indic language configuration and split:

### 2.1 Available Language Configurations (14 Languages)
| Code | ISO / Script | Language | File Prefix |
|---|---|---|---|
| `hi` | `hin_Deva` | Hindi | `hin` |
| `bn` | `ben_Beng` | Bengali | `ben` |
| `gu` | `guj_Gujr` | Gujarati | `guj` |
| `kn` | `kan_Knda` | Kannada | `kan` |
| `ml` | `mal_Mlym` | Malayalam | `mal` |
| `mr` | `mar_Deva` | Marathi | `mar` |
| `ne` | `nep_Deva` | Nepali | `nep` |
| `or` | `ori_Orya` | Odia | `ori` |
| `pa` | `pan_Guru` | Punjabi | `pan` |
| `sa` | `san_Deva` | Sanskrit | `san` |
| `ta` | `tam_Taml` | Tamil | `tam` |
| `te` | `tel_Telu` | Telugu | `tel` |
| `ur` | `urd_Arab` | Urdu | `urd` |
| `as` | `asm_Beng` | Assamese | `asm` |

### 2.2 Splits
- `train`: `train/{lang3}train.parquet` (Full training partition)
- `validation`: `validation/{lang3}val.parquet` (Evaluation partition)

---

## 3. Raw vs. Canonical Schema

### 3.1 Raw Schema (Hugging Face / Parquet)
```json
{
  "source_lang": "string",
  "target_lang": "string",
  "meta": {
    "model_name": "string",
    "temperature": "float",
    "max_tokens": "int",
    "top_p": "float",
    "frequency_penalty": "float",
    "presence_penalty": "float"
  },
  "query": "string",
  "Answer": "string",
  "query_id": "int64",
  "query_type": "string",
  "passages": {
    "is_selected": "list[int32]",
    "English_passages": "list[string]",
    "Translated_passages": "list[string]"
  },
  "Eng_Query": "string",
  "Eng_Answer": "string"
}
```

### 3.2 Canonical Normalized Schema
```json
{
  "record_id": "1001_hi",
  "query_id": 1001,
  "query": "भारत की राजधानी क्या है?",
  "answer": "नई दिल्ली",
  "query_type": "DESCRIPTION",
  "source_lang": "eng_Latn",
  "target_lang": "hin_Deva",
  "passages": [
    {
      "passage_id": "1001_hi_0",
      "passage_index": 0,
      "text": "नई दिल्ली भारत की आधिकारिक राजधानी है।",
      "english_text": "New Delhi is the official capital of India.",
      "is_selected": true
    }
  ],
  "original": {
    "eng_query": "What is the capital of India?",
    "eng_answer": "New Delhi"
  },
  "metadata": {
    "model_name": "indic-trans"
  }
}
```

---

## 4. Ingestion Pipeline Mechanics

### 4.1 Memory-Efficient Streaming
- Avoids loading entire multi-gigabyte files into RAM.
- Utilizes `pyarrow.parquet.ParquetFile.iter_batches(batch_size=500)` to stream records row-by-row in bounded chunks.
- Writes to disk using `pyarrow.parquet.ParquetWriter` incrementally.
- Memory footprint remains strictly bounded ($\mathcal{O}(\text{batch\_size})$).

### 4.2 Indic-Aware Normalization (`ingestion/normalize.py`)
1. **Unicode NFC Composition**: Ensures consistent character codepoints for Indic composite glyphs and matras.
2. **Preservation of Indic Characters**: Halant / virama (`्`), nuktas (`़`), chandrabindu (`ँ`), and ligatures (`\u200C`, `\u200D`) are strictly preserved.
3. **Control Character Stripping**: Strips invalid non-printable characters while preserving intended spaces.
4. **Whitespace Collapsing**: Collapses multi-space sequences and trims extraneous margins.
5. **No Destructive Lowercasing / Punctuation Removal**: Preserves case distinctions for English and full stops / danda (`।`) for Indic sentences.

### 4.3 Validation (`ingestion/validate.py`)
- Evaluates query existence, non-emptiness, valid passage arrays, and language tags.
- Logs and isolates invalid records into `data/statistics/validation_errors.json` without interrupting stream processing.

### 4.4 Context-Aware Deduplication (`ingestion/deduplicate.py`)
- Tracks unique `query_id` occurrences and SHA-256 content hashes of query texts.
- Discards duplicate questions while preserving identical passages attached to distinct query contexts.

---

## 5. CLI Usage & Pipeline Commands

### Inspect Dataset Schema
```bash
python ingestion/pipeline.py --inspect
```

### Ingest 100-Record Development Sample
```bash
python ingestion/pipeline.py --language hi --split train --sample-size 100
```

### Ingest 10,000-Record Benchmark Sample
```bash
python ingestion/pipeline.py --language hi --split train --sample-size 10000 --stream
```

### Change Language or Split
```bash
python ingestion/pipeline.py --language ta --split validation --sample-size 500
```

---

## 6. Output Artifacts

- **Parquet Dataset**: `data/processed/msmarco_xi_<lang>_<split>.parquet`
- **Raw JSONL Sample**: `data/samples/raw_<lang>_<split>.jsonl`
- **Processed JSONL Sample**: `data/processed/sample_<lang>_<split>.jsonl`
- **Statistics JSON**: `data/statistics/dataset_stats_<lang>_<split>.json`
- **Statistics Markdown**: `data/statistics/dataset_stats_<lang>_<split>.md`
- **Validation Errors**: `data/statistics/validation_errors.json`
