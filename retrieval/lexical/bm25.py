import os
import math
import time
import json
import pickle
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.lexical.tokenizer import MultilingualTokenizer
from retrieval.faiss.persistence import METADATA_SCHEMA

logger = logging.getLogger("voice_rag.lexical.bm25")


class BM25Index:
    """
    Persistent Okapi BM25 Index with inverted postings.
    Score formula:
        BM25(D, Q) = sum_{t in Q} IDF(t) * [ TF(t, D) * (k1 + 1) ] / [ TF(t, D) + k1 * (1 - b + b * (|D| / avgdl)) ]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: Optional[MultilingualTokenizer] = None):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or MultilingualTokenizer()

        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_ids: List[int] = []  # Maps internal doc index -> integer vector_id / doc_id

        # Inverted index: term -> list of (doc_index, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Term IDFs: term -> float
        self.idf: Dict[str, float] = {}

    @property
    def num_docs(self) -> int:
        """Alias for total indexed document count."""
        return self.corpus_size

    def fit(self, documents: List[str], doc_ids: Optional[List[int]] = None):
        """
        Build BM25 index from list of document text strings.
        """
        self.corpus_size = len(documents)
        self.doc_ids = doc_ids if doc_ids is not None else list(range(self.corpus_size))
        self.doc_lengths = []
        self.inverted_index.clear()
        self.idf.clear()

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        total_tokens = 0
        doc_freqs: Dict[str, int] = defaultdict(int)

        logger.info(f"Tokenizing and indexing {self.corpus_size:,} documents for BM25...")
        start_t = time.perf_counter()

        for doc_idx, doc_text in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc_text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

            counts = Counter(tokens)
            for term, count in counts.items():
                self.inverted_index[term].append((doc_idx, count))
                doc_freqs[term] += 1

        self.avg_doc_len = total_tokens / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate Okapi BM25 IDFs
        for term, df in doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

        elapsed = time.perf_counter() - start_t
        logger.info(f"BM25 Index built: {self.corpus_size:,} docs, {len(self.inverted_index):,} terms, avgdl={self.avg_doc_len:.1f} in {elapsed:.2f}s")

    def search(self, query: str, top_k: int = 20) -> Tuple[List[int], List[float]]:
        """
        Score documents for a query and return top_k (doc_ids, scores).
        """
        if self.corpus_size == 0 or not query:
            return [], []

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return [], []

        doc_scores: Dict[int, float] = defaultdict(float)
        k1 = self.k1
        b = self.b
        avgdl = self.avg_doc_len or 1.0

        # Accumulate BM25 scores across matching inverted lists
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            idf_val = self.idf[term]
            for doc_idx, tf in self.inverted_index[term]:
                dl = self.doc_lengths[doc_idx]
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                score = idf_val * (tf * (k1 + 1.0)) / denom
                doc_scores[doc_idx] += score

        if not doc_scores:
            return [], []

        # Extract top_k highest scoring documents
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        res_ids = [self.doc_ids[doc_idx] for doc_idx, _ in sorted_docs]
        res_scores = [round(score, 4) for _, score in sorted_docs]

        return res_ids, res_scores

    def save(self, index_file: Path):
        """Save serialized BM25 index to disk."""
        index_file = Path(index_file)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "doc_ids": self.doc_ids,
            "inverted_index": dict(self.inverted_index),
            "idf": self.idf,
        }
        with open(index_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, index_file: Path) -> "BM25Index":
        """Load serialized BM25 index from disk."""
        index_file = Path(index_file)
        if not index_file.exists():
            raise FileNotFoundError(f"BM25 index not found at: {index_file}")

        with open(index_file, "rb") as f:
            data = pickle.load(f)

        idx = cls(k1=data["k1"], b=data["b"])
        idx.corpus_size = data["corpus_size"]
        idx.avg_doc_len = data["avg_doc_len"]
        idx.doc_lengths = data["doc_lengths"]
        idx.doc_ids = data["doc_ids"]
        idx.inverted_index = data["inverted_index"]
        idx.idf = data["idf"]
        return idx


class BM25Retriever:
    """
    Production-grade BM25 Lexical Retriever with persistent disk loading and metadata resolution.
    """

    def __init__(self, strategy: str = "adaptive", index_dir: Optional[Path] = None):
        self.strategy = strategy
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.index_dir = Path(index_dir) if index_dir else base_dir / "indexes" / "bm25" / strategy
        self.index_path = self.index_dir / "bm25_index.pkl"
        self.metadata_path = self.index_dir / "metadata.parquet"
        self.config_path = self.index_dir / "config.json"
        self.manifest_path = self.index_dir / "manifest.json"

        self.bm25: Optional[BM25Index] = None
        self.metadata_lookup: Dict[int, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.index_path.exists():
            self.bm25 = BM25Index.load(self.index_path)
            if self.metadata_path.exists():
                df = pd.read_parquet(self.metadata_path)
                for _, row in df.iterrows():
                    v_id = int(row["vector_id"])
                    self.metadata_lookup[v_id] = {
                        "vector_id": v_id,
                        "chunk_id": row["chunk_id"],
                        "record_id": row["record_id"],
                        "query_id": int(row["query_id"]),
                        "passage_id": row["passage_id"],
                        "language": row["language"],
                        "strategy": row["strategy"],
                        "query_type": row["query_type"],
                        "is_selected": bool(row["is_selected"]),
                        "token_count": int(row["token_count"]),
                        "text": row["text"],
                        "metadata": json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
                    }

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 sparse search and return standardized candidate results.
        """
        if not self.bm25 or self.bm25.corpus_size == 0:
            return []

        doc_ids, scores = self.bm25.search(query, top_k=top_k)

        bm25_candidates: List[Dict[str, Any]] = []
        for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
            meta = self.metadata_lookup.get(int(doc_id), {})
            bm25_candidates.append({
                "source": "bm25",
                "rank": rank,
                "score": float(score),
                "chunk_id": meta.get("chunk_id", f"{self.strategy}_{doc_id}"),
                "record_id": meta.get("record_id", ""),
                "query_id": meta.get("query_id", 0),
                "passage_id": meta.get("passage_id", ""),
                "language": meta.get("language", "hi"),
                "strategy": meta.get("strategy", self.strategy),
                "query_type": meta.get("query_type", "standard"),
                "is_selected": meta.get("is_selected", False),
                "token_count": meta.get("token_count", 0),
                "text": meta.get("text", ""),
                "metadata": meta.get("metadata", {}),
            })

        return bm25_candidates
