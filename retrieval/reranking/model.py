import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from retrieval.reranking.base import BaseReranker
from retrieval.reranking.cache import RerankerCache
from retrieval.lexical.tokenizer import MultilingualTokenizer
from retrieval.embeddings.provider import get_default_embedder

logger = logging.getLogger("voice_rag.retrieval.reranking")


class CrossEncoderReranker(BaseReranker):
    """
    Genuine Multilingual Cross-Encoder Reranker using Pretrained Transformers.
    Jointly evaluates (query, candidate passage) token interactions with multi-head self-attention.
    Supported models:
      - 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1' (Default, 117M params, fast multilingual CPU inference)
      - 'BAAI/bge-reranker-v2-m3' (568M params, high capacity XLM-RoBERTa multilingual reranker)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        model_version: str = "v1.0",
        device: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 16,
        use_cache: bool = True,
        cache_db_path: Optional[Path] = None,
        lazy_load: bool = True,
    ):
        self._model_name = model_name
        self._model_version = model_version
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_cache = use_cache

        # Auto-detect CUDA or CPU
        if device is not None:
            self._device = device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._device == "cpu":
            import os
            try:
                torch.set_num_threads(min(8, os.cpu_count() or 4))
            except Exception:
                pass

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.cache = (
            RerankerCache(db_path=cache_db_path, model_name=model_name, model_version=model_version)
            if use_cache
            else None
        )

        if not lazy_load:
            self.load()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def device(self) -> str:
        return self._device

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load tokenizer and model exactly once."""
        if self.is_loaded():
            return

        logger.info(f"Loading CrossEncoder model '{self._model_name}' on device '{self._device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self.model.to(self._device)
        self.model.eval()
        logger.info(f"CrossEncoder model '{self._model_name}' loaded successfully on {self._device}.")

    def warmup(self) -> None:
        """Perform a single dummy inference for JIT / kernel warmup."""
        if not self.is_loaded():
            self.load()
        dummy_query = "भारत"
        dummy_passages = ["भारत एक विशाल देश है।"]
        _ = self.score(dummy_query, dummy_passages)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model": self._model_name,
            "model_version": self._model_version,
            "device": self._device,
            "loaded": self.is_loaded(),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "use_cache": self.use_cache,
        }

    def score(self, query: str, passages: List[str]) -> List[float]:
        """
        Jointly score (query, passage) pairs using the cross-encoder model.
        """
        if not passages:
            return []

        if not self.is_loaded():
            self.load()

        scores: List[float] = []

        # Process in batches
        for i in range(0, len(passages), self.batch_size):
            batch_passages = passages[i : i + self.batch_size]
            # Construct joint (query, passage) pairs
            pairs = [[query, p] for p in batch_passages]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Handle 1D (regression) vs 2D (binary classification) logits
                if logits.dim() > 1 and logits.shape[1] > 1:
                    raw_scores = logits[:, 1]
                else:
                    raw_scores = logits.view(-1)

                # Calibrated sigmoid activation to produce [0.0, 1.0] relevance scores
                probs = torch.sigmoid(raw_scores).cpu().tolist()

                if isinstance(probs, float):
                    probs = [probs]
                scores.extend([round(float(p), 4) for p in probs])

        return scores

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate list using the cross-encoder model.
        Checks SQLite persistent cache before batch inference.
        """
        if not candidates:
            return []

        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        cached_scores: Dict[int, float] = {}

        for i, candidate in enumerate(candidates):
            chunk_id = candidate.get("chunk_id", "")
            cached_val = None
            if self.use_cache and self.cache and chunk_id:
                cached_val = self.cache.get(query, chunk_id)

            if cached_val is not None:
                cached_scores[i] = cached_val
            else:
                uncached_indices.append(i)
                uncached_texts.append(candidate.get("text", ""))

        # Run batch inference on uncached items
        if uncached_texts:
            new_scores = self.score(query=query, passages=uncached_texts)
            for local_idx, orig_idx in enumerate(uncached_indices):
                score = new_scores[local_idx]
                cached_scores[orig_idx] = score

                chunk_id = candidates[orig_idx].get("chunk_id", "")
                if self.use_cache and self.cache and chunk_id:
                    self.cache.set(query, chunk_id, score)

        reranked: List[Dict[str, Any]] = []
        for i, candidate in enumerate(candidates):
            rec = dict(candidate)
            rec["reranker_score"] = cached_scores.get(i, 0.0)
            reranked.append(rec)

        # Sort descending by reranker_score
        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

        for rank, item in enumerate(reranked, start=1):
            item["rerank_rank"] = rank

        return reranked[:top_k]


class CustomReranker(BaseReranker):
    """
    Legacy Heuristic / Dual-Embedding Reranker from Module 5.
    Combines subword TF-IDF coverage, phrase proximity bonus, and 384-d semantic projection.
    Kept for empirical comparison and ablation against the genuine CrossEncoder.
    """

    def __init__(
        self,
        model_name: str = "custom-heuristic-reranker-v1",
        cache_db_path: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self._model_name = model_name
        self.use_cache = use_cache
        self.tokenizer = MultilingualTokenizer(use_subwords=True)
        self.embedder = get_default_embedder()
        self.cache = (
            RerankerCache(db_path=cache_db_path, model_name=model_name, model_version="v1.0")
            if use_cache
            else None
        )
        self._loaded = True

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return "cpu"

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def warmup(self) -> None:
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self._model_name,
            "model_version": "v1.0",
            "device": "cpu",
            "loaded": True,
            "max_length": 512,
            "batch_size": 1,
            "type": "heuristic_custom",
        }

    def score_pair(self, query: str, document_text: str) -> float:
        q_tokens = self.tokenizer.tokenize(query)
        d_tokens = self.tokenizer.tokenize(document_text)
        if not q_tokens or not d_tokens:
            return 0.0

        q_set = set(q_tokens)
        d_set = set(d_tokens)
        overlap = len(q_set.intersection(d_set))
        lexical_coverage = overlap / len(q_set) if q_set else 0.0

        q_clean = query.strip().lower()
        d_clean = document_text.strip().lower()
        exact_match_bonus = 0.25 if (len(q_clean) > 3 and q_clean in d_clean) else 0.0

        q_vec = self.embedder.embed_query(query)
        d_vec = self.embedder.embed_text(document_text)
        semantic_sim = float(np.dot(q_vec, d_vec))

        logit = (2.5 * lexical_coverage) + (1.8 * semantic_sim) + exact_match_bonus - 1.0
        score = 1.0 / (1.0 + math.exp(-logit))
        return round(score, 4)

    def score(self, query: str, passages: List[str]) -> List[float]:
        return [self.score_pair(query, p) for p in passages]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        reranked = []
        for cand in candidates:
            chunk_id = cand.get("chunk_id", "")
            cached_score = None
            if self.use_cache and self.cache and chunk_id:
                cached_score = self.cache.get(query, chunk_id)

            if cached_score is not None:
                score = cached_score
            else:
                score = self.score_pair(query, cand.get("text", ""))
                if self.use_cache and self.cache and chunk_id:
                    self.cache.set(query, chunk_id, score)

            rec = dict(cand)
            rec["reranker_score"] = score
            reranked.append(rec)

        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
        for rank, item in enumerate(reranked, start=1):
            item["rerank_rank"] = rank
        return reranked[:top_k]


# Backwards compatibility alias
MultilingualCrossEncoderReranker = CustomReranker
