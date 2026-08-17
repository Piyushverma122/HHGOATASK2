import hashlib
import math
from typing import List, Dict, Any, Optional
from ingestion.chunking.base import Chunker
from ingestion.chunking.models import Chunk
from ingestion.chunking.utils import split_sentences


class LightweightSentenceVectorizer:
    """
    Lightweight, deterministic multilingual vectorizer.
    Uses character 3-gram frequencies to compute semantic similarity
    between sentences across Indic and Latin scripts without heavyweight dependencies.
    Includes a deterministic in-memory cache.
    """

    def __init__(self):
        self._cache: Dict[str, float] = {}

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_char_ngrams(self, text: str, n: int = 3) -> Dict[str, int]:
        clean = text.strip().lower()
        if len(clean) < n:
            return {clean: 1}
        counts: Dict[str, int] = {}
        for i in range(len(clean) - n + 1):
            gram = clean[i:i + n]
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    def compute_similarity(self, sent1: str, sent2: str) -> float:
        """
        Compute cosine similarity between two sentences based on character n-gram profiles.
        Returns a float between 0.0 and 1.0.
        """
        if not sent1 or not sent2:
            return 0.0
        if sent1.strip() == sent2.strip():
            return 1.0

        cache_key = f"{self._get_hash(sent1)}_{self._get_hash(sent2)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        vec1 = self._get_char_ngrams(sent1, 3)
        vec2 = self._get_char_ngrams(sent2, 3)

        dot_product = sum(count * vec2.get(gram, 0) for gram, count in vec1.items())
        norm1 = math.sqrt(sum(c * c for c in vec1.values()))
        norm2 = math.sqrt(sum(c * c for c in vec2.values()))

        if norm1 == 0.0 or norm2 == 0.0:
            sim = 0.0
        else:
            sim = dot_product / (norm1 * norm2)
            sim = max(0.0, min(1.0, float(sim)))

        self._cache[cache_key] = sim
        return sim


# Global shared vectorizer instance with cache
sentence_vectorizer = LightweightSentenceVectorizer()


class SemanticChunker(Chunker):
    """
    Strategy 5: Semantic Similarity Chunking.
    Splits text into sentences, measures semantic similarity between adjacent sentences,
    and inserts chunk boundaries at semantic shift valleys (similarity < threshold).
    Guarantees that no chunk exceeds max_chunk_tokens by applying safe fallback slicing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="semantic", config=config)
        self.target_chunk_tokens = int(self.config.get("target_chunk_tokens", 256))
        self.min_chunk_tokens = int(self.config.get("min_chunk_tokens", 64))
        self.max_chunk_tokens = int(self.config.get("max_chunk_tokens", 384))
        self.semantic_threshold = float(self.config.get("semantic_threshold", 0.65))
        self.vectorizer = sentence_vectorizer

    def _split_oversized_text(
        self,
        text: str,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
        start_chunk_idx: int,
    ) -> List[Chunk]:
        """Fallback to split any single text/sentence that exceeds max_chunk_tokens."""
        raw_tokens = self.token_counter.encode(text)
        chunks: List[Chunk] = []
        chunk_idx = start_chunk_idx

        for start in range(0, len(raw_tokens), self.target_chunk_tokens):
            end = min(start + self.target_chunk_tokens, len(raw_tokens))
            chunk_text = self.token_counter.decode(raw_tokens[start:end]).strip()
            if chunk_text:
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=chunk_idx,
                        extra_metadata={"is_oversized_split": True},
                    )
                )
                chunk_idx += 1
        return chunks

    def chunk_passage(
        self,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
    ) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []

        sentences = split_sentences(text)
        if not sentences:
            return []

        # If single sentence or entire passage within max_chunk_tokens
        total_tokens = self.token_counter.count(text)
        if len(sentences) <= 1:
            if total_tokens <= self.max_chunk_tokens:
                return [
                    self._create_chunk(
                        text=text,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=0,
                        extra_metadata={"semantic_boundary_detected": False, "sentence_count": 1},
                    )
                ]
            else:
                return self._split_oversized_text(text, passage, record_context, 0)

        # 1. Compute pairwise similarity between consecutive sentences
        similarities: List[float] = []
        for i in range(len(sentences) - 1):
            sim = self.vectorizer.compute_similarity(sentences[i], sentences[i + 1])
            similarities.append(sim)

        # 2. Group sentences into semantic clusters based on similarity drops and token constraints
        chunks: List[Chunk] = []
        current_cluster: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        for idx, sent in enumerate(sentences):
            sent_tokens = self.token_counter.count(sent)

            # If individual sentence exceeds max size, flush cluster then split sentence
            if sent_tokens > self.max_chunk_tokens:
                if current_cluster:
                    combined = " ".join(current_cluster).strip()
                    if combined:
                        chunks.append(
                            self._create_chunk(
                                text=combined,
                                passage=passage,
                                record_context=record_context,
                                chunk_index=chunk_idx,
                                extra_metadata={"sentence_count": len(current_cluster)},
                            )
                        )
                        chunk_idx += 1
                    current_cluster = []
                    current_tokens = 0

                oversized_chunks = self._split_oversized_text(sent, passage, record_context, chunk_idx)
                chunks.extend(oversized_chunks)
                chunk_idx += len(oversized_chunks)
                continue

            if not current_cluster:
                current_cluster.append(sent)
                current_tokens = sent_tokens
                continue

            # Prior similarity to this sentence
            prev_sim = similarities[idx - 1] if idx - 1 < len(similarities) else 1.0

            is_semantic_shift = (prev_sim < self.semantic_threshold) and (current_tokens >= self.min_chunk_tokens)
            is_overflow = (current_tokens + sent_tokens) > self.max_chunk_tokens

            if is_semantic_shift or is_overflow:
                combined_text = " ".join(current_cluster).strip()
                if combined_text:
                    chunks.append(
                        self._create_chunk(
                            text=combined_text,
                            passage=passage,
                            record_context=record_context,
                            chunk_index=chunk_idx,
                            extra_metadata={
                                "sentence_count": len(current_cluster),
                                "semantic_boundary_detected": is_semantic_shift,
                                "boundary_similarity": round(prev_sim, 4),
                            },
                        )
                    )
                    chunk_idx += 1
                current_cluster = [sent]
                current_tokens = sent_tokens
            else:
                current_cluster.append(sent)
                current_tokens += sent_tokens

        # Flush final cluster
        if current_cluster:
            combined_text = " ".join(current_cluster).strip()
            if combined_text:
                # Merge tiny trailing cluster with previous chunk if within max tokens
                if chunks and current_tokens < self.min_chunk_tokens:
                    last_chunk = chunks[-1]
                    merged = f"{last_chunk.text} {combined_text}".strip()
                    if self.token_counter.count(merged) <= self.max_chunk_tokens:
                        chunks[-1] = self._create_chunk(
                            text=merged,
                            passage=passage,
                            record_context=record_context,
                            chunk_index=last_chunk.chunk_index,
                            extra_metadata={
                                "sentence_count": last_chunk.metadata.get("sentence_count", 1) + len(current_cluster),
                                "merged_trailing_cluster": True,
                            },
                        )
                        return chunks

                chunks.append(
                    self._create_chunk(
                        text=combined_text,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=chunk_idx,
                        extra_metadata={
                            "sentence_count": len(current_cluster),
                            "semantic_boundary_detected": False,
                        },
                    )
                )

        return chunks
