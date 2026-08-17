from typing import List, Dict, Any, Optional
from ingestion.chunking.base import Chunker
from ingestion.chunking.models import Chunk
from ingestion.chunking.utils import split_sentences


class SentenceChunker(Chunker):
    """
    Strategy 3: Sentence-Aware Chunking.
    Preserves complete sentence structures (supporting Indic ।, ॥ as well as Latin .?!).
    Accumulates whole sentences until target_chunk_tokens is met without exceeding max_chunk_tokens.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="sentence", config=config)
        self.target_chunk_tokens = int(self.config.get("target_chunk_tokens", 256))
        self.max_chunk_tokens = int(self.config.get("max_chunk_tokens", 384))
        self.min_chunk_tokens = int(self.config.get("min_chunk_tokens", 32))

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

        # If total passage is already within target, keep as single chunk
        total_tokens = self.token_counter.count(text)
        if total_tokens <= self.target_chunk_tokens:
            return [
                self._create_chunk(
                    text=text,
                    passage=passage,
                    record_context=record_context,
                    chunk_index=0,
                    start_pos=0,
                    end_pos=len(text),
                    extra_metadata={"sentence_count": len(sentences)},
                )
            ]

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        for sent in sentences:
            sent_tokens = self.token_counter.count(sent)

            # Fallback if a single sentence is larger than max_chunk_tokens
            if sent_tokens > self.max_chunk_tokens:
                # Flush existing buffer first
                if current_sentences:
                    combined = " ".join(current_sentences).strip()
                    if combined:
                        chunks.append(
                            self._create_chunk(
                                text=combined,
                                passage=passage,
                                record_context=record_context,
                                chunk_index=chunk_idx,
                                extra_metadata={"sentence_count": len(current_sentences)},
                            )
                        )
                        chunk_idx += 1
                    current_sentences.clear()
                    current_tokens = 0

                # Split oversized sentence into token slices
                sent_raw_tokens = self.token_counter.encode(sent)
                for s_start in range(0, len(sent_raw_tokens), self.target_chunk_tokens):
                    s_end = min(s_start + self.target_chunk_tokens, len(sent_raw_tokens))
                    s_slice = self.token_counter.decode(sent_raw_tokens[s_start:s_end]).strip()
                    if s_slice:
                        chunks.append(
                            self._create_chunk(
                                text=s_slice,
                                passage=passage,
                                record_context=record_context,
                                chunk_index=chunk_idx,
                                extra_metadata={"sentence_count": 1, "is_oversized_split": True},
                            )
                        )
                        chunk_idx += 1
                continue

            # If adding this sentence exceeds max_chunk_tokens, flush current
            if current_sentences and (current_tokens + sent_tokens > self.max_chunk_tokens):
                combined = " ".join(current_sentences).strip()
                if combined:
                    chunks.append(
                        self._create_chunk(
                            text=combined,
                            passage=passage,
                            record_context=record_context,
                            chunk_index=chunk_idx,
                            extra_metadata={"sentence_count": len(current_sentences)},
                        )
                    )
                    chunk_idx += 1
                current_sentences = [sent]
                current_tokens = sent_tokens
            else:
                current_sentences.append(sent)
                current_tokens += sent_tokens

                # If reached target_chunk_tokens, flush
                if current_tokens >= self.target_chunk_tokens:
                    combined = " ".join(current_sentences).strip()
                    if combined:
                        chunks.append(
                            self._create_chunk(
                                text=combined,
                                passage=passage,
                                record_context=record_context,
                                chunk_index=chunk_idx,
                                extra_metadata={"sentence_count": len(current_sentences)},
                            )
                        )
                        chunk_idx += 1
                    current_sentences.clear()
                    current_tokens = 0

        # Flush trailing sentences
        if current_sentences:
            combined = " ".join(current_sentences).strip()
            if combined:
                # If very small and previous chunk exists, optionally merge if under max_chunk_tokens
                if chunks and current_tokens < self.min_chunk_tokens:
                    last_chunk = chunks[-1]
                    merged_text = f"{last_chunk.text} {combined}".strip()
                    if self.token_counter.count(merged_text) <= self.max_chunk_tokens:
                        chunks[-1] = self._create_chunk(
                            text=merged_text,
                            passage=passage,
                            record_context=record_context,
                            chunk_index=last_chunk.chunk_index,
                            extra_metadata={"sentence_count": last_chunk.metadata.get("sentence_count", 1) + len(current_sentences)},
                        )
                        return chunks

                chunks.append(
                    self._create_chunk(
                        text=combined,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=chunk_idx,
                        extra_metadata={"sentence_count": len(current_sentences)},
                    )
                )

        return chunks
