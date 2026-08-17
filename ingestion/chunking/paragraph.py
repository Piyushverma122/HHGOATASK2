from typing import List, Dict, Any, Optional
from ingestion.chunking.base import Chunker
from ingestion.chunking.models import Chunk
from ingestion.chunking.utils import split_paragraphs
from ingestion.chunking.sentence import SentenceChunker


class ParagraphChunker(Chunker):
    """
    Strategy 4: Paragraph-Aware Chunking.
    Respects document and passage paragraph structure.
    Merges small adjacent paragraphs within the same passage up to target size.
    Splits oversized paragraphs using sentence-aware fallback.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="paragraph", config=config)
        self.target_chunk_tokens = int(self.config.get("target_chunk_tokens", 256))
        self.max_chunk_tokens = int(self.config.get("max_chunk_tokens", 384))
        self.min_chunk_tokens = int(self.config.get("min_chunk_tokens", 32))
        self.sentence_fallback = SentenceChunker(config=self.config)

    def chunk_passage(
        self,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
    ) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []

        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return []

        # If single paragraph and within limit, return as 1 chunk
        if len(paragraphs) == 1:
            p_tokens = self.token_counter.count(paragraphs[0])
            if p_tokens <= self.max_chunk_tokens:
                return [
                    self._create_chunk(
                        text=paragraphs[0],
                        passage=passage,
                        record_context=record_context,
                        chunk_index=0,
                        extra_metadata={"paragraph_count": 1},
                    )
                ]
            else:
                return self.sentence_fallback.chunk_passage(passage, record_context)

        chunks: List[Chunk] = []
        current_paras: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        for p in paragraphs:
            p_tokens = self.token_counter.count(p)

            # If an individual paragraph exceeds max_chunk_tokens, split via sentence fallback
            if p_tokens > self.max_chunk_tokens:
                # Flush previous paragraph buffer
                if current_paras:
                    combined = "\n\n".join(current_paras).strip()
                    if combined:
                        chunks.append(
                            self._create_chunk(
                                text=combined,
                                passage=passage,
                                record_context=record_context,
                                chunk_index=chunk_idx,
                                extra_metadata={"paragraph_count": len(current_paras)},
                            )
                        )
                        chunk_idx += 1
                    current_paras.clear()
                    current_tokens = 0

                # Sentence-chunk this large paragraph
                temp_passage = dict(passage)
                temp_passage["text"] = p
                sub_chunks = self.sentence_fallback.chunk_passage(temp_passage, record_context)
                for sc in sub_chunks:
                    chunks.append(
                        self._create_chunk(
                            text=sc.text,
                            passage=passage,
                            record_context=record_context,
                            chunk_index=chunk_idx,
                            extra_metadata={"paragraph_count": 1, "is_oversized_para_split": True},
                        )
                    )
                    chunk_idx += 1
                continue

            # If adding this paragraph exceeds max tokens, flush current
            if current_paras and (current_tokens + p_tokens > self.max_chunk_tokens):
                combined = "\n\n".join(current_paras).strip()
                if combined:
                    chunks.append(
                        self._create_chunk(
                            text=combined,
                            passage=passage,
                            record_context=record_context,
                            chunk_index=chunk_idx,
                            extra_metadata={"paragraph_count": len(current_paras)},
                        )
                    )
                    chunk_idx += 1
                current_paras = [p]
                current_tokens = p_tokens
            else:
                current_paras.append(p)
                current_tokens += p_tokens

                # If reached target tokens, flush
                if current_tokens >= self.target_chunk_tokens:
                    combined = "\n\n".join(current_paras).strip()
                    if combined:
                        chunks.append(
                            self._create_chunk(
                                text=combined,
                                passage=passage,
                                record_context=record_context,
                                chunk_index=chunk_idx,
                                extra_metadata={"paragraph_count": len(current_paras)},
                            )
                        )
                        chunk_idx += 1
                    current_paras.clear()
                    current_tokens = 0

        # Flush trailing paragraphs
        if current_paras:
            combined = "\n\n".join(current_paras).strip()
            if combined:
                if chunks and current_tokens < self.min_chunk_tokens:
                    last_chunk = chunks[-1]
                    merged_text = f"{last_chunk.text}\n\n{combined}".strip()
                    if self.token_counter.count(merged_text) <= self.max_chunk_tokens:
                        chunks[-1] = self._create_chunk(
                            text=merged_text,
                            passage=passage,
                            record_context=record_context,
                            chunk_index=last_chunk.chunk_index,
                            extra_metadata={"paragraph_count": last_chunk.metadata.get("paragraph_count", 1) + len(current_paras)},
                        )
                        return chunks

                chunks.append(
                    self._create_chunk(
                        text=combined,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=chunk_idx,
                        extra_metadata={"paragraph_count": len(current_paras)},
                    )
                )

        return chunks
