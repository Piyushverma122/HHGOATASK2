from typing import List, Dict, Any, Optional
from ingestion.chunking.base import Chunker
from ingestion.chunking.models import Chunk


class FixedChunker(Chunker):
    """
    Strategy 1: Fixed-Size Token Chunking.
    Splits passage text into deterministic, fixed-sized token windows.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="fixed", config=config)
        self.chunk_size = int(self.config.get("chunk_size", 256))

    def chunk_passage(
        self,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
    ) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []

        tokens = self.token_counter.encode(text)
        if not tokens:
            return []

        if len(tokens) <= self.chunk_size:
            return [
                self._create_chunk(
                    text=text,
                    passage=passage,
                    record_context=record_context,
                    chunk_index=0,
                    start_pos=0,
                    end_pos=len(text),
                )
            ]

        chunks: List[Chunk] = []
        chunk_idx = 0

        for start in range(0, len(tokens), self.chunk_size):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.token_counter.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=chunk_idx,
                        start_pos=start,
                        end_pos=end,
                    )
                )
                chunk_idx += 1

        return chunks


class OverlapChunker(Chunker):
    """
    Strategy 2: Fixed-Size Token Chunking with Sliding Overlap.
    Maintains context across chunk boundaries by sliding a window with stride (chunk_size - overlap).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="overlap", config=config)
        self.chunk_size = int(self.config.get("chunk_size", 256))
        self.overlap = int(self.config.get("overlap", 32))

        if self.overlap >= self.chunk_size:
            raise ValueError(f"Overlap ({self.overlap}) must be strictly less than chunk_size ({self.chunk_size})")
        if self.overlap < 0:
            raise ValueError("Overlap must be non-negative")

    def chunk_passage(
        self,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
    ) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []

        tokens = self.token_counter.encode(text)
        if not tokens:
            return []

        if len(tokens) <= self.chunk_size:
            return [
                self._create_chunk(
                    text=text,
                    passage=passage,
                    record_context=record_context,
                    chunk_index=0,
                    start_pos=0,
                    end_pos=len(text),
                )
            ]

        chunks: List[Chunk] = []
        stride = self.chunk_size - self.overlap
        chunk_idx = 0
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.token_counter.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=chunk_idx,
                        start_pos=start,
                        end_pos=end,
                        extra_metadata={"overlap_tokens": self.overlap},
                    )
                )
                chunk_idx += 1

            if end >= len(tokens):
                break
            start += stride

        return chunks
