from typing import List, Dict, Any, Optional
from ingestion.chunking.base import Chunker
from ingestion.chunking.models import Chunk
from ingestion.chunking.utils import split_sentences, split_paragraphs
from ingestion.chunking.sentence import SentenceChunker
from ingestion.chunking.paragraph import ParagraphChunker
from ingestion.chunking.semantic import SemanticChunker


class AdaptiveChunker(Chunker):
    """
    Strategy 7: Adaptive Multi-Factor Chunking.
    Dynamically routes each passage to the most optimal chunking algorithm
    based on a deterministic decision tree inspecting:
    1. Passage token length (short, medium, long).
    2. Sentence and paragraph structural density.
    3. Query type (NUMERIC, ENTITY, DESCRIPTION).
    4. Semantic topic shift boundaries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="adaptive", config=config)
        self.short_threshold = int(self.config.get("short_passage_threshold", 64))
        self.long_threshold = int(self.config.get("long_passage_threshold", 256))
        self.max_chunk_tokens = int(self.config.get("max_chunk_tokens", 384))

        # Internal strategy delegates
        self.sentence_chunker = SentenceChunker(config=self.config)
        self.paragraph_chunker = ParagraphChunker(config=self.config)
        self.semantic_chunker = SemanticChunker(config=self.config)

    def determine_adaptive_strategy(
        self,
        token_count: int,
        sentence_count: int,
        paragraph_count: int,
        query_type: str,
    ) -> str:
        """
        Deterministic decision tree:
        1. Token count <= short_threshold or single sentence (within max_chunk_tokens): 'atomic_single'
        2. Paragraph count > 1 and token_count >= short_threshold: 'paragraph'
        3. Query type is NUMERIC and token_count > short_threshold: 'fine_sentence'
        4. Token count >= long_threshold and sentence_count >= 3: 'semantic'
        5. Default medium structured: 'sentence'
        """
        if token_count <= self.short_threshold or (sentence_count <= 1 and token_count <= self.max_chunk_tokens):
            return "atomic_single"
        if paragraph_count > 1:
            return "paragraph"
        if query_type == "NUMERIC":
            return "fine_sentence"
        if token_count >= self.long_threshold and sentence_count >= 3:
            return "semantic"
        return "sentence"

    def chunk_passage(
        self,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
    ) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []

        tokens_count = self.token_counter.count(text)
        sentences = split_sentences(text)
        paragraphs = split_paragraphs(text)
        query_type = str(record_context.get("query_type", "DESCRIPTION")).upper()

        decision = self.determine_adaptive_strategy(
            token_count=tokens_count,
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            query_type=query_type,
        )

        if decision == "atomic_single":
            if tokens_count <= self.max_chunk_tokens:
                chunks = [
                    self._create_chunk(
                        text=text,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=0,
                        extra_metadata={
                            "adaptive_route": "atomic_single",
                            "sentence_count": len(sentences),
                        },
                    )
                ]
            else:
                # Oversized single passage fallback
                raw_chunks = self.sentence_chunker.chunk_passage(passage, record_context)
                chunks = [
                    self._create_chunk(
                        text=c.text,
                        passage=passage,
                        record_context=record_context,
                        chunk_index=c.chunk_index,
                        extra_metadata={"adaptive_route": "sentence_fallback", "sentence_count": len(sentences)},
                    )
                    for c in raw_chunks
                ]
        elif decision == "paragraph":
            raw_chunks = self.paragraph_chunker.chunk_passage(passage, record_context)
            chunks = [
                self._create_chunk(
                    text=c.text,
                    passage=passage,
                    record_context=record_context,
                    chunk_index=c.chunk_index,
                    extra_metadata={"adaptive_route": "paragraph", "paragraph_count": len(paragraphs)},
                )
                for c in raw_chunks
            ]
        elif decision == "fine_sentence":
            fine_chunker = SentenceChunker(
                config={"target_chunk_tokens": 128, "max_chunk_tokens": 192, "min_chunk_tokens": 24}
            )
            raw_chunks = fine_chunker.chunk_passage(passage, record_context)
            chunks = [
                self._create_chunk(
                    text=c.text,
                    passage=passage,
                    record_context=record_context,
                    chunk_index=c.chunk_index,
                    extra_metadata={"adaptive_route": "fine_sentence", "query_type": "NUMERIC"},
                )
                for c in raw_chunks
            ]
        elif decision == "semantic":
            raw_chunks = self.semantic_chunker.chunk_passage(passage, record_context)
            chunks = [
                self._create_chunk(
                    text=c.text,
                    passage=passage,
                    record_context=record_context,
                    chunk_index=c.chunk_index,
                    extra_metadata={"adaptive_route": "semantic", "sentence_count": len(sentences)},
                )
                for c in raw_chunks
            ]
        else:  # sentence
            raw_chunks = self.sentence_chunker.chunk_passage(passage, record_context)
            chunks = [
                self._create_chunk(
                    text=c.text,
                    passage=passage,
                    record_context=record_context,
                    chunk_index=c.chunk_index,
                    extra_metadata={"adaptive_route": "sentence", "sentence_count": len(sentences)},
                )
                for c in raw_chunks
            ]

        return chunks
