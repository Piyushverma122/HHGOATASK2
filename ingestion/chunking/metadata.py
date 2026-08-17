from typing import List, Dict, Any, Optional
from ingestion.chunking.base import Chunker
from ingestion.chunking.models import Chunk
from ingestion.chunking.sentence import SentenceChunker
from ingestion.chunking.fixed import FixedChunker


class MetadataChunker(Chunker):
    """
    Strategy 6: Metadata-Aware Chunking.
    Adjusts chunking granularity and behavior dynamically based on query_type
    (NUMERIC, ENTITY, LOCATION, PERSON, DESCRIPTION) while strictly preserving
    all query/passage provenance and ground-truth selection metadata.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="metadata", config=config)
        self.numeric_chunk_size = int(self.config.get("numeric_chunk_size", 128))
        self.entity_chunk_size = int(self.config.get("entity_chunk_size", 192))
        self.description_chunk_size = int(self.config.get("description_chunk_size", 256))

    def chunk_passage(
        self,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
    ) -> List[Chunk]:
        text = passage.get("text", "").strip()
        if not text:
            return []

        query_type = str(record_context.get("query_type", "DESCRIPTION")).upper()

        # Query-type specific routing
        if query_type == "NUMERIC":
            # For numerical queries (dates, statistics, quantities), use tighter sentence/fixed chunks
            sub_chunker = SentenceChunker(
                config={
                    "target_chunk_tokens": self.numeric_chunk_size,
                    "max_chunk_tokens": self.numeric_chunk_size + 64,
                    "min_chunk_tokens": 24,
                }
            )
            strategy_tag = "metadata_numeric"
        elif query_type in ("ENTITY", "PERSON", "LOCATION"):
            # For entity-based queries, preserve complete sentence units with moderate window
            sub_chunker = SentenceChunker(
                config={
                    "target_chunk_tokens": self.entity_chunk_size,
                    "max_chunk_tokens": self.entity_chunk_size + 64,
                    "min_chunk_tokens": 32,
                }
            )
            strategy_tag = f"metadata_{query_type.lower()}"
        else:
            # For general descriptive queries, use standard descriptive sentence chunking
            sub_chunker = SentenceChunker(
                config={
                    "target_chunk_tokens": self.description_chunk_size,
                    "max_chunk_tokens": self.description_chunk_size + 96,
                    "min_chunk_tokens": 48,
                }
            )
            strategy_tag = "metadata_description"

        raw_chunks = sub_chunker.chunk_passage(passage, record_context)

        # Re-tag and enrich chunks with metadata strategy provenance
        chunks: List[Chunk] = []
        for c in raw_chunks:
            enriched_chunk = self._create_chunk(
                text=c.text,
                passage=passage,
                record_context=record_context,
                chunk_index=c.chunk_index,
                start_pos=c.start_position,
                end_pos=c.end_position,
                extra_metadata={
                    "query_type": query_type,
                    "metadata_strategy": strategy_tag,
                    "is_selected": bool(passage.get("is_selected", False)),
                },
            )
            chunks.append(enriched_chunk)

        return chunks
