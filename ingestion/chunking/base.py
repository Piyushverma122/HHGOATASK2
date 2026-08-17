from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ingestion.chunking.models import Chunk
from ingestion.chunking.utils import token_counter


class Chunker(ABC):
    """
    Abstract base class for all chunking strategies.
    Enforces a consistent interface and provides common utilities.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.token_counter = token_counter

    @abstractmethod
    def chunk_passage(
        self,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Chunk an individual passage belonging to a query record.
        """
        pass

    def chunk_record(self, record: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk all passages in a canonical record while maintaining traceability.
        """
        all_chunks: List[Chunk] = []
        passages = record.get("passages", [])
        for passage in passages:
            chunks = self.chunk_passage(passage, record)
            all_chunks.extend(chunks)
        return all_chunks

    def _create_chunk(
        self,
        text: str,
        passage: Dict[str, Any],
        record_context: Dict[str, Any],
        chunk_index: int,
        start_pos: int = 0,
        end_pos: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """Helper to construct a strongly-typed, traceable Chunk object."""
        passage_id = str(passage.get("passage_id", "unknown_p"))
        chunk_id = Chunk.generate_chunk_id(
            passage_id=passage_id,
            strategy=self.name,
            chunk_index=chunk_index,
            text=text,
        )

        metadata = dict(record_context.get("metadata", {}))
        if extra_metadata:
            metadata.update(extra_metadata)

        token_cnt = self.token_counter.count(text)
        char_cnt = len(text)

        return Chunk(
            chunk_id=chunk_id,
            record_id=str(record_context.get("record_id", "")),
            query_id=int(record_context.get("query_id", 0)),
            passage_id=passage_id,
            text=text,
            strategy=self.name,
            language=str(record_context.get("target_lang", "hi")),
            source_lang=str(record_context.get("source_lang", "eng_Latn")),
            target_lang=str(record_context.get("target_lang", "hin_Deva")),
            query_type=str(record_context.get("query_type", "standard")),
            chunk_index=chunk_index,
            start_position=start_pos,
            end_position=end_pos or char_cnt,
            token_count=token_cnt,
            character_count=char_cnt,
            is_selected_passage=bool(passage.get("is_selected", False)),
            metadata=metadata,
        )
