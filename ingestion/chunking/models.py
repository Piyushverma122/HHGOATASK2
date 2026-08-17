import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class Chunk:
    """
    Strongly-typed canonical chunk model representing a discrete segment of text.
    Ensures complete end-to-end traceability back to the source query and passage.
    """
    chunk_id: str
    record_id: str
    query_id: int
    passage_id: str
    text: str
    strategy: str
    language: str
    source_lang: str
    target_lang: str
    query_type: str
    chunk_index: int
    start_position: int
    end_position: int
    token_count: int
    character_count: int
    is_selected_passage: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def generate_chunk_id(
        passage_id: str,
        strategy: str,
        chunk_index: int,
        text: Optional[str] = None,
    ) -> str:
        """
        Generate a deterministic chunk ID.
        Format: {passage_id}_{strategy}_{chunk_index}
        Appends deterministic 8-char SHA-256 hash to ensure uniqueness even under heavy overlap.
        """
        base_id = f"{passage_id}_{strategy}_{chunk_index}"
        if text is not None:
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
            return f"{base_id}_{text_hash}"
        return base_id
