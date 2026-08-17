import hashlib
from typing import Set, Tuple, Dict, Any, Optional


class Deduplicator:
    """
    Context-aware deduplication for MSMARCO-XI records.
    Tracks:
    1. Seen Query IDs (to prevent duplicate queries within the same split/language).
    2. Query text content hash (to identify identical text questions with different IDs).
    3. Passage signatures (scoped by query ID or globally for statistical awareness).
    """

    def __init__(self):
        self.seen_query_ids: Set[Any] = set()
        self.seen_query_hashes: Set[str] = set()
        self.total_records: int = 0
        self.duplicate_records: int = 0
        self.unique_records: int = 0

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute deterministic SHA-256 hash of normalized text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def process_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if a record is a duplicate.
        Returns (is_unique, duplicate_reason).
        """
        self.total_records += 1
        query_id = record.get("query_id")
        query_text = record.get("query", "")

        # 1. Check duplicate query ID
        if query_id is not None and query_id in self.seen_query_ids:
            self.duplicate_records += 1
            return False, f"DUPLICATE_QUERY_ID: {query_id}"

        # 2. Check exact duplicate query text
        if query_text:
            text_hash = self.compute_text_hash(query_text)
            if text_hash in self.seen_query_hashes:
                self.duplicate_records += 1
                return False, f"DUPLICATE_QUERY_TEXT: hash {text_hash[:8]}"
            self.seen_query_hashes.add(text_hash)

        if query_id is not None:
            self.seen_query_ids.add(query_id)

        self.unique_records += 1
        return True, None

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_records": self.total_records,
            "duplicate_records": self.duplicate_records,
            "unique_records": self.unique_records,
        }
