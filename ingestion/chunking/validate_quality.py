import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ingestion.chunking.models import Chunk


def validate_chunk_quality(
    chunks: List[Chunk],
    max_token_limit: int = 512,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates a list of Chunk objects against strict production quality criteria.
    """
    seen_chunk_ids = set()
    duplicate_ids = []
    empty_chunks = []
    broken_unicode_chunks = []
    missing_query_id = []
    missing_passage_id = []
    invalid_token_counts = []
    oversized_chunks = []
    selected_passage_found = 0

    for idx, c in enumerate(chunks):
        # 1. Duplicate ID check
        if c.chunk_id in seen_chunk_ids:
            duplicate_ids.append(c.chunk_id)
        seen_chunk_ids.add(c.chunk_id)

        # 2. Empty text check
        if not c.text or not c.text.strip():
            empty_chunks.append(c.chunk_id)

        # 3. Unicode check
        try:
            c.text.encode("utf-8")
        except UnicodeError:
            broken_unicode_chunks.append(c.chunk_id)

        # 4. Query ID check
        if c.query_id is None or c.query_id == 0:
            missing_query_id.append(c.chunk_id)

        # 5. Passage ID check
        if not c.passage_id or c.passage_id == "unknown_p":
            missing_passage_id.append(c.chunk_id)

        # 6. Token count check
        if c.token_count <= 0 and len(c.text.strip()) > 0:
            invalid_token_counts.append(c.chunk_id)

        # 7. Max size check
        if c.token_count > max_token_limit:
            oversized_chunks.append({
                "chunk_id": c.chunk_id,
                "token_count": c.token_count,
                "limit": max_token_limit,
            })

        # 8. Selected passage tracking
        if c.is_selected_passage:
            selected_passage_found += 1

    issues_found = (
        len(duplicate_ids) > 0
        or len(empty_chunks) > 0
        or len(broken_unicode_chunks) > 0
        or len(missing_query_id) > 0
        or len(missing_passage_id) > 0
        or len(invalid_token_counts) > 0
        or len(oversized_chunks) > 0
    )

    report = {
        "total_chunks_evaluated": len(chunks),
        "unique_chunk_ids": len(seen_chunk_ids),
        "selected_passage_chunks": selected_passage_found,
        "is_valid": not issues_found,
        "issues": {
            "duplicate_chunk_ids_count": len(duplicate_ids),
            "empty_chunks_count": len(empty_chunks),
            "broken_unicode_count": len(broken_unicode_chunks),
            "missing_query_id_count": len(missing_query_id),
            "missing_passage_id_count": len(missing_passage_id),
            "invalid_token_counts_count": len(invalid_token_counts),
            "oversized_chunks_count": len(oversized_chunks),
        },
        "sample_duplicates": duplicate_ids[:5],
        "sample_oversized": oversized_chunks[:5],
    }

    return (not issues_found), report
