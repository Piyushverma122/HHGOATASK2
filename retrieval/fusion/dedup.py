from typing import List, Dict, Any, Optional


def deduplicate_candidates(
    dense_candidates: List[Dict[str, Any]],
    bm25_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deduplicate retrieval candidates by `chunk_id` while preserving dual provenance,
    individual dense & BM25 ranks, and raw scores.
    """
    merged_map: Dict[str, Dict[str, Any]] = {}

    # 1. Ingest Dense candidates
    for item in dense_candidates:
        cid = item.get("chunk_id")
        if not cid:
            continue
        merged_map[cid] = {
            "chunk_id": cid,
            "dense_rank": item.get("rank"),
            "dense_score": item.get("score"),
            "bm25_rank": None,
            "bm25_score": None,
            "record_id": item.get("record_id", ""),
            "query_id": item.get("query_id", 0),
            "passage_id": item.get("passage_id", ""),
            "language": item.get("language", "hi"),
            "strategy": item.get("strategy", ""),
            "query_type": item.get("query_type", "standard"),
            "is_selected": item.get("is_selected", False),
            "token_count": item.get("token_count", 0),
            "text": item.get("text", ""),
            "metadata": item.get("metadata", {}),
        }

    # 2. Ingest BM25 candidates
    for item in bm25_candidates:
        cid = item.get("chunk_id")
        if not cid:
            continue
        if cid in merged_map:
            merged_map[cid]["bm25_rank"] = item.get("rank")
            merged_map[cid]["bm25_score"] = item.get("score")
            # Fill in any missing metadata
            if not merged_map[cid]["text"] and item.get("text"):
                merged_map[cid]["text"] = item.get("text")
            if not merged_map[cid]["is_selected"] and item.get("is_selected"):
                merged_map[cid]["is_selected"] = item.get("is_selected")
        else:
            merged_map[cid] = {
                "chunk_id": cid,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": item.get("rank"),
                "bm25_score": item.get("score"),
                "record_id": item.get("record_id", ""),
                "query_id": item.get("query_id", 0),
                "passage_id": item.get("passage_id", ""),
                "language": item.get("language", "hi"),
                "strategy": item.get("strategy", ""),
                "query_type": item.get("query_type", "standard"),
                "is_selected": item.get("is_selected", False),
                "token_count": item.get("token_count", 0),
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
            }

    return list(merged_map.values())
