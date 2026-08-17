from typing import List, Dict, Any, Optional


def reciprocal_rank_fusion(
    deduped_candidates: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: Optional[int] = 20,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm:
        RRF_Score(d) = sum_{m in {dense, bm25}} 1.0 / (rrf_k + rank_m(d))
    
    Args:
        deduped_candidates: List of deduplicated candidates containing dense_rank and bm25_rank.
        rrf_k: Smoothing constant k (default: 60).
        top_k: Maximum number of fused candidates to return (default: 20).
    
    Returns:
        List of candidate dictionaries sorted by rrf_score in descending order.
    """
    fused_results: List[Dict[str, Any]] = []

    for item in deduped_candidates:
        score = 0.0
        d_rank = item.get("dense_rank")
        b_rank = item.get("bm25_rank")

        if d_rank is not None and d_rank > 0:
            score += 1.0 / (rrf_k + d_rank)
        if b_rank is not None and b_rank > 0:
            score += 1.0 / (rrf_k + b_rank)

        record = dict(item)
        record["rrf_score"] = round(score, 6)
        fused_results.append(record)

    # Sort descending by rrf_score, tie-breaking by dense_rank then bm25_rank
    fused_results.sort(
        key=lambda x: (
            x["rrf_score"],
            -(x["dense_rank"] or 9999),
            -(x["bm25_rank"] or 9999),
        ),
        reverse=True,
    )

    # Assign fused_rank
    for rank, candidate in enumerate(fused_results, start=1):
        candidate["fused_rank"] = rank

    if top_k is not None:
        return fused_results[:top_k]
    return fused_results
