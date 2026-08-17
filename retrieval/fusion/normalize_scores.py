from typing import List, Dict, Any


def min_max_normalize(scores: List[float]) -> List[float]:
    """Normalize a list of float scores to [0.0, 1.0]."""
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    if max_val == min_val:
        return [1.0] * len(scores)
    return [(s - min_val) / (max_val - min_val) for s in scores]


def linear_score_fusion(
    deduped_candidates: List[Dict[str, Any]],
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Alternative linear interpolation fusion after min-max scaling."""
    if not deduped_candidates:
        return []

    dense_scores = [c["dense_score"] if c["dense_score"] is not None else 0.0 for c in deduped_candidates]
    bm25_scores = [c["bm25_score"] if c["bm25_score"] is not None else 0.0 for c in deduped_candidates]

    norm_dense = min_max_normalize(dense_scores)
    norm_bm25 = min_max_normalize(bm25_scores)

    fused = []
    for i, c in enumerate(deduped_candidates):
        item = dict(c)
        score = (dense_weight * norm_dense[i]) + (bm25_weight * norm_bm25[i])
        item["linear_score"] = round(score, 5)
        fused.append(item)

    fused.sort(key=lambda x: x["linear_score"], reverse=True)
    return fused[:top_k]
