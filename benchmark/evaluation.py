from typing import List, Dict, Any, Set


def compute_retrieval_metrics(
    eval_records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute standard IR metrics: Recall@1, Recall@5, Recall@10, and MRR.
    Each eval_record contains:
      - 'retrieved_passage_ids': List[str] in ranked order
      - 'relevant_passage_ids': Set[str] or List[str] ground truth relevant IDs
    """
    if not eval_records:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}

    r1_count = 0
    r5_count = 0
    r10_count = 0
    reciprocal_ranks = []

    for rec in eval_records:
        retrieved = rec.get("retrieved_passage_ids", [])
        relevant = set(rec.get("relevant_passage_ids", []))

        if not relevant:
            continue

        # Recall@1
        if retrieved and retrieved[0] in relevant:
            r1_count += 1

        # Recall@5
        top5 = set(retrieved[:5])
        if top5.intersection(relevant):
            r5_count += 1

        # Recall@10
        top10 = set(retrieved[:10])
        if top10.intersection(relevant):
            r10_count += 1

        # MRR (Mean Reciprocal Rank)
        rr = 0.0
        for rank, pid in enumerate(retrieved, start=1):
            if pid in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    total = max(len(eval_records), 1)
    return {
        "recall@1": round((r1_count / total) * 100.0, 2),
        "recall@5": round((r5_count / total) * 100.0, 2),
        "recall@10": round((r10_count / total) * 100.0, 2),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
    }
