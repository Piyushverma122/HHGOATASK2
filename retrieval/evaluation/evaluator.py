from typing import List, Dict, Any, Set
import numpy as np


class RetrievalEvaluator:
    """
    Evaluates retrieval candidate rankings against ground-truth MSMARCO-XI selected passages.
    Calculates:
    - Recall@1, Recall@3, Recall@5, Recall@10, Recall@20
    - Mean Reciprocal Rank (MRR)
    """

    @staticmethod
    def is_hit(candidate: Dict[str, Any], ground_truth_passage_ids: Set[str], query_id: int) -> bool:
        """
        Determine if candidate corresponds to ground truth.
        """
        if candidate.get("is_selected", False):
            return True
        p_id = candidate.get("passage_id", "")
        if p_id and p_id in ground_truth_passage_ids:
            return True
        c_qid = candidate.get("query_id")
        if c_qid == query_id and candidate.get("is_selected", False):
            return True
        return False

    def evaluate_query(
        self,
        retrieved_candidates: List[Dict[str, Any]],
        ground_truth_passage_ids: Set[str],
        query_id: int,
        ks: List[int] = [1, 3, 5, 10, 20],
    ) -> Dict[str, float]:
        """
        Compute recall@K and reciprocal rank for a single query.
        """
        recalls = {f"recall@{k}": 0.0 for k in ks}
        first_hit_rank = 0

        for rank, cand in enumerate(retrieved_candidates, start=1):
            if self.is_hit(cand, ground_truth_passage_ids, query_id):
                if first_hit_rank == 0:
                    first_hit_rank = rank
                for k in ks:
                    if rank <= k:
                        recalls[f"recall@{k}"] = 1.0

        rr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
        return {**recalls, "reciprocal_rank": rr, "first_hit_rank": first_hit_rank}

    def aggregate_metrics(self, query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate individual query metrics into mean benchmark scores.
        """
        if not query_metrics:
            return {
                "recall@1": 0.0,
                "recall@3": 0.0,
                "recall@5": 0.0,
                "recall@10": 0.0,
                "recall@20": 0.0,
                "mrr": 0.0,
                "total_queries": 0,
            }

        total = len(query_metrics)
        agg: Dict[str, float] = {}

        for k in [1, 3, 5, 10, 20]:
            key = f"recall@{k}"
            vals = [m.get(key, 0.0) for m in query_metrics]
            agg[key] = round(float(np.mean(vals)), 4)

        rrs = [m.get("reciprocal_rank", 0.0) for m in query_metrics]
        agg["mrr"] = round(float(np.mean(rrs)), 4)
        agg["total_queries"] = total

        return agg
