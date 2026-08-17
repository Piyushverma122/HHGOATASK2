import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


class FailureAnalyzer:
    """
    Analyzes retrieval misses and categorizes failure root causes.
    """

    @staticmethod
    def classify_failure(
        query: str,
        dense_candidates: List[Dict[str, Any]],
        bm25_candidates: List[Dict[str, Any]],
        fused_candidates: List[Dict[str, Any]],
        reranked_candidates: List[Dict[str, Any]],
        has_hit_in_dense: bool,
        has_hit_in_bm25: bool,
        has_hit_in_fused: bool,
        has_hit_in_rerank: bool,
    ) -> str:
        """
        Classifies failure mechanism based on pipeline stage drop-offs.
        """
        if has_hit_in_fused and not has_hit_in_rerank:
            return "reranker_failure"
        if has_hit_in_dense and not has_hit_in_bm25:
            return "lexical_mismatch"
        if has_hit_in_bm25 and not has_hit_in_dense:
            return "semantic_mismatch"
        if not has_hit_in_dense and not has_hit_in_bm25:
            words = query.strip().split()
            if len(words) <= 2:
                return "query_ambiguity"
            return "insufficient_candidate_pool"
        return "chunk_boundary_issue"

    @classmethod
    def record_failure(
        cls,
        query_id: int,
        query: str,
        expected_passage_ids: Set[str],
        pipeline_output: Dict[str, Any],
        ground_truth_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        dense = pipeline_output.get("dense_candidates", [])
        bm25 = pipeline_output.get("bm25_candidates", [])
        fused = pipeline_output.get("fused_candidates", [])
        reranked = pipeline_output.get("reranked_results", [])

        dense_hit = any(c.get("is_selected", False) or c.get("passage_id") in expected_passage_ids for c in dense)
        bm25_hit = any(c.get("is_selected", False) or c.get("passage_id") in expected_passage_ids for c in bm25)
        fused_hit = any(c.get("is_selected", False) or c.get("passage_id") in expected_passage_ids for c in fused)
        rerank_hit = any(c.get("is_selected", False) or c.get("passage_id") in expected_passage_ids for c in reranked)

        classification = cls.classify_failure(
            query=query,
            dense_candidates=dense,
            bm25_candidates=bm25,
            fused_candidates=fused,
            reranked_candidates=reranked,
            has_hit_in_dense=dense_hit,
            has_hit_in_bm25=bm25_hit,
            has_hit_in_fused=fused_hit,
            has_hit_in_rerank=rerank_hit,
        )

        top_retrieved = []
        for c in (reranked or fused)[:3]:
            top_retrieved.append({
                "chunk_id": c.get("chunk_id"),
                "passage_id": c.get("passage_id"),
                "text": c.get("text", "")[:120] + "...",
                "dense_score": c.get("dense_score"),
                "bm25_score": c.get("bm25_score"),
                "rrf_score": c.get("rrf_score"),
                "reranker_score": c.get("reranker_score"),
            })

        return {
            "query_id": query_id,
            "query": query,
            "expected_passage_ids": list(expected_passage_ids),
            "ground_truth_snippet": (ground_truth_text or "")[:120],
            "top_retrieved_passages": top_retrieved,
            "stage_hits": {
                "dense_hit": dense_hit,
                "bm25_hit": bm25_hit,
                "fused_hit": fused_hit,
                "rerank_hit": rerank_hit,
            },
            "failure_classification": classification,
        }

    @staticmethod
    def save_failures(failures: List[Dict[str, Any]], output_file: Path):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
