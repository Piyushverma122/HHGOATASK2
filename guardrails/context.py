import logging
from typing import List, Dict, Any, Optional
from guardrails.models import ContextGuardrailResult, AbstentionReason

logger = logging.getLogger("voice_rag.guardrails.context")


class ContextGuardrail:
    """
    Validates retrieved context chunks, enforces relevance thresholds,
    and applies context window budgeting.
    """

    def __init__(
        self,
        max_chunks: int = 5,
        max_context_chars: int = 8000,
        relevance_threshold: float = 0.0001,
    ):
        self.max_chunks = max_chunks
        self.max_context_chars = max_context_chars
        self.relevance_threshold = relevance_threshold

    def validate_and_budget(
        self,
        candidates: Optional[List[Dict[str, Any]]],
    ) -> ContextGuardrailResult:
        if not candidates or len(candidates) == 0:
            return ContextGuardrailResult(
                passed=False,
                relevance_score=0.0,
                selected_chunks=[],
                abstention_reason=AbstentionReason.INSUFFICIENT_CONTEXT,
                message="Retrieval returned 0 candidates for this query.",
            )

        # Validate candidate structure integrity
        cleaned_candidates = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            text = c.get("text", "")
            chunk_id = c.get("chunk_id", "")
            if text and chunk_id:
                cleaned_candidates.append(c)

        if not cleaned_candidates:
            return ContextGuardrailResult(
                passed=False,
                relevance_score=0.0,
                selected_chunks=[],
                abstention_reason=AbstentionReason.MALFORMED_CONTEXT,
                message="Retrieved candidates contain malformed or missing text payloads.",
            )

        # Determine highest relevance score (prioritize cross-encoder reranker score, then RRF, then dense)
        top_cand = cleaned_candidates[0]
        top_score = (
            top_cand.get("reranker_score")
            or top_cand.get("dense_score")
            or top_cand.get("rrf_score")
            or 0.0
        )

        # Check relevance threshold (Off-Topic / Insufficient Context detection)
        # Note: Cross-encoder sigmoid scores range in [0, 1]. Dense cosine ranges in [0, 1].
        if top_score < self.relevance_threshold and top_score > 0.0:
            logger.info(
                f"Context relevance score ({top_score:.4f}) below threshold ({self.relevance_threshold:.4f}). Abstaining."
            )
            return ContextGuardrailResult(
                passed=False,
                relevance_score=round(float(top_score), 4),
                selected_chunks=cleaned_candidates[:self.max_chunks],
                abstention_reason=AbstentionReason.INSUFFICIENT_CONTEXT,
                message="Retrieved context has insufficient semantic relevance to answer accurately.",
            )

        # Apply Context Budget (Max Chunks + Character Budget with Unicode integrity)
        budgeted_chunks = []
        accumulated_chars = 0

        for chunk in cleaned_candidates[:self.max_chunks]:
            chunk_len = len(chunk.get("text", ""))
            if accumulated_chars + chunk_len > self.max_context_chars and len(budgeted_chunks) >= 1:
                # Character budget full, preserve at least 1-2 full chunks without arbitrary Unicode cutting
                break
            budgeted_chunks.append(chunk)
            accumulated_chars += chunk_len

        return ContextGuardrailResult(
            passed=True,
            relevance_score=round(float(top_score), 4),
            selected_chunks=budgeted_chunks,
            abstention_reason=None,
            message=None,
        )
