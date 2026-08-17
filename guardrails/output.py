import re
import logging
from typing import List, Dict, Any, Optional
from guardrails.models import GroundingCheckResult

logger = logging.getLogger("voice_rag.guardrails.output")


class GroundingVerifier:
    """
    Evaluates factual grounding, citation validity, and hallucination scores
    between generated answers and supplied context chunks.
    """

    def __init__(self, grounding_threshold: float = 0.65):
        self.grounding_threshold = grounding_threshold

    def verify(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> GroundingCheckResult:
        if not answer or not answer.strip():
            return GroundingCheckResult(
                grounded=False,
                score=0.0,
                supported_claims=[],
                unsupported_claims=["Generated answer is empty."],
                missing_citations=[],
                invalid_citations=[],
            )

        if not context_chunks:
            return GroundingCheckResult(
                grounded=False,
                score=0.0,
                supported_claims=[],
                unsupported_claims=["No retrieved context was provided for grounding."],
                missing_citations=[],
                invalid_citations=[],
            )

        # Aggregate all context text
        context_corpus = " ".join([c.get("text", "") for c in context_chunks]).lower()
        valid_chunk_ids = {c.get("chunk_id") for c in context_chunks if c.get("chunk_id")}

        # 1. Citation Validity Check
        invalid_citations = []
        if citations:
            for cit in citations:
                cit_id = cit.get("chunk_id") if isinstance(cit, dict) else getattr(cit, "chunk_id", None)
                if cit_id and cit_id not in valid_chunk_ids:
                    invalid_citations.append(str(cit_id))

        # 2. Extract Claim Sentences / Entities / Numbers
        # Split answer into sentence-level claims
        sentences = [s.strip() for s in re.split(r"[।\.\?\!\n]+", answer) if len(s.strip()) > 5]
        if not sentences:
            sentences = [answer.strip()]

        supported_claims = []
        unsupported_claims = []

        STOPWORDS = {
            "की", "का", "के", "में", "है", "हैं", "से", "पर", "को", "और", "या", "था", "थी", "थे", "यह", "वह",
            "to", "is", "are", "a", "an", "the", "in", "on", "of", "and", "or", "was", "were", "it", "this"
        }

        for sent in sentences:
            words = [w.lower().strip("।,?.!;:'\"()[]{}") for w in sent.split()]
            tokens = [w for w in words if len(w) >= 2 and w not in STOPWORDS]
            if not tokens:
                supported_claims.append(sent)
                continue

            # Check overlap against context corpus
            matched_tokens = [t for t in tokens if t in context_corpus]
            overlap_ratio = len(matched_tokens) / max(len(tokens), 1)

            # Also check numbers and dates
            numbers = re.findall(r"\b\d+[\.,]?\d*\b", sent)
            unsupported_numbers = [n for n in numbers if n not in context_corpus]

            if overlap_ratio >= 0.50 and len(unsupported_numbers) == 0:
                supported_claims.append(sent)
            else:
                unsupported_claims.append(sent)

        # 3. Compute Composite Grounding Score
        total_claims = len(supported_claims) + len(unsupported_claims)
        claim_grounding_rate = len(supported_claims) / max(total_claims, 1)

        # Penalize if citations are invalid or empty when context was used
        penalty = 0.0
        if invalid_citations:
            penalty += 0.30

        final_score = max(0.0, min(1.0, round(claim_grounding_rate - penalty, 4)))
        is_grounded = bool(final_score >= self.grounding_threshold and len(invalid_citations) == 0)

        missing_citations = []
        if is_grounded and not citations:
            missing_citations = ["Answer appears grounded but missing explicit chunk citations."]

        return GroundingCheckResult(
            grounded=is_grounded,
            score=final_score,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            missing_citations=missing_citations,
            invalid_citations=invalid_citations,
        )
