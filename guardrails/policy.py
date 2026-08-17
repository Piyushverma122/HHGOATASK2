import logging
from typing import List, Dict, Any, Optional, Tuple
from guardrails.models import (
    InputGuardrailResult,
    ContextGuardrailResult,
    GroundingCheckResult,
    PostGuardrailResult,
    AbstentionReason,
)
from guardrails.input import InputGuardrail
from guardrails.context import ContextGuardrail
from guardrails.output import GroundingVerifier

logger = logging.getLogger("voice_rag.guardrails.policy")


class GuardrailPolicy:
    """
    Centralized pre- and post-generation policy coordinator.
    Determines whether a query is safe to execute and whether an output is safe to return.
    """

    def __init__(
        self,
        input_guardrail: Optional[InputGuardrail] = None,
        context_guardrail: Optional[ContextGuardrail] = None,
        grounding_verifier: Optional[GroundingVerifier] = None,
        max_regeneration_attempts: int = 1,
    ):
        self.input_guardrail = input_guardrail or InputGuardrail()
        self.context_guardrail = context_guardrail or ContextGuardrail()
        self.grounding_verifier = grounding_verifier or GroundingVerifier()
        self.max_regeneration_attempts = max_regeneration_attempts

    def check_input(self, query: str) -> InputGuardrailResult:
        return self.input_guardrail.validate(query)

    def check_context(self, candidates: List[Dict[str, Any]]) -> ContextGuardrailResult:
        return self.context_guardrail.validate_and_budget(candidates)

    def check_output(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        citations: Optional[List[Dict[str, Any]]] = None,
        attempt: int = 1,
    ) -> PostGuardrailResult:
        grounding_res = self.grounding_verifier.verify(
            query=query,
            answer=answer,
            context_chunks=context_chunks,
            citations=citations,
        )

        if grounding_res.grounded:
            return PostGuardrailResult(
                passed=True,
                should_retry=False,
                abstention_reason=None,
                grounding_result=grounding_res,
                message="Answer passed factual grounding verification.",
            )

        # Grounding check failed: Decide whether to retry or abstain
        if attempt <= self.max_regeneration_attempts:
            logger.info(f"Grounding score {grounding_res.score:.2f} below threshold. Requesting 1 regeneration attempt.")
            return PostGuardrailResult(
                passed=False,
                should_retry=True,
                abstention_reason=AbstentionReason.GROUNDING_FAILURE,
                grounding_result=grounding_res,
                message="Answer failed grounding; triggering stricter regeneration attempt.",
            )

        logger.warning(f"Regeneration attempt exhausted. Abstaining due to ungrounded claims.")
        return PostGuardrailResult(
            passed=False,
            should_retry=False,
            abstention_reason=AbstentionReason.GROUNDING_FAILURE,
            grounding_result=grounding_res,
            message="Answer contains unverified or hallucinated claims not supported by retrieved context.",
        )

    @staticmethod
    def get_abstention_text(reason: AbstentionReason, language: str = "hi") -> str:
        """Returns natural, helpful abstention messages in user's query language."""
        if language in ["hi", "hi-IN", "hindi"]:
            templates = {
                AbstentionReason.EMPTY_QUERY: "कृपया अपना प्रश्न दर्ज करें या बोलें।",
                AbstentionReason.QUERY_TOO_LONG: "आपका प्रश्न बहुत लंबा है। कृपया इसे संक्षिप्त करके पुनः पूछें।",
                AbstentionReason.PROMPT_INJECTION: "असुरक्षित अथवा अमान्य निर्देश का पता चला है।",
                AbstentionReason.UNSAFE_CONTENT: "यह अनुरोध हमारी सुरक्षा नीति के अनुरूप नहीं है।",
                AbstentionReason.INSUFFICIENT_CONTEXT: "उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
                AbstentionReason.OFF_TOPIC: "यह प्रश्न उपलब्ध ज्ञानकोश के विषय से बाहर प्रतीत होता है।",
                AbstentionReason.RETRIEVAL_FAILURE: "ज्ञानकोश से जानकारी प्राप्त करने में असमर्थ।",
                AbstentionReason.GROUNDING_FAILURE: "स्रोतों के आधार पर इस प्रश्न का सटीक उत्तर सत्यापित नहीं किया जा सका।",
                AbstentionReason.MALFORMED_CONTEXT: "स्रोतों का डेटा स्वरूप अमान्य है।",
            }
            return templates.get(reason, "स्रोतों में पर्याप्त जानकारी उपलब्ध नहीं है।")
        else:
            templates = {
                AbstentionReason.EMPTY_QUERY: "Please enter or speak a valid query.",
                AbstentionReason.QUERY_TOO_LONG: "Your query is too long. Please shorten it and try again.",
                AbstentionReason.PROMPT_INJECTION: "Unauthorized system instruction override detected.",
                AbstentionReason.UNSAFE_CONTENT: "This query violates our safety policies.",
                AbstentionReason.INSUFFICIENT_CONTEXT: "I do not have enough relevant information in the provided sources to answer that.",
                AbstentionReason.OFF_TOPIC: "This query appears outside the scope of the available knowledge base.",
                AbstentionReason.RETRIEVAL_FAILURE: "Unable to retrieve information from knowledge base.",
                AbstentionReason.GROUNDING_FAILURE: "Unable to verify a factual answer from the retrieved sources.",
                AbstentionReason.MALFORMED_CONTEXT: "Retrieved source format is invalid.",
            }
            return templates.get(reason, "Insufficient relevant information in provided sources.")
