import time
import logging
from typing import Optional, Dict, Any, List

from generation.base import LLMProvider
from generation.provider import get_llm_provider
from generation.schemas import (
    AnswerResponse,
    Citation,
    RAGQueryResponse,
    RAGLatencyBreakdown,
)
from generation.prompts import (
    RAG_SYSTEM_PROMPT,
    format_rag_context,
    build_rag_user_prompt,
)
from generation.cache import GenerationCache
from retrieval.cache.query_cache import QueryCache
from guardrails.policy import GuardrailPolicy
from guardrails.models import AbstentionReason
from retrieval.pipeline import RetrievalPipeline
from retrieval.query.normalize import normalize_query
from retrieval.query.analyze import analyze_query

logger = logging.getLogger("voice_rag.generation.harness")


class RAGHarness:
    """
    Production-grade Grounded RAG Orchestration Harness.
    Coordinates:
    Query -> Pre-Guardrail -> Parallel Retrieval -> Context Budgeting -> LLM Generation ->
    Grounding Verification -> Post-Guardrail -> Retry/Abstain -> Structured Response.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
        guardrail_policy: Optional[GuardrailPolicy] = None,
        cache: Optional[GenerationCache] = None,
        query_cache: Optional[QueryCache] = None,
    ):
        self.llm_provider = llm_provider or get_llm_provider()
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(strategy="adaptive")
        self.guardrail_policy = guardrail_policy or GuardrailPolicy()
        self.cache = cache or GenerationCache()
        self.query_cache = query_cache or QueryCache()

    def process_rag_query(
        self,
        query: str,
        strategy: str = "adaptive",
        top_k: int = 5,
        enable_reranking: bool = True,
        parallel: bool = True,
        request_id: Optional[str] = None,
    ) -> RAGQueryResponse:
        """
        Execute full lifecycle Grounded RAG query.
        """
        total_start = time.perf_counter()

        # ----------------------------------------------------
        # Stage 1: Normalization & Input Pre-Guardrail Check
        # ----------------------------------------------------
        norm_start = time.perf_counter()
        normalized_q = normalize_query(query)
        norm_ms = (time.perf_counter() - norm_start) * 1000.0

        analysis_start = time.perf_counter()
        analysis = analyze_query(normalized_q)
        analysis_ms = (time.perf_counter() - analysis_start) * 1000.0

        guard_pre_start = time.perf_counter()
        input_guard_res = self.guardrail_policy.check_input(normalized_q)

        if not input_guard_res.passed:
            guard_pre_ms = (time.perf_counter() - guard_pre_start) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0
            abstention_text = self.guardrail_policy.get_abstention_text(
                input_guard_res.abstention_reason or AbstentionReason.EMPTY_QUERY,
                language=analysis.language,
            )
            return RAGQueryResponse(
                query=query,
                normalized_query=normalized_q,
                detected_language=analysis.language,
                strategy=strategy,
                answer=abstention_text,
                grounded=False,
                confidence=0.0,
                citations=[],
                abstained=True,
                abstention_reason=input_guard_res.violation_type or "INPUT_GUARDRAIL_VIOLATION",
                retrieved_chunks=[],
                latency=RAGLatencyBreakdown(
                    normalization_ms=round(norm_ms, 3),
                    analysis_ms=round(analysis_ms, 3),
                    guardrail_pre_ms=round(guard_pre_ms, 3),
                    total_ms=round(total_ms, 3),
                ),
                request_id=request_id,
            )

        # ----------------------------------------------------
        # Stage 2: Module 5 Hybrid Retrieval & Cross-Encoder
        # ----------------------------------------------------
        retrieval_start = time.perf_counter()
        retrieval_out = self.retrieval_pipeline.retrieve(
            query=normalized_q,
            strategy=strategy,
            dense_k=20,
            bm25_k=20,
            hybrid_k=20,
            rerank_top_k=top_k,
            enable_reranking=enable_reranking,
            parallel=parallel,
        )
        retrieval_total_ms = (time.perf_counter() - retrieval_start) * 1000.0
        retrieval_latencies = retrieval_out.get("latency", {})

        raw_candidates = retrieval_out.get("final_context", [])

        # ----------------------------------------------------
        # Stage 3: Context Relevance & Budgeting Guardrail
        # ----------------------------------------------------
        context_guard_res = self.guardrail_policy.check_context(raw_candidates)
        guard_pre_ms = (time.perf_counter() - guard_pre_start) * 1000.0

        if not context_guard_res.passed:
            total_ms = (time.perf_counter() - total_start) * 1000.0
            abstention_text = self.guardrail_policy.get_abstention_text(
                context_guard_res.abstention_reason or AbstentionReason.INSUFFICIENT_CONTEXT,
                language=analysis.language,
            )
            return RAGQueryResponse(
                query=query,
                normalized_query=normalized_q,
                detected_language=analysis.language,
                strategy=strategy,
                answer=abstention_text,
                grounded=False,
                confidence=0.0,
                citations=[],
                abstained=True,
                abstention_reason=context_guard_res.abstention_reason.value if context_guard_res.abstention_reason else "INSUFFICIENT_CONTEXT",
                retrieved_chunks=context_guard_res.selected_chunks,
                latency=RAGLatencyBreakdown(
                    normalization_ms=round(norm_ms, 3),
                    analysis_ms=round(analysis_ms, 3),
                    guardrail_pre_ms=round(guard_pre_ms, 3),
                    dense_retrieval_ms=retrieval_latencies.get("dense_ms", 0.0),
                    bm25_retrieval_ms=retrieval_latencies.get("bm25_ms", 0.0),
                    reranking_ms=retrieval_latencies.get("rerank_ms", 0.0),
                    retrieval_total_ms=round(retrieval_total_ms, 3),
                    total_ms=round(total_ms, 3),
                ),
                request_id=request_id,
            )

        budgeted_chunks = context_guard_res.selected_chunks
        chunk_ids = [c.get("chunk_id", "") for c in budgeted_chunks]

        # ----------------------------------------------------
        # Stage 4: Cache Lookup
        # ----------------------------------------------------
        cached_answer = self.cache.get(
            model_name=self.llm_provider.model_name,
            query=normalized_q,
            chunk_ids=chunk_ids,
        )
        if cached_answer:
            total_ms = (time.perf_counter() - total_start) * 1000.0
            return RAGQueryResponse(
                query=query,
                normalized_query=normalized_q,
                detected_language=cached_answer.language,
                strategy=strategy,
                answer=cached_answer.answer,
                grounded=cached_answer.grounded,
                confidence=cached_answer.confidence,
                citations=cached_answer.citations,
                abstained=cached_answer.abstained,
                abstention_reason=cached_answer.abstention_reason,
                retrieved_chunks=budgeted_chunks,
                latency=RAGLatencyBreakdown(
                    normalization_ms=round(norm_ms, 3),
                    analysis_ms=round(analysis_ms, 3),
                    guardrail_pre_ms=round(guard_pre_ms, 3),
                    dense_retrieval_ms=retrieval_latencies.get("dense_ms", 0.0),
                    bm25_retrieval_ms=retrieval_latencies.get("bm25_ms", 0.0),
                    reranking_ms=retrieval_latencies.get("rerank_ms", 0.0),
                    retrieval_total_ms=round(retrieval_total_ms, 3),
                    context_prep_ms=0.0,
                    generation_ms=0.0,
                    verification_ms=0.0,
                    total_ms=round(total_ms, 3),
                ),
                request_id=request_id,
            )

        # ----------------------------------------------------
        # Stage 5: Context Formatting & Prompt Construction
        # ----------------------------------------------------
        prep_start = time.perf_counter()
        formatted_context = format_rag_context(budgeted_chunks)
        user_prompt = build_rag_user_prompt(normalized_q, formatted_context, is_retry=False)
        context_prep_ms = (time.perf_counter() - prep_start) * 1000.0

        # ----------------------------------------------------
        # Stage 6 & 7: LLM Generation & Grounding Verification Loop
        # ----------------------------------------------------
        gen_ms = 0.0
        verif_ms = 0.0
        final_answer_obj: Optional[AnswerResponse] = None

        max_attempts = 2  # 1 initial + max 1 regeneration
        for attempt in range(1, max_attempts + 1):
            is_retry = bool(attempt > 1)
            if is_retry:
                user_prompt = build_rag_user_prompt(normalized_q, formatted_context, is_retry=True)

            g_start = time.perf_counter()
            answer_obj = self.llm_provider.generate_structured(
                prompt=user_prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
            )
            gen_ms += (time.perf_counter() - g_start) * 1000.0

            # Output Grounding Verification
            v_start = time.perf_counter()
            post_guard = self.guardrail_policy.check_output(
                query=normalized_q,
                answer=answer_obj.answer,
                context_chunks=budgeted_chunks,
                citations=answer_obj.citations,
                attempt=attempt,
            )
            verif_ms += (time.perf_counter() - v_start) * 1000.0

            if post_guard.passed or answer_obj.abstained:
                final_answer_obj = answer_obj
                final_answer_obj.grounded = True
                break

            if post_guard.should_retry and attempt < max_attempts:
                logger.info("Triggering grounded regeneration attempt...")
                continue

            # If failed grounding after all retries -> Abstain
            abstention_text = self.guardrail_policy.get_abstention_text(
                AbstentionReason.GROUNDING_FAILURE,
                language=analysis.language,
            )
            final_answer_obj = AnswerResponse(
                answer=abstention_text,
                language=analysis.language,
                grounded=False,
                confidence=0.0,
                citations=[],
                abstained=True,
                abstention_reason=AbstentionReason.GROUNDING_FAILURE.value,
            )
            break

        if final_answer_obj is None:
            final_answer_obj = AnswerResponse(
                answer="स्रोतों में पर्याप्त जानकारी उपलब्ध नहीं है।",
                language=analysis.language,
                grounded=False,
                confidence=0.0,
                citations=[],
                abstained=True,
                abstention_reason="INSUFFICIENT_CONTEXT",
            )

        # Store grounded answer in cache
        if final_answer_obj.grounded and not final_answer_obj.abstained:
            self.cache.put(
                model_name=self.llm_provider.model_name,
                query=normalized_q,
                chunk_ids=chunk_ids,
                response=final_answer_obj,
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return RAGQueryResponse(
            query=query,
            normalized_query=normalized_q,
            detected_language=final_answer_obj.language or analysis.language,
            strategy=strategy,
            answer=final_answer_obj.answer,
            grounded=final_answer_obj.grounded,
            confidence=final_answer_obj.confidence,
            citations=final_answer_obj.citations,
            abstained=final_answer_obj.abstained,
            abstention_reason=final_answer_obj.abstention_reason,
            retrieved_chunks=budgeted_chunks,
            latency=RAGLatencyBreakdown(
                normalization_ms=round(norm_ms, 3),
                analysis_ms=round(analysis_ms, 3),
                guardrail_pre_ms=round(guard_pre_ms, 3),
                dense_retrieval_ms=retrieval_latencies.get("dense_ms", 0.0),
                bm25_retrieval_ms=retrieval_latencies.get("bm25_ms", 0.0),
                reranking_ms=retrieval_latencies.get("rerank_ms", 0.0),
                retrieval_total_ms=round(retrieval_total_ms, 3),
                context_prep_ms=round(context_prep_ms, 3),
                generation_ms=round(gen_ms, 3),
                verification_ms=round(verif_ms, 3),
                total_ms=round(total_ms, 3),
            ),
            request_id=request_id,
        )
