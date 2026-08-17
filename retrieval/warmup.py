import time
import logging
from typing import Dict, Any, Optional

from retrieval.pipeline import RetrievalPipeline
from retrieval.embeddings.provider import get_embedding_provider
from retrieval.reranking.reranker import RerankerService
from generation.provider import get_llm_provider
from generation.harness import RAGHarness

logger = logging.getLogger("voice_rag.warmup")

_WARMUP_RESULTS: Optional[Dict[str, Any]] = None


def warmup_system(verbose: bool = True) -> Dict[str, Any]:
    """
    Explicit warmup lifecycle for all ML models and index stores.
    Pre-allocates memory, triggers PyTorch JIT execution, and pre-loads Transformer weights.
    Returns cold-start vs warm-request timing benchmarks.
    """
    global _WARMUP_RESULTS
    total_start = time.perf_counter()
    timings = {}

    if verbose:
        logger.info("Initiating Voice RAG system warmup...")

    # 1. Warmup Embedding Provider
    t0 = time.perf_counter()
    embedder = get_embedding_provider()
    # Dummy embedding inference
    _ = embedder.embed_query("भारत की राजधानी")
    embed_ms = (time.perf_counter() - t0) * 1000.0
    timings["embedding_warmup_ms"] = round(embed_ms, 3)

    # 2. Warmup Cross-Encoder Reranker
    t0 = time.perf_counter()
    reranker = RerankerService(default_top_k=5)
    _ = reranker.rerank_candidates(
        query="भारत की राजधानी",
        candidates=[
            {"chunk_id": "c1", "passage_id": "p1", "text": "भारत की राजधानी नई दिल्ली है।", "rrf_score": 0.5},
            {"chunk_id": "c2", "passage_id": "p2", "text": "मुंबई महाराष्ट्र की राजधानी है।", "rrf_score": 0.4},
        ],
        top_k=2,
    )
    rerank_ms = (time.perf_counter() - t0) * 1000.0
    timings["reranker_warmup_ms"] = round(rerank_ms, 3)

    # 3. Warmup Hybrid Retrieval Pipeline
    t0 = time.perf_counter()
    pipeline = RetrievalPipeline(strategy="adaptive")
    _ = pipeline.retrieve("भारत की राजधानी", parallel=True, rerank_top_k=3)
    retrieval_ms = (time.perf_counter() - t0) * 1000.0
    timings["retrieval_pipeline_warmup_ms"] = round(retrieval_ms, 3)

    # 4. Warmup LLM Provider & RAG Harness
    t0 = time.perf_counter()
    harness = RAGHarness()
    _ = harness.process_rag_query("भारत की राजधानी", parallel=True)
    rag_ms = (time.perf_counter() - t0) * 1000.0
    timings["rag_harness_warmup_ms"] = round(rag_ms, 3)

    total_warmup_ms = (time.perf_counter() - total_start) * 1000.0
    timings["total_warmup_ms"] = round(total_warmup_ms, 3)

    # Measure Warm Request Latency immediately after warmup
    t_warm = time.perf_counter()
    _ = harness.process_rag_query("भारत की राजधानी क्या है?", parallel=True)
    warm_request_ms = (time.perf_counter() - t_warm) * 1000.0
    timings["warm_request_ms"] = round(warm_request_ms, 3)

    _WARMUP_RESULTS = {
        "status": "WARMED",
        "timings": timings,
        "cold_start_total_ms": timings["total_warmup_ms"],
        "warm_request_ms": timings["warm_request_ms"],
        "warmup_timestamp": time.time(),
    }

    if verbose:
        logger.info(
            f"Warmup complete in {total_warmup_ms:.2f}ms. Warm request baseline: {warm_request_ms:.2f}ms."
        )

    return _WARMUP_RESULTS


def get_warmup_status() -> Optional[Dict[str, Any]]:
    """Return status of the warmup lifecycle."""
    return _WARMUP_RESULTS
