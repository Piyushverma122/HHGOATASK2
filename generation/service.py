import logging
from typing import Optional
from generation.harness import RAGHarness

logger = logging.getLogger("voice_rag.generation.service")

_GLOBAL_RAG_HARNESS: Optional[RAGHarness] = None


def get_rag_harness() -> RAGHarness:
    """
    Singleton getter for global RAGHarness orchestrator.
    """
    global _GLOBAL_RAG_HARNESS
    if _GLOBAL_RAG_HARNESS is None:
        _GLOBAL_RAG_HARNESS = RAGHarness()
    return _GLOBAL_RAG_HARNESS
