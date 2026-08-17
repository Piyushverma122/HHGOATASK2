from pathlib import Path
from typing import Dict, Any, Optional
from retrieval.faiss.search import StrategyVectorSearcher
from retrieval.embeddings.provider import EmbeddingProviderFactory

# Cache initialized searchers per strategy to avoid per-query file IO
_SEARCHER_CACHE: Dict[str, StrategyVectorSearcher] = {}


def search(
    query: str,
    strategy: str = "adaptive",
    top_k: int = 10,
    index_dir: Optional[Path] = None,
    model_name: str = "multilingual-dense-e5",
) -> Dict[str, Any]:
    """
    Primary API for Vector Similarity Search across chunking strategy indexes.

    Args:
        query: Search query text (Hindi, English, Hinglish, etc.).
        strategy: Chunking strategy index to query (e.g. 'adaptive', 'sentence', 'fixed').
        top_k: Number of nearest neighbor chunks to return (default: 10).
        index_dir: Optional custom index directory.
        model_name: Embedding model identifier.

    Returns:
        Structured dictionary containing:
        - query: Original query text
        - strategy: Strategy queried
        - results: List of ranked chunk results with similarity score and metadata
        - latencies: Millisecond breakdown (query_embed_ms, faiss_search_ms, metadata_lookup_ms, total_ms)
        - total_vectors_in_index: Total chunks in index
    """
    cache_key = f"{strategy}_{str(index_dir)}_{model_name}"
    if cache_key not in _SEARCHER_CACHE:
        embedder = EmbeddingProviderFactory.get_provider(model_name=model_name)
        _SEARCHER_CACHE[cache_key] = StrategyVectorSearcher(
            strategy=strategy,
            index_dir=index_dir,
            embedder=embedder,
        )

    searcher = _SEARCHER_CACHE[cache_key]
    return searcher.search(query=query, top_k=top_k)
