from typing import List, Dict, Any, Optional
from pathlib import Path

from retrieval.vector_search import search
from retrieval.faiss.search import StrategyVectorSearcher
from retrieval.embeddings.provider import EmbeddingProviderFactory


class DenseRetriever:
    """
    Dense Vector Retriever wrapping FAISS and MultilingualDenseEmbedder from Module 4.
    Reuses existing index persistence, embeddings, and vector stores without duplication.
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        index_dir: Optional[Path] = None,
        model_name: str = "multilingual-dense-e5",
    ):
        self.strategy = strategy
        self.index_dir = index_dir
        self.model_name = model_name

    def search(
        self,
        query: str,
        strategy: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Execute dense vector retrieval.

        Args:
            query: User search query string.
            strategy: Chunking strategy (defaults to initialized strategy).
            top_k: Number of dense candidate vectors to retrieve (default: 20).

        Returns:
            List of standardized result dicts:
            [
                {
                    "source": "dense",
                    "rank": 1,
                    "score": 0.91,
                    "chunk_id": "...",
                    "query_id": "...",
                    "passage_id": "...",
                    "text": "...",
                    "is_selected": True,
                    "metadata": {}
                },
                ...
            ]
        """
        active_strategy = strategy or self.strategy
        raw_res = search(
            query=query,
            strategy=active_strategy,
            top_k=top_k,
            index_dir=self.index_dir,
            model_name=self.model_name,
        )

        dense_candidates: List[Dict[str, Any]] = []
        for item in raw_res.get("results", []):
            dense_candidates.append({
                "source": "dense",
                "rank": item.get("rank", len(dense_candidates) + 1),
                "score": round(float(item.get("score", 0.0)), 4),
                "chunk_id": str(item.get("chunk_id", "")),
                "record_id": str(item.get("record_id", "")),
                "query_id": int(item.get("query_id", 0)),
                "passage_id": str(item.get("passage_id", "")),
                "language": str(item.get("language", "hi")),
                "strategy": str(item.get("strategy", active_strategy)),
                "query_type": str(item.get("query_type", "standard")),
                "is_selected": bool(item.get("is_selected", False)),
                "token_count": int(item.get("token_count", 0)),
                "text": str(item.get("text", "")),
                "metadata": item.get("metadata", {}),
            })

        return dense_candidates
