from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseReranker(ABC):
    """
    Abstract Base Class for Multilingual Cross-Encoder Rerankers.
    Jointly evaluates (query, candidate passage) pairs.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier / HuggingFace ID of the reranker model."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Runtime execution device ('cpu' or 'cuda')."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer into memory once."""
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Perform a single warm-up inference pass."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (name, device, loaded, max_length, batch_size)."""
        pass

    @abstractmethod
    def score(self, query: str, passages: List[str]) -> List[float]:
        """
        Jointly score a query against a list of passage texts.
        Returns a list of float relevance scores (higher means more relevant).
        """
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate dictionaries and return top_k candidates sorted by relevance.
        """
        pass
