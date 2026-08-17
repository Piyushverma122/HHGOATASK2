from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class EmbeddingProvider(ABC):
    """
    Abstract Base Class for Multilingual Embedding Providers.
    All embedding implementations must adhere to this interface.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name and identifier of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector embedding dimensionality (e.g., 384, 512, 768)."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an L2-normalized embedding vector for a single document / passage text.
        Returns a 1D float32 numpy array of shape (dimension,).
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate L2-normalized embedding vectors for a batch of document texts.
        Returns a 2D float32 numpy array of shape (len(texts), dimension).
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an L2-normalized embedding vector for a search query.
        Applies query-specific formatting/prefixes where required (e.g. 'query: ' for E5).
        Returns a 1D float32 numpy array of shape (dimension,).
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """
        Explicit model warmup mechanism to load weights and execute dry-run inference.
        Ensures runtime queries do not incur initialization overhead.
        """
        pass

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalize vectors along the last dimension so that cosine_similarity(A, B) == inner_product(A, B).
        Handles both 1D and 2D arrays safely.
        """
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / (norm + 1e-12) if norm > 0 else vectors
        elif vectors.ndim == 2:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            return vectors / norms
        return vectors
