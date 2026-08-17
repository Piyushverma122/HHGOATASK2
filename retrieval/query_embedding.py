from typing import Optional
import numpy as np
from retrieval.embeddings.provider import EmbeddingProviderFactory, get_default_embedder


def embed_query(
    query: str,
    model_name: str = "multilingual-dense-e5",
    dimension: int = 384,
) -> np.ndarray:
    """
    Generate an L2-normalized vector embedding for a user query.
    Validates model compatibility, dimension, and normalization invariants.
    """
    if not query or not query.strip():
        return np.zeros(dimension, dtype=np.float32)

    embedder = EmbeddingProviderFactory.get_provider(
        model_name=model_name,
        dimension=dimension,
    )
    vec = embedder.embed_query(query)

    # Invariant validation
    if len(vec) != dimension:
        raise ValueError(f"Query vector dimension mismatch: expected {dimension}, got {len(vec)}")

    norm = np.linalg.norm(vec)
    if norm < 0.99 or norm > 1.01:
        # Re-enforce L2 normalization if float drift occurred
        vec = vec / (norm + 1e-12)

    return vec
