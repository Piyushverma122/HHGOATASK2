import logging
from typing import Dict, Any, Optional, Type
from retrieval.embeddings.base import EmbeddingProvider
from retrieval.embeddings.model import MultilingualDenseEmbedder

logger = logging.getLogger("voice_rag.retrieval.embeddings")


class EmbeddingProviderFactory:
    """
    Factory for initializing and configuring multilingual embedding providers.
    Supports candidate models:
    - 'multilingual-dense-e5' (Default high-speed 384-d dense multilingual embedder)
    - 'intfloat/multilingual-e5-small' (384-d candidate)
    - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (384-d candidate)
    - 'BAAI/bge-m3' (1024-d candidate)
    """

    _REGISTRY: Dict[str, Type[EmbeddingProvider]] = {
        "multilingual-dense-e5": MultilingualDenseEmbedder,
        "multilingual-e5-small": MultilingualDenseEmbedder,
        "paraphrase-multilingual": MultilingualDenseEmbedder,
    }

    _SINGLETON_INSTANCES: Dict[str, EmbeddingProvider] = {}

    @classmethod
    def get_provider(
        cls,
        model_name: str = "multilingual-dense-e5",
        dimension: int = 384,
        use_cache: bool = True,
        force_reload: bool = False,
    ) -> EmbeddingProvider:
        key = f"{model_name}_{dimension}_{use_cache}"
        if not force_reload and key in cls._SINGLETON_INSTANCES:
            return cls._SINGLETON_INSTANCES[key]

        provider_cls = cls._REGISTRY.get(model_name, MultilingualDenseEmbedder)
        logger.info(f"Initializing EmbeddingProvider: {model_name} (dim={dimension}, cache={use_cache})")

        instance = provider_cls(
            model_name=model_name,
            dimension=dimension,
            use_cache=use_cache,
        )
        instance.warmup()

        cls._SINGLETON_INSTANCES[key] = instance
        return instance

    @classmethod
    def clear_cache(cls) -> None:
        cls._SINGLETON_INSTANCES.clear()


get_embedding_provider = EmbeddingProviderFactory.get_provider


# Global default helper
def get_default_embedder() -> EmbeddingProvider:
    return EmbeddingProviderFactory.get_provider(
        model_name="multilingual-dense-e5",
        dimension=384,
        use_cache=True,
    )
