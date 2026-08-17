from typing import Dict, Any, Optional
from ingestion.chunking.base import Chunker
from ingestion.chunking.fixed import FixedChunker, OverlapChunker
from ingestion.chunking.sentence import SentenceChunker
from ingestion.chunking.paragraph import ParagraphChunker
from ingestion.chunking.semantic import SemanticChunker
from ingestion.chunking.metadata import MetadataChunker
from ingestion.chunking.adaptive import AdaptiveChunker
from ingestion.chunking.config import chunking_settings


class ChunkerFactory:
    """
    Factory to construct and configure chunking strategy instances.
    """

    _REGISTRY = {
        "fixed": FixedChunker,
        "overlap": OverlapChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "semantic": SemanticChunker,
        "metadata": MetadataChunker,
        "adaptive": AdaptiveChunker,
    }

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        return list(cls._REGISTRY.keys())

    @classmethod
    def create(
        cls,
        strategy_name: str,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Chunker:
        name = strategy_name.lower()
        if name not in cls._REGISTRY:
            raise ValueError(
                f"Unknown chunking strategy '{strategy_name}'. "
                f"Available strategies: {cls.get_available_strategies()}"
            )

        # Merge base default settings with strategy-specific config and user override
        merged_config = dict(chunking_settings.STRATEGIES.get(name, {}))
        if config_override:
            merged_config.update(config_override)

        chunker_cls = cls._REGISTRY[name]
        return chunker_cls(config=merged_config)
