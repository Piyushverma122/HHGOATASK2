from typing import Dict, Any, Optional, Union
from ingestion.chunking.models import Chunk


def prepare_embedding_text(
    chunk_or_dict: Union[Chunk, Dict[str, Any]],
    include_query_context: bool = False,
    prefix: Optional[str] = None,
) -> str:
    """
    Extracts and prepares the raw text content to be embedded.
    Preserves all chunk metadata separately without polluting the vector representation.

    Args:
        chunk_or_dict: A Chunk object or a dictionary representing a chunk.
        include_query_context: If True, optionally prepends query context (default: False).
        prefix: Optional prefix for asymmetric models (e.g. 'passage: ').

    Returns:
        Prepared clean text string for embedding.
    """
    if isinstance(chunk_or_dict, Chunk):
        raw_text = chunk_or_dict.text or ""
    elif isinstance(chunk_or_dict, dict):
        raw_text = chunk_or_dict.get("text", "") or ""
    else:
        raw_text = str(chunk_or_dict)

    clean_text = raw_text.strip()

    if prefix:
        clean_text = f"{prefix.strip()} {clean_text}"

    return clean_text


def prepare_query_text(
    query: str,
    prefix: Optional[str] = None,
) -> str:
    """
    Prepares a search query string for embedding.
    Applies any model-specific prefix (e.g., 'query: ').
    """
    clean_query = (query or "").strip()
    if prefix:
        clean_query = f"{prefix.strip()} {clean_query}"
    return clean_query
