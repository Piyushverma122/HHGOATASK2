import pytest
import numpy as np
import tempfile
from pathlib import Path

from retrieval.embeddings.base import EmbeddingProvider
from retrieval.embeddings.cache import EmbeddingCache
from retrieval.embeddings.text import prepare_embedding_text, prepare_query_text
from retrieval.embeddings.model import MultilingualDenseEmbedder
from retrieval.embeddings.provider import EmbeddingProviderFactory, get_default_embedder


def test_embedding_provider_interface():
    embedder = get_default_embedder()
    assert isinstance(embedder, EmbeddingProvider)
    assert embedder.dimension == 384
    assert embedder.model_name == "multilingual-dense-e5"


def test_embed_text_shape_and_normalization():
    embedder = get_default_embedder()
    text = "यह एक परीक्षण वाक्य है।"
    vec = embedder.embed_text(text)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

    # L2 norm must be 1.0 (within float precision)
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_embed_batch():
    embedder = get_default_embedder()
    texts = [
        "पहला परीक्षण वाक्य।",
        "दूसरा परीक्षण वाक्य।",
        "Third English test sentence.",
    ]
    vectors = embedder.embed_batch(texts, batch_size=2)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32

    norms = np.linalg.norm(vectors, axis=1)
    for n in norms:
        assert pytest.approx(n, abs=1e-3) == 1.0


def test_embed_query_asymmetric_prefix():
    embedder = get_default_embedder()
    query = "निगम क्या है?"
    vec = embedder.embed_query(query)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_empty_and_whitespace_input():
    embedder = get_default_embedder()
    empty_vec = embedder.embed_text("")
    assert empty_vec.shape == (384,)

    ws_vec = embedder.embed_query("    ")
    assert ws_vec.shape == (384,)


def test_embedding_cache_persistence():
    import gc
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_cache.sqlite3"
        cache = EmbeddingCache(db_path=db_path, model_name="test_model", model_version="1.0")

        text1 = "परीक्षण वाक्य 1"
        vec1 = np.ones(384, dtype=np.float32) / np.sqrt(384)

        assert cache.get(text1) is None
        cache.set(text1, vec1)

        loaded = cache.get(text1)
        assert loaded is not None
        assert np.allclose(loaded, vec1, atol=1e-5)
        assert cache.count() == 1
    gc.collect()


def test_text_preparation_utilities():
    raw = "  हेलो दुनिया  "
    prep_doc = prepare_embedding_text(raw, prefix="passage:")
    assert prep_doc == "passage: हेलो दुनिया"

    prep_query = prepare_query_text(raw, prefix="query:")
    assert prep_query == "query: हेलो दुनिया"
