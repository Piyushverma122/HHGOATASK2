import pytest
import numpy as np

from retrieval.query_embedding import embed_query
from retrieval.vector_search import search


def test_embed_query_module():
    q = "निगम की कानूनी शक्तियाँ"
    vec = embed_query(q)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 384
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_search_on_primary_strategies():
    query = "निगम क्या है?"
    for strategy in ["fixed", "sentence", "adaptive"]:
        out = search(query=query, strategy=strategy, top_k=5)

        assert out["query"] == query
        assert out["strategy"] == strategy
        assert len(out["results"]) <= 5
        assert "latencies" in out
        assert "query_embed_ms" in out["latencies"]
        assert "faiss_search_ms" in out["latencies"]
        assert "metadata_lookup_ms" in out["latencies"]
        assert "total_ms" in out["latencies"]
        assert out["total_vectors_in_index"] > 0

        # Validate result structure
        if out["results"]:
            top = out["results"][0]
            assert "rank" in top
            assert "vector_id" in top
            assert "score" in top
            assert "chunk_id" in top
            assert "text" in top
            assert len(top["text"]) > 0


def test_search_top_k_parameter():
    query = "कानूनी इकाई"
    out_top3 = search(query=query, strategy="adaptive", top_k=3)
    out_top7 = search(query=query, strategy="adaptive", top_k=7)

    assert len(out_top3["results"]) == 3
    assert len(out_top7["results"]) == 7
    # Top 1 must match
    assert out_top3["results"][0]["vector_id"] == out_top7["results"][0]["vector_id"]
