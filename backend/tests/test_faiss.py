import pytest
import numpy as np
import tempfile
from pathlib import Path

from retrieval.faiss.index import FaissVectorStore
from retrieval.faiss.persistence import IndexPersistenceManager


def test_faiss_vector_store_exact_flat():
    dim = 64
    store = FaissVectorStore(dimension=dim, index_type="flat")
    assert store.size() == 0

    # Generate 10 normalized vectors
    rng = np.random.RandomState(42)
    vecs = rng.randn(10, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = np.arange(100, 110, dtype=np.int64)

    added = store.add_vectors(vecs, ids)
    assert added == 10
    assert store.size() == 10

    # Search with exact first vector -> should return id 100 with score ~1.0
    query_vec = vecs[0]
    scores, res_ids, latency_ms = store.search(query_vec, top_k=3)

    assert len(scores) == 3
    assert len(res_ids) == 3
    assert res_ids[0] == 100
    assert pytest.approx(scores[0], abs=1e-4) == 1.0
    assert latency_ms >= 0.0


def test_faiss_vector_store_hnsw():
    dim = 64
    store = FaissVectorStore(dimension=dim, index_type="hnsw", hnsw_m=16)

    rng = np.random.RandomState(42)
    vecs = rng.randn(20, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = np.arange(20, dtype=np.int64)

    store.add_vectors(vecs, ids)
    assert store.size() == 20

    scores, res_ids, _ = store.search(vecs[5], top_k=1)
    assert res_ids[0] == 5
    assert pytest.approx(scores[0], abs=1e-4) == 1.0


def test_faiss_save_and_reload_parity():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir) / "test_index"
        persistence = IndexPersistenceManager(dir_path)

        dim = 32
        store = FaissVectorStore(dimension=dim, index_type="flat")
        vecs = np.eye(4, dim, dtype=np.float32)
        ids = np.array([10, 20, 30, 40], dtype=np.int64)
        store.add_vectors(vecs, ids)

        metadata = [
            {"vector_id": 10, "chunk_id": "c1", "record_id": "r1", "query_id": 1, "passage_id": "p1", "language": "hi", "strategy": "fixed", "query_type": "s", "is_selected": True, "token_count": 50, "text": "t1", "metadata_json": "{}"},
            {"vector_id": 20, "chunk_id": "c2", "record_id": "r2", "query_id": 2, "passage_id": "p2", "language": "hi", "strategy": "fixed", "query_type": "s", "is_selected": False, "token_count": 60, "text": "t2", "metadata_json": "{}"},
        ]
        config = {"strategy": "fixed", "embedding_model": "test", "embedding_dimension": dim}

        persistence.save(store, metadata, config)

        # Reload
        reloaded_store = persistence.load_index()
        assert reloaded_store.size() == 4
        assert reloaded_store.dimension == dim

        reloaded_meta = persistence.load_metadata_lookup()
        assert 10 in reloaded_meta
        assert reloaded_meta[10]["chunk_id"] == "c1"
        assert reloaded_meta[10]["is_selected"] is True

        # Search reload parity
        scores, found_ids, _ = reloaded_store.search(vecs[0], top_k=1)
        assert found_ids[0] == 10
        assert pytest.approx(scores[0], abs=1e-4) == 1.0
