import pytest

from hnsw.index import HNSW
from hnsw.node import Node


def test_hnsw_simple():
    vectors = [
        [0.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0]
    ]
    query = [1.1, 1.1]

    hnsw = HNSW(distance="euclidean")
    for idx, vec in enumerate(vectors):
        hnsw.insert(vector=vec, metadata=idx)
    result = hnsw.k_nn_search(query=Node(query, -1), k=1, ef=100)[0]

    assert result.metadata == 1
