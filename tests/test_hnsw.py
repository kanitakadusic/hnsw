import pytest
import numpy as np

from hnsw.index import Index
from hnsw.node import Node


def test_simple():
    vectors = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0]
    ])
    query_vector = np.array([1.1, 1.1])

    hnsw = Index(space="euclidean")
    for idx, vec in enumerate(vectors):
        hnsw.insert(vector=vec, metadata=idx)

    result = hnsw.k_nn_search(query=Node(query_vector, -1), k=1, ef=50)[0]
    assert result.metadata == 1, "HNSW did not find the true nearest neighbor"


def test_recall():
    np.random.seed(42)
    space = "cosine"
    num_vectors = 100
    dim = 20
    k = 3
    ef = 50

    vectors = np.random.rand(num_vectors, dim)
    query_vector = np.random.rand(dim)

    hnsw = Index(space)
    nodes = []
    for idx, vec in enumerate(vectors):
        node = hnsw.insert(vec, idx)
        nodes.append(node)

    query_node = Node(query_vector, -1)

    hnsw_results = hnsw.k_nn_search(query_node, k, ef)
    hnsw_indices = [node.metadata for node in hnsw_results]

    distances = [hnsw.distance(query_node, node) for node in nodes]
    bf_indices = np.argsort(distances)[:k]

    overlap = set(hnsw_indices).intersection(bf_indices)
    assert len(overlap) > 0, "HNSW did not find any of the true nearest neighbors"
