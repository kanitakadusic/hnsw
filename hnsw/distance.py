import numpy as np

from hnsw.node import Node


class Distance:
    _DIST_MAP = {}

    def __init__(self, space: str):
        self._init_distances()
        if space not in Distance._DIST_MAP.keys():
            raise ValueError(f"Invalid distance space. Choose from {list(Distance._DIST_MAP.keys())}.")
        self._distance_func = self._DIST_MAP[space]

    @classmethod
    def _init_distances(cls):
        if not cls._DIST_MAP:
            cls._DIST_MAP = {
                "cosine": cls._cosine,
                "euclidean": cls._euclidean
            }

    def __call__(self, n1: Node, n2: Node) -> float:
        return self._distance_func(n1, n2)

    @staticmethod
    def _cosine(n1: Node, n2: Node) -> float:
        if n1.magnitude == 0 or n2.magnitude == 0:
            return 1.0
        return 1 - np.dot(n1.vector, n2.vector) / (n1.magnitude * n2.magnitude)

    @staticmethod
    def _euclidean(n1: Node, n2: Node) -> float:
        return np.linalg.norm(n1.vector - n2.vector)
