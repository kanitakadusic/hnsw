import numpy as np


class Node:
    _global_id = 0

    def __init__(self, vector: np.ndarray, layer: int, metadata=None):
        self.id = Node._global_id
        Node._global_id += 1

        self.vector = vector
        self.layer = layer
        self.metadata = metadata

        self.neighbors: dict[int, list["Node"]] = {i: [] for i in range(layer + 1)}
        self.is_deleted = False
        self.magnitude = np.linalg.norm(self.vector)

    def __eq__(self, other: "Node"):
        return isinstance(other, Node) and self.id == other.id

    def __lt__(self, other: "Node"):
        return self.magnitude < other.magnitude

    def __gt__(self, other: "Node"):
        return self.magnitude > other.magnitude

    def __hash__(self):
        return self.id
