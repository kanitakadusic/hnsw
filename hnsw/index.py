import heapq
import numpy as np

from hnsw.distance import Distance
from hnsw.node import Node


class HNSW:
    def __init__(self, distance: str, M: int = 16, ef_construction: int = 200):
        self.distance = Distance(distance)
        self.entry_point = None

        self.M = M
        self.M_max = self.M
        self.M_max0 = 2 * self.M
        self.m_L = 1 / np.log(self.M)
        self.ef_construction = ef_construction

    def __random_layer(self) -> int:
        return int(np.floor(-np.log(np.random.uniform(0, 1)) * self.m_L))

    def insert(self, vector: list[float], metadata=None):
        new_point = Node(vector, self.__random_layer(), metadata)

        if self.entry_point is None:
            self.entry_point = new_point

        entry_point = self.entry_point

        for layer in range(entry_point.layer, new_point.layer, -1):
            entry_point = self.search_layer(new_point, entry_point, 1, layer)[0]

        for layer in range(min(entry_point.layer, new_point.layer), -1, -1):
            candidates = self.search_layer(new_point, entry_point, self.ef_construction, layer)
            neighbors = self.select_neighbours_simple(new_point, candidates, self.M)

            # add neighbors bidirectionally
            for node in neighbors:
                if node not in new_point.neighbors[layer]:
                    new_point.neighbors[layer].append(node)
                if new_point not in node.neighbors[layer]:
                    node.neighbors[layer].append(new_point)

            # shrink neighbors' neighborhood if needed
            for node in neighbors:
                node_neighbors = node.neighbors[layer]
                max_neighbors = self.M_max if layer > 0 else self.M_max0
                if len(node_neighbors) > max_neighbors:
                    node.neighbors[layer] = self.select_neighbours_simple(node, node_neighbors, max_neighbors)

            entry_point = candidates[0]

        if new_point.layer > entry_point.layer:
            self.entry_point = new_point

        return new_point

    def k_nn_search(self, query: Node, k: int, ef: int) -> list[Node]:
        entry_point = self.entry_point

        for layer in range(entry_point.layer, 0, -1):
            entry_point = self.search_layer(query, entry_point, 1, layer)[0]
        candidates = self.search_layer(query, entry_point, ef, 0)

        return self.select_neighbours_simple(query, candidates, k)

    def select_neighbours_simple(self, query: Node, candidates: list[Node], top_n: int) -> list[Node]:
        candidate_distance = [(c, self.distance(query, c)) for c in candidates]
        candidate_distance.sort(key=lambda pair: pair[1])
        return [neighbor for neighbor, _ in candidate_distance[:top_n]]

    def search_layer(self, query: Node, entry_point: Node, ef: int, layer: int) -> list[Node]:
        visited = set()
        visited.add(entry_point)

        entry_point_query_distance = self.distance(entry_point, query)

        candidates = []
        heapq.heappush(candidates, (entry_point_query_distance, entry_point))

        neighbors = []
        heapq.heappush(neighbors, (-entry_point_query_distance, entry_point))

        while len(candidates) > 0:
            closest_candidate = heapq.heappop(candidates)

            furthest_neighbor_distance = (-neighbors[0][0] if len(neighbors) > 0 else float("inf"))
            if closest_candidate[0] > furthest_neighbor_distance:
                break

            for node in closest_candidate[1].neighbors[layer]:
                if node not in visited and not node.is_deleted:
                    visited.add(node)

                    node_query_distance = self.distance(node, query)

                    furthest_neighbor_distance = (-neighbors[0][0] if len(neighbors) > 0 else float("inf"))
                    if node_query_distance < furthest_neighbor_distance or len(neighbors) < ef:
                        heapq.heappush(candidates, (node_query_distance, node))
                        if len(neighbors) + 1 > ef:
                            heapq.heappushpop(neighbors, (-node_query_distance, node))
                        else:
                            heapq.heappush(neighbors, (-node_query_distance, node))

        ef_neighbors = heapq.nlargest(ef, neighbors)
        return [neighbor for _, neighbor in ef_neighbors]
