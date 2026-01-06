import time
import numpy as np
import matplotlib.pyplot as plt
import os

from hnsw.index import Index
from hnsw.node import Node

np.random.seed(42)
n_runs = 50  # number of repetitions per search to average out timing
max_power = 17  # maximum power of 2 for dataset size
dim = 50

space = "euclidean"
M = 16
ef_construction = 200

k = 5
ef = 50

# dataset sizes
sizes = [2 ** i for i in range(max_power + 1)]

# generate all vectors once
all_vectors = np.random.rand(sizes[-1], dim)
query_vector = np.random.rand(dim)
query_node = Node(query_vector, layer=-1)

hnsw_times = []
bf_times = []
recalls = []

# initialize hnsw
hnsw = Index(space, M, ef_construction)
nodes = []
current_count = 0
power_count = 0

header = f"| seed: 42 | n_runs: {n_runs} | max_power: {max_power} | dim: {dim} | k: {k} | ef: {ef} | space: {space} |"
print("-" * len(header))
print(header)
print("-" * len(header))

for n in sizes:
    # add only new vectors to hnsw
    for vec in all_vectors[current_count:n]:
        node = hnsw.insert(vec, current_count)
        nodes.append(node)
        current_count += 1

    # hnsw timing
    hnsw_time = 0
    results = []
    for _ in range(n_runs):
        start = time.time()
        results = hnsw.k_nn_search(query_node, k, ef)
        hnsw_time += (time.time() - start) * 1000.0
    hnsw_times.append(hnsw_time / n_runs)

    # brute force timing
    bf_time = 0
    for _ in range(n_runs):
        start = time.time()
        distances = [hnsw.distance(query_node, node) for node in nodes]
        bf_indices = np.argsort(distances)[:k]
        bf_time += (time.time() - start) * 1000.0
    bf_times.append(bf_time / n_runs)

    # recall calculation
    hnsw_indices = [node.metadata for node in results]
    bf_indices = np.argsort([hnsw.distance(query_node, node) for node in nodes])[:k]
    overlap = set(hnsw_indices).intersection(bf_indices)
    recall = len(overlap) / min(k, n)
    recalls.append(recall)

    print(
        f"n: 2 ^ {power_count:>2} = {n:>10} | "
        f"HNSW avg: {hnsw_times[-1]:>10.3f} | "
        f"BF avg:   {bf_times[-1]:>10.3f} | "
        f"Recall: {recall:>4.2f}"
    )
    power_count += 1

# plot with dual y-axis
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(sizes, hnsw_times, marker="s", color="blue", label="HNSW time")
ax1.plot(sizes, bf_times, marker="^", color="red", label="BF time")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xlabel("Number of vectors")
ax1.set_ylabel("Search time [ms]")
ax1.grid(True)

# secondary y-axis for recall
ax2 = ax1.twinx()
ax2.plot(sizes, recalls, marker="o", color="green", linestyle="", label="Recall")
ax2.set_ylabel("Recall", color="green")
ax2.set_ylim(0, 1)
ax2.tick_params(axis="y", colors="green")
ax2.grid(False)

# combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

plt.title("HNSW vs Brute Force: Search Time Scaling", pad=25)
param_text = f"runs: {n_runs} | dim: {dim} | k: {k} | ef: {ef} | space: {space}"
plt.gcf().text(0.507, 0.9, param_text, ha="center", fontsize=10, color="gray")
plt.tight_layout()

# save plot
plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(plots_dir, exist_ok=True)
plot_path = os.path.join(plots_dir, "search_time_scaling.png")
plt.savefig(plot_path, dpi=300)

plt.show()
