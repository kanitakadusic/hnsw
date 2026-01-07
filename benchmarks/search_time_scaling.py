import time
import numpy as np
import matplotlib.pyplot as plt
import os

from hnsw.index import Index
from hnsw.node import Node

##################################################

np.random.seed(42)

n_runs = 30  # number of repetitions/queries per dataset size to average search timing
max_power = 17  # maximum power of 2 for dataset size
dim = 32

space = "euclidean"
M = 16
ef_construction = 100

k = 5
ef = 100

##################################################

h1 = f"| n_runs: {n_runs} | max_power: {max_power} | dim: {dim} |"
h2 = f"| space: {space} | M: {M} | ef_construction: {ef_construction} |"
h3 = f"| k: {k} | ef: {ef} |"

dashes = "-" * max(len(h1), len(h2), len(h3))
full_output = dashes + "\n" + h1 + "\n" + h2 + "\n" + h3 + "\n" + dashes
print(full_output)

# dataset sizes
sizes = [2 ** i for i in range(max_power + 1)]

# generate all vectors once
dataset = np.random.rand(sizes[-1], dim)
queries = np.random.rand(n_runs, dim)
query_nodes = [Node(vec, layer=-1) for vec in queries]

# results
hnsw_times = []
bf_times = []
recalls = []

# initialize hnsw
hnsw = Index(space, M, ef_construction)
nodes = []
current_count = 0
power_count = 0

for n in sizes:
    # add only new vectors to hnsw
    for vec in dataset[current_count:n]:
        node = hnsw.insert(vec, current_count)
        nodes.append(node)
        current_count += 1

    hnsw_time = 0.0
    bf_time = 0.0
    recall = 0.0

    for i in range(n_runs):
        start = time.perf_counter()
        bf_indices = np.argsort([hnsw.distance(query_nodes[i], node) for node in nodes])[:k]
        bf_time += (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        hnsw_results = hnsw.k_nn_search(query_nodes[i], k, ef)
        hnsw_time += (time.perf_counter() - start) * 1000.0

        hnsw_indices = {node.metadata for node in hnsw_results}
        recall += len(hnsw_indices & set(bf_indices)) / min(k, n)

    hnsw_times.append(hnsw_time / n_runs)
    bf_times.append(bf_time / n_runs)
    recalls.append(recall / n_runs)

    output = \
        f"n: 2 ^ {power_count:>2} = {n:>10} | " \
        f"HNSW avg: {hnsw_times[-1]:>10.3f} | " \
        f"BF avg: {bf_times[-1]:>10.3f} | " \
        f"recall: {recalls[-1]:>4.2f}"

    full_output += "\n" + output
    print(output)

    power_count += 1

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

# save output
output_path = os.path.join(results_dir, "search_time_scaling.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(full_output)

# plot with dual y-axis
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(sizes, hnsw_times, marker="s", color="blue", label="HNSW time")
ax1.plot(sizes, bf_times, marker="^", color="red", label="BF time")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xlabel("Dataset size")
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
param_text = f"dim: {dim} | space: {space} | M: {M} | ef construction: {ef_construction} | k: {k} | ef: {ef}"
plt.gcf().text(0.507, 0.9, param_text, ha="center", fontsize=10, color="gray")
plt.tight_layout()

# save plot
plot_path = os.path.join(results_dir, "search_time_scaling.png")
plt.savefig(plot_path, dpi=300)
