"""
Node-Level Permutation Test: CBF Network Degree Centrality (Schizophrenia vs Controls)

Tests whether the observed difference in node-level degree centrality between
groups is statistically significant, using a non-parametric permutation test
(shuffling group labels) rather than assuming a parametric distribution.
FDR correction is applied across all nodes to control for multiple comparisons.
"""

import networkx as nx
import numpy as np
from scipy.stats import false_discovery_control

# Load group-averaged CBF similarity matrices (aggregate data, no individual
# subject-level values)
schiz_matrices = list(np.load("/data/raulab/asl2/results/matrices/schiz_matrices.npy"))
control_matrices = list(np.load("/data/raulab/asl2/results/matrices/control_matrices.npy"))

schiz_mean = np.mean(schiz_matrices, axis=0)
control_mean = np.mean(control_matrices, axis=0)


def threshold_matrix(matrix, density=0.20):
    """Convert a weighted similarity matrix into a binary graph by keeping
    only the top X% strongest connections (density = proportion kept)."""
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    upper_tri = mat[np.triu_indices_from(mat, k=1)]
    threshold = np.percentile(upper_tri, 100 * (1 - density))
    return (mat >= threshold).astype(float), threshold


# Observed (real) node-level degree centrality difference between groups
s_bin, _ = threshold_matrix(schiz_mean, density=0.20)
c_bin, _ = threshold_matrix(control_mean, density=0.20)

s_degree = np.array(list(nx.degree_centrality(nx.from_numpy_array(s_bin)).values()))
c_degree = np.array(list(nx.degree_centrality(nx.from_numpy_array(c_bin)).values()))
obs_diff = s_degree - c_degree

# ─────────────────────────────────────────────
# Node-level permutation test
# ─────────────────────────────────────────────
# Logic: repeatedly shuffle which subjects are labeled "schizophrenia" vs
# "control" (regardless of their true diagnosis), recompute the group
# difference each time, and see how often a random shuffle produces a
# difference as extreme as what we actually observed. If the real difference
# is rare among random shuffles, it's likely a genuine group effect rather
# than noise.

n_permutations = 1000
all_matrices = schiz_matrices + control_matrices
n_schiz = len(schiz_matrices)
n_rois = obs_diff.shape[0]
perm_diffs = np.zeros((n_permutations, n_rois))

print(f"Running {n_permutations} node-level permutations...")
for i in range(n_permutations):
    # Randomly reassign matrices to "schiz" and "control" groups, ignoring
    # their true labels, to build a null distribution
    perm_idx = np.random.permutation(len(all_matrices))
    perm_schiz = [all_matrices[j] for j in perm_idx[:n_schiz]]
    perm_control = [all_matrices[j] for j in perm_idx[n_schiz:]]

    ps_bin, _ = threshold_matrix(np.mean(perm_schiz, axis=0), density=0.20)
    pc_bin, _ = threshold_matrix(np.mean(perm_control, axis=0), density=0.20)

    ps_deg = np.array(list(nx.degree_centrality(nx.from_numpy_array(ps_bin)).values()))
    pc_deg = np.array(list(nx.degree_centrality(nx.from_numpy_array(pc_bin)).values()))
    perm_diffs[i] = ps_deg - pc_deg

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_permutations} done...")

# p-value per node: proportion of random shuffles with a difference at least
# as extreme as the real observed difference (two-sided)
p_values = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff), axis=0)

# FDR correction across all nodes to control for testing many ROIs at once
p_fdr = false_discovery_control(p_values)

n_sig = np.sum(p_fdr < 0.05)
print(f"\nSignificant nodes after FDR correction: {n_sig}/{n_rois}")

# Save results — raw p-values, FDR-corrected p-values, and the observed
# degree difference per node, for downstream visualization
np.save("/data/raulab/asl2/results/matrices/node_pvalues.npy", p_values)
np.save("/data/raulab/asl2/results/matrices/node_pvalues_fdr.npy", p_fdr)
np.save("/data/raulab/asl2/results/matrices/node_degree_diff.npy", obs_diff)
print("Results saved!")
