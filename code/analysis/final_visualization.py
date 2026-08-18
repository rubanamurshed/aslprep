"""
Final Visualization: CBF-based Brain Network Analysis Summary Figure

Combines results from cbf_correlation_analysis.py, smallworld.py, and
node_permutation.py into one comprehensive summary figure comparing
schizophrenia patients and controls: similarity matrices, global graph
metrics, small-world statistics, network-level connectivity, node-level
differences, and top significant regions.

Note: some numeric values below (clustering, path length, sigma/gamma/lambda)
are hardcoded from earlier script runs rather than recomputed here — update
these manually if the underlying analysis is rerun with different data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, needed for headless cluster environments
import matplotlib.pyplot as plt
import networkx as nx
from nilearn.datasets import fetch_atlas_schaefer_2018

# ─────────────────────────────────────────────
# Load all upstream results
# ─────────────────────────────────────────────
schiz_matrices = list(np.load("/data/raulab/asl2/results/matrices/schiz_matrices.npy"))
control_matrices = list(np.load("/data/raulab/asl2/results/matrices/control_matrices.npy"))
schiz_mean = np.mean(schiz_matrices, axis=0)
control_mean = np.mean(control_matrices, axis=0)
p_values = np.load("/data/raulab/asl2/results/matrices/node_pvalues.npy")
degree_diff = np.load("/data/raulab/asl2/results/matrices/node_degree_diff.npy")

# Atlas labels, used to identify which brain region/network each node belongs to
atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]


def threshold_matrix(matrix, density=0.20):
    """Convert a weighted similarity matrix into a binary graph by keeping
    only the top X% strongest connections (density = proportion kept)."""
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    upper_tri = mat[np.triu_indices_from(mat, k=1)]
    threshold = np.percentile(upper_tri, 100 * (1 - density))
    return (mat >= threshold).astype(float), threshold


s_bin, _ = threshold_matrix(schiz_mean, density=0.20)
c_bin, _ = threshold_matrix(control_mean, density=0.20)

# ─────────────────────────────────────────────
# Network-level summary: average degree difference within each of the 7
# canonical resting-state networks (Vis, SomMot, DorsAttn, etc.)
# ─────────────────────────────────────────────
def get_network(label):
    """Extract the network name from a Schaefer atlas label string,
    e.g. '7Networks_LH_Vis_1' -> 'Vis'"""
    parts = label.split('_')
    return parts[2] if len(parts) >= 3 else 'Other'

networks = {}
for idx, label in enumerate(labels[1:]):  # skip Background label
    network = get_network(label)
    if network not in networks:
        networks[network] = []
    networks[network].append(degree_diff[idx])

network_means = {n: np.mean(d) for n, d in networks.items()}
if 'Other' in network_means:
    del network_means['Other']

# ─────────────────────────────────────────────
# === FIGURE: 3x3 grid summarizing all key results ===
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle('CBF-based Brain Network Analysis: Schizophrenia vs Controls\n(N=9 schizophrenia, N=5 controls)',
             fontsize=16, fontweight='bold', y=0.98)

# 1-3. Mean similarity matrices (schiz, control) and their difference
ax1 = fig.add_subplot(3, 3, 1)
im1 = ax1.imshow(schiz_mean, cmap='hot', vmin=0.5, vmax=0.8)
ax1.set_title('Schizophrenia\nMean Similarity Matrix')
plt.colorbar(im1, ax=ax1)

ax2 = fig.add_subplot(3, 3, 2)
im2 = ax2.imshow(control_mean, cmap='hot', vmin=0.5, vmax=0.8)
ax2.set_title('Control\nMean Similarity Matrix')
plt.colorbar(im2, ax=ax2)

ax3 = fig.add_subplot(3, 3, 3)
diff_matrix = schiz_mean - control_mean
im3 = ax3.imshow(diff_matrix, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
ax3.set_title('Difference Matrix\n(Schiz - Control)')
plt.colorbar(im3, ax=ax3)

# 4. Global graph metrics (clustering, path length) — values from earlier analysis
ax4 = fig.add_subplot(3, 3, 4)
metrics = ['Clustering\n(p=0.014)', 'Path Length\n(p=0.001)']
schiz_vals = [0.7154, 1.7852]
control_vals = [0.6507, 1.8194]
x = np.arange(len(metrics))
width = 0.35
ax4.bar(x - width/2, schiz_vals, width, label='Schizophrenia', color='#E74C3C', alpha=0.8)
ax4.bar(x + width/2, control_vals, width, label='Controls', color='#3498DB', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.set_title('Global Graph Metrics')
ax4.legend()
ax4.set_ylabel('Value')

# 5. Small-world metrics (sigma/gamma/lambda) — values from smallworld.py output
ax5 = fig.add_subplot(3, 3, 5)
sw_metrics = ['Sigma', 'Gamma', 'Lambda']
schiz_sw = [2.4271, 2.5330, 1.0436]
control_sw = [2.4151, 2.6952, 1.1160]
x = np.arange(len(sw_metrics))
ax5.bar(x - width/2, schiz_sw, width, label='Schizophrenia', color='#E74C3C', alpha=0.8)
ax5.bar(x + width/2, control_sw, width, label='Controls', color='#3498DB', alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(sw_metrics)
ax5.set_title('Small-World Metrics\n(Both groups: Small-world)')
ax5.legend()
ax5.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Sigma=1 threshold')

# 6. Network-level connectivity summary (bar chart, red=hyper, blue=hypo)
ax6 = fig.add_subplot(3, 3, 6)
net_names = sorted(network_means.keys())
net_vals = [network_means[n] for n in net_names]
colors = ['#E74C3C' if v > 0 else '#3498DB' for v in net_vals]
bars = ax6.bar(net_names, net_vals, color=colors, alpha=0.8, edgecolor='black')
ax6.axhline(y=0, color='black', linestyle='--')
ax6.set_title('Network-level Connectivity\n(Red=Hyper, Blue=Hypo)')
ax6.set_ylabel('Mean Degree Diff')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

# 7. Distribution of node-level degree differences, with significant nodes marked
ax7 = fig.add_subplot(3, 3, 7)
ax7.hist(degree_diff, bins=40, color='purple', alpha=0.7, edgecolor='black')
ax7.axvline(x=0, color='black', linestyle='--', linewidth=2)
sig_mask = p_values < 0.01
ax7.scatter(degree_diff[sig_mask],
            np.zeros(sig_mask.sum()) + 2,
            color='gold', zorder=5, s=80, label='p<0.01', marker='*')
ax7.set_xlabel('Degree Centrality Difference')
ax7.set_ylabel('Number of ROIs')
ax7.set_title('Node-level Differences\n(★ = p<0.01 uncorrected)')
ax7.legend()

# 8. Top 6 hyper- and hypo-connected regions specifically
ax8 = fig.add_subplot(3, 3, 8)
sig_idx = np.where(p_values < 0.01)[0]
all_top_idx = list(np.argsort(degree_diff)[-6:][::-1]) + list(np.argsort(degree_diff)[:6])
top_labs = [labels[i].replace('7Networks_', '').replace('_', ' ') for i in all_top_idx]
top_diffs = degree_diff[all_top_idx]
top_colors = ['#E74C3C' if d > 0 else '#3498DB' for d in top_diffs]
top_sig = [i in sig_idx for i in all_top_idx]
bars = ax8.barh(range(len(top_labs)), top_diffs, color=top_colors, alpha=0.8)
for i, (bar, is_sig) in enumerate(zip(bars, top_sig)):
    if is_sig:
        ax8.text(bar.get_width() + 0.01 if bar.get_width() > 0 else bar.get_width() - 0.01,
                 i, '★', ha='left' if bar.get_width() > 0 else 'right', va='center', fontsize=12)
ax8.set_yticks(range(len(top_labs)))
ax8.set_yticklabels(top_labs, fontsize=7)
ax8.axvline(x=0, color='black', linestyle='--')
ax8.set_title('Top Regions\n(★ = p<0.01)')
ax8.set_xlabel('Degree Centrality Difference')

# 9. Plain-text summary panel of key findings
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
summary = """KEY FINDINGS

Global Metrics:
- Higher clustering in schizophrenia
  (0.715 vs 0.651, p=0.014)
- Shorter path length in schizophrenia
  (1.785 vs 1.819, p=0.001)

Small-World Organization:
- Both groups show small-world topology
- Schizophrenia: lower lambda (1.04 vs 1.12)

Network Level:
- Hypoconnected: Default, Limbic, SomMot
- Hyperconnected: DorsAttn, SalVentAttn

Significant Nodes (p<0.01):
- Hypo: pCunPCC, Visual cortex
- Hyper: Somatomotor cortex

Sample: N=9 schizophrenia, N=5 controls
Method: JSD-based CBF network analysis
Atlas: Schaefer 400-parcel, 7-network"""

ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/data/raulab/asl2/results/figures/final_results.png', dpi=150, bbox_inches='tight')
print("Final figure saved to /data/raulab/asl2/results/figures/final_results.png")
