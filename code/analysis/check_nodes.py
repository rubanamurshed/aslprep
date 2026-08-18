"""
Check Significant Nodes: Label Lookup for Node-Level Permutation Results

Loads the uncorrected p-values and degree centrality differences from
node_permutation.py, and prints the anatomical/network label (from the
Schaefer 400-parcel atlas) for each node significant at p<0.01 uncorrected.

Useful as a quick sanity check / exploratory look before applying FDR
correction — helps confirm the significant regions make anatomical sense
before finalizing results.
"""

import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018

# Load node-level results from node_permutation.py
p_values = np.load('/data/raulab/asl2/results/matrices/node_pvalues.npy')
degree_diff = np.load('/data/raulab/asl2/results/matrices/node_degree_diff.npy')

# Load the same atlas used during graph construction, to map node index -> region name
atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]
labels = ['Background'] + labels  # atlas labels are 1-indexed, so index 0 = background

# Print any node significant at the (uncorrected) p<0.01 threshold, along
# with its direction of effect (positive = higher in schizophrenia group)
sig_idx = np.where(p_values < 0.01)[0]
print('Nodes significant at p<0.01 (uncorrected):')
for idx in sig_idx:
    print(f'  {labels[idx]}: diff={degree_diff[idx]:.3f}, p={p_values[idx]:.4f}')
