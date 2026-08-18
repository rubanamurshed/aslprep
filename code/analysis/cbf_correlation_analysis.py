"""
CBF Brain Network Analysis: Per-Subject Graph Metrics + Correlation with Demographics & SANS
Author: Rubana
Method: JSD-based CBF shape similarity (He et al., NeuroImage 2025)
Atlas: Schaefer 400-parcel, 7-network
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. CONFIGURATION — edit paths here if needed
# ─────────────────────────────────────────────

BASE_PATH = "/data/raulab/asl2/output"
OUTPUT_DIR = "/data/raulab/asl2/correlation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCHIZ_SUBJECTS = ["sub-4028", "sub-4042", "sub-4045", "sub-4046", "sub-4049",
                  "sub-4061", "sub-4062", "sub-4063", "sub-4067"]
CONTROL_SUBJECTS = ["sub-5002", "sub-5008", "sub-5009", "sub-5012", "sub-5016"]

DEMOGRAPHICS_FILE = "/data/raulab/asl2/config/demographics.csv"
demo_df = pd.read_csv(DEMOGRAPHICS_FILE)
DEMOGRAPHICS = demo_df.set_index("subject").to_dict(orient="index")
SCHIZ_SUBJECTS = demo_df.loc[demo_df["group"] == "Schizophrenia", "subject"].tolist()
CONTROL_SUBJECTS = demo_df.loc[demo_df["group"] == "Control", "subject"].tolist()
# 7-network labels (Schaefer order)
NETWORK_NAMES = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_BOUNDARIES = [0, 58, 116, 166, 206, 236, 282, 400]  # approximate

DENSITY = 0.20  # 20% proportional threshold (good balance for this sample size)

# ─────────────────────────────────────────────
# 1. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_roi_distributions(cbf_data, atlas_data, unique_labels, n_bins=10):
    distributions = []
    for label in unique_labels:
        mask = atlas_data == label
        voxels = cbf_data[mask]
        voxels = voxels[(voxels != 0) & ~np.isnan(voxels)]
        if len(voxels) == 0:
            hist = np.ones(n_bins) / n_bins
        else:
            hist, _ = np.histogram(voxels, bins=n_bins, density=False)
            hist = hist + 1e-10
            hist = hist / hist.sum()
        distributions.append(hist)
    return np.array(distributions)

def build_similarity_matrix(distributions):
    n = len(distributions)
    jsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            jsd = jensenshannon(distributions[i], distributions[j])
            jsd_matrix[i, j] = jsd
            jsd_matrix[j, i] = jsd
    return 1 - jsd_matrix

def threshold_matrix(matrix, density=0.20):
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    upper_tri = mat[np.triu_indices_from(mat, k=1)]
    threshold = np.percentile(upper_tri, 100 * (1 - density))
    binary = (mat >= threshold).astype(float)
    return binary

def compute_graph_metrics(binary_matrix, atlas_labels):
    """Compute global and network-level graph metrics for one subject"""
    G = nx.from_numpy_array(binary_matrix)

    clustering = nx.average_clustering(G)

    # Path length on largest connected component
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    G_lcc = G.subgraph(largest)
    path_length = nx.average_shortest_path_length(G_lcc)

    # Degree centrality per node
    degree_cent = np.array(list(nx.degree_centrality(G).values()))

    # Network-level mean degree centrality
    network_degrees = {}
    for net_idx, net_name in enumerate(NETWORK_NAMES):
        start = NETWORK_BOUNDARIES[net_idx]
        end = NETWORK_BOUNDARIES[net_idx + 1]
        network_degrees[net_name] = degree_cent[start:end].mean()

    return {
        "clustering": clustering,
        "path_length": path_length,
        "mean_degree": degree_cent.mean(),
        "degree_centrality": degree_cent,
        **{f"net_{k}": v for k, v in network_degrees.items()}
    }

def partial_correlation(x, y, covariates):
    """
    Partial correlation between x and y controlling for covariates.
    Uses residual method.
    """
    from numpy.linalg import lstsq

    def residualize(var, covs):
        covs_with_intercept = np.column_stack([np.ones(len(var)), covs])
        coeffs, _, _, _ = lstsq(covs_with_intercept, var, rcond=None)
        predicted = covs_with_intercept @ coeffs
        return var - predicted

    x_resid = residualize(x, covariates)
    y_resid = residualize(y, covariates)
    r, p = pearsonr(x_resid, y_resid)
    return r, p

# ─────────────────────────────────────────────
# 2. LOAD ATLAS (once, reused for all subjects)
# ─────────────────────────────────────────────

print("=" * 60)
print("CBF CORRELATION ANALYSIS")
print("=" * 60)

print("\n[1/4] Loading atlas...")
first_subj = SCHIZ_SUBJECTS[0]
ref_cbf_file = f"{BASE_PATH}/{first_subj}/ses-01/perf/{first_subj}_ses-01_space-MNI152NLin2009aSym_desc-basilGM_cbf.nii.gz"
ref_img = nib.load(ref_cbf_file)

atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
atlas_img = nib.load(atlas.maps)
atlas_labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]

atlas_resampled = resample_to_img(atlas_img, ref_img, interpolation='nearest')
atlas_data = atlas_resampled.get_fdata().squeeze()
unique_labels = np.unique(atlas_data)
unique_labels = unique_labels[unique_labels != 0]

print(f"  Atlas loaded: {len(unique_labels)} parcels")

# ─────────────────────────────────────────────
# 3. PROCESS EACH SUBJECT
# ─────────────────────────────────────────────

print(f"\n[2/4] Computing per-subject matrices & graph metrics (density={DENSITY})...")

all_subjects = SCHIZ_SUBJECTS + CONTROL_SUBJECTS
results = []

for subj in all_subjects:
    print(f"  Processing {subj}...", end=" ")
    cbf_file = f"{BASE_PATH}/{subj}/ses-01/perf/{subj}_ses-01_space-MNI152NLin2009aSym_desc-basilGM_cbf.nii.gz"

    if not os.path.exists(cbf_file):
        print("FILE NOT FOUND, skipping")
        continue

    img = nib.load(cbf_file)
    cbf = img.get_fdata().squeeze()

    dists = get_roi_distributions(cbf, atlas_data, unique_labels)
    sim_matrix = build_similarity_matrix(dists)
    binary = threshold_matrix(sim_matrix, density=DENSITY)
    metrics = compute_graph_metrics(binary, atlas_labels)

    row = {
        "subject": subj,
        "group": DEMOGRAPHICS[subj]["group"],
        "age": DEMOGRAPHICS[subj]["age"],
        "sex": DEMOGRAPHICS[subj]["sex"],
        "SANS": DEMOGRAPHICS[subj]["SANS"],
        "mean_GM_CBF": np.nanmean(cbf[(cbf != 0) & ~np.isnan(cbf)]),
        **{k: v for k, v in metrics.items() if k != "degree_centrality"}
    }
    results.append(row)
    print(f"clustering={metrics['clustering']:.3f}, path={metrics['path_length']:.3f}")

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_metrics.csv"), index=False)
print(f"\n  Saved per_subject_metrics.csv ({len(df)} subjects)")

# ─────────────────────────────────────────────
# 4. CORRELATION ANALYSIS
# ─────────────────────────────────────────────

print("\n[3/4] Running correlation analyses...")

# Variables to correlate
connectivity_vars = ["clustering", "path_length", "mean_degree",
                     "net_Vis", "net_SomMot", "net_DorsAttn",
                     "net_SalVentAttn", "net_Limbic", "net_Cont", "net_Default"]

var_labels = {
    "clustering": "Clustering Coefficient",
    "path_length": "Path Length",
    "mean_degree": "Mean Degree Centrality",
    "net_Vis": "Visual Network",
    "net_SomMot": "Somatomotor Network",
    "net_DorsAttn": "Dorsal Attention Network",
    "net_SalVentAttn": "Salience/VentAttn Network",
    "net_Limbic": "Limbic Network",
    "net_Cont": "Control Network",
    "net_Default": "Default Mode Network",
}

corr_results = []

# --- A) SANS correlations (schizophrenia only, controlling for age + sex) ---
df_schiz = df[df["group"] == "Schizophrenia"].dropna(subset=["SANS"]).copy()
print(f"\n  SANS correlations (N={len(df_schiz)} schizophrenia patients):")

for var in connectivity_vars:
    x = df_schiz[var].values.astype(float)
    y = df_schiz["SANS"].values.astype(float)
    covs = df_schiz[["age", "sex"]].values.astype(float)

    # Need at least 5 subjects for partial correlation to be meaningful
    if len(x) < 4:
        continue

    r, p = partial_correlation(x, y, covs)
    corr_results.append({"variable": var, "covariate_controlled": "age, sex",
                         "correlation_with": "SANS", "sample": "Schizophrenia only",
                         "r": r, "p": p, "N": len(x)})
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"    {var_labels[var]:35s}: r={r:+.3f}, p={p:.3f} {sig}")

# --- B) Age correlations (all subjects, controlling for group + sex) ---
print(f"\n  Age correlations (N={len(df)} all subjects, controlling for group + sex):")
df["group_bin"] = (df["group"] == "Schizophrenia").astype(float)

for var in connectivity_vars:
    x = df[var].values.astype(float)
    y = df["age"].values.astype(float)
    covs = df[["group_bin", "sex"]].values.astype(float)
    r, p = partial_correlation(x, y, covs)
    corr_results.append({"variable": var, "covariate_controlled": "group, sex",
                         "correlation_with": "age", "sample": "All subjects",
                         "r": r, "p": p, "N": len(x)})
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"    {var_labels[var]:35s}: r={r:+.3f}, p={p:.3f} {sig}")

# Save correlation table
corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlation_results.csv"), index=False)

# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────

print("\n[4/4] Generating figures...")

GROUP_COLORS = {"Schizophrenia": "#E8735A", "Control": "#6BAED6"}

fig = plt.figure(figsize=(20, 16))
fig.suptitle("CBF Network Connectivity: Correlations with Demographics & Negative Symptoms\n"
             "(Partial correlations controlling for age, sex, and/or group)",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

# ── Row 1: SANS scatter plots (top 4 most interesting variables) ──
sans_vars = ["clustering", "net_Default", "net_Limbic", "net_DorsAttn"]
sans_titles = ["Clustering\nvs SANS", "Default Mode\nvs SANS",
               "Limbic Network\nvs SANS", "Dorsal Attention\nvs SANS"]

for col, (var, title) in enumerate(zip(sans_vars, sans_titles)):
    ax = fig.add_subplot(gs[0, col])
    x_vals = df_schiz[var].values
    y_vals = df_schiz["SANS"].values

    ax.scatter(x_vals, y_vals, color=GROUP_COLORS["Schizophrenia"],
               s=80, zorder=3, edgecolors='white', linewidth=0.8)

    # Add regression line
    if len(x_vals) > 2:
        z = np.polyfit(x_vals, y_vals, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, p_line(x_line), color="#333333", linewidth=1.5, linestyle="--", alpha=0.7)

    # Label points with subject IDs
    for i, row in df_schiz.iterrows():
        ax.annotate(row["subject"].replace("sub-", ""),
                    (row[var], row["SANS"]),
                    fontsize=6, ha='left', va='bottom', color='gray',
                    xytext=(2, 2), textcoords='offset points')

    # Get r and p from results
    match = corr_df[(corr_df["variable"] == var) & (corr_df["correlation_with"] == "SANS")]
    if len(match):
        r_val = match["r"].values[0]
        p_val = match["p"].values[0]
        ax.set_title(f"{title}\nr={r_val:+.3f}, p={p_val:.3f}", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)

    ax.set_xlabel(var_labels[var], fontsize=8)
    ax.set_ylabel("SANS Total Score", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor("#fafafa")

# ── Row 2: Age scatter plots (all subjects) ──
age_vars = ["clustering", "net_Default", "net_Limbic", "net_DorsAttn"]
age_titles = ["Clustering\nvs Age", "Default Mode\nvs Age",
              "Limbic Network\nvs Age", "Dorsal Attention\nvs Age"]

for col, (var, title) in enumerate(zip(age_vars, age_titles)):
    ax = fig.add_subplot(gs[1, col])

    for _, row in df.iterrows():
        ax.scatter(row[var], row["age"],
                   color=GROUP_COLORS[row["group"]],
                   s=80, zorder=3, edgecolors='white', linewidth=0.8)

    # Regression line across all subjects
    x_vals = df[var].values
    y_vals = df["age"].values
    z = np.polyfit(x_vals, y_vals, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, p_line(x_line), color="#333333", linewidth=1.5, linestyle="--", alpha=0.7)

    match = corr_df[(corr_df["variable"] == var) & (corr_df["correlation_with"] == "age")]
    if len(match):
        r_val = match["r"].values[0]
        p_val = match["p"].values[0]
        ax.set_title(f"{title}\nr={r_val:+.3f}, p={p_val:.3f}", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)

    ax.set_xlabel(var_labels[var], fontsize=8)
    ax.set_ylabel("Age (years)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor("#fafafa")

    # Legend on first plot only
    if col == 0:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=GROUP_COLORS["Schizophrenia"], label="Schizophrenia"),
                           Patch(facecolor=GROUP_COLORS["Control"], label="Control")]
        ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

# ── Row 3: Correlation heatmap ──
ax_heat = fig.add_subplot(gs[2, :2])

# Build heatmap matrix: rows = connectivity vars, cols = [SANS, Age]
heat_r = np.zeros((len(connectivity_vars), 2))
heat_p = np.zeros((len(connectivity_vars), 2))

for i, var in enumerate(connectivity_vars):
    for j, corr_with in enumerate(["SANS", "age"]):
        match = corr_df[(corr_df["variable"] == var) & (corr_df["correlation_with"] == corr_with)]
        if len(match):
            heat_r[i, j] = match["r"].values[0]
            heat_p[i, j] = match["p"].values[0]

im = ax_heat.imshow(heat_r, cmap="RdBu_r", vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax_heat, label="Partial r")

ax_heat.set_xticks([0, 1])
ax_heat.set_xticklabels(["SANS\n(Schiz only,\nctrl age+sex)",
                          "Age\n(All subj,\nctrl group+sex)"], fontsize=8)
ax_heat.set_yticks(range(len(connectivity_vars)))
ax_heat.set_yticklabels([var_labels[v] for v in connectivity_vars], fontsize=8)
ax_heat.set_title("Partial Correlation Heatmap\n(* p<0.05, ** p<0.01)", fontsize=10, fontweight="bold")

# Annotate with r values and significance stars
for i in range(len(connectivity_vars)):
    for j in range(2):
        r_val = heat_r[i, j]
        p_val = heat_p[i, j]
        sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        text_color = "white" if abs(r_val) > 0.5 else "black"
        ax_heat.text(j, i, f"{r_val:+.2f}{sig}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold" if sig else "normal")

# ── Row 3: Box plots of key metrics by group ──
ax_box1 = fig.add_subplot(gs[2, 2])
ax_box2 = fig.add_subplot(gs[2, 3])

for ax_box, var, ylabel in [(ax_box1, "clustering", "Clustering Coefficient"),
                              (ax_box2, "net_Default", "Mean Degree Centrality")]:
    schiz_vals = df[df["group"] == "Schizophrenia"][var].values
    ctrl_vals  = df[df["group"] == "Control"][var].values

    bp = ax_box.boxplot([schiz_vals, ctrl_vals],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=5))

    bp["boxes"][0].set_facecolor(GROUP_COLORS["Schizophrenia"])
    bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor(GROUP_COLORS["Control"])
    bp["boxes"][1].set_alpha(0.8)

    # Overlay individual points
    for i, (vals, color) in enumerate([(schiz_vals, GROUP_COLORS["Schizophrenia"]),
                                        (ctrl_vals, GROUP_COLORS["Control"])]):
        jitter = np.random.uniform(-0.08, 0.08, len(vals))
        ax_box.scatter([i+1+j for j in jitter], vals,
                       color=color, s=40, zorder=5, edgecolors='white', linewidth=0.5)

    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(["Schizophrenia", "Controls"], fontsize=8)
    ax_box.set_ylabel(ylabel, fontsize=8)
    ax_box.set_title(f"{ylabel}\nby Group", fontsize=9, fontweight="bold")
    ax_box.tick_params(labelsize=7)
    ax_box.spines[['top', 'right']].set_visible(False)
    ax_box.set_facecolor("#fafafa")

plt.savefig(os.path.join(OUTPUT_DIR, "correlation_analysis.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\n  Saved correlation_analysis.png")
print(f"  Saved per_subject_metrics.csv")
print(f"  Saved correlation_results.csv")
print(f"\n{'='*60}")
print("DONE! All outputs saved to:", OUTPUT_DIR)
print("="*60)

# Print final summary table
print("\nFINAL CORRELATION SUMMARY:")
print(corr_df[["variable", "correlation_with", "r", "p", "N"]].to_string(index=False))
