"""
CBF Graph Pipeline: Per-Subject Correlation-Based Graph Metrics + Group Tables

For each subject/session, loads a parcel-level CBF timeseries, computes a
correlation matrix between parcels, thresholds it into a weighted adjacency
matrix, and derives node-level graph metrics (strength, clustering,
betweenness centrality). Merges these with parcel-level CBF values, then
combines everyone into group-level long-format tables for downstream
statistics (see cbf_correlation_analysis.py, smallworld.py).

Atlas: 4S256 (256-parcel, this is a different/coarser atlas than the
Schaefer-400 used in the other analysis scripts — used here specifically
for the timeseries-based graph pipeline).
"""

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx


# ---------- USER SETTINGS ----------

# Root where all subject folders live
ROOT = "/data/raulab/asl2/output"

# Subject codes only (no clinical/demographic data) — matches Participants
# table in top-level README
SUBJECTS = [
    "sub-4028", "sub-4042", "sub-4045", "sub-4046", "sub-4049",
    "sub-4061", "sub-4062", "sub-4063", "sub-4067",
    "sub-5002", "sub-5008", "sub-5009", "sub-5012", "sub-5016"
]

SESSIONS = ["ses-01"]          # adjust if you later add more sessions
ATLAS = "4S256"
THRESH = 0.2                   # |r| threshold for adjacency

# Group-level output directory (matches results/stats/graph_cbf/ in top-level README)
GROUP_OUTPUT_DIR = "/data/raulab/asl2/results/stats/graph_cbf"
OUT_GROUP_MERGED = "allSubs_4S256_cbf_graphMerged.tsv"
OUT_GROUP_GLOBAL = "allSubs_4S256_globalMetrics.tsv"


# ---------- HELPER: MAKE GRAPH METRICS FOR ONE SUBJECT ----------

def compute_graph_metrics_from_corr(corr_mat, thresh):
    """
    corr_mat: 2D numpy array (parcels x parcels)
    thresh: absolute correlation threshold
    Returns a DataFrame with parcelID, strength, clustering, betweenness
    """
    # Threshold: keep weights where |r| >= thresh, zero otherwise
    adj = corr_mat.copy()
    adj[np.abs(adj) < thresh] = 0

    # Build weighted undirected graph (nodes 0..N-1)
    G = nx.from_numpy_array(adj)

    # Node strength = sum of weights
    strength_dict = dict(G.degree(weight="weight"))

    # Weighted clustering (networkx uses edge weights if present)
    clustering_dict = nx.clustering(G, weight="weight")

    # Betweenness with edge length = 1/weight (so stronger edges are shorter,
    # i.e. more strongly correlated parcels are treated as "closer")
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0.0)
        if w != 0:
            H.add_edge(u, v, weight=w, length=1.0 / abs(w))

    betw_dict = nx.betweenness_centrality(H, weight="length", normalized=True)

    # Pack into DataFrame (parcelID 1..N)
    n_parcels = corr_mat.shape[0]
    df = pd.DataFrame({
        "parcelID": np.arange(1, n_parcels + 1),
        "strength": [strength_dict.get(i, 0.0) for i in range(n_parcels)],
        "clustering": [clustering_dict.get(i, 0.0) for i in range(n_parcels)],
        "betweenness": [betw_dict.get(i, 0.0) for i in range(n_parcels)],
    })
    return df, adj


# ---------- MAIN SUBJECT-LEVEL LOOP ----------

def run_subject_pipeline():
    """
    Processes every subject/session: loads their CBF timeseries (produced by
    make_4S256_cbf_timeseries.py), builds a correlation-based graph, computes
    node metrics, and merges with parcel-level CBF values. Saves per-subject
    outputs alongside their perf/ data, and returns a list of output paths
    for the group-table step.
    """
    merged_paths = []

    for sub in SUBJECTS:
        for ses in SESSIONS:
            perf_dir = os.path.join(ROOT, sub, ses, "perf")
            if not os.path.isdir(perf_dir):
                print(f"[SKIP] {sub} {ses}: perf directory not found")
                continue

            print(f"[INFO] Processing {sub} {ses}")

            # ----- 1. Load timeseries and compute correlation -----
            ts_path = os.path.join(
                perf_dir, f"{sub}_{ses}_atlas-{ATLAS}_cbfTimeseries.npy"
            )
            if not os.path.exists(ts_path):
                print(f"   [WARN] Missing timeseries: {ts_path}")
                continue

            ts = np.load(ts_path)               # time x parcels
            corr = np.corrcoef(ts, rowvar=False)

            corr_path = os.path.join(
                perf_dir, f"{sub}_{ses}_atlas-{ATLAS}_cbfCorr.npy"
            )
            np.save(corr_path, corr)

            # ----- 2. Graph metrics from correlation -----
            metrics_df, adj = compute_graph_metrics_from_corr(corr, THRESH)

            adj_path = os.path.join(
                perf_dir, f"{sub}_{ses}_atlas-{ATLAS}_cbfAdj.npy"
            )
            np.save(adj_path, adj)

            metrics_path = os.path.join(
                perf_dir, f"{sub}_{ses}_atlas-{ATLAS}_graphMetrics.tsv"
            )
            metrics_df.to_csv(metrics_path, sep="\t", index=False)

            # ----- 3. Merge graph metrics with parcel-level CBF values -----
            cbf_tsv = os.path.join(
                perf_dir, f"{sub}_{ses}_atlas-{ATLAS}Parcels_cbf.tsv"
            )
            if not os.path.exists(cbf_tsv):
                print(f"   [WARN] Missing parcel CBF TSV: {cbf_tsv}")
                continue

            cbf = pd.read_csv(cbf_tsv, sep="\t")

            # one row, columns = parcel names in order
            cols = cbf.columns.tolist()
            cbf_values = cbf.iloc[0].values

            cbf_parcels = pd.DataFrame({
                "parcelID": range(1, len(cols) + 1),
                "parcel_name": cols,
                "cbf": cbf_values,
            })

            merged = metrics_df.merge(cbf_parcels, on="parcelID")

            merged_path = os.path.join(
                perf_dir, f"{sub}_{ses}_atlas-{ATLAS}_cbf_graphMerged.tsv"
            )
            merged.to_csv(merged_path, sep="\t", index=False)
            merged_paths.append((sub, ses, merged_path))

    return merged_paths


# ---------- GROUP-LEVEL TABLES ----------

def build_group_tables(merged_paths):
    """
    Combines every subject's per-parcel merged table into one long-format
    group table (one row per subject x parcel), then collapses that into a
    global metrics table (one row per subject, averaged across parcels).
    Both are saved to results/stats/graph_cbf/ per the project README.
    """
    os.makedirs(GROUP_OUTPUT_DIR, exist_ok=True)

    all_rows = []
    for sub, ses, path in merged_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t")
        df["sub"] = sub
        df["ses"] = ses
        all_rows.append(df)

    if not all_rows:
        print("[ERROR] No merged TSVs found; nothing to build group tables from.")
        return

    group_df = pd.concat(all_rows, ignore_index=True)

    # Save long table: subject x parcel
    group_merged_path = os.path.join(GROUP_OUTPUT_DIR, OUT_GROUP_MERGED)
    group_df.to_csv(group_merged_path, sep="\t", index=False)
    print(f"[INFO] Wrote parcel-level table: {group_merged_path}")

    # Global (mean across parcels per subject)
    global_df = (
        group_df
        .groupby(["sub", "ses"])[["cbf", "strength", "clustering", "betweenness"]]
        .mean()
        .reset_index()
    )

    global_path = os.path.join(GROUP_OUTPUT_DIR, OUT_GROUP_GLOBAL)
    global_df.to_csv(global_path, sep="\t", index=False)
    print(f"[INFO] Wrote global metrics table: {global_path}")


# ---------- ENTRY POINT ----------

if __name__ == "__main__":
    merged_paths = run_subject_pipeline()
    build_group_tables(merged_paths)
