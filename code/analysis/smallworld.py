import networkx as nx
import numpy as np

schiz_matrices = list(np.load("/data/raulab/asl2/schiz_matrices.npy"))
control_matrices = list(np.load("/data/raulab/asl2/control_matrices.npy"))

schiz_mean = np.mean(schiz_matrices, axis=0)
control_mean = np.mean(control_matrices, axis=0)

def threshold_matrix(matrix, density=0.20):
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    upper_tri = mat[np.triu_indices_from(mat, k=1)]
    threshold = np.percentile(upper_tri, 100 * (1 - density))
    return (mat >= threshold).astype(float), threshold

def compute_smallworld(G, n_random=100):
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes)
    real_clustering = nx.average_clustering(G_lcc)
    real_path = nx.average_shortest_path_length(G_lcc)
    n_nodes = G_lcc.number_of_nodes()
    n_edges = G_lcc.number_of_edges()
    rand_clusterings, rand_paths = [], []
    print(f"Generating {n_random} random graphs...")
    for i in range(n_random):
        G_rand = nx.gnm_random_graph(n_nodes, n_edges)
        if not nx.is_connected(G_rand):
            lcc_rand = max(nx.connected_components(G_rand), key=len)
            G_rand = G_rand.subgraph(lcc_rand)
        rand_clusterings.append(nx.average_clustering(G_rand))
        rand_paths.append(nx.average_shortest_path_length(G_rand))
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{n_random} done...")
    gamma = real_clustering / np.mean(rand_clusterings)
    lambda_ = real_path / np.mean(rand_paths)
    sigma = gamma / lambda_
    print(f"Real clustering: {real_clustering:.4f}, Random: {np.mean(rand_clusterings):.4f}")
    print(f"Real path length: {real_path:.4f}, Random: {np.mean(rand_paths):.4f}")
    print(f"Gamma: {gamma:.4f}, Lambda: {lambda_:.4f}, Sigma: {sigma:.4f}")
    print(f"Small-world? {'YES' if sigma > 1 else 'NO'}")
    return sigma, gamma, lambda_

s_bin, _ = threshold_matrix(schiz_mean, density=0.20)
c_bin, _ = threshold_matrix(control_mean, density=0.20)

print("=== SCHIZOPHRENIA ===")
s_sigma, s_gamma, s_lambda = compute_smallworld(nx.from_numpy_array(s_bin))

print("\n=== CONTROLS ===")
c_sigma, c_gamma, c_lambda = compute_smallworld(nx.from_numpy_array(c_bin))

print(f"\n=== FINAL COMPARISON ===")
print(f"Sigma:  Schiz={s_sigma:.4f}, Control={c_sigma:.4f}")
print(f"Gamma:  Schiz={s_gamma:.4f}, Control={c_gamma:.4f}")
print(f"Lambda: Schiz={s_lambda:.4f}, Control={c_lambda:.4f}")
