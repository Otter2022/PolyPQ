import os
import pickle
import shelve
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import re

# ---------- I/O Functions ----------

def read_fvecs(filename):
    """
    Reads a .fvecs file into a numpy array of shape (num_vectors, d).
    Each vector is stored as:
       [d (int32 little-endian), component1 (float32), component2, ..., component_d]
    """
    with open(filename, 'rb') as f:
        data = f.read()
    d = int.from_bytes(data[:4], byteorder='little', signed=True)
    vec_size = 1 + d  # one int header + d float32 values per vector
    total_vecs = len(data) // (4 * vec_size)
    all_data = np.frombuffer(data, dtype='<f4').reshape(total_vecs, vec_size)
    return all_data[:, 1:]  # drop the header column

def load_groundtruth_ivecs(filename):
    """
    Reads a groundtruth file in .ivecs format.
    It is assumed that each row is stored as:
         [k, neighbor1, neighbor2, ..., neighbor_k]
    Returns a dictionary mapping the query index (0-indexed) to a set of true neighbor IDs.
    """
    # We reuse the read_fvecs function since .ivecs files follow a similar structure:
    gt_array = read_fvecs(filename).astype(np.int32)
    groundtruth = {}
    for i, row in enumerate(gt_array):
        groundtruth[i] = set(row)
    return groundtruth

# ---------- Distance and Query Functions ----------

def jaccard_distance(a, b):
    """
    Computes the Jaccard distance between two binary 1D numpy arrays.
    (Distance = 1 - (intersection / union))
    """
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = np.sum(a_bool & b_bool)
    union = np.sum(a_bool | b_bool)
    return 1 - (inter / union) if union != 0 else 1

def query_database(query_vector, codebook, pq_index, grid_size, m, k=10, metric='euclidean'):
    """
    Given a query vector, this function:
      1. Splits the query into m subvectors.
      2. For each subspace, computes the distances between the query subvector and all cluster centers.
      3. Uses these lookup distances plus the PQ code (for each database vector) to compute an approximate distance.
    
    Parameters:
      query_vector: A 1D numpy array representing the query.
      codebook: A list of m arrays (cluster centers) created during PQ encoding.
      pq_index: A dictionary mapping database vector IDs to a tuple of cluster assignments (one per subspace).
      grid_size: The dimensionality of a full vector.
      m: The number of subspaces.
      k: The number of neighbors to return.
      metric: 'euclidean' or 'jaccard' for the distance measure.
    
    Returns:
      A tuple (top_keys, top_distances) where top_keys is a list of the top-k database vector IDs,
      and top_distances their corresponding approximate distances.
    """
    subvector_size = grid_size // m
    # Build a lookup table: For each subspace, compute the distance between the query subvector and each center.
    query_distance_table = []
    for i in range(m):
        subquery = query_vector[i * subvector_size:(i+1) * subvector_size]
        centers = codebook[i]
        if metric == 'jaccard':
            distances = np.array([jaccard_distance(subquery, center) for center in centers])
        else:
            distances = np.linalg.norm(centers - subquery, axis=1)
        query_distance_table.append(distances)
    query_distance_table = np.array(query_distance_table)  # shape: (m, n_clusters)

    # For each database vector (represented by its PQ code), compute approximate distance.
    db_keys = list(pq_index.keys())
    approx_distances = []
    for key in db_keys:
        code = pq_index[key]  # a tuple/list of length m (one cluster assignment per subspace)
        d = 0.0
        for i in range(m):
            d += query_distance_table[i, code[i]]
        approx_distances.append(d)
    approx_distances = np.array(approx_distances)
    sorted_indices = np.argsort(approx_distances)
    top_keys = [db_keys[i] for i in sorted_indices[:k]]
    top_distances = approx_distances[sorted_indices[:k]]
    return top_keys, top_distances

# ---------- Helper Functions for Evaluation ----------

def compute_std_dev(data, mean):
    return np.sqrt(np.mean((np.array(data) - mean) ** 2))

def plot_recall_distribution(recall_values, k):
    cnt = 0
    filename = f"recall_distribution_{k}().png"
    while filename in os.listdir():
        cnt += 1
        filename = f"recall_distribution_{k}({cnt}).png"
    plt.figure(figsize=(8, 6))
    plt.hist(recall_values, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(f"Recall@{k}")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Recall@{k} over Queries")
    plt.savefig(filename, dpi=300)
    print(f"Recall distribution plotted to {filename}")

def evaluate_recall(gt, queries, codebook, pq_index, grid_size, m, k=10):
    """
    Evaluates recall for a set of queries.
    
    gt: Dictionary mapping query index to a set of true neighbor IDs.
    queries: A numpy array of query vectors (one per row, from a .fvecs file).
    codebook, pq_index, grid_size, m: PQ parameters (should match those used in encoding).
    k: The number of neighbors to return per query.
    
    Returns:
      (average_recall, total_query_time, recall_values)
    """
    total_recall = 0.0
    valid_queries = 0
    recall_values = []
    total_query_time = 0.0
    for query_id, query_vector in enumerate(queries):
        start_time = time.perf_counter()
        candidate_ids, _ = query_database(query_vector, codebook, pq_index, grid_size, m, k=k)
        end_time = time.perf_counter()
        total_query_time += (end_time - start_time)
        retrieved_set = set(candidate_ids)
        if query_id in gt:
            gt_neighbors = gt[query_id]
            if len(gt_neighbors) > 0:
                recall = len(gt_neighbors & retrieved_set) / float(len(gt_neighbors))
                recall_values.append(recall)
                total_recall += recall
                valid_queries += 1
    average_recall = total_recall / valid_queries if valid_queries > 0 else 0.0
    print(f"Average Recall@{k}: {average_recall:.3f} over {valid_queries} queries")
    return average_recall, total_query_time, recall_values

# ---------- Main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query PQ-based approximate search using a codebook and PQ index; evaluate recall using groundtruth."
    )
    parser.add_argument("--gt_file", type=str, required=True,
                        help="Path to the groundtruth file in .ivecs format.")
    parser.add_argument("--query_file", type=str, required=True,
                        help="Path to the query vectors file in .fvecs format.")
    parser.add_argument("--codebook_file", type=str, default="codebook.pkl",
                        help="Pickle file containing the codebook.")
    parser.add_argument("--pq_index_file", type=str, default="pq_index.db",
                        help="Shelve file containing the PQ index.")
    parser.add_argument("--grid_size", type=int, default=128,
                        help="Dimensionality of full vectors.")
    parser.add_argument("--m", type=int, required=True,
                        help="Number of subspaces used in PQ.")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest neighbors to return per query.")
    parser.add_argument("--metric", type=str, default="euclidean",
                        help="Distance metric: 'euclidean' (default) or 'jaccard'.")
    args = parser.parse_args()

    # Load groundtruth (assumed to be in .ivecs format)
    gt = load_groundtruth_ivecs(args.gt_file)
    print(f"Loaded groundtruth for {len(gt)} queries from '{args.gt_file}'.")

    # Load query vectors from .fvecs file
    queries = read_fvecs(args.query_file)
    print(f"Loaded {queries.shape[0]} query vectors with dimension {queries.shape[1]} from '{args.query_file}'.")

    # Load the codebook (list of m arrays) from pickle.
    with open(args.codebook_file, "rb") as f:
        codebook = pickle.load(f)
    print(f"Loaded codebook from '{args.codebook_file}'.")

    # Load the PQ index from shelve.
    with shelve.open(args.pq_index_file) as db:
        # Convert keys (stored as strings) back to integers.
        pq_index = {int(key): db[key] for key in db.keys()}
    print(f"Loaded PQ index with {len(pq_index)} entries from '{args.pq_index_file}'.")

    # Evaluate query performance and recall.
    avg_recall, total_query_time, recall_values = evaluate_recall(
        gt, queries, codebook, pq_index, args.grid_size, args.m, k=args.k
    )
    avg_query_time = total_query_time / queries.shape[0]
    print(f"Average query time per query: {avg_query_time:.6f} seconds")
    print(f"Standard deviation of recall: {compute_std_dev(recall_values, avg_recall):.3f}")

    # Plot recall distribution
    plot_recall_distribution(recall_values, args.k)
