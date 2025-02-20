import os
import csv
import shelve
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
import argparse
import time  # For timing queries

# ---------- Helper Functions ----------

def readEncodeFile(filepath):
    """
    Reads a file containing encoded sparse vectors.
    Each line corresponds to a vector (as a space-separated string of indices).
    """
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def reconstruct_dense_vector(vec_str, grid_size):
    """
    Reconstructs a dense binary vector of length grid_size from its sparse string.
    """
    dense_vec = [0] * grid_size
    for idx in vec_str.split():
        dense_vec[int(idx)] = 1
    return np.array(dense_vec)

def load_groundtruth(filepath):
    """
    Loads a ground–truth CSV file where each row is:
      query_id, true_neighbor1, true_neighbor2, ...
    Returns a dictionary mapping query_id (int) to a set of true neighbor IDs.
    """
    groundtruth = {}
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or all(x.strip() == "" for x in row):
                continue
            try:
                query_id = int(row[0])
            except ValueError:
                continue
            neighbors = set()
            for r in row[1:]:
                try:
                    neighbors.add(int(r))
                except ValueError:
                    continue
            groundtruth[query_id] = neighbors
    return groundtruth

def load_all_groundtruth(gt_dir):
    """
    Iterates over all files in the provided directory, loads the groundtruth from each,
    and merges them into a single dictionary.
    """
    all_gt = {}
    for fname in os.listdir(gt_dir):
        fpath = os.path.join(gt_dir, fname)
        gt = load_groundtruth(fpath)
        all_gt.update(gt)
    return all_gt

def jaccard_distance(a, b):
    """
    Computes the Jaccard distance between two binary 1D numpy arrays.
    Distance = 1 - (intersection / union)
    """
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = np.sum(a_bool & b_bool)
    union = np.sum(a_bool | b_bool)
    return 1 - (inter / union) if union != 0 else 1

def query_database(query_vector, codebook, pq_index, grid_size, m, k=10, metric='jaccard'):
    """
    Given a query vector, splits it into m subvectors and builds a lookup table
    of distances between the query subvector and all centers in each subspace.
    Then, for each database vector (represented by its PQ code in pq_index),
    it sums the corresponding lookup distances to compute an approximate distance.
    Returns the top-k database vector indices (keys) and their approximate distances.
    """
    subvector_size = grid_size // m
    query_distance_table = []
    for i in range(m):
        subquery = query_vector[i*subvector_size:(i+1)*subvector_size]
        centers = codebook[i]
        if metric == 'jaccard':
            distances = np.array([jaccard_distance(subquery, center) for center in centers])
        else:
            distances = np.linalg.norm(centers - subquery, axis=1)
        query_distance_table.append(distances)
    query_distance_table = np.array(query_distance_table)  # shape: (m, n_clusters)

    db_keys = list(pq_index.keys())
    approx_distances = []
    for key in db_keys:
        code = pq_index[key]  # a tuple/list of length m
        d = 0.0
        for i in range(m):
            d += query_distance_table[i, code[i]]
        approx_distances.append(d)
    approx_distances = np.array(approx_distances)
    sorted_indices = np.argsort(approx_distances)
    top_keys = [db_keys[i] for i in sorted_indices[:k]]
    top_distances = approx_distances[sorted_indices[:k]]
    return top_keys, top_distances

# ---------- Evaluation Function ----------

def evaluate_recall_from_gt_dir(gt_dir, query_file, codebook, pq_index, grid_size, m, k=10, metric='jaccard', query_offset=0):
    """
    Given a directory of groundtruth files and a query file, this function:
      1. Loads and merges all groundtruth data (mapping query_id to a set of true neighbor IDs).
      2. Reads the query vectors from the query file.
      3. For each query, performs PQ-based approximate search (using the provided codebook and pq_index),
         computes the recall@k, and times how long the query takes.
      4. Prints and returns the average recall over all queries, along with total query time.
    
    It is assumed that the queries in query_file correspond (in order) to the groundtruth query IDs.
    If 'query_offset' is provided, the first query is assigned that ID; otherwise, the offset is taken as min(groundtruth.keys()).
    """
    groundtruth = load_all_groundtruth(gt_dir)
    print(f"Loaded groundtruth for {len(groundtruth)} queries from directory '{gt_dir}'.")
    
    query_strings = readEncodeFile(query_file)
    if not groundtruth:
        print("No groundtruth found!")
        return 0.0, 0, [], 0.0
    
    total_recall = 0.0
    valid_queries = 0
    recall_values = []
    total_query_time = 0.0  # Total time taken by all queries
    for i, q_str in enumerate(query_strings):
        query_vector = reconstruct_dense_vector(q_str, grid_size)
        start_time = time.perf_counter()
        candidate_ids, _ = query_database(query_vector, codebook, pq_index, grid_size, m, k=k, metric=metric)
        end_time = time.perf_counter()
        total_query_time += (end_time - start_time)
        
        retrieved_set = set(candidate_ids)
        query_id = i + query_offset  # Use the provided offset
        if query_id in groundtruth:
            gt_neighbors = groundtruth[query_id]
            if len(gt_neighbors) > 0:
                recall = len(gt_neighbors & retrieved_set) / float(len(gt_neighbors))
                recall_values.append(recall)
                total_recall += recall
                valid_queries += 1
    average_recall = total_recall / valid_queries if valid_queries > 0 else 0.0
    print(f"\nAverage Recall@{k}: {average_recall:.3f} over {valid_queries} queries")
    return total_recall, valid_queries, recall_values, total_query_time

def compute_std_dev(data, mean):
    """
    Given a list of data points and their mean, computes the standard deviation.
    """
    return np.sqrt(np.mean([(x - mean)**2 for x in data]))

def check_file_in_dir(directory, filename):
    """
    Checks if a file with the given name exists in the provided directory.
    """
    return os.path.isfile(os.path.join(directory, filename))

def plot_recall_distribution(recall_values, k):
    """
    Plots a histogram of the recall values.
    """
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

# ---------- Example Usage ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate recall for PQ-based approximate search using a given codebook and PQ index."
    )
    parser.add_argument("--k", type=int, default=250, help="The number of top results to return per query.")
    parser.add_argument("--db_file", type=str, default="pq_index.db", help="Path to the shelve db file for the PQ index.")
    parser.add_argument("--codebook_file", type=str, default="codebook.pkl", help="Path to the pickle file containing the codebook.")
    args = parser.parse_args()

    # Parameters
    grid_size = 18382   # Dimensionality of the full vectors
    m = 14              # Number of subspaces (as used during clustering)
    k = args.k        # Number of neighbors to return (from command line)
    gt_dir = "../data/pk-query-50k"  # Directory containing groundtruth files (CSV format)
    query_dir = "../data/tmp/shared/encoding/pk-50k0.002"  # Directory containing query vector files

    # Load the PQ codebook (list of m arrays) from file.
    with open(args.codebook_file, "rb") as f:
        codebook = pickle.load(f)
    print(f"Codebook loaded from '{args.codebook_file}'.")

    # Open the PQ index (persistent key–value store) and load it into a dictionary.
    with shelve.open(args.db_file) as db:
        # Keys in shelve are stored as strings; convert them back to integers.
        pq_index = {int(key): db[key] for key in db.keys()}
    print(f"PQ index loaded from '{args.db_file}'.")

    count = 0
    recall_sum = 0
    recall_values = []
    total_query_time_sum = 0.0

    for i in os.listdir(query_dir):
        # Extract starting index from the filename (assumes a number is present)
        starting_index = re.findall(r'\d+', i)[0]
        if int(starting_index) >= 40000:
            query_file = os.path.join(query_dir, i)
            rec_total, valid_queries, _recall_values, query_time = evaluate_recall_from_gt_dir(
                gt_dir, query_file, codebook, pq_index, grid_size, m, k=k, metric='jaccard', query_offset=int(starting_index)
            )
            count += valid_queries
            recall_sum += rec_total
            recall_values.extend(_recall_values)
            total_query_time_sum += query_time
    
    if count > 0:
        avg_recall = recall_sum / count
        avg_query_time = total_query_time_sum / count
        print(f"\nAverage Recall@{k} over {count} queries: {avg_recall:.3f}, std dev: {compute_std_dev(recall_values, avg_recall):.3f}")
        print(f"Average Query Time: {avg_query_time:.6f} seconds per query")
    else:
        print("No valid queries evaluated.")
    plot_recall_distribution(recall_values, k)
