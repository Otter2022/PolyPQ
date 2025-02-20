import os
import re
import pickle
import shelve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ------------- Helper Functions ------------- #

def sortFilesByIdData(files):
    """
    Sorts files based on the numeric part of their names.
    For example, a filename like 'weightint_47000.txt' will be sorted by 47000.
    """
    def extract_id(filename):
        base = os.path.splitext(filename)[0]
        try:
            return int(base.split('_')[1])
        except (IndexError, ValueError):
            return 0
    return sorted(files, key=extract_id)

def readEncodeFile(filepath):
    """
    Reads a file containing encoded sparse vectors.
    Each line corresponds to a vector represented as a string.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def readAllSparseStr(path, exclusion_regex=None):
    """
    Reads all sparse vector strings from files in the given folder.
    """
    files = os.listdir(path)
    files = sortFilesByIdData(files)
    
    if exclusion_regex:
        files = [f for f in files if not re.match(exclusion_regex, f)]
    
    vector_str = []
    for file in files:
        full_path = os.path.join(path, file)
        vector_str.extend(readEncodeFile(full_path))
    return vector_str

def reconstruct_dense_vector(vec_str, grid_size):
    """
    Reconstructs a dense binary vector from its sparse representation.
    The sparse representation is assumed to be a string of space-separated indices
    (as strings) where the value is 1. All other positions (0 ... grid_size-1) are 0.
    """
    dense_vec = [0] * grid_size
    if vec_str:
        for idx in vec_str.split():
            dense_vec[int(idx)] = 1
    return dense_vec

def split_into_subvectors(dense_vec, subvector_size):
    """
    Splits a dense vector into subvectors of the given size.
    """
    return [dense_vec[i:i+subvector_size] for i in range(0, len(dense_vec), subvector_size)]

def make_subvector_groups(sparse_vecs, m, d):
    """
    Splits each sparse vector into subvectors and groups them by subspace.
    Returns a list of m groups, where each group is a list of subvectors.
    The order of vectors is preserved.
    """
    dense_vecs = []
    for vec_str in sparse_vecs:
        dense_vec = reconstruct_dense_vector(vec_str, d)
        dense_vecs.append(dense_vec)

    subvector_size = len(dense_vecs[0]) // m
    if len(dense_vecs[0]) % m != 0:
        raise ValueError("The dense vector length is not divisible by the number of subvectors (m).")

    # Create m empty groups
    subvector_groups = [[] for _ in range(m)]
    for vec in dense_vecs:
        subvectors = split_into_subvectors(vec, subvector_size)
        for j in range(m):
            subvector_groups[j].append(subvectors[j])
    return subvector_groups

def jaccard_distance(a, b):
    """
    Computes the Jaccard distance between two binary 1D arrays.
    Distance = 1 - (intersection / union)
    """
    a_bool = np.array(a, dtype=bool)
    b_bool = np.array(b, dtype=bool)
    inter = np.sum(a_bool & b_bool)
    union = np.sum(a_bool | b_bool)
    return 1 - (inter / union) if union != 0 else 1

def compute_medoid(cluster_points):
    """
    Computes the medoid of a set of points based on Jaccard distance.
    Returns the point (from cluster_points) with minimal total distance to others.
    """
    if len(cluster_points) == 0:
        return None
    cluster_points = np.array(cluster_points)
    best_index = 0
    best_sum = float('inf')
    for i in range(len(cluster_points)):
        candidate = cluster_points[i]
        s = 0.0
        for j in range(len(cluster_points)):
            if i == j:
                continue
            s += jaccard_distance(candidate, cluster_points[j])
        if s < best_sum:
            best_sum = s
            best_index = i
    return cluster_points[best_index]

# ------------- Main Script ------------- #

if __name__ == "__main__":
    # --- Configuration ---
    data_path = "../data/tmp/shared/encoding/pk-50k0.002/"
    d = 18382            # Dimensionality of the full vector
    m = 14               # Number of subvectors/subspaces
    # DBSCAN parameters (tweak these as needed)
    eps = 0.3
    min_samples = 5

    # --- Read and process vectors ---
    all_vector_str = readAllSparseStr(data_path, exclusion_regex=r'^.*_(?:[4-9]\d{4}|\d{6,}).*$')
    print(f"Read {len(all_vector_str)} sparse vectors.")

    # Group subvectors by subspace (order is preserved)
    subvectors = make_subvector_groups(all_vector_str, m, d)
    print(f"Split vectors into {m} subvectors of size {d//m}.")

    # --- Build the codebook using DBSCAN ---
    codebook = []       # List to hold the codebook for each subspace
    all_labels = []     # List to hold the cluster labels (assignments) for each subspace

    for group_index, subvector_group in enumerate(subvectors):
        group_data = np.array(subvector_group)  # shape: (num_vectors, subvector_size)
        print(f"\nProcessing subvector group {group_index} with shape {group_data.shape} ...")
        
        # Cluster using DBSCAN with Jaccard distance
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='jaccard')
        labels = dbscan.fit_predict(group_data)
        
        # DBSCAN labels noise as -1. First, get unique non-noise labels.
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        unique_labels = sorted(unique_labels)
        centers = {}
        
        # Compute medoid (cluster center) for each cluster
        for label in unique_labels:
            cluster_points = group_data[labels == label]
            medoid = compute_medoid(cluster_points)
            centers[label] = medoid
        
        # If no clusters were found, assign all points to a single cluster.
        if len(unique_labels) == 0:
            print(f"Group {group_index}: No clusters found, assigning all points to cluster 0.")
            labels = np.zeros(len(group_data), dtype=int)
            centers[0] = group_data[0]  # Arbitrary choice.
            unique_labels = [0]
        else:
            # For noise points (label -1), assign to the nearest cluster center.
            for i in range(len(labels)):
                if labels[i] == -1:
                    distances = [jaccard_distance(group_data[i], centers[ul]) for ul in unique_labels]
                    nearest = unique_labels[np.argmin(distances)]
                    labels[i] = nearest

        # Remap labels to consecutive integers (0, 1, 2, ...)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        remapped_labels = np.array([label_mapping[label] for label in labels])
        all_labels.append(remapped_labels)
        
        # Build codebook for this subspace: centers in remapped order.
        codebook_centers = [centers[old_label] for old_label in unique_labels]
        codebook.append(np.array(codebook_centers))
        
        print(f"Group {group_index}: Found clusters: {np.unique(remapped_labels)}")

    # Save the codebook (to be used later for querying)
    with open("DBscan_codebook.pkl", "wb") as f:
        pickle.dump(codebook, f)
    print("\nCodebook saved to 'codebook.pkl'.")

    # --- Build and save the PQ index ---
    num_vectors = len(all_vector_str)
    pq_index = {}
    for i in range(num_vectors):
        # The PQ code is a tuple: (label from subspace 0, label from subspace 1, ..., label from subspace m-1)
        code = tuple(all_labels[j][i] for j in range(m))
        pq_index[i] = code

    with shelve.open('DBScan_pq_index.db') as db:
        for key, value in pq_index.items():
            db[str(key)] = value
    print("PQ index saved to 'pq_index.db'.")


