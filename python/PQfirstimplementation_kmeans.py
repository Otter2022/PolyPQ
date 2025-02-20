import os
import numpy as np
import matplotlib.pyplot as plt
import pickle  # For saving the codebook
import shelve  # For creating a persistent key–value store
import re
import random
import PolyPQ  # Our C-based k-means module
from joblib import Parallel, delayed  # For parallel processing

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

# --- Helper functions for medoid computation using Jaccard distance ---

def jaccard_distance(a, b):
    """
    Computes the Jaccard distance between two binary vectors.
    """
    a = np.array(a)
    b = np.array(b)
    inter = np.sum((a != 0) & (b != 0))
    union = np.sum((a != 0) | (b != 0))
    return 1 - (inter / union) if union != 0 else 1

def compute_medoid(points):
    """
    Computes the medoid of a list of points (each a list of numbers) based on Jaccard distance.
    Returns the point with the minimal total distance to all other points.
    """
    if not points:
        return None
    best_point = points[0]
    best_total = float('inf')
    for p in points:
        total = sum(jaccard_distance(p, q) for q in points)
        if total < best_total:
            best_total = total
            best_point = p
    return best_point

def compute_cluster_medoid(c, group_points, labels, initial_center):
    """
    Computes the medoid for cluster 'c' from group_points given the cluster assignments in labels.
    If no points are assigned to the cluster, returns the initial_center.
    """
    cluster_points = [pt for pt, lab in zip(group_points, labels) if lab == c]
    if cluster_points:
        return compute_medoid(cluster_points)
    else:
        return initial_center

if __name__ == "__main__":
    # --- Configuration ---
    data_path = "../data/tmp/shared/encoding/pk-50k0.002/"
    d = 18382            # Dimensionality of the full vector
    m = 14               # Number of subvectors/subspaces
    num_clusters = 256   # Number of clusters per subspace

    # Exclude files with numbers >= 40000 in their name
    exclusion_regex = r'^.*_(?:[4-9]\d{4}|\d{6,}).*$'
    
    # --- Read and process vectors ---
    all_vector_str = readAllSparseStr(data_path, exclusion_regex=exclusion_regex)
    print(f"Read {len(all_vector_str)} sparse vectors.")

    # Group subvectors by subspace (order is preserved)
    subvectors = make_subvector_groups(all_vector_str, m, d)
    print(f"Split vectors into {m} subvectors of size {d//m}.")

    # --- Build the codebook using PolyPQ.kmeans ---
    # For each subspace, we will cluster the subvectors using our k-means implementation.
    codebook = []       # List to hold the codebook for each subspace (medoids)
    all_labels = []     # List to hold the cluster labels (assignments) for each subspace

    for group_index, subvector_group in enumerate(subvectors):
        # Convert each subvector to a list of floats
        group_points = [list(map(float, vec)) for vec in subvector_group]
        num_points = len(group_points)
        
        # Adjust the number of clusters if needed
        current_clusters = num_clusters if num_points >= num_clusters else num_points
        if current_clusters != num_clusters:
            print(f"Group {group_index}: Reducing number of clusters to {current_clusters}")

        # Randomly select initial centers from the group points
        initial_centers = random.sample(group_points, current_clusters)

        # Run k-means using the PolyPQ module with Jaccard distance
        labels = PolyPQ.kmeans(group_points, initial_centers, max_iterations=999999, metric="jaccard")
        all_labels.append(labels)

        print("Created labels for group", group_index)
        
        # **** remove because slow ****
        # --- Parallelized medoid computation for each cluster ---
        # centers = Parallel(n_jobs=16)(
        #     delayed(compute_cluster_medoid)(c, group_points, labels, initial_centers[c])
        #     for c in range(current_clusters)
        # )

        centers = initial_centers
        
        print(f"Group {group_index}: Found clusters: {np.unique(labels)}")
        codebook.append(np.array(centers))

    # Save the codebook (this can later be used for query processing)
    with open("Kmeans_codebook.pkl", "wb") as f:
        pickle.dump(codebook, f)
    print("Codebook saved to 'Kmeans_codebook.pkl'.")

    # --- Build and save the PQ index ---
    # For each original vector (order preserved), combine the cluster assignments from all m subspaces.
    num_vectors = len(all_vector_str)
    pq_index = {}
    for i in range(num_vectors):
        # The PQ code is a tuple: (label from subspace 0, label from subspace 1, ..., label from subspace m-1)
        code = tuple(all_labels[j][i] for j in range(m))
        pq_index[i] = code

    # Save the PQ index in a persistent key–value store using shelve.
    with shelve.open('Kmeans_pq_index.db') as db:
        for key, value in pq_index.items():
            db[str(key)] = value
    print("PQ index saved to 'Kmeans_pq_index.db'.")
