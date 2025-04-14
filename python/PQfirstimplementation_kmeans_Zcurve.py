import os
import numpy as np
import matplotlib.pyplot as plt
import pickle  # For saving the codebook
import shelve  # For creating a persistent key–value store
import re
import random
import argparse
from sklearn.cluster import KMeans

import dbm.dumb
import sys
sys.modules['dbm'] = dbm.dumb

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

    for ind, line in enumerate(lines):
        if ',' in line:
            lines[ind] = line.replace(',', '')
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

# --- Z-curve (Morton order) helper functions ---
def interleave_bits(x, y):
    """
    Interleaves the bits of x and y.
    Returns the Morton code for the coordinate (x, y).
    """
    z = 0
    max_bits = max(x.bit_length(), y.bit_length())
    for i in range(max_bits):
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z

def get_zorder_indices(rows, cols):
    """
    Returns a list of indices (0 ... rows*cols - 1) corresponding to the Z-curve (Morton order)
    of a grid with the given dimensions.
    """
    indices = []
    for r in range(rows):
        for c in range(cols):
            morton = interleave_bits(r, c)
            index = r * cols + c
            indices.append((morton, index))
    # Sort by the computed Morton code
    indices.sort(key=lambda x: x[0])
    return [idx for (_, idx) in indices]

def make_subvector_groups(sparse_vecs, m, d):
    """
    Splits each sparse vector into subvectors and groups them by subspace.
    First, each sparse vector is reconstructed into a dense vector.
    If possible, the vector is then reordered using Z-curve (Morton order) assuming the vector
    represents a square grid. Finally, the reordered vector is split into m subvectors.
    
    Returns a list of m groups, where each group is a list of subvectors.
    The order of vectors is preserved.
    """
    dense_vecs = []
    for vec_str in sparse_vecs:
        dense_vec = reconstruct_dense_vector(vec_str, d)
        dense_vecs.append(dense_vec)

    # If d is a perfect square, reorder each dense vector using Z-curve order.
    side = int(np.sqrt(d))
    if side * side == d:
        z_indices = get_zorder_indices(side, side)
        # Reorder each dense vector based on z_indices.
        dense_vecs = [[vec[i] for i in z_indices] for vec in dense_vecs]
    else:
        print("Warning: grid size is not a perfect square, Z-curve ordering skipped.")

    subvector_size = d // m
    if d % m != 0:
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
    
def jaccard_distance_dense(a, b):
    """
    Computes the Jaccard distance for dense binary vectors.
    """
    intersection = np.sum(np.logical_and(a, b))
    union = np.sum(np.logical_or(a, b))
    if union == 0:
        return 0.0
    similarity = intersection / union
    return 1.0 - similarity

if __name__ == "__main__":
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="Clustering using sklearn KMeans with unique vector initialization"
    )
    parser.add_argument("--data_path", type=str, default="../../../uni_filtered_pk-5e-06-5e-05-147-147",
                        help="Path to the folder containing the encoded files.")
    parser.add_argument("--d", type=int, default=21609,
                        help="Dimensionality of the full vector.")
    parser.add_argument("--m", type=int, required=True,
                        help="Number of subvectors/subspaces.")
    parser.add_argument("--num_clusters", type=int, default=1024,
                        help="Number of clusters per subspace.")
    parser.add_argument("--codebook_out", type=str, default="Kmeans_codebook.pkl",
                        help="Output file for the codebook (pickle file).")
    parser.add_argument("--pq_index_out", type=str, default="Kmeans_pq_index.db",
                        help="Output file for the PQ index (shelve database).")
    args = parser.parse_args()

    data_path = args.data_path
    d = args.d
    m = args.m
    num_clusters = args.num_clusters

    # Exclude files with numbers >= 40000 in their name
    exclusion_regex = r'^.*_(?:[1-9][0-9]\d{3}).*$'
    
    # --- Read and process vectors ---
    all_vector_str = readAllSparseStr(data_path, exclusion_regex=exclusion_regex)
    print(f"Read {len(all_vector_str)} sparse vectors.")

    # Group subvectors by subspace (order is preserved)
    subvectors = make_subvector_groups(all_vector_str, m, d)
    print(f"Split vectors into {m} subvectors of size {d//m}.")
    print("Example subvector:", subvectors[0][0])

    # --- Build the codebook using sklearn KMeans ---
    codebook = []       # List to hold the codebook for each subspace (cluster centers)
    all_labels = []     # List to hold the cluster labels (assignments) for each subspace

    for group_index, subvector_group in enumerate(subvectors):
        # Convert each subvector to a list of floats
        group_points = [list(map(float, vec)) for vec in subvector_group]
        num_points = len(group_points)
        
        # Adjust the number of clusters if needed
        current_clusters = num_clusters if num_points >= num_clusters else num_points
        if current_clusters != num_clusters:
            print(f"Group {group_index}: Reducing number of clusters to {current_clusters}")
        
        # Convert to a NumPy array for easier processing
        dense_points = np.array(group_points)
        
        # Use unique points as initial centers.
        unique_points = np.unique(dense_points, axis=0)
        print(f"Group {group_index}: Found {len(unique_points)} unique points.")
        
        # If there are fewer unique points than desired, use them all.
        if len(unique_points) < num_clusters:
            print(f"Group {group_index}: Using all {len(unique_points)} unique points as initial centers.")
            initial_centers = unique_points
        else:
            # Otherwise, take the first 'num_clusters' unique points.
            initial_centers = unique_points[:num_clusters]
        
        # Use the number of unique initial centers as the number of clusters.
        n_clusters = initial_centers.shape[0]
        # Run sklearn's KMeans clustering.
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, max_iter=500, n_init=1, random_state=42)
        labels = kmeans.fit_predict(dense_points)
        all_labels.append(labels)
        centers = kmeans.cluster_centers_
        print("Created labels for group", group_index)
        print(f"Group {group_index}: Found clusters: {np.unique(labels)}")
        codebook.append(centers)
    
    # Save the codebook (this can later be used for query processing)
    with open(args.codebook_out, "wb") as f:
        pickle.dump(codebook, f)
    print(f"Codebook saved to '{args.codebook_out}'.")

    # --- (Optional) Build and save the PQ index ---
    # For each original vector (order is preserved), combine the cluster assignments from all m subspaces.
    num_vectors = len(all_vector_str)
    pq_index = {}
    for i in range(num_vectors):
        # The PQ code is a tuple: (label from subspace 0, label from subspace 1, ..., label from subspace m-1)
        code = tuple(all_labels[j][i] for j in range(m))
        pq_index[i] = code

    with shelve.open(args.pq_index_out) as db:
        for key, value in pq_index.items():
            db[str(key)] = value
    print(f"PQ index saved to '{args.pq_index_out}'.")
