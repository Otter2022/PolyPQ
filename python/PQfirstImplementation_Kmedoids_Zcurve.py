import os
import numpy as np
import pickle  # For saving the codebook
import shelve  # For creating a persistent key–value store
import re
import random
import argparse
from sklearn.cluster import KMeans
from numba import njit
from scipy.spatial.distance import jaccard
import dbm.dumb
import sys
sys.modules['dbm'] = dbm.dumb

# Import KMedoids from scikit-learn-extra
from sklearn_extra.cluster import KMedoids

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
    indices.sort(key=lambda x: x[0])
    return [idx for (_, idx) in indices]

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
        print(files)
    vector_str = []
    for file in files:
        full_path = os.path.join(path, file)
        vector_str.extend(readEncodeFile(full_path))
    return vector_str

def reconstruct_dense_vector(vec_str, grid_size):
    """
    Reconstructs a dense binary vector from its sparse representation.
    The sparse representation is assumed to be a string of space-separated indices
    where the value is 1. All other positions (0 ... grid_size-1) are 0.
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
    First, each sparse vector is reconstructed into a dense vector.
    If d is a perfect square, the dense vector is reordered using Z-curve (Morton order)
    before being split into m subvectors.
    Returns a list of m groups, where each group is a list of subvectors.
    The order of vectors is preserved.
    """
    dense_vecs = []
    for vec_str in sparse_vecs:
        dense_vec = reconstruct_dense_vector(vec_str, d)
        dense_vecs.append(dense_vec)
    
    # If d is a perfect square, reorder each dense vector using Z-curve.
    side = int(np.sqrt(d))
    if side * side == d:
        z_indices = get_zorder_indices(side, side)
        dense_vecs = [[vec[i] for i in z_indices] for vec in dense_vecs]
    else:
        print("Warning: grid size is not a perfect square, Z-curve ordering skipped.")
    
    subvector_size = len(dense_vecs[0]) // m
    if len(dense_vecs[0]) % m != 0:
        raise ValueError("The dense vector length is not divisible by the number of subvectors (m).")
    subvector_groups = [[] for _ in range(m)]
    for vec in dense_vecs:
        subvectors = split_into_subvectors(vec, subvector_size)
        for j in range(m):
            subvector_groups[j].append(subvectors[j])
    return subvector_groups

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clustering using scikit-learn-extra's KMedoids with a weighted Jaccard metric"
    )
    parser.add_argument("--data_path", type=str, default="../../../uni_filtered_pk-5e-06-5e-05-147-147",
                        help="Path to the folder containing the encoded files.")
    parser.add_argument("--d", type=int, default=21609,
                        help="Dimensionality of the full vector.")
    parser.add_argument("--m", type=int, required=True,
                        help="Number of subvectors/subspaces.")
    parser.add_argument("--num_clusters", type=int, default=1024,
                        help="Number of clusters per subspace.")
    parser.add_argument("--codebook_out", type=str, default="Kmedoids_codebook.pkl",
                        help="Output file for the codebook (pickle file).")
    parser.add_argument("--pq_index_out", type=str, default="Kmedoids_pq_index.db",
                        help="Output file for the PQ index (shelve database).")
    args = parser.parse_args()

    data_path = args.data_path
    d = args.d
    m = args.m
    num_clusters = args.num_clusters

    exclusion_regex = r'^.*_(?:[1-9][0-9]\d{3}).*$'
    
    all_vector_str = readAllSparseStr(data_path, exclusion_regex=exclusion_regex)
    print(f"Read {len(all_vector_str)} sparse vectors.")

    # Group subvectors by subspace (with Z-curve reordering if possible).
    subvectors = make_subvector_groups(all_vector_str, m, d)
    print(f"Split vectors into {m} subvectors of size {d//m}.")

    # Build the codebook and PQ index using k-medoids clustering.
    codebook = []       # Will store one codebook per subspace (using candidate centers)
    all_labels = []     # Holds the cluster assignments for each subspace

    for group_index, subvector_group in enumerate(subvectors):
        # Convert each subvector to a list of booleans.
        group_points = [list(map(bool, vec)) for vec in subvector_group]
        num_points = len(group_points)
        
        # Run k-medoids clustering using scikit-learn-extra.
        kmedoids = KMedoids(
            n_clusters=num_clusters, 
            metric='jaccard', 
            init='k-medoids++',
            random_state=0
        )
        kmedoids.fit(group_points)
        labels = kmedoids.labels_.tolist()
        all_labels.append(labels)
        print("Created labels for group", group_index)
        
        # Use the cluster centers as the codebook for this subspace.
        codebook_group = kmedoids.cluster_centers_
        codebook.append(np.array(codebook_group))
        print(f"Group {group_index}: Codebook (candidate centers) has shape {codebook_group.shape}")
    
    # Save the codebook for later use.
    with open(args.codebook_out, "wb") as f:
        pickle.dump(codebook, f)
    print(f"Codebook saved to '{args.codebook_out}'.")

    # Build and save the PQ index.
    num_vectors = len(all_vector_str)
    pq_index = {}
    for i in range(num_vectors):
        code = tuple(all_labels[j][i] for j in range(m))
        pq_index[i] = code

    with shelve.open(args.pq_index_out) as db:
        for key, value in pq_index.items():
            db[str(key)] = value
    print(f"PQ index saved to '{args.pq_index_out}'.")
