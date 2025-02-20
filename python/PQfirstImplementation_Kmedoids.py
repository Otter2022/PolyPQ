
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
import pickle  # For saving the codebook
import shelve  # For creating a persistent key-value store
import re

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

if __name__ == "__main__":
    # --- Configuration ---
    data_path = "../data/tmp/shared/encoding/pk-50k0.002/"
    d = 18382            # Dimensionality of the full vector
    m = 14               # Number of subvectors/subspaces
    num_clusters = 256   # Number of clusters per subspace

    # --- Read and process vectors ---
    all_vector_str = readAllSparseStr(data_path, exclusion_regex=r'^.*_(?:[4-9]\d{4}|\d{6,}).*$')
    print(f"Read {len(all_vector_str)} sparse vectors.")

    # (Optional) If you need the full dense vectors later, you can reconstruct them here:
    # dense_vectors = [reconstruct_dense_vector(s, d) for s in all_vector_str]

    # Group subvectors by subspace (order is preserved)
    subvectors = make_subvector_groups(all_vector_str, m, d)
    print(f"Split vectors into {m} subvectors of size {d//m}.")

    # --- Build the codebook ---
    # For each subspace, perform clustering to get the codebook (cluster centers)
    codebook = []       # List to hold the codebook for each subspace
    all_labels = []     # List to hold the cluster labels (assignments) for each subspace

    for group_index, subvector_group in enumerate(subvectors):
        group_data = np.array(subvector_group)  # shape: (num_vectors, subvector_size)
        print(f"Processing subvector group {group_index} with shape {group_data.shape} ...")
        
        # Adjust the number of clusters if necessary
        current_clusters = num_clusters if group_data.shape[0] >= num_clusters else group_data.shape[0]
        if current_clusters != num_clusters:
            print(f"Group {group_index}: Reducing number of clusters to {current_clusters}")

        # Cluster using KMedoids with Jaccard distance
        kmedoids = KMedoids(n_clusters=current_clusters, metric='jaccard', init='k-medoids++', random_state=42)
        labels = kmedoids.fit_predict(group_data)
        all_labels.append(labels)  # Save the labels for this subspace

        centers = kmedoids.cluster_centers_
        print(f"Group {group_index}: Found clusters: {np.unique(labels)}")
        codebook.append(centers)

    # Save the codebook (this can later be used for query processing)
    with open("Kmedoids_codebook.pkl", "wb") as f:
        pickle.dump(codebook, f)
    print("Codebook saved to 'codebook.pkl'.")

    # --- Build and save the PQ index ---
    # For each original vector (preserved in order), combine the cluster assignments from all m groups.
    num_vectors = len(all_vector_str)
    pq_index = {}
    for i in range(num_vectors):
        # The PQ code is a tuple: (label from subspace 0, label from subspace 1, ..., label from subspace m-1)
        code = tuple(all_labels[j][i] for j in range(m))
        pq_index[i] = code

    # Save the PQ index in a binary key–value store using shelve.
    # This creates a persistent dictionary-like database file.
    with shelve.open('Kmedoids_pq_index.db') as db:
        for key, value in pq_index.items():
            db[str(key)] = value
    print("PQ index saved to 'pq_index.db'.")
