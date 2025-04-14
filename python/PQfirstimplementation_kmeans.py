import os
import numpy as np
import pickle  # For saving the codebook
import shelve  # For creating a persistent key–value store
import argparse
from sklearn.cluster import KMeans
import dbm.dumb
import sys
sys.modules['dbm'] = dbm.dumb


def read_fvecs(filename):
    """
    Reads a .fvecs file into a numpy array of shape (num_vectors, dimension).
    The file format:
      - Each vector is stored as: [d, component_1, component_2, ..., component_d]
      - The first 4 bytes (an int32 in little-endian) specify the vector dimension d.
      - The next d*4 bytes are float32 values (little-endian) for the vector components.
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

def split_into_subvectors(vectors, m):
    """
    Splits a 2D numpy array of vectors (shape: num_vectors x d) into m subspaces.
    Each subvector will have size d_sub = d // m.
    
    Returns:
        List of m arrays, each of shape (num_vectors, d_sub).
    """
    num_vectors, d = vectors.shape
    if d % m != 0:
        raise ValueError("Vector dimension must be divisible by m.")
    subvector_size = d // m
    subvectors = []
    for i in range(m):
        subvectors.append(vectors[:, i * subvector_size : (i+1) * subvector_size])
    return subvectors

def main():
    parser = argparse.ArgumentParser(
        description="Product Quantization using .fvecs vectors and KMeans clustering"
    )
    parser.add_argument("--data_path", type=str, default="../data/tmp/shared/encoding/pk-50k0.002/",
                        help="Path to the folder containing the encoded files.")
    parser.add_argument("--d", type=int, default=18382,
                        help="Dimensionality of the full vector.")
    parser.add_argument("--m", type=int, required=True,
                        help="Number of subvectors/subspaces.")
    parser.add_argument("--num_clusters", type=int, default=512,
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
    exclusion_regex = r'^.*_(?:[4-9]\d{4}|\d{6,}).*$'
    
    # --- Read and process vectors ---
    all_vector_str = readAllSparseStr(data_path, exclusion_regex=exclusion_regex)
    print(f"Read {len(all_vector_str)} sparse vectors.")

    # --- Split vectors into m subspaces ---
    subvectors = split_into_subvectors(all_vectors, args.m)
    subvector_size = args.d // args.m
    print(f"Split vectors into {args.m} subspaces, each of size {subvector_size}.")

    # --- Build the codebook using sklearn KMeans ---
    codebook = []       # To store the cluster centers for each subspace
    all_labels = []     # To store the cluster assignments for each subspace

    for group_index, subvector_group in enumerate(subvectors):
        # Convert the subspace data to float32 (if not already)
        dense_points = subvector_group.astype(np.float32)
        num_points = dense_points.shape[0]
        
        # Adjust the number of clusters if there are fewer points than desired
        current_clusters = args.num_clusters if num_points >= args.num_clusters else num_points
        if current_clusters != args.num_clusters:
            print(f"Subspace {group_index}: Reducing number of clusters to {current_clusters}")
        
        # Use unique points as initial centers if possible.
        unique_points = np.unique(dense_points, axis=0)
        print(f"Subspace {group_index}: Found {len(unique_points)} unique points.")
        if len(unique_points) < args.num_clusters:
            print(f"Subspace {group_index}: Using all {len(unique_points)} unique points as initial centers.")
            initial_centers = unique_points
        else:
            # Otherwise, take the first args.num_clusters unique points.
            initial_centers = unique_points[:args.num_clusters]
        
        n_clusters = initial_centers.shape[0]
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, max_iter=500, n_init=1, random_state=42)
        labels = kmeans.fit_predict(dense_points)
        all_labels.append(labels)
        centers = kmeans.cluster_centers_
        print(f"Subspace {group_index}: Cluster labels: {np.unique(labels)}")
        codebook.append(centers)
    
    # --- Save the codebook ---
    with open(args.codebook_out, "wb") as f:
        pickle.dump(codebook, f)
    print(f"Codebook saved to '{args.codebook_out}'.")

    # --- Build and save the PQ index ---
    # For each original vector (order preserved), combine the cluster assignments from all m subspaces.
    pq_index = {}
    for i in range(num_vectors):
        # The PQ code is a tuple: (label from subspace 0, label from subspace 1, ..., label from subspace m-1)
        code = tuple(all_labels[j][i] for j in range(args.m))
        pq_index[i] = code

    with shelve.open(args.pq_index_out) as db:
        for key, value in pq_index.items():
            db[str(key)] = value
    print(f"PQ index saved to '{args.pq_index_out}'.")

if __name__ == "__main__":
    main()
