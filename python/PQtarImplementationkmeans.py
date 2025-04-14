import os
import numpy as np
import pickle  # For saving the codebook
import shelve  # For creating a persistent key–value store
import argparse
from sklearn.cluster import KMeans

def read_fvecs(filename):
    """
    Reads a .fvecs file into a numpy array of shape (num_vectors, dimension).
    The file format:
      - Each vector is stored as: [d, component_1, component_2, ..., component_d]
      - The first 4 bytes represent an int32 (little endian) specifying the vector dimension d.
      - The following d*4 bytes are float32 values (little endian) for the vector components.
    """
    with open(filename, 'rb') as f:
        data = f.read()
        
    # Read dimension of the first vector from the first 4 bytes
    d = int.from_bytes(data[:4], byteorder='little', signed=True)
    vec_size = 1 + d   # one header + d float32 values per vector
    total_vecs = len(data) // (4 * vec_size)
    
    # Interpret the entire file as float32 values.
    # Note: The header (dimension) is stored as a float32 in this array,
    # but we will discard it.
    all_data = np.frombuffer(data, dtype='<f4').reshape(total_vecs, vec_size)
    return all_data[:, 1:]  # drop the header column

def split_into_subvectors(vectors, m):
    """
    Splits the given 2D numpy array of vectors (shape: num_vectors x d)
    into m subspaces. Each subvector is of size d_sub = d // m.
    
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
    parser.add_argument("--data_path", type=str, default="../data/sift",
                        help="Path to the folder containing the .fvecs file.")
    parser.add_argument("--fvecs_file", type=str, default="sift_base.fvecs",
                        help="The name of the .fvecs file to read.")
    parser.add_argument("--d", type=int, default=128,
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

    # Construct full file path for the .fvecs file
    fvecs_file_path = os.path.join(args.data_path, args.fvecs_file)

    # --- Read .fvecs vectors ---
    all_vectors = read_fvecs(fvecs_file_path)
    num_vectors, actual_d = all_vectors.shape
    print(f"Read {num_vectors} fvecs vectors with dimension {actual_d}.")

    if actual_d != args.d:
        raise ValueError(f"Expected vectors of dimension {args.d}, but got {actual_d}")

    # --- Split vectors into m subspaces ---
    subvectors = split_into_subvectors(all_vectors, args.m)
    subvector_size = args.d // args.m
    print(f"Split vectors into {args.m} subspaces, each of size {subvector_size}.")

    # --- Build the codebook using sklearn KMeans ---
    codebook = []       # List to store the codebook (cluster centers) for each subspace
    all_labels = []     # List to store the cluster labels for each subspace

    for group_index, subvector_group in enumerate(subvectors):
        # Ensure data is float32 for clustering (it should already be)
        dense_points = subvector_group.astype(np.float32)
        num_points = dense_points.shape[0]
        
        # Adjust the number of clusters if there are fewer points than desired
        current_clusters = args.num_clusters if num_points >= args.num_clusters else num_points
        if current_clusters != args.num_clusters:
            print(f"Subspace {group_index}: Reducing number of clusters to {current_clusters}")
        
        # Use unique points as initial centers if possible
        unique_points = np.unique(dense_points, axis=0)
        print(f"Subspace {group_index}: Found {len(unique_points)} unique points.")
        if len(unique_points) < args.num_clusters:
            print(f"Subspace {group_index}: Using all {len(unique_points)} unique points as initial centers.")
            initial_centers = unique_points
        else:
            # Otherwise, select the first 'args.num_clusters' unique points as initial centers.
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
