import numpy as np
import argparse
import os
from sklearn.cluster import KMeans
import utils

# If using Jaccard clustering, we define our custom Jaccard distance function.
def jaccard_distance(x, y, eps=1e-10):
    """
    Compute the generalized Jaccard distance between two nonnegative vectors x and y.
    d(x, y) = 1 - (sum(min(x, y)) / (sum(max(x, y)) + eps))
    """
    num = np.sum(np.minimum(x, y))
    den = np.sum(np.maximum(x, y)) + eps
    return 1 - (num / den)

def product_quantization(descriptors, M=4, Ks=256, clustering_metric="l2"):
    """
    Compute product quantization on the input descriptors by splitting each
    descriptor into M subvectors. For each subvector, clustering is performed
    using either Euclidean distance (via KMeans) or generalized Jaccard distance 
    (via KMedoids from scikit-learn-extra).
    
    Parameters:
        descriptors (np.ndarray): Array of shape (n_samples, D).
        M (int): Number of subvectors.
        Ks (int): Number of clusters (centroids) per subvector.
        clustering_metric (str): "l2" (default) for Euclidean distance clustering,
                                 or "jaccard" for clustering with generalized Jaccard distance.
    
    Returns:
        quantized_codes (np.ndarray): Array of shape (n_samples, M) containing the cluster assignments.
        codebooks (list): List of M arrays, each representing the codebook for that subvector.
    """
    n_samples, D = descriptors.shape
    if D % M != 0:
        raise ValueError("Descriptor dimension must be divisible by M (the number of subvectors).")
    
    subvector_dim = D // M
    quantized_codes = np.empty((n_samples, M), dtype=np.int32)
    codebooks = []
    
    for m in range(M):
        sub_vectors = descriptors[:, m * subvector_dim : (m+1) * subvector_dim]
        if clustering_metric.lower() == "l2":
            # Use standard KMeans with Euclidean distance.
            kmeans = KMeans(n_clusters=Ks, random_state=42)
            kmeans.fit(sub_vectors)
            codebook = kmeans.cluster_centers_
            assignments = kmeans.labels_
        elif clustering_metric.lower() == "jaccard":
            # Use KMedoids from scikit-learn-extra with the custom Jaccard distance.
            # Note: scikit-learn-extra's KMedoids accepts a callable for the metric.
            try:
                from sklearn_extra.cluster import KMedoids
            except ImportError:
                raise ImportError("scikit-learn-extra is required for Jaccard clustering. Install with: pip install scikit-learn-extra")
            kmedoids = KMedoids(n_clusters=Ks, metric=jaccard_distance, random_state=42)
            kmedoids.fit(sub_vectors)
            codebook = kmedoids.cluster_centers_
            assignments = kmedoids.labels_
        else:
            raise ValueError("Unsupported clustering metric: " + clustering_metric)
        
        codebooks.append(codebook)
        quantized_codes[:, m] = assignments
    
    return quantized_codes, codebooks

def parse_args():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Product Quantization for SIFT descriptors using NumPy, scikit-learn and scikit-learn-extra (for Jaccard clustering)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the binary file containing SIFT descriptors")
    parser.add_argument("--M", type=int, default=4,
                        help="Number of subvectors to split each descriptor into (default: 4)")
    parser.add_argument("--Ks", type=int, default=256,
                        help="Number of clusters per subvector (default: 256)")
    parser.add_argument("--clustering_metric", type=str, default="l2",
                        choices=["l2", "jaccard"],
                        help="Clustering metric to use: 'l2' (default) for Euclidean KMeans or 'jaccard' for KMedoids with Jaccard distance")
    parser.add_argument("--output_prefix", type=str, default="output",
                        help="Prefix for the output files (default: 'output')")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save the output files (default: current directory)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Maximum number of vectors to load (default: -1 means load all vectors)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load descriptors using your utility functions.
    descriptors = utils.load_vectors(args.data_file, sparse=True, dimension=21609)
    if descriptors.dtype != np.float32:
        descriptors = descriptors.astype(np.float32)
    print("Loaded descriptors shape:", descriptors.shape)

    if args.limit > 0:
        descriptors = descriptors[:args.limit]
        print(f"Limiting descriptors to the first {args.limit} vectors. New shape: {descriptors.shape}")

    # Perform product quantization with the chosen clustering metric.
    quantized_codes, codebooks = product_quantization(descriptors, M=args.M, Ks=args.Ks, clustering_metric=args.clustering_metric)
    print("Quantized codes shape:", quantized_codes.shape)
    
    # Save the quantized codes.
    np.save(f"{args.output_dir}/{args.output_prefix}__quantized_codes.npy", quantized_codes)
    
    # Save each codebook.
    for m, cb in enumerate(codebooks):
        np.save(f"{args.output_dir}/{args.output_prefix}_codebook_{m}_{args.Ks}.npy", cb)

if __name__ == '__main__':
    main()
