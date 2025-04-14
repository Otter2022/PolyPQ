import numpy as np
import argparse
import os
import utils

def compute_adc_distances(query, codebooks, database_codes, M, adc_metric="l2"):
    """
    Compute the asymmetric distance between a query vector and all database vectors.
    
    For each subvector, this function computes a lookup table between the query subvector
    and all centroids in the corresponding codebook, using the specified adc_metric.
    For each database vector (represented by its subquantizer indices), it sums the corresponding table values.
    
    Parameters:
        query (np.ndarray): A query vector of shape (D,).
        codebooks (list): List of M codebooks; each is an array of shape (Ks, D/M).
        database_codes (np.ndarray): Array of shape (n_database, M) containing PQ codes.
        M (int): Number of subvectors.
        adc_metric (str): 'l2' (default) or 'jaccard' to choose the distance measure.
    
    Returns:
        distances (np.ndarray): A 1D array of length n_database with the estimated distances.
    """
    D = query.shape[0]
    subvector_dim = D // M
    tables = []
    
    for m in range(M):
        start = m * subvector_dim
        end = (m + 1) * subvector_dim
        query_sub = query[start:end]  # shape (subvector_dim,)
        centroids = codebooks[m]       # shape (Ks, subvector_dim)
        
        if adc_metric.lower() == "l2":
            # Squared L2 distance
            diff = centroids - query_sub  # broadcasting; shape (Ks, subvector_dim)
            table = np.sum(diff ** 2, axis=1)  # shape (Ks,)
        elif adc_metric.lower() == "jaccard":
            # Generalized Jaccard distance: 1 - (sum(min)/sum(max))
            min_sum = np.sum(np.minimum(centroids, query_sub), axis=1)
            max_sum = np.sum(np.maximum(centroids, query_sub), axis=1)
            epsilon = 1e-10
            table = 1 - (min_sum / (max_sum + epsilon))
        else:
            raise ValueError("Unsupported ADC distance metric: " + adc_metric)
        
        tables.append(table)
    
    # Sum the lookup values for each database vector.
    n_database = database_codes.shape[0]
    distances = np.zeros(n_database, dtype=np.float32)
    for m in range(M):
        indices = database_codes[:, m]  # shape (n_database,)
        distances += tables[m][indices]
    
    return distances

def quantize_query(query, codebooks, M):
    """
    Quantize a query vector symmetrically using the provided codebooks.
    
    For each subvector, this function finds the nearest centroid index.
    
    Parameters:
        query (np.ndarray): A query vector of shape (D,).
        codebooks (list): List of codebooks; each has shape (Ks, D/M).
        M (int): Number of subvectors.
    
    Returns:
        query_codes (np.ndarray): A 1D array of length M containing the quantization indices.
    """
    D = query.shape[0]
    subvector_dim = D // M
    query_codes = np.empty(M, dtype=np.int32)
    
    for m in range(M):
        start = m * subvector_dim
        end = (m + 1) * subvector_dim
        query_sub = query[start:end]
        diff = codebooks[m] - query_sub  # shape (Ks, subvector_dim)
        dists = np.sum(diff ** 2, axis=1)
        query_codes[m] = np.argmin(dists)
    
    return query_codes

def evaluate_recall(query_vectors, codebooks, database_codes, groundtruth, M, topk=100, distance_metric="adc", adc_metric="l2"):
    """
    Evaluate recall@topk for the given distance metric.
    
    If distance_metric is "adc", the function computes asymmetric distances
    using the full query vector and the specified adc_metric ("l2" or "jaccard").
    If distance_metric is "hamming", it symmetrically quantizes the query and compares codes.
    
    Parameters:
        query_vectors (np.ndarray): Array of query vectors, shape (n_queries, D).
        codebooks (list): List of codebooks for each subvector.
        database_codes (np.ndarray): Array of database PQ codes, shape (n_database, M).
        groundtruth (np.ndarray): Groundtruth array (groundtruth[i,0] is assumed to be the true NN index).
        M (int): Number of subvectors.
        topk (int): Number of top candidates to consider.
        distance_metric (str): "adc" or "hamming".
        adc_metric (str): For ADC, choose between "l2" and "jaccard".
    
    Returns:
        recall (float): The recall@topk value.
    """
    correct = 0
    n_queries = query_vectors.shape[0]
    
    for i, query in enumerate(query_vectors):
        if distance_metric.lower() == "adc":
            distances = compute_adc_distances(query, codebooks, database_codes, M, adc_metric=adc_metric)
            retrieved_indices = np.argsort(distances)[:topk]
        elif distance_metric.lower() == "hamming":
            q_codes = quantize_query(query, codebooks, M)
            distances = np.sum(database_codes != q_codes, axis=1)
            retrieved_indices = np.argsort(distances)[:topk]
        else:
            raise ValueError("Unsupported distance metric: " + distance_metric)
        
        if groundtruth[i, 0] in retrieved_indices:
            correct += 1
            
    recall = correct / n_queries
    print(f"Recall@{topk} ({distance_metric} with adc_metric={adc_metric if distance_metric=='adc' else 'N/A'}): {recall:.4f}")
    return recall

def load_codebooks(codebooks_prefix, output_dir, M, Ks):
    """
    Load codebooks from disk.
    
    Expected filenames: {output_dir}/{codebooks_prefix}_codebook_{m}_{Ks}.npy for m = 0,...,M-1.
    
    Returns:
        codebooks (list): A list of loaded codebooks.
    """
    codebooks = []
    for m in range(M):
        filename = os.path.join(output_dir, f"{codebooks_prefix}_codebook_{m}_{Ks}.npy")
        if not os.path.exists(filename):
            raise IOError(f"Codebook file not found: {filename}")
        cb = np.load(filename)
        codebooks.append(cb)
    return codebooks

def load_groundtruth(fname):
    """
    Load groundtruth data from a file. Supports .npy and .ivecs formats.
    """
    _, ext = os.path.splitext(fname)
    if ext.lower() == ".npy":
        groundtruth = np.load(fname)
    elif ext.lower() == ".ivecs":
        groundtruth = utils.read_ivecs(fname)
    else:
        raise ValueError("Unsupported groundtruth file format: " + ext)
    return groundtruth

def parse_args():
    """
    Parse command-line arguments for the query evaluation script.
    """
    parser = argparse.ArgumentParser(
        description="Query PQ codebooks and evaluate recall with selectable ADC distance measures")
    parser.add_argument("--query_file", type=str, required=True,
                        help="Path to the file containing query vectors (e.g., .fvecs, .bvecs, or .npy)")
    parser.add_argument("--database_codes_file", type=str, required=True,
                        help="Path to the .npy file containing database PQ codes")
    parser.add_argument("--codebooks_prefix", type=str, required=True,
                        help="Prefix of the saved codebooks files (must match encoding)")
    parser.add_argument("--groundtruth_file", type=str, required=True,
                        help="Path to the groundtruth file (.ivecs or .npy)")
    parser.add_argument("--M", type=int, default=8,
                        help="Number of subvectors (must match encoding; default: 8)")
    parser.add_argument("--Ks", type=int, default=256,
                        help="Number of clusters per subvector (default: 256)")
    parser.add_argument("--topk", type=int, default=100,
                        help="Number of top items to consider for recall evaluation (default: 100)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory containing the codebook files (default: current directory)")
    parser.add_argument("--distance_metric", type=str, default="adc",
                        choices=["adc", "hamming"],
                        help="Distance metric to use: 'adc' (default) or 'hamming'")
    parser.add_argument("--adc_metric", type=str, default="l2",
                        choices=["l2", "jaccard"],
                        help="For ADC mode, choose the distance metric: 'l2' (default) or 'jaccard'")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load query vectors.
    query_vectors = utils.load_vectors(args.query_file)
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)
    print("Loaded query vectors, shape:", query_vectors.shape)
    
    # Load database PQ codes.
    if not os.path.exists(args.database_codes_file):
        raise IOError("Database codes file not found: " + args.database_codes_file)
    database_codes = np.load(args.database_codes_file)
    print("Loaded database codes, shape:", database_codes.shape)
    
    # Load the codebooks.
    codebooks = load_codebooks(args.codebooks_prefix, args.output_dir, args.M, args.Ks)
    print(f"Loaded {len(codebooks)} codebooks.")
    
    # Load groundtruth.
    groundtruth = load_groundtruth(args.groundtruth_file)
    print("Loaded groundtruth, shape:", groundtruth.shape)
    
    # Evaluate recall.
    evaluate_recall(query_vectors, codebooks, database_codes, groundtruth,
                    args.M, topk=args.topk, distance_metric=args.distance_metric,
                    adc_metric=args.adc_metric)

if __name__ == "__main__":
    main()
