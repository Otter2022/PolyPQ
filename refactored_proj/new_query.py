import os
import csv
import numpy as np
import argparse
import utils

# ----------------- ADC and quantization functions ----------------- #
def compute_adc_distances(query, codebooks, database_codes, M, adc_metric="l2"):
    D = query.shape[0]
    subvector_dim = D // M
    tables = []
    for m in range(M):
        qs = query[m*subvector_dim:(m+1)*subvector_dim]
        cents = codebooks[m]
        if adc_metric.lower() == "l2":
            diff = cents - qs
            table = np.sum(diff**2, axis=1)
        elif adc_metric.lower() == "jaccard":
            eps = 1e-10
            min_sum = np.sum(np.minimum(cents, qs), axis=1)
            max_sum = np.sum(np.maximum(cents, qs), axis=1)
            table = 1 - (min_sum / (max_sum + eps))
        else:
            raise ValueError("Unsupported ADC metric: " + adc_metric)
        tables.append(table)
    n_db = database_codes.shape[0]
    dists = np.zeros(n_db, dtype=np.float32)
    for m in range(M):
        idxs = database_codes[:, m]
        dists += tables[m][idxs]
    return dists


def quantize_query(query, codebooks, M):
    D = query.shape[0]
    subvector_dim = D // M
    q_codes = np.empty(M, dtype=np.int32)
    for m in range(M):
        qs = query[m*subvector_dim:(m+1)*subvector_dim]
        cents = codebooks[m]
        dists = np.sum((cents - qs)**2, axis=1)
        q_codes[m] = np.argmin(dists)
    return q_codes

# ---------- Groundtruth loading as dict of sets ----------
def load_groundtruth(filepath):
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
    all_gt = {}
    for fname in os.listdir(gt_dir):
        fpath = os.path.join(gt_dir, fname)
        gt = load_groundtruth(fpath)
        all_gt.update(gt)
    return all_gt


def filter_queries_by_groundtruth(query_vectors, gt_dict):
    available = set(range(query_vectors.shape[0]))
    valid_ids = sorted(available & set(gt_dict.keys()))
    if not valid_ids:
        raise ValueError("No matching query indices in groundtruth.")
    filtered_qs = query_vectors[valid_ids]
    filtered_gt = {qid: gt_dict[qid] for qid in valid_ids}
    print(f"Filtered queries: using {len(valid_ids)} queries out of {query_vectors.shape[0]}")
    return valid_ids, filtered_qs, filtered_gt


def evaluate_recall(query_vectors, codebooks, database_codes, valid_ids, gt_dict,
                    M, topk=100, distance_metric="adc", adc_metric="l2"):
    recall_values = []
    for i, q in enumerate(query_vectors):
        qid = valid_ids[i]
        if distance_metric.lower() == "adc":
            dists = compute_adc_distances(q, codebooks, database_codes, M, adc_metric)
            retrieved = np.argsort(dists)[:topk]
        elif distance_metric.lower() == "hamming":
            q_codes = quantize_query(q, codebooks, M)
            dists = np.sum(database_codes != q_codes, axis=1)
            retrieved = np.argsort(dists)[:topk]
        else:
            raise ValueError("Unsupported distance metric: " + distance_metric)
        retrieved_set = set(retrieved)
        gt_set = gt_dict.get(qid, set())
        if gt_set:
            intersect = gt_set & retrieved_set
            recall_values.append(len(intersect) / len(gt_set))
    if recall_values:
        avg_recall = sum(recall_values) / len(recall_values)
    else:
        avg_recall = 0.0
    print(f"Average Recall@{topk}: {avg_recall:.4f} over {len(recall_values)} queries")
    return avg_recall


def parse_args():
    p = argparse.ArgumentParser(description="PQ query + recall eval with dict groundtruth")
    p.add_argument("--query_file", required=True)
    p.add_argument("--database_codes_file", required=True)
    p.add_argument("--codebooks_prefix", required=True)
    p.add_argument("--groundtruth_dir", required=True)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--Ks", type=int, default=256)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--distance_metric", choices=["adc","hamming"], default="adc")
    p.add_argument("--adc_metric", choices=["l2","jaccard"], default="l2")
    p.add_argument("--sparse", action="store_true")
    p.add_argument("--dimension", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    qvs = utils.load_vectors(args.query_file, dimension=args.dimension, sparse=args.sparse)
    if qvs.dtype != np.float32:
        qvs = qvs.astype(np.float32)
    print("Loaded query vectors, shape:", qvs.shape)
    db_codes = np.load(args.database_codes_file)
    print("Loaded database codes, shape:", db_codes.shape)
    cbs = []
    for m in range(args.M):
        fn = os.path.join(args.codebooks_prefix + f"_codebook_{m}_{args.Ks}.npy")
        cbs.append(np.load(fn))
    print(f"Loaded {len(cbs)} codebooks.")
    # Load groundtruth as dict of sets
    gt_dict = load_all_groundtruth(args.groundtruth_dir)
    print(f"Loaded groundtruth for {len(gt_dict)} queries.")
    valid_ids, fqs, fgt = filter_queries_by_groundtruth(qvs, gt_dict)
    evaluate_recall(fqs, cbs, db_codes, valid_ids, fgt,
                    args.M, topk=args.topk,
                    distance_metric=args.distance_metric,
                    adc_metric=args.adc_metric)

if __name__ == "__main__":
    main()
#__________________________________________________________


# import numpy as np
# import argparse
# import os
# import utils

# # ----------------- ADC and quantization functions ----------------- #
# def compute_adc_distances(query, codebooks, database_codes, M, adc_metric="l2"):
#     D = query.shape[0]
#     subvector_dim = D // M
#     tables = []
#     for m in range(M):
#         qs = query[m*subvector_dim:(m+1)*subvector_dim]
#         cents = codebooks[m]
#         if adc_metric.lower() == "l2":
#             diff = cents - qs
#             table = np.sum(diff**2, axis=1)
#         elif adc_metric.lower() == "jaccard":
#             eps = 1e-10
#             min_sum = np.sum(np.minimum(cents, qs), axis=1)
#             max_sum = np.sum(np.maximum(cents, qs), axis=1)
#             table = 1 - (min_sum / (max_sum + eps))
#         else:
#             raise ValueError("Unsupported ADC metric: " + adc_metric)
#         tables.append(table)
#     n_db = database_codes.shape[0]
#     dists = np.zeros(n_db, dtype=np.float32)
#     for m in range(M):
#         idxs = database_codes[:, m]
#         dists += tables[m][idxs]
#     return dists


# def quantize_query(query, codebooks, M):
#     D = query.shape[0]
#     subvector_dim = D // M
#     q_codes = np.empty(M, dtype=np.int32)
#     for m in range(M):
#         qs = query[m*subvector_dim:(m+1)*subvector_dim]
#         cents = codebooks[m]
#         dists = np.sum((cents - qs)**2, axis=1)
#         q_codes[m] = np.argmin(dists)
#     return q_codes


# def evaluate_recall(query_vectors, codebooks, database_codes, valid_ids, gt_dict,
#                     M, topk=100, distance_metric="adc", adc_metric="l2"):
#     correct = 0
#     for i, q in enumerate(query_vectors):
#         qid = valid_ids[i]
#         if distance_metric.lower() == "adc":
#             dists = compute_adc_distances(q, codebooks, database_codes, M, adc_metric)
#             retrieved = np.argsort(dists)[:topk]
#         elif distance_metric.lower() == "hamming":
#             q_codes = quantize_query(q, codebooks, M)
#             dists = np.sum(database_codes != q_codes, axis=1)
#             retrieved = np.argsort(dists)[:topk]
#         else:
#             raise ValueError("Unsupported distance metric: " + distance_metric)
#         if any(nb in retrieved for nb in gt_dict[qid]):
#             correct += 1
#     recall = correct / len(query_vectors)
#     print(f"Recall@{topk}: {recall:.4f}")
#     return recall


# def load_groundtruth_from_dir(gt_dir, valid_db_ids=None):
#     gt = {}
#     for fname in os.listdir(gt_dir):
#         path = os.path.join(gt_dir, fname)
#         with open(path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split(',')
#                 if len(parts) < 2: continue
#                 try:
#                     qid = int(parts[0])
#                 except ValueError:
#                     continue
#                 nbrs = [int(x) for x in parts[1:] if x.isdigit() and (valid_db_ids is None or int(x) in valid_db_ids)]
#                 if nbrs:
#                     gt[qid] = nbrs
#     return gt


# def filter_queries_by_groundtruth(query_vectors, gt_dict):
#     available = set(range(query_vectors.shape[0]))
#     valid_ids = sorted(available & set(gt_dict.keys()))
#     if not valid_ids:
#         raise ValueError("No matching query indices in groundtruth.")
#     filtered_qs = query_vectors[valid_ids]
#     filtered_gt = {qid: gt_dict[qid] for qid in valid_ids}
#     print(f"Filtered queries: using {len(valid_ids)} queries out of {query_vectors.shape[0]}")
#     return valid_ids, filtered_qs, filtered_gt


# def parse_args():
#     p = argparse.ArgumentParser(description="PQ query + recall eval with dict groundtruth")
#     p.add_argument("--query_file", required=True)
#     p.add_argument("--database_codes_file", required=True)
#     p.add_argument("--codebooks_prefix", required=True)
#     p.add_argument("--groundtruth_dir", required=True)
#     p.add_argument("--M", type=int, default=8)
#     p.add_argument("--Ks", type=int, default=256)
#     p.add_argument("--topk", type=int, default=100)
#     p.add_argument("--distance_metric", choices=["adc","hamming"], default="adc")
#     p.add_argument("--adc_metric", choices=["l2","jaccard"], default="l2")
#     p.add_argument("--sparse", action="store_true")
#     p.add_argument("--dimension", type=int)
#     return p.parse_args()


# def main():
#     args = parse_args()
#     # load queries
#     qvs = utils.load_vectors(args.query_file, dimension=args.dimension, sparse=args.sparse)
#     if qvs.dtype != np.float32:
#         qvs = qvs.astype(np.float32)
#     print("Loaded query vectors, shape:", qvs.shape)
#     # load database codes
#     db_codes = np.load(args.database_codes_file)
#     print("Loaded database codes, shape:", db_codes.shape)
#     # load codebooks
#     cbs = []
#     for m in range(args.M):
#         fn = os.path.join(args.codebooks_prefix + f"_codebook_{m}_{args.Ks}.npy")
#         cbs.append(np.load(fn))
#     print(f"Loaded {len(cbs)} codebooks.")
#     # groundtruth dict
#     gt_dict = load_groundtruth_from_dir(args.groundtruth_dir, set(range(db_codes.shape[0])))
#     print(f"Loaded groundtruth for {len(gt_dict)} queries.")
#     # filter
#     valid_ids, fqs, fgt = filter_queries_by_groundtruth(qvs, gt_dict)
#     # eval
#     evaluate_recall(fqs, cbs, db_codes, valid_ids, fgt,
#                     args.M, topk=args.topk,
#                     distance_metric=args.distance_metric,
#                     adc_metric=args.adc_metric)

# if __name__ == "__main__":
#     print("Running query evaluation script...")
#     main()

#__________________________________________________________

# import numpy as np
# import argparse
# import os
# import utils

# # ----------------- ADC and quantization functions from before ----------------- #
# def compute_adc_distances(query, codebooks, database_codes, M, adc_metric="l2"):
#     """
#     Compute the asymmetric distance between a query vector and all database vectors.

#     For each subvector, compute a lookup table between the query subvector and the corresponding centroids.
#     Then, for each database vector (its PQ codes), sum the corresponding table values using the specified ADC metric.
#     """
#     D = query.shape[0]
#     subvector_dim = D // M
#     tables = []
#     for m in range(M):
#         start = m * subvector_dim
#         end = (m + 1) * subvector_dim
#         query_sub = query[start:end]
#         centroids = codebooks[m]
#         if adc_metric.lower() == "l2":
#             diff = centroids - query_sub
#             table = np.sum(diff ** 2, axis=1)
#         elif adc_metric.lower() == "jaccard":
#             eps = 1e-10
#             min_sum = np.sum(np.minimum(centroids, query_sub), axis=1)
#             max_sum = np.sum(np.maximum(centroids, query_sub), axis=1)
#             table = 1 - (min_sum / (max_sum + eps))
#         else:
#             raise ValueError("Unsupported ADC distance metric: " + adc_metric)
#         tables.append(table)
#     n_database = database_codes.shape[0]
#     distances = np.zeros(n_database, dtype=np.float32)
#     for m in range(M):
#         indices = database_codes[:, m]
#         distances += tables[m][indices]
#     return distances

# def quantize_query(query, codebooks, M):
#     """
#     Quantize a query vector symmetrically using the provided codebooks.
#     Returns an array of subvector cluster indices.
#     """
#     D = query.shape[0]
#     subvector_dim = D // M
#     query_codes = np.empty(M, dtype=np.int32)
#     for m in range(M):
#         start = m * subvector_dim
#         end = (m + 1) * subvector_dim
#         query_sub = query[start:end]
#         diff = codebooks[m] - query_sub
#         dists = np.sum(diff ** 2, axis=1)
#         query_codes[m] = np.argmin(dists)
#     return query_codes

# def evaluate_recall(query_vectors, codebooks, database_codes, groundtruth_arr, M, topk=100,
#                     distance_metric="adc", adc_metric="l2"):
#     """
#     Evaluate recall@topk for each query using either ADC or symmetric (Hamming) matching.
    
#     groundtruth_arr is an array of shape (n_queries,) containing the true nearest neighbor index for each query.
#     """
#     correct = 0
#     n_queries = query_vectors.shape[0]
#     for i, query in enumerate(query_vectors):
#         if distance_metric.lower() == "adc":
#             distances = compute_adc_distances(query, codebooks, database_codes, M, adc_metric=adc_metric)
#             retrieved = np.argsort(distances)[:topk]
#         elif distance_metric.lower() == "hamming":
#             q_codes = quantize_query(query, codebooks, M)
#             distances = np.sum(database_codes != q_codes, axis=1)
#             retrieved = np.argsort(distances)[:topk]
#         else:
#             raise ValueError("Unsupported distance metric: " + distance_metric)
#         true_nn = groundtruth_arr[i]
#         if true_nn in retrieved:
#             correct += 1
#     recall = correct / n_queries
#     print(f"Recall@{topk} ({distance_metric} with adc_metric={adc_metric if distance_metric=='adc' else 'N/A'}): {recall:.4f}")
#     return recall

# # ----------------- Groundtruth loading and filtering functions ----------------- #
# def load_groundtruth_from_dir(gt_dir, valid_db_ids):
#     """
#     Read groundtruth entries from all files in the gt_dir that start with "similarityMap".
    
#     Each file is expected to have one line with comma-separated integers, where the first integer is the query id,
#     and the rest are candidate neighbor indices. Only neighbors that are in valid_db_ids are kept.
    
#     Returns:
#         gt_dict: A dictionary mapping each query id (int) to a list of valid neighbor indices.
#     """
#     gt_dict = {}
#     for fname in os.listdir(gt_dir):
#         fpath = os.path.join(gt_dir, fname)
#         with open(fpath, "r") as f:
#             for line in f:
#                 parts = line.strip().split(',')
#                 if len(parts) < 2:
#                     continue
#                 try:
#                     qid = int(parts[0].strip())
#                 except ValueError:
#                     continue
#                 neighbors = []
#                 for token in parts[1:]:
#                     try:
#                         nb = int(token.strip())
#                         if nb in valid_db_ids:
#                             neighbors.append(nb)
#                     except ValueError:
#                         continue
#                 if neighbors:
#                     gt_dict[qid] = neighbors
#     return gt_dict

# def filter_queries_by_groundtruth(query_vectors, gt_dict):
#     """
#     Given query_vectors (loaded from an fvecs file) and a groundtruth dictionary (keys: query ids),
#     select only those query vectors whose index is present in gt_dict.
    
#     Returns:
#         valid_query_ids: Sorted list of query ids that exist in both the groundtruth dictionary and the available queries.
#         filtered_queries: A numpy array containing only the query vectors whose index is in valid_query_ids.
#         gt_array: A numpy array of groundtruth neighbor indices (taking the first neighbor for each query).
#     """
#     available_query_ids = set(range(query_vectors.shape[0]))
#     valid_query_ids = sorted(list(available_query_ids.intersection(set(gt_dict.keys()))))
    
#     if not valid_query_ids:
#         raise ValueError("No query vector indices match those provided in the groundtruth files.")
    
#     filtered_queries = query_vectors[valid_query_ids]
#     # For each valid query id, take the first valid neighbor from the groundtruth list.
#     gt_array = np.array([gt_dict[qid][0] for qid in valid_query_ids], dtype=np.int32)
    
#     print(f"Filtered queries: using {len(valid_query_ids)} queries out of {query_vectors.shape[0]}")
#     return valid_query_ids, filtered_queries, gt_array

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Query PQ codebooks and evaluate recall using only the query vectors matching indices in the groundtruth files")
#     parser.add_argument("--query_file", type=str, required=True,
#                         help="Path to the fvecs file containing query vectors (e.g., vectors 0-31142)")
#     parser.add_argument("--database_codes_file", type=str, required=True,
#                         help="Path to the .npy file containing database PQ codes")
#     parser.add_argument("--codebooks_prefix", type=str, required=True,
#                         help="Prefix of the saved codebooks files (must match encoding)")
#     parser.add_argument("--groundtruth_dir", type=str, required=True,
#                         help="Directory containing groundtruth files (e.g., similarityMap_*)")
#     parser.add_argument("--M", type=int, default=8,
#                         help="Number of subvectors (must match encoding; default: 8)")
#     parser.add_argument("--Ks", type=int, default=256,
#                         help="Number of clusters per subvector (default: 256)")
#     parser.add_argument("--topk", type=int, default=100,
#                         help="Number of top items to consider for recall evaluation (default: 100)")
#     parser.add_argument("--output_dir", type=str, default=".",
#                         help="Directory containing the codebook files (default: current directory)")
#     parser.add_argument("--distance_metric", type=str, default="adc",
#                         choices=["adc", "hamming"],
#                         help="Distance metric to use: 'adc' (default) or 'hamming'")
#     parser.add_argument("--adc_metric", type=str, default="l2",
#                         choices=["l2", "jaccard"],
#                         help="For ADC mode, choose the distance metric: 'l2' (default) or 'jaccard'")
#     return parser.parse_args()

# def main():
#     args = parse_args()
    
#     # Load all query vectors from the fvecs file.
#     query_vectors = utils.load_vectors(args.query_file)
#     if query_vectors.dtype != np.float32:
#         query_vectors = query_vectors.astype(np.float32)
#     print("Loaded query vectors, shape:", query_vectors.shape)
    
#     # Load the database PQ codes.
#     if not os.path.exists(args.database_codes_file):
#         raise IOError("Database codes file not found: " + args.database_codes_file)
#     database_codes = np.load(args.database_codes_file)
#     print("Loaded database codes, shape:", database_codes.shape)
#     valid_db_ids = set(range(database_codes.shape[0]))
    
#     # Load codebooks from the output directory.
#     def load_codebooks(prefix, output_dir, M, Ks):
#         codebooks = []
#         for m in range(M):
#             filename = os.path.join(output_dir, f"{prefix}_codebook_{m}_{Ks}.npy")
#             if not os.path.exists(filename):
#                 raise IOError("Codebook file not found: " + filename)
#             codebooks.append(np.load(filename))
#         return codebooks
#     codebooks = load_codebooks(args.codebooks_prefix, args.output_dir, args.M, args.Ks)
#     print(f"Loaded {len(codebooks)} codebooks.")
    
#     # Load the groundtruth dictionary from the specified directory.
#     gt_dict = load_groundtruth_from_dir(args.groundtruth_dir, valid_db_ids)
#     print(f"Loaded groundtruth for {len(gt_dict)} queries from groundtruth files.")
    
#     # Filter query vectors to only those whose indices appear in the groundtruth.
#     valid_query_ids, filtered_queries, gt_array = filter_queries_by_groundtruth(query_vectors, gt_dict)
    
#     # Evaluate recall using the filtered set of queries.
#     evaluate_recall(filtered_queries, codebooks, database_codes, gt_array,
#                     args.M, topk=args.topk, distance_metric=args.distance_metric,
#                     adc_metric=args.adc_metric)

# if __name__ == "__main__":
#     main()


#-----------------------------------------------------------------------------
# import numpy as np
# import argparse
# import os
# import utils

# def compute_adc_distances(query, codebooks, database_codes, M, adc_metric="l2"):
#     """
#     Compute the asymmetric distance between a query vector and all database vectors.
    
#     For each subvector, this function computes a lookup table between the query subvector
#     and all centroids in the corresponding codebook, using the specified adc_metric.
#     For each database vector (represented by its subquantizer indices), it sums the corresponding table values.
    
#     Parameters:
#         query (np.ndarray): A query vector of shape (D,).
#         codebooks (list): List of M codebooks; each is an array of shape (Ks, D/M).
#         database_codes (np.ndarray): Array of shape (n_database, M) containing PQ codes.
#         M (int): Number of subvectors.
#         adc_metric (str): 'l2' (default) or 'jaccard' to choose the distance measure.
    
#     Returns:
#         distances (np.ndarray): A 1D array of length n_database with the estimated distances.
#     """
#     D = query.shape[0]
#     subvector_dim = D // M
#     tables = []
    
#     for m in range(M):
#         start = m * subvector_dim
#         end = (m + 1) * subvector_dim
#         query_sub = query[start:end]  # shape (subvector_dim,)
#         centroids = codebooks[m]       # shape (Ks, subvector_dim)
        
#         if adc_metric.lower() == "l2":
#             # Squared L2 distance
#             diff = centroids - query_sub  # broadcasting; shape (Ks, subvector_dim)
#             table = np.sum(diff ** 2, axis=1)  # shape (Ks,)
#         elif adc_metric.lower() == "jaccard":
#             # Generalized Jaccard distance: 1 - (sum(min)/sum(max))
#             min_sum = np.sum(np.minimum(centroids, query_sub), axis=1)
#             max_sum = np.sum(np.maximum(centroids, query_sub), axis=1)
#             epsilon = 1e-10
#             table = 1 - (min_sum / (max_sum + epsilon))
#         else:
#             raise ValueError("Unsupported ADC distance metric: " + adc_metric)
        
#         tables.append(table)
    
#     # Sum the lookup values for each database vector.
#     n_database = database_codes.shape[0]
#     distances = np.zeros(n_database, dtype=np.float32)
#     for m in range(M):
#         indices = database_codes[:, m]  # shape (n_database,)
#         distances += tables[m][indices]
    
#     return distances

# def quantize_query(query, codebooks, M):
#     """
#     Quantize a query vector symmetrically using the provided codebooks.
    
#     For each subvector, this function finds the nearest centroid index.
    
#     Parameters:
#         query (np.ndarray): A query vector of shape (D,).
#         codebooks (list): List of codebooks; each has shape (Ks, D/M).
#         M (int): Number of subvectors.
    
#     Returns:
#         query_codes (np.ndarray): A 1D array of length M containing the quantization indices.
#     """
#     D = query.shape[0]
#     subvector_dim = D // M
#     query_codes = np.empty(M, dtype=np.int32)
    
#     for m in range(M):
#         start = m * subvector_dim
#         end = (m + 1) * subvector_dim
#         query_sub = query[start:end]
#         diff = codebooks[m] - query_sub  # shape (Ks, subvector_dim)
#         dists = np.sum(diff ** 2, axis=1)
#         query_codes[m] = np.argmin(dists)
    
#     return query_codes

# def evaluate_recall(query_vectors, codebooks, database_codes, groundtruth, M, topk=100, distance_metric="adc", adc_metric="l2"):
#     """
#     Evaluate recall@topk for the given distance metric.
    
#     If distance_metric is "adc", the function computes asymmetric distances
#     using the full query vector and the specified adc_metric ("l2" or "jaccard").
#     If distance_metric is "hamming", it symmetrically quantizes the query and compares codes.
    
#     Parameters:
#         query_vectors (np.ndarray): Array of query vectors, shape (n_queries, D).
#         codebooks (list): List of codebooks for each subvector.
#         database_codes (np.ndarray): Array of database PQ codes, shape (n_database, M).
#         groundtruth (np.ndarray): Groundtruth array (groundtruth[i,0] is assumed to be the true NN index).
#         M (int): Number of subvectors.
#         topk (int): Number of top candidates to consider.
#         distance_metric (str): "adc" or "hamming".
#         adc_metric (str): For ADC, choose between "l2" and "jaccard".
    
#     Returns:
#         recall (float): The recall@topk value.
#     """
#     correct = 0
#     n_queries = query_vectors.shape[0]
    
#     for i, query in enumerate(query_vectors):
#         if distance_metric.lower() == "adc":
#             distances = compute_adc_distances(query, codebooks, database_codes, M, adc_metric=adc_metric)
#             retrieved_indices = np.argsort(distances)[:topk]
#         elif distance_metric.lower() == "hamming":
#             q_codes = quantize_query(query, codebooks, M)
#             distances = np.sum(database_codes != q_codes, axis=1)
#             retrieved_indices = np.argsort(distances)[:topk]
#         else:
#             raise ValueError("Unsupported distance metric: " + distance_metric)
        
#         if groundtruth[i, 0] in retrieved_indices:
#             correct += 1
            
#     recall = correct / n_queries
#     print(f"Recall@{topk} ({distance_metric} with adc_metric={adc_metric if distance_metric=='adc' else 'N/A'}): {recall:.4f}")
#     return recall

# def load_codebooks(codebooks_prefix, output_dir, M, Ks):
#     """
#     Load codebooks from disk.
    
#     Expected filenames: {output_dir}/{codebooks_prefix}_codebook_{m}_{Ks}.npy for m = 0,...,M-1.
    
#     Returns:
#         codebooks (list): A list of loaded codebooks.
#     """
#     codebooks = []
#     for m in range(M):
#         filename = os.path.join(output_dir, f"{codebooks_prefix}_codebook_{m}_{Ks}.npy")
#         if not os.path.exists(filename):
#             raise IOError(f"Codebook file not found: {filename}")
#         cb = np.load(filename)
#         codebooks.append(cb)
#     return codebooks

# def load_groundtruth(fname):
#     """
#     Load groundtruth data from a file. Supports .npy and .ivecs formats.
#     """
#     _, ext = os.path.splitext(fname)
#     if ext.lower() == ".npy":
#         groundtruth = np.load(fname)
#     elif ext.lower() == ".ivecs":
#         groundtruth = utils.read_ivecs(fname)
#     else:
#         raise ValueError("Unsupported groundtruth file format: " + ext)
#     return groundtruth

# def parse_args():
#     """
#     Parse command-line arguments for the query evaluation script.
#     """
#     parser = argparse.ArgumentParser(
#         description="Query PQ codebooks and evaluate recall with selectable ADC distance measures")
#     parser.add_argument("--query_file", type=str, required=True,
#                         help="Path to the file containing query vectors (e.g., .fvecs, .bvecs, or .npy)")
#     parser.add_argument("--database_codes_file", type=str, required=True,
#                         help="Path to the .npy file containing database PQ codes")
#     parser.add_argument("--codebooks_prefix", type=str, required=True,
#                         help="Prefix of the saved codebooks files (must match encoding)")
#     parser.add_argument("--groundtruth_file", type=str, required=True,
#                         help="Path to the groundtruth file (.ivecs or .npy)")
#     parser.add_argument("--M", type=int, default=8,
#                         help="Number of subvectors (must match encoding; default: 8)")
#     parser.add_argument("--Ks", type=int, default=256,
#                         help="Number of clusters per subvector (default: 256)")
#     parser.add_argument("--topk", type=int, default=100,
#                         help="Number of top items to consider for recall evaluation (default: 100)")
#     parser.add_argument("--output_dir", type=str, default=".",
#                         help="Directory containing the codebook files (default: current directory)")
#     parser.add_argument("--distance_metric", type=str, default="adc",
#                         choices=["adc", "hamming"],
#                         help="Distance metric to use: 'adc' (default) or 'hamming'")
#     parser.add_argument("--adc_metric", type=str, default="l2",
#                         choices=["l2", "jaccard"],
#                         help="For ADC mode, choose the distance metric: 'l2' (default) or 'jaccard'")
#     return parser.parse_args()

# def main():
#     args = parse_args()
    
#     # Load query vectors.
#     query_vectors = utils.load_vectors(args.query_file)
#     if query_vectors.dtype != np.float32:
#         query_vectors = query_vectors.astype(np.float32)
#     print("Loaded query vectors, shape:", query_vectors.shape)
    
#     # Load database PQ codes.
#     if not os.path.exists(args.database_codes_file):
#         raise IOError("Database codes file not found: " + args.database_codes_file)
#     database_codes = np.load(args.database_codes_file)
#     print("Loaded database codes, shape:", database_codes.shape)
    
#     # Load the codebooks.
#     codebooks = load_codebooks(args.codebooks_prefix, args.output_dir, args.M, args.Ks)
#     print(f"Loaded {len(codebooks)} codebooks.")
    
#     # Load groundtruth.
#     groundtruth = load_groundtruth(args.groundtruth_file)
#     print("Loaded groundtruth, shape:", groundtruth.shape)
    
#     # Evaluate recall.
#     evaluate_recall(query_vectors, codebooks, database_codes, groundtruth,
#                     args.M, topk=args.topk, distance_metric=args.distance_metric,
#                     adc_metric=args.adc_metric)

# if __name__ == "__main__":
#     main()
