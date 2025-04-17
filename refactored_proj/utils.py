import numpy as np
import os


def read_fvecs(fname):
    """
    Load vectors from a .fvecs file.
    """
    a = np.fromfile(fname, dtype=np.int32)
    if a.size == 0:
        raise IOError(f"File is empty or cannot be read: {fname}")
    d = a[0]
    total_vectors = a.size // (d + 1)
    a = a.reshape(total_vectors, d + 1)
    return a[:, 1:].view(np.float32)


def read_ivecs(fname):
    """
    Load vectors from an .ivecs file.
    """
    a = np.fromfile(fname, dtype=np.int32)
    if a.size == 0:
        raise IOError(f"File is empty or cannot be read: {fname}")
    d = a[0]
    total_vectors = a.size // (d + 1)
    a = a.reshape(total_vectors, d + 1)
    return a[:, 1:]


def read_bvecs(fname):
    """
    Load vectors from a .bvecs file.
    """
    with open(fname, 'rb') as f:
        content = f.read()
    d = np.frombuffer(content[:4], dtype=np.int32)[0]
    vector_byte_size = 4 + d
    total_vectors = len(content) // vector_byte_size
    data = np.empty((total_vectors, d), dtype=np.uint8)
    for i in range(total_vectors):
        start = i * vector_byte_size + 4
        end = start + d
        data[i, :] = np.frombuffer(content[start:end], dtype=np.uint8)
    return data


def reconstruct_dense_vector(vec_str, grid_size):
    """
    Build a dense binary vector from sparse index string.
    """
    dense_vec = np.zeros(grid_size, dtype=np.float32)
    for idx in vec_str.replace(',', ' ').split():
        dense_vec[int(idx)] = 1.0
    return dense_vec


def read_txt(fname, sparse=False, dimension=None):
    """
    Load vectors from a .txt file.
    """
    if sparse:
        if dimension is None:
            raise ValueError("Dimension must be specified for sparse vectors.")
        vectors = []
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vectors.append(reconstruct_dense_vector(line, dimension))
        return np.array(vectors, dtype=np.float32)
    else:
        try:
            return np.loadtxt(fname, dtype=np.float32, delimiter=',')
        except ValueError:
            return np.loadtxt(fname, dtype=np.float32)


def load_vectors(fname, dimension=None, sparse=False):
    """
    Load vectors from a file or directory by extension.
    """
    def multi_file():
        if not os.path.isdir(fname):
            raise IOError(f"Not a valid file or directory: {fname}")
        vectors = []
        for file in sorted(os.listdir(fname)):
            path = os.path.join(fname, file)
            if file.endswith('.fvecs'):
                vectors.append(read_fvecs(path))
            elif file.endswith('.ivecs'):
                vectors.append(read_ivecs(path))
            elif file.endswith('.bvecs'):
                vectors.append(read_bvecs(path))
            elif file.endswith('.txt'):
                vectors.append(read_txt(path, sparse=sparse, dimension=dimension))
            else:
                vectors.append(read_txt(path, sparse=sparse, dimension=dimension))
        return np.concatenate(vectors, axis=0)

    _, ext = os.path.splitext(fname)
    ext = ext.lower()
    if ext == '.fvecs':
        return read_fvecs(fname)
    elif ext == '.ivecs':
        return read_ivecs(fname)
    elif ext == '.bvecs':
        return read_bvecs(fname)
    elif ext == '.txt':
        return read_txt(fname, sparse=sparse, dimension=dimension)
    else:
        return multi_file()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python utils.py <path_to_vector_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    try:
        vectors = load_vectors(file_path)
        print("Loaded vectors shape:", vectors.shape)
    except Exception as e:
        print("Error loading vectors:", e)


#______________________________________

# import numpy as np
# import os


# def read_fvecs(fname):
#     """
#     Load vectors from a .fvecs file.
#     Each vector block: [d (int32), d float32 values].
#     Returns: (num_vectors, d) float32 array.
#     """
#     a = np.fromfile(fname, dtype=np.int32)
#     if a.size == 0:
#         raise IOError(f"File is empty or cannot be read: {fname}")
#     d = a[0]
#     total_vectors = a.size // (d + 1)
#     a = a.reshape(total_vectors, d + 1)
#     return a[:, 1:].view(np.float32)


# def read_ivecs(fname):
#     """
#     Load vectors from an .ivecs file (int32 components).
#     Returns: (num_vectors, d) int32 array.
#     """
#     a = np.fromfile(fname, dtype=np.int32)
#     if a.size == 0:
#         raise IOError(f"File is empty or cannot be read: {fname}")
#     d = a[0]
#     total_vectors = a.size // (d + 1)
#     a = a.reshape(total_vectors, d + 1)
#     return a[:, 1:]


# def read_bvecs(fname):
#     """
#     Load vectors from a .bvecs file (uint8 components).
#     Returns: (num_vectors, d) uint8 array.
#     """
#     with open(fname, 'rb') as f:
#         content = f.read()
#     d = np.frombuffer(content[:4], dtype=np.int32)[0]
#     vector_byte_size = 4 + d
#     total_vectors = len(content) // vector_byte_size
#     data = np.empty((total_vectors, d), dtype=np.uint8)
#     for i in range(total_vectors):
#         start = i * vector_byte_size + 4
#         end = start + d
#         data[i, :] = np.frombuffer(content[start:end], dtype=np.uint8)
#     return data


# def reconstruct_dense_vector(vec_str, grid_size):
#     """
#     Build a dense binary vector from sparse index string.
#     """
#     dense_vec = np.zeros(grid_size, dtype=np.float32)
#     for idx in vec_str.replace(',', ' ').split():
#         dense_vec[int(idx)] = 1.0
#     return dense_vec


# def read_txt(fname, sparse=False, dimension=None):
#     """
#     Load vectors from a .txt file.
#     If sparse: each line is space/comma-separated indices of 1s (requires dimension).
#     If dense: attempts comma-delim then whitespace-delim float loading.
#     """
#     if sparse:
#         if dimension is None:
#             raise ValueError("Dimension must be specified for sparse vectors.")
#         vectors = []
#         with open(fname, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 vectors.append(reconstruct_dense_vector(line, dimension))
#         return np.array(vectors, dtype=np.float32)
#     else:
#         try:
#             return np.loadtxt(fname, dtype=np.float32, delimiter=',')
#         except ValueError:
#             return np.loadtxt(fname, dtype=np.float32)


# def load_vectors(fname, dimension=None, sparse=False):
#     """
#     Load vectors from a file or directory by extension:
#       .fvecs, .ivecs, .bvecs, .txt, or folder of these.
#     """
#     def multi_file():
#         if not os.path.isdir(fname):
#             raise IOError(f"Not a valid file or directory: {fname}")
#         vectors = []
#         for file in sorted(os.listdir(fname)):
#             path = os.path.join(fname, file)
#             if file.endswith('.fvecs'):
#                 vectors.append(read_fvecs(path))
#             elif file.endswith('.ivecs'):
#                 vectors.append(read_ivecs(path))
#             elif file.endswith('.bvecs'):
#                 vectors.append(read_bvecs(path))
#             elif file.endswith('.txt'):
#                 vectors.append(read_txt(path, sparse=sparse, dimension=dimension))
#             else:
#                 vectors.append(read_txt(path, sparse=sparse, dimension=dimension))
#         return np.concatenate(vectors, axis=0)

#     _, ext = os.path.splitext(fname)
#     ext = ext.lower()
#     if ext == '.fvecs':
#         return read_fvecs(fname)
#     elif ext == '.ivecs':
#         return read_ivecs(fname)
#     elif ext == '.bvecs':
#         return read_bvecs(fname)
#     elif ext == '.txt':
#         return read_txt(fname, sparse=sparse, dimension=dimension)
#     else:
#         return multi_file()


# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python utils.py <path_to_vector_file>")
#         sys.exit(1)
#     file_path = sys.argv[1]
#     try:
#         vectors = load_vectors(file_path)
#         print("Loaded vectors shape:", vectors.shape)
#     except Exception as e:
#         print("Error loading vectors:", e)

#_______________________________________

# import numpy as np
# import os


# def read_fvecs(fname):
#     """
#     Load vectors from a .fvecs file.
    
#     Each vector in the .fvecs file is stored as:
#        [d (int32), component_1 (float32), component_2 (float32), ..., component_d (float32)]
#     where d is the dimension of the vector.
    
#     Returns:
#         A numpy array of shape (num_vectors, d) with dtype=np.float32.
#     """
#     # Read the file as raw int32 data. The file is structured in blocks:
#     # first value in each block is the dimension 'd', then d float32 values follow.
#     a = np.fromfile(fname, dtype=np.int32)
#     if a.size == 0:
#         raise IOError("File is empty or cannot be read: %s" % fname)
    
#     # Get the vector dimension from the first 4 bytes.
#     d = a[0]
#     # Each vector is stored as (d + 1) numbers: one int32 for d and d numbers for the vector.
#     total_vectors = a.size // (d + 1)
    
#     # Reshape into (total_vectors, d+1) so that each row starts with the dimension.
#     a = a.reshape(total_vectors, d + 1)
    
#     # The remaining part (columns 1: end) stores the float32 components.
#     # We use view(np.float32) to reinterpret the bits as float32.
#     return a[:, 1:].view(np.float32)

# def read_ivecs(fname):
#     """
#     Load vectors from an .ivecs file.
    
#     Each vector in the .ivecs file is stored as:
#        [d (int32), component_1 (int32), component_2 (int32), ..., component_d (int32)]
    
#     Returns:
#         A numpy array of shape (num_vectors, d) with dtype=np.int32.
#     """
#     a = np.fromfile(fname, dtype=np.int32)
#     if a.size == 0:
#         raise IOError("File is empty or cannot be read: %s" % fname)
    
#     d = a[0]
#     total_vectors = a.size // (d + 1)
#     a = a.reshape(total_vectors, d + 1)
#     return a[:, 1:]

# def read_bvecs(fname):
#     """
#     Load vectors from a .bvecs file.
    
#     Each vector in the .bvecs file is stored as:
#        [d (int32), component_1 (uint8), component_2 (uint8), ..., component_d (uint8)]
    
#     Returns:
#         A numpy array of shape (num_vectors, d) with dtype=np.uint8.
#     """
#     with open(fname, 'rb') as f:
#         content = f.read()
    
#     # Get the dimension from the first 4 bytes.
#     d = np.frombuffer(content[:4], dtype=np.int32)[0]
#     vector_byte_size = 4 + d  # 4 bytes for dimension + d bytes for the vector.
#     total_vectors = len(content) // vector_byte_size
    
#     # Preallocate an array for efficiency.
#     data = np.empty((total_vectors, d), dtype=np.uint8)
#     for i in range(total_vectors):
#         start = i * vector_byte_size + 4  # Skip the 4 bytes (dimension) for each vector.
#         end = start + d
#         data[i, :] = np.frombuffer(content[start:end], dtype=np.uint8)
#     return data

# def reconstruct_dense_vector(vec_str, grid_size):
#     """
#     Reconstructs a dense binary vector from its sparse representation.
#     The sparse representation is assumed to be a string of space-separated indices
#     where the value is 1. All other positions (0 ... grid_size-1) are 0.
#     """
#     dense_vec = np.zeros(grid_size, dtype=int)
#     if vec_str:
#         # Replace commas with spaces and collapse multiple spaces if necessary.
#         vec_str = vec_str.replace(',', ' ')
#         # Optionally: you might use split() without replacing "  " explicitly,
#         # as split() will handle multiple whitespace characters by default.
#         for idx in vec_str.split():
#             dense_vec[int(idx)] = 1
#     return dense_vec

# def read_txt(fname, sparse, dimension):
#     """
#     Load vectors from a .txt file.
    
#     For dense vectors:
#         Each line is expected to be a space-separated list of float values.
#     For sparse vectors:
#         Each line should be a sparse representation (indices where value is 1).
    
#     Returns:
#         A numpy array of shape (num_vectors, d) with dtype=np.float32 for dense vectors or int for sparse.
#     """
#     if sparse and dimension is None:
#         raise ValueError("Dimension must be specified for sparse vectors.")
    
#     if sparse:
#         vectors = []  # Use a list to accumulate vectors.
#         with open(fname, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:  # Skip empty lines.
#                     continue
#                 # Process the current line rather than the file name.
#                 vectors.append(reconstruct_dense_vector(line, dimension))
#         return np.array(vectors)
#     else:
#         # If the file is space separated, you can omit the delimiter.
#         try: 
#             return np.loadtxt(fname, dtype=np.float32, delimiter=',')
#         except:
#             return np.loadtxt(fname, dtype=np.float32)

# def load_vectors(fname, dimension=None, sparse=False):
#     """
#     Load vectors from a given binary file.
    
#     The function will detect the file format based on the extension.
#       - .fvecs: returns float32 vectors.
#       - .ivecs: returns int32 vectors.
#       - .bvecs: returns uint8 vectors.
    
#     Parameters:
#         fname (str): Path to the vector file.
    
#     Returns:
#         A numpy array of shape (num_vectors, d) containing the vectors.
#     """
    
#     def multi_file():
#         if not os.path.isdir(fname):
#             raise IOError("Not a valid file or directory: %s" % fname)

#         vectors = []
#         for file in sorted(os.listdir(fname)):
#             if file.endswith('.fvecs'):
#                 vectors.append(read_fvecs(os.path.join(fname, file)))
#             elif file.endswith('.ivecs'):
#                 vectors.append(read_ivecs(os.path.join(fname, file)))
#             elif file.endswith('.bvecs'):
#                 vectors.append(read_bvecs(os.path.join(fname, file)))
#             elif file.endswith('.txt'):
#                 vectors.append(read_txt(os.path.join(fname, file), sparse, dimension))
#             else:
#                 vectors.append(read_txt(os.path.join(fname, file), sparse, dimension))
#         return np.concatenate(vectors, axis=0)
    
#     _, ext = os.path.splitext(fname)
#     ext = ext.lower()
#     if ext == '.fvecs':
#         return read_fvecs(fname)
#     elif ext == '.ivecs':
#         return read_ivecs(fname)
#     elif ext == '.bvecs':
#         return read_bvecs(fname)
#     elif ext == '.txt':
#         return read_txt(fname, dimension=dimension, sparse=sparse)
#     else:
#         return multi_file()
        

# if __name__ == '__main__':
#     # Example usage:
#     #   python utils.py path/to/sift_descriptors.fvecs
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python utils.py <path_to_vector_file>")
#         sys.exit(1)
    
#     file_path = sys.argv[1]
#     try:
#         vectors = load_vectors(file_path)
#         print("Loaded vectors shape:", vectors.shape)
#     except Exception as e:
#         print("Error loading vectors:", e)
