import numpy as np
import cv2
import struct
import math
import utils  # Assuming utils.py contains the read_fvecs function

### ------------------------
### UTILITY FUNCTIONS
### ------------------------

def gaussian_kernel(sigma, kernel_size=None):
    """Generate a 1D Gaussian kernel."""
    if kernel_size is None:
        kernel_size = int(6 * sigma) + 1  # ensure odd size
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    return kernel

def fix_descriptor_dimension(descriptor, target_dim=512):
    """
    Force the descriptor vector to be exactly target_dim in length.
    If shorter, pad with zeros; if longer, truncate.
    """
    descriptor = np.asarray(descriptor, dtype=np.float32)
    current_dim = descriptor.size
    if current_dim < target_dim:
        pad = np.zeros(target_dim - current_dim, dtype=np.float32)
        fixed = np.concatenate([descriptor, pad])
    else:
        fixed = descriptor[:target_dim]
    return fixed

### ------------------------
### FOURIER DESCRIPTOR (FD)
### ------------------------
def compute_fourier_descriptor(contour, n_coeffs=256):
    """
    Compute a Fourier descriptor for the given contour.
    
    The contour is converted into a complex-valued 1D signal and its FFT is computed.
    The DC component is removed, and then the first n_coeffs coefficients (ignoring DC)
    are concatenated (real and imaginary parts), yielding a vector of 2*n_coeffs dimensions.
    """
    contour = contour.squeeze()
    if len(contour.shape) == 1:
        contour = contour[np.newaxis, :]
    
    # Represent (x,y) as a complex number.
    complex_contour = np.empty(contour.shape[0], dtype=complex)
    complex_contour.real = contour[:, 0]
    complex_contour.imag = contour[:, 1]
    
    # Compute FFT.
    fourier_result = np.fft.fft(complex_contour)
    # Remove the DC component.
    fourier_result[0] = 0
    # Normalize for scale invariance using the first nonzero coefficient.
    norm = np.abs(fourier_result[1]) if len(fourier_result) > 1 else 1
    if norm != 0:
        fourier_result /= norm
    # Ensure we have at least n_coeffs+1 coefficients, pad if needed.
    if len(fourier_result) < n_coeffs + 1:
        padded = np.zeros(n_coeffs + 1, dtype=fourier_result.dtype)
        padded[:len(fourier_result)] = fourier_result
        fourier_result = padded
    # Select coefficients 1 to n_coeffs (skip DC).
    descriptor = fourier_result[1:n_coeffs+1]
    feature_vector = np.concatenate([descriptor.real, descriptor.imag])
    return feature_vector

### ------------------------
### CSS DESCRIPTOR
### ------------------------
def compute_css_descriptor(contour, css_scales=None, sample_length=32):
    """
    Compute a Curvature Scale Space (CSS) descriptor for the given contour.
    
    For each smoothing scale, the contour is smoothed using a Gaussian filter applied to x and y,
    derivatives are computed to estimate curvature, and then the curvature signal is uniformly resampled.
    By default (if css_scales is None), scales 1 to 16 are used yielding a raw descriptor of length 16*32 = 512.
    """
    if css_scales is None:
        css_scales = list(range(1, 17))  # scales 1 through 16
    contour = contour.squeeze()
    if len(contour.shape) == 1:
        contour = contour[np.newaxis, :]
    x = contour[:, 0].astype(np.float32)
    y = contour[:, 1].astype(np.float32)
    css_signals = []
    for sigma in css_scales:
        kernel_size = int(6 * sigma) + 1
        kernel = gaussian_kernel(sigma, kernel_size)
        smooth_x = np.convolve(x, kernel, mode='same')
        smooth_y = np.convolve(y, kernel, mode='same')
        dx = np.gradient(smooth_x)
        dy = np.gradient(smooth_y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        eps = 1e-8
        curvature = (dx * ddy - dy * ddx) / (np.power(dx**2 + dy**2, 1.5) + eps)
        # Resample curvature to sample_length points.
        indices = np.linspace(0, len(curvature) - 1, sample_length)
        sampled = np.interp(indices, np.arange(len(curvature)), curvature)
        css_signals.append(sampled)
    css_descriptor = np.concatenate(css_signals)
    return css_descriptor

### ------------------------
### EDGE HISTOGRAM DESCRIPTOR (EHD)
### ------------------------
def compute_ehd_descriptor(image, num_blocks=(4, 4)):
    """
    Compute an Edge Histogram Descriptor (EHD) for the binary image.
    
    Five edge filters (vertical, horizontal, 45° diagonal, 135° diagonal, and non-directional)
    are applied, and the image is subdivided into blocks. For each block and each edge type,
    the sum of absolute filter responses is computed. The responses are then concatenated and normalized.
    """
    edge_kernels = {
        'vertical': np.array([[-1, 2, -1],
                              [-1, 2, -1],
                              [-1, 2, -1]], dtype=np.float32),
        'horizontal': np.array([[-1, -1, -1],
                                [2, 2, 2],
                                [-1, -1, -1]], dtype=np.float32),
        'diagonal_45': np.array([[2, 2, -1],
                                 [2, -1, -1],
                                 [-1, -1, -1]], dtype=np.float32),
        'diagonal_135': np.array([[-1, 2, 2],
                                  [-1, -1, 2],
                                  [-1, -1, -1]], dtype=np.float32),
        'non_directional': np.array([[1, 1, 1],
                                     [1, -8, 1],
                                     [1, 1, 1]], dtype=np.float32)
    }
    descriptor = []
    h, w = image.shape
    blk_h = h // num_blocks[0]
    blk_w = w // num_blocks[1]
    for key, kernel in edge_kernels.items():
        response = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
        response = np.abs(response)
        # Sum filter responses in each block.
        for i in range(num_blocks[0]):
            for j in range(num_blocks[1]):
                block = response[i * blk_h:(i + 1) * blk_h, j * blk_w:(j + 1) * blk_w]
                descriptor.append(np.sum(block))
    ehd_descriptor = np.array(descriptor, dtype=np.float32)
    norm = np.linalg.norm(ehd_descriptor)
    if norm > 0:
        ehd_descriptor = ehd_descriptor / norm
    return ehd_descriptor

### ------------------------
### HOG DESCRIPTOR (Histogram of Oriented Gradients)
### ------------------------
def compute_hog_descriptor(image, cell_size=(21, 21), nbins=9):
    """
    Compute a Histogram of Oriented Gradients (HOG) descriptor.
    
    The image gradients are computed via the Sobel operator; then the image is divided into cells.
    For each cell, a histogram of gradient orientations (weighted by magnitude) is computed.
    All cell histograms are concatenated to form the descriptor.
    """
    image = image.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = (np.arctan2(gy, gx) * (180 / np.pi)) % 180  # angles in [0, 180)
    
    h, w = image.shape
    cell_h, cell_w = cell_size
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w
    hog_descriptor = []
    bin_width = 180.0 / nbins
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_angle = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            hist = np.zeros(nbins, dtype=np.float32)
            for m in range(cell_mag.shape[0]):
                for n in range(cell_mag.shape[1]):
                    a = cell_angle[m, n]
                    mag_val = cell_mag[m, n]
                    bin_idx = int(a // bin_width) % nbins
                    hist[bin_idx] += mag_val
            hog_descriptor.extend(hist)
    hog_descriptor = np.array(hog_descriptor, dtype=np.float32)
    return hog_descriptor

### ------------------------
### SIFT DESCRIPTOR
### ------------------------
def compute_sift_descriptor(image):
    """
    Compute a SIFT descriptor by detecting keypoints and their descriptors on the image.
    
    The image is pre-processed (Gaussian blur) to enhance gradient information.
    The SIFT descriptors (128-d each) are aggregated using mean pooling to produce a single descriptor.
    If no keypoints are detected, a zero vector of length 128 is returned.
    """
    # Preprocess image with a slight blur.
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=1)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(blurred, None)
    if descriptors is None or len(descriptors)==0:
        return np.zeros(128, dtype=np.float32)
    # Simple aggregation: mean of all descriptor vectors.
    aggregated = np.mean(descriptors, axis=0)
    return aggregated

### ------------------------
### GIST DESCRIPTOR
### ------------------------
def compute_gist_descriptor(image, num_blocks=(4, 4), orientations_per_scale=[8, 8, 8, 8]):
    """
    Compute a simplified GIST descriptor for the image.
    
    A bank of Gabor filters is applied at multiple scales and orientations.
    The absolute responses are pooled over a spatial grid defined by num_blocks.
    The responses from all filters are concatenated.
    By default (with orientations_per_scale=[8,8,8,8] and num_blocks=(4,4)), the output is 512-dimensional.
    """
    image = image.astype(np.float32)
    h, w = image.shape
    responses = []
    # For each scale and each orientation, construct a Gabor filter and convolve.
    n_scales = len(orientations_per_scale)
    for scale in range(n_scales):
        num_orientations = orientations_per_scale[scale]
        sigma = 0.5 * (scale+1) * min(h, w) / 100  # example: scale-dependent sigma
        lamda = np.pi / 4  # wavelength
        gamma = 0.5       # spatial aspect ratio
        psi = 0
        ksize = int(6 * sigma + 1)
        for o in range(num_orientations):
            theta = o * math.pi / num_orientations
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(np.abs(filtered))
    # Now, pool the responses over a grid.
    gist_descriptor = []
    blk_h = h // num_blocks[0]
    blk_w = w // num_blocks[1]
    for response in responses:
        for i in range(num_blocks[0]):
            for j in range(num_blocks[1]):
                block = response[i*blk_h:(i+1)*blk_h, j*blk_w:(j+1)*blk_w]
                gist_descriptor.append(np.mean(block))
    gist_descriptor = np.array(gist_descriptor, dtype=np.float32)
    return gist_descriptor

### ------------------------
### PROCESSING A SINGLE FLAT VECTOR
### ------------------------
def process_flat_vector(flat_array, shape, 
                        n_coeffs=256,  # For Fourier
                        css_scales=None, css_sample_length=32,  # For CSS
                        ehd_blocks=(4,4),  # For EHD
                        hog_params={'cell_size': (21,21), 'nbins': 9}):
    """
    Process one flat vector (1D array representing a 2D grid) to compute six descriptors:
      - Fourier Descriptor (forced to 512 dimensions)
      - CSS Descriptor (forced to 512 dimensions)
      - EHD Descriptor (forced to 512 dimensions)
      - HOG Descriptor (forced to 512 dimensions)
      - SIFT Descriptor (aggregated, natural dimension e.g. 128)
      - GIST Descriptor (using default parameters; typically 512-d with these defaults)
    """
    # Reshape flat vector into a 2D matrix and create binary image.
    matrix = np.array(flat_array, dtype=np.uint8).reshape(shape)
    img = matrix * 255  # binary image
    
    # Extract contour for Fourier and CSS descriptors.
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No contours found in the image.")
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute shape descriptors.
    fourier_desc = fix_descriptor_dimension(compute_fourier_descriptor(largest_contour, n_coeffs=n_coeffs), target_dim=512)
    css_desc = fix_descriptor_dimension(compute_css_descriptor(largest_contour, css_scales=css_scales, sample_length=css_sample_length), target_dim=512)
    ehd_desc = fix_descriptor_dimension(compute_ehd_descriptor(img, num_blocks=ehd_blocks), target_dim=512)
    hog_desc = fix_descriptor_dimension(compute_hog_descriptor(img, cell_size=hog_params.get('cell_size', (21,21)), nbins=hog_params.get('nbins', 9)), target_dim=512)
    
    # Compute texture/gradient-based descriptors (without forcing dimension).
    sift_desc = compute_sift_descriptor(img)
    gist_desc = compute_gist_descriptor(img, num_blocks=(4,4), orientations_per_scale=[8,8,8,8])
    
    return fourier_desc, css_desc, ehd_desc, hog_desc, sift_desc, gist_desc

### ------------------------
### PROCESSING A BIG 2D ARRAY OF FLAT VECTORS
### ------------------------
def process_flat_array_matrix(flat_matrix, shape, 
                              n_coeffs=256,
                              css_scales=None, css_sample_length=32, ehd_blocks=(4,4),
                              hog_params={'cell_size': (21,21), 'nbins': 9}):
    """
    Process a big 2D array where each row is a flat vector.
    Returns six lists of descriptors (one per descriptor type):
      - Fourier, CSS, EHD, HOG descriptors are forced to 512 dimensions.
      - SIFT and GIST descriptors are output in their natural dimensions.
    """
    fourier_list, css_list, ehd_list, hog_list, sift_list, gist_list = [], [], [], [], [], []
    for idx, flat_array in enumerate(flat_matrix):
        try:
            f_desc, c_desc, e_desc, h_desc, s_desc, g_desc = process_flat_vector(
                flat_array, shape,
                n_coeffs=n_coeffs,
                css_scales=css_scales,
                css_sample_length=css_sample_length,
                ehd_blocks=ehd_blocks,
                hog_params=hog_params)
            fourier_list.append(f_desc)
            css_list.append(c_desc)
            ehd_list.append(e_desc)
            hog_list.append(h_desc)
            sift_list.append(s_desc)
            gist_list.append(g_desc)
        except ValueError as e:
            print(f"Warning: Skipping vector {idx} due to error: {e}")
            fourier_list.append(np.zeros(512, dtype=np.float32))
            css_list.append(np.zeros(512, dtype=np.float32))
            ehd_list.append(np.zeros(512, dtype=np.float32))
            hog_list.append(np.zeros(512, dtype=np.float32))
            sift_list.append(np.zeros(128, dtype=np.float32))
            gist_list.append(np.zeros(512, dtype=np.float32))
    return fourier_list, css_list, ehd_list, hog_list, sift_list, gist_list

### ------------------------
### WRITE .fvecs FILE FUNCTION
### ------------------------
def write_fvecs(filename, descriptors):
    """
    Write a list of descriptors to an .fvecs file.
    
    Each descriptor is written as:
      - A 4-byte little-endian integer for the dimension.
      - Followed by that many 4-byte little-endian floats.
    """
    with open(filename, "wb") as f:
        for vec in descriptors:
            vec = np.asarray(vec, dtype=np.float32)
            d = vec.size
            f.write(struct.pack("<i", d))
            f.write(vec.tobytes())

### ------------------------
### MAIN EXECUTION
### ------------------------
if __name__ == '__main__':

    query_vectors = utils.load_vectors("/home/otto/uni_filtered_pk-5e-06-5e-05-147-147", dimension=21609, sparse=True)
    # Combine the sample flat vectors into one "big" 2D array.
    flat_matrix = query_vectors

    # Define the grid shape for each sample.
    grid_shape = (147, 147)

    # Process all the flat vectors.
    fourier_descs, css_descs, ehd_descs, hog_descs, sift_descs, gist_descs = process_flat_array_matrix(
        flat_matrix, grid_shape,
        n_coeffs=256,  # for Fourier to produce raw 512-d vectors
        css_scales=None,  # defaults to scales 1..16 for CSS (512-d)
        css_sample_length=32,
        ehd_blocks=(4,4),
        hog_params={'cell_size': (21,21), 'nbins': 9}
    )
    
    # Write the descriptors to separate .fvecs files.
    write_fvecs("fourier_descriptors.fvecs", fourier_descs)
    write_fvecs("css_descriptors.fvecs", css_descs)
    write_fvecs("ehd_descriptors.fvecs", ehd_descs)
    write_fvecs("hog_descriptors.fvecs", hog_descs)
    write_fvecs("sift_descriptors.fvecs", sift_descs)   # SIFT is not forced to 512-d (typically 128-d)
    write_fvecs("gist_descriptors.fvecs", gist_descs)   # GIST here is computed with default parameters
    
    # Print out some information.
    print("Fourier Descriptors:")
    for i, d in enumerate(fourier_descs):
        print(f"  Sample {i}: dimension = {d.size}")
    print("\nCSS Descriptors:")
    for i, d in enumerate(css_descs):
        print(f"  Sample {i}: dimension = {d.size}")
    print("\nEHD Descriptors:")
    for i, d in enumerate(ehd_descs):
        print(f"  Sample {i}: dimension = {d.size}")
    print("\nHOG Descriptors:")
    for i, d in enumerate(hog_descs):
        print(f"  Sample {i}: dimension = {d.size}")
    print("\nSIFT Descriptors:")
    for i, d in enumerate(sift_descs):
        print(f"  Sample {i}: dimension = {d.size}")
    print("\nGIST Descriptors:")
    for i, d in enumerate(gist_descs):
        print(f"  Sample {i}: dimension = {d.size}")
    
    print("\nAll descriptors have been written to their respective .fvecs files.")