import numpy as np
from .block_svd import Block_SVD
from ..utils.image_loader import load_image, save_image, plot_image
from ..utils.metrics import time_function, compute_snr


def extract_patches(image, patch_size, stride):
    """
    Extract overlapping patches from an image.
    :param image: Grayscale image (2D numpy array)
    :param patch_size: Size of square patches (p x p)
    :param stride: Step size for moving the patch window
    :return: Matrix Y where each column is a vectorized patch
    """
    patches = []
    positions = []
    for i in range(0, image.shape[0] - patch_size + 1, stride):
        for j in range(0, image.shape[1] - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size].flatten()
            patches.append(patch)
            positions.append((i, j))
    return np.array(patches).T, positions, image.shape

def omp(D, Y, sparsity):
    """
    Orthogonal Matching Pursuit (OMP) algorithm using Least Squares (LS).
    :param D: Dictionary matrix (atoms as columns)
    :param Y: Input signals (each column is a signal)
    :param sparsity: Desired sparsity level
    :return: Sparse coefficient matrix X
    """
    X = np.zeros((D.shape[1], Y.shape[1]))  # Initialize sparse coefficient matrix
    
    for i in range(Y.shape[1]):  # Process each signal (patch) separately
        residual = Y[:, i]  # Initial residual
        index_set = []  # Indices of selected atoms
        x_temp = np.zeros(D.shape[1])  # Sparse representation vector
        
        for _ in range(sparsity):
            projections = D.T @ residual  # Compute projections
            idx = np.argmax(np.abs(projections))  # Select best atom
            index_set.append(idx)  # Store selected atom index
            
            # Solve least squares problem using the selected atoms
            selected_atoms = D[:, index_set]
            x_subset, _, _, _ = np.linalg.lstsq(selected_atoms, Y[:, i], rcond=None)
            
            # Update residual
            residual = Y[:, i] - selected_atoms @ x_subset
            
            # Store coefficients in the correct indices
            x_temp[index_set] = x_subset
        
        X[:, i] = x_temp  # Store sparse representation for this patch
    
    return X

@time_function("K-SVD time taken")


def ksvd(IMAGE : str, patch_size = 8, stride = 8, sparsity = 5, max_iter=20):
    """
    K-SVD Algorithm for Dictionary Learning and Image Denoising.
    :param image: Input grayscale image
    :param patch_size: Size of square patches (p x p)
    :param stride: Step size for moving the patch window
    :param sparsity: Desired sparsity level
    :param max_iter: Maximum number of iterations
    :return: Denoised image
    """
    image = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
    Y, positions, image_shape = extract_patches(image, patch_size, stride)
    
    # Initialize dictionary with random normalized patches
    D = np.random.randn(patch_size**2, 2*patch_size**2)
    D /= np.linalg.norm(D, axis=0)
    
    for _ in range(max_iter):
        X = omp(D, Y, sparsity)
        
        for j in range(D.shape[1]):
            index_j = np.where(X[j, :] != 0)[0]
            if len(index_j) <= 1 :
                continue
            E_j = Y[:, index_j] - D @ X[:, index_j] + np.outer(D[:, j], X[j, index_j])
            #U, S, Vt = np.linalg.svd(E_j, full_matrices=False)
            U, S, Vt = Block_SVD(E_j, 1)
            D[:,j] = U[: ,0]
            X[j, index_j] = S[0]*Vt[0 ,:]
    
    # Image reconstruction
    reconstructed = np.zeros(image_shape)
    weight = np.zeros(image_shape)
    
    Y_reconstructed = D @ X  # Reconstruct patches
    
    patch_idx = 0
    for (i, j) in positions:
        patch = Y_reconstructed[:, patch_idx].reshape((patch_size, patch_size))
        reconstructed[i:i + patch_size, j:j + patch_size] += patch
        weight[i:i + patch_size, j:j + patch_size] += 1
        patch_idx += 1

    reconstructed /= (weight + 1e-8)  # Avoid division by zero
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def ksvd_image(orginal_image,path,save = True,patch_size=8,stride=8,sparsity=5,max_iter=20):
    org = load_image(orginal_image)
    im = ksvd(orginal_image,patch_size,stride,sparsity,max_iter)
    snr_value =compute_snr(org,im)
    print(f"SNR: {snr_value:.2f} dB")
    if save:
        save_image(im,path)
    plot_image(im)
	
