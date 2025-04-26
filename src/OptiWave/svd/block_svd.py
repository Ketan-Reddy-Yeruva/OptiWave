import numpy as np
import time



def Block_SVD(A, s=None, tol=1e-6, max_iter=50, power_iter=1):
    
    n, m = A.shape
    if s is None:
        s = min(n, m)
    
    # Initialize with random matrix and one QR
    V = np.random.randn(m, s)
    V, _ = np.linalg.qr(V)
    
    # Preallocate memory
    AV = np.empty((n, s))
    AtU = np.empty((m, s))
    prev_sig = np.zeros(s)
    
    for _ in range(max_iter):
        # Compute A @ V with power iterations
        np.matmul(A, V, out=AV)
        for _ in range(power_iter - 1):
            np.matmul(A, np.matmul(A.T, AV, out=AtU), out=AV)
        
        # Orthogonalize U
        U, R1 = np.linalg.qr(AV, mode='reduced')
        
        # Compute A.T @ U with power iterations
        np.matmul(A.T, U, out=AtU)
        for _ in range(power_iter - 1):
            np.matmul(A.T, np.matmul(A, AtU, out=AV), out=AtU)
        
        # Orthogonalize V and get singular values
        V, R2 = np.linalg.qr(AtU, mode='reduced')
        sig = np.abs(np.diag(R2)[:s])
        
        # Early termination check
        err = np.linalg.norm(sig - prev_sig) / np.linalg.norm(sig)
        if err < tol:
            break
        prev_sig = sig
    
    # Final computation of singular values
    np.matmul(A, V, out=AV)
    sig = np.linalg.norm(AV, axis=0)
    
    # Sort results
    idx = np.argsort(sig)[::-1]
    sig = sig[idx]
    U = U[:, idx]
    V = V[:, idx]
    
    return U, sig, V.T



# Example usage (not reqd)
if __name__ == "__main__":
    import time
    import numpy as np

    A = np.random.randn(1000, 800)
    s = 150

    # Block power SVD
    a = time.time()
    U, sig, Vt = Block_SVD(A, s)
    b = time.time()
    print("Block Power SVD time:", b - a)

    # Full NumPy SVD
    c = time.time()
    U_n, sig_n, Vt_n = np.linalg.svd(A, full_matrices=False)
    d = time.time()
    print("NumPy SVD time:", d - c)

    # Compare singular values
    rel_sig_err = np.linalg.norm(sig - sig_n[:s]) / np.linalg.norm(sig_n[:s])
    print("Relative error in singular values:", rel_sig_err)

    # Compare Vt subspace error
    proj_err = np.linalg.norm(Vt @ Vt.T - Vt_n[:s] @ Vt_n[:s].T)
    print("Vt subspace projection error:", proj_err)

    
    
