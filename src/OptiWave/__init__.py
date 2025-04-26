"""
SVD and Dictionary Learning Package

This package provides:
1. Block_SVD - Randomized block SVD computation
2. svd_bpso - Image compression using SVD and PSO optimization
3. ksvd - Dictionary learning for image denoising
"""
from .svd import Block_SVD, base_svd
from .hybrid import denoise_compress, ksvd
from .bpso import svd_bpso

__all__ = ['Block_SVD', 'svd_bpso', 'ksvd', 'denoise_compress', 'base_svd']
__version__ = '0.1.0'
