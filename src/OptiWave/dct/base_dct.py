import numpy as np
import cv2
from ..utils.image_loader import load_image
from ..utils.metrics import time_function


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the range [0, 1].
    Args:
        image (np.ndarray): Input grayscale image with pixel values in range [0, 255].
    Returns:
        np.ndarray: Normalized image with pixel values in range [0.0, 1.0].
    """
    return np.float32(image) / 255.0

def apply_dct(image: np.ndarray) -> np.ndarray:
    """
    Apply 2D Discrete Cosine Transform (DCT) to the input image.
    Args:
        image (np.ndarray): Normalized grayscale image (2D array).
    Returns:
        np.ndarray: DCT-transformed image (2D array).
    """
    return cv2.dct(image)


def apply_idct(dct_coeffs: np.ndarray) -> np.ndarray:
    """
    Apply Inverse 2D Discrete Cosine Transform (IDCT) to the DCT coefficients.
    Args:
        dct_coeffs (np.ndarray): DCT coefficients of the image (2D array).
    Returns:
        np.ndarray: IDCT-transformed image (2D array, float values in range ~[0.0, 1.0]).
    """
    return cv2.idct(dct_coeffs)


def reconstruct_image(normalized_image: np.ndarray) -> np.ndarray:
    """
    Reconstruct an image from normalized float format to standard 8-bit format.
    Args:
        normalized_image (np.ndarray): Normalized image with pixel values in range [0.0, 1.0].
    Returns:
        np.ndarray: Reconstructed 8-bit image with pixel values in range [0, 255].
    """
    return np.clip(normalized_image * 255.0, 0, 255).astype(np.uint8)


@time_function("DCT")
def dct_pipeline() -> np.ndarray:
    """
    Perform the full DCT transformation and reconstruction pipeline.

    This includes:
    - Loading the image using the external image_loader module.
    - Normalizing the image.
    - Applying DCT and IDCT.
    - Converting the image back to 8-bit format.

    Returns:
        np.ndarray: Final reconstructed 8-bit grayscale image.
    """
    image = load_image()  # Assumes grayscale image is returned as a NumPy 2D array

    normalized = normalize_image(image)
    dct_result = apply_dct(normalized)
    idct_result = apply_idct(dct_result)
    reconstructed = reconstruct_image(idct_result)

    return reconstructed
