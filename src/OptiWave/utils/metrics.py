from time import perf_counter
import numpy as np

def time_function(name: str):
    """
    Decorator to measure the execution time of a function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = perf_counter()
            result = func(*args, **kwargs)
            end = perf_counter()
            print(f"[{name}] Time taken: {end - start:.4f} seconds")
            return result
        return wrapper
    return decorator

def compute_snr(original: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        original (np.ndarray): Original image (clean signal).
        noisy (np.ndarray): Noisy or reconstructed image.

    Returns:
        float: SNR in decibels.
    """
    noise = original - noisy
    snr = 10 * np.log10(np.sum(original ** 2) / (np.sum(noise ** 2) + 1e-10))
    return snr
    


