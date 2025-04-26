import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No such file: {image_path}")

    img = mpimg.imread(image_path)
    # Convert to grayscale if RGB
    if img.ndim == 3:
        # If image is in [0,255] range (e.g., uint8), normalize for float ops
        if img.dtype == np.uint8:
            img = img / 255.0
        gray = img @ [0.2989, 0.5870, 0.1140]  # Luminosity method
    else:
        gray = img  # Already grayscale
    return gray
def save_image(image: np.ndarray, path: str = "output/denoised.png"):
    """
    Save a grayscale image matrix to a given file path.

    Args:
        image (np.ndarray): Grayscale image (2D NumPy array).
        path (str): Full path (with filename) to save the image.
    """
    # Ensure image is uint8 type
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Create directory only if path has a folder
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # Save image
    success = cv2.imwrite(path, image)
    if success:
        print(f"Image saved to: {path}")
    else:
        raise IOError(f"Failed to save image to: {path}")

def plot_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

