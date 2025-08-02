import numpy as np
from PIL import Image
import os

def load_image_as_array(image_path: str) -> np.ndarray:
    """Loads an image, converts to grayscale, and normalizes to [0.0, 1.0]."""
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            return np.array(img_gray, dtype=np.float64) / 255.0
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Input file '{image_path}' not found.")
    except Exception as e:
        raise IOError(f"Error loading image: {e}")

def save_array_as_image(image_array: np.ndarray, output_path: str):
    """Saves a NumPy array as a grayscale image, clipping values to [0, 1]."""
    try:
        # Clip values to the valid range [0, 1] before scaling
        clipped_array = np.clip(image_array, 0.0, 1.0)
        img_data = (clipped_array * 255).astype(np.uint8)
        Image.fromarray(img_data, 'L').save(output_path)
        print(f"Image successfully saved to '{output_path}'")
    except Exception as e:
        raise IOError(f"Error saving image: {e}")

def save_state_to_npz(output_path: str, u: np.ndarray, v: np.ndarray, shape: tuple):
    """Saves the encrypted u and v states and original shape to a compressed .npz file."""
    try:
        if not output_path.endswith('.npz'):
            output_path += '.npz'
        np.savez_compressed(output_path, u=u, v=v, shape=shape)
        print(f"Encrypted state saved to '{output_path}'")
    except Exception as e:
        raise IOError(f"Error saving .npz data: {e}")

def load_state_from_npz(input_path: str) -> tuple:
    """Loads the u, v, and shape states from an .npz file."""
    try:
        if not input_path.endswith('.npz'):
            input_path += '.npz'
        with np.load(input_path) as data:
            u = data['u']
            v = data['v']
            shape = data['shape']
        print(f"Encrypted state loaded from '{input_path}'")
        return u, v, tuple(shape)
    except FileNotFoundError:
        raise FileNotFoundError(f"Encrypted data file not found: '{input_path}'")
    except Exception as e:
        raise IOError(f"Error loading .npz data: {e}")