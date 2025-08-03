import json
import numpy as np
from PIL import Image
from typing import Dict, Any


def load_image_as_array(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    Loads an image and normalizes it to [0.0, 1.0].
    
    Args:
        image_path (str): Path to the source image.
        grayscale (bool): If True, converts the image to grayscale. 
                          Otherwise, keeps it as RGB.

    Returns:
        np.ndarray: The normalized image array (H, W) or (H, W, C).
    """
    try:
        with Image.open(image_path) as img:
            if grayscale:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            return np.array(img, dtype=np.float64) / 255.0
    except Exception as e:
        raise IOError(f"Error loading image '{image_path}': {e}")

def save_array_as_image(image_array: np.ndarray, output_path: str):
    """
    Saves a NumPy array as an image. Handles both grayscale and RGB.
    Values are clipped to [0, 1] then scaled to [0, 255].
    """
    try:
        if image_array.ndim not in [2, 3]:
            raise ValueError("Input array must be 2D (grayscale) or 3D (RGB).")

        clipped_array = np.clip(image_array, 0.0, 1.0)
        img_data = (clipped_array * 255).astype(np.uint8)
        
        mode = 'L' if image_array.ndim == 2 else 'RGB'
        Image.fromarray(img_data, mode).save(output_path)
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

def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.

    Args:
        config_path (str): The path to the .json configuration file.

    Returns:
        Dict[str, Any]: A dictionary with the loaded parameters.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from '{config_path}'")
        return config if config else {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: '{config_path}'")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from '{config_path}'. Check for syntax errors.")
    except Exception as e:
        raise IOError(f"Error reading config file: {e}")