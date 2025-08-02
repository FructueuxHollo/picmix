__version__ = "1.0.0"

import numpy as np
import os
from typing import Dict, Any

# Import the core engine and IO functions
from .engine import VortexCryptEngine
from .io import (load_image_as_array, save_array_as_image, 
               save_state_to_npz, load_state_from_npz)

def encrypt(
    image_path: str,
    output_path_npz: str,
    key: str,
    config: Dict[str, Any] = None,
    save_preview: bool = True
):
    """
    Encrypts an image using the VortexCrypt algorithm.

    The primary output is a .npz file containing the full encrypted state {u(T), v(T)}.
    A visual preview of the encrypted image (u-field) can also be saved.

    Args:
        image_path (str): Path to the source image.
        output_path_npz (str): Path to save the output .npz data file.
        key (str): Secret key (8-24 characters).
        config (Dict, optional): Dictionary to override simulation parameters.
        save_preview (bool): If True, saves a .png preview of the encrypted image.
    """
    print("--- VortexCrypt Encryption ---")
    original_array = load_image_as_array(image_path)
    
    engine = VortexCryptEngine(key=key, image_shape=original_array.shape, config=config)
    
    u_final_flat, v_final_flat = engine.encrypt(original_array)
    
    save_state_to_npz(
        output_path_npz, 
        u=u_final_flat, 
        v=v_final_flat, 
        shape=engine.padded_shape
    )
    
    if save_preview:
        u_final_2d = u_final_flat.reshape(engine.padded_shape)
        pad = engine.config['pad_width']
        preview_path = os.path.splitext(output_path_npz)[0] + ".png"
        save_array_as_image(u_final_2d[pad:-pad, pad:-pad], preview_path)
    
    print("--- Encryption Complete ---")

def decrypt(
    encrypted_state_path_npz: str,
    output_path_png: str,
    key: str,
    config: Dict[str, Any] = None
):
    """
    Decrypts an image from a .npz state file using the VortexCrypt algorithm.

    Args:
        encrypted_state_path_npz (str): Path to the encrypted .npz data file.
        output_path_png (str): Path to save the decrypted .png image.
        key (str): The SAME secret key used for encryption.
        config (Dict, optional): The SAME config dictionary, if used for encryption.
    """
    print("--- VortexCrypt Decryption ---")
    u_final_flat, v_final_flat, padded_shape = load_state_from_npz(encrypted_state_path_npz)
    
    pad = config.get('pad_width', 1) if config else 1
    original_shape = (padded_shape[0] - 2 * pad, padded_shape[1] - 2 * pad)
    
    engine = VortexCryptEngine(key=key, image_shape=original_shape, config=config)
    
    decrypted_image = engine.decrypt(u_final_flat, v_final_flat)
    
    save_array_as_image(decrypted_image, output_path_png)
    print("--- Decryption Complete ---")