import picmix
import os
import numpy as np
from PIL import Image

def load_image_as_array(image_path: str) -> np.ndarray:
    """
    Charge une image, la convertit en niveaux de gris et la retourne
    sous forme de tableau NumPy normalisé (0.0 à 1.0).
    """
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            return np.array(img_gray, dtype=np.float64) / 255.0
    except FileNotFoundError:
        raise FileNotFoundError(f"Erreur: Le fichier '{image_path}' n'a pas été trouvé.")
    except Exception as e:
        raise IOError(f"Erreur lors du chargement de l'image: {e}")

SECRET_PASSPHRASE = "mon_secret_passphrase"
output_dir = "crypto_files"
os.makedirs(output_dir, exist_ok=True)

original_image_path = "test_image.png"
# Le fichier de données est maintenant un .npz
encrypted_data_path = os.path.join(output_dir, "encrypted_data.npz")
# Ajout d'un chemin pour la preview
preview_image_path = os.path.join(output_dir, "encrypted_preview.png") 
decrypted_image_path = os.path.join(output_dir, "decrypted_final.png")

# --- 1. Chiffrement ---
print(f"Chiffrement de '{original_image_path}'...")
img_array = load_image_as_array(original_image_path)
parameters = picmix._derive_parameters("mon_secret_passphrase",img_shape=img_array.shape) # Assurez-vous que cette fonction est définie dans picmix
# picmix.encrypt(
#     image_path=original_image_path,
#     key_string=SECRET_PASSPHRASE,
#     output_data_path=encrypted_data_path,
#     preview_path=preview_image_path
# )
# print("\n" + "="*50 + "\n")

# # --- 2. Déchiffrement ---
# print(f"Déchiffrement de '{encrypted_data_path}'...")
# picmix.decrypt(
#     input_data_path=encrypted_data_path,
#     key_string=SECRET_PASSPHRASE,
#     decrypted_output_path=decrypted_image_path
# )

# print(f"\nProcessus terminé. L'image déchiffrée est : '{decrypted_image_path}'")