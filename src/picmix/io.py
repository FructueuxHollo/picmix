import numpy as np
from PIL import Image

def load_image_as_array(image_path: str) -> np.ndarray:
    """
    Charge une image à partir d'un chemin de fichier, la convertit en niveaux de gris
    et la retourne sous forme de tableau NumPy normalisé (valeurs entre 0.0 et 1.0).

    Args:
        image_path (str): Le chemin vers le fichier image.

    Returns:
        np.ndarray: Le tableau NumPy représentant l'image en niveaux de gris.
    """
    try:
        img = Image.open(image_path)
        # Convertir en niveaux de gris ('L' pour Luminance)
        img_gray = img.convert('L')
        # Convertir en tableau NumPy et normaliser
        img_array = np.array(img_gray, dtype=np.float64) / 255.0
        return img_array
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{image_path}' n'a pas été trouvé.")
        raise
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {e}")
        raise

def save_array_as_image(image_array: np.ndarray, output_path: str):
    """
    Sauvegarde un tableau NumPy en tant que fichier image.
    Le tableau est dé-normalisé (multiplié par 255) et converti en entiers 8 bits.

    Args:
        image_array (np.ndarray): Le tableau à sauvegarder.
        output_path (str): Le chemin où sauvegarder l'image.
    """
    try:
        # S'assurer que les valeurs sont bien entre 0 et 1 avant de convertir
        image_array = np.clip(image_array, 0.0, 1.0)
        
        # Dé-normaliser et convertir en entiers 8-bits non signés
        img_data = (image_array * 255).astype(np.uint8)
        
        # Créer une image PIL à partir du tableau
        img = Image.fromarray(img_data, 'L')
        
        # Sauvegarder l'image
        img.save(output_path)
        print(f"Image sauvegardée avec succès à l'emplacement : '{output_path}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image: {e}")
        raise