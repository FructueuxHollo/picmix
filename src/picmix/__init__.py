__version__ = "0.1.0"

from .engine import PicMixEngine
from .io import load_image_as_array, save_array_as_image
from typing import Dict, Any

def encrypt(
    image_path: str,
    output_path: str,
    key: str,
    config: Dict[str, Any] = None
):
    """
    Chiffre une image en utilisant l'algorithme PicMix.

    Args:
        image_path (str): Chemin vers l'image d'origine.
        output_path (str): Chemin pour sauvegarder l'image chiffrée.
        key (str): Clé secrète pour le chiffrement. Doit être identique pour le déchiffrement.
        config (Dict, optional): Dictionnaire pour surcharger les paramètres de simulation
                                 par défaut (ex: {'time_steps': 100}).
    """
    print(f"Chargement de l'image '{image_path}'...")
    original_image_array = load_image_as_array(image_path)
    
    # Initialisation du moteur de simulation
    engine = PicMixEngine(image=original_image_array, key=key, config=config)
    
    # Exécution du chiffrement
    encrypted_image_array = engine.run_encryption()
    
    # Sauvegarde du résultat
    save_array_as_image(encrypted_image_array, output_path)
    print("Chiffrement terminé.")


def decrypt(
    encrypted_image_path: str,
    output_path: str,
    key: str,
    config: Dict[str, Any] = None
):
    """
    Déchiffre une image chiffrée avec l'algorithme PicMix.

    Args:
        encrypted_image_path (str): Chemin vers l'image chiffrée.
        output_path (str): Chemin pour sauvegarder l'image déchiffrée.
        key (str): Clé secrète utilisée pour le chiffrement.
        config (Dict, optional): Dictionnaire pour surcharger les paramètres. Doit être
                                 identique à celui utilisé pour le chiffrement.
    """
    print(f"Chargement de l'image chiffrée '{encrypted_image_path}'...")
    encrypted_image_array = load_image_as_array(encrypted_image_path)
    
    # Initialisation du moteur avec l'image chiffrée
    engine = PicMixEngine(image=encrypted_image_array, key=key, config=config)
    
    # Exécution du déchiffrement
    decrypted_image_array = engine.run_decryption()
    
    # Sauvegarde du résultat
    save_array_as_image(decrypted_image_array, output_path)
    print("Déchiffrement terminé.")