# run_test.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import picmix
import numpy as np
from PIL import Image

def compare_images(path1, path2):
    """Compare deux images et retourne leur Mean Squared Error (MSE)."""
    img1 = np.array(Image.open(path1).convert('L'), dtype=np.float64)
    img2 = np.array(Image.open(path2).convert('L'), dtype=np.float64)
    
    if img1.shape != img2.shape:
        print("Les images n'ont pas la même taille !")
        return float('inf')
        
    mse = np.mean((img1 - img2) ** 2)
    return mse

def main():
    print(f"Test du package PicMix version {picmix.__version__}")

    # --- Paramètres du test ---
    CLE_SECRETE = "fluid-dynamics-are-cool-42"
    IMAGE_ORIGINALE = "test_image.png"
    IMAGE_CHIFFREE = "encrypted_output.png"
    IMAGE_DECHIFFREE = "decrypted_output.png"
    
    # Paramètres de simulation (pour un test plus rapide)
    # Plus time_steps est élevé, plus le "mélange" est fort.
    # Pour un premier test, gardons des valeurs basses pour que ce soit rapide.
    simulation_config = {
        'time_steps': 20, # Réduit pour un test rapide
        'dt': 0.1
    }
    
    # Vérifier que l'image de test existe
    if not os.path.exists(IMAGE_ORIGINALE):
        print(f"ERREUR: L'image de test '{IMAGE_ORIGINALE}' n'a pas été trouvée à la racine du projet.")
        return

    # --- ÉTAPE 1: Chiffrement ---
    print("\n" + "="*20 + " DÉBUT DU CHIFFREMENT " + "="*20)
    try:
        picmix.encrypt(
            image_path=IMAGE_ORIGINALE,
            output_path=IMAGE_CHIFFREE,
            key=CLE_SECRETE,
            config=simulation_config
        )
        print("Chiffrement terminé avec succès.")
    except Exception as e:
        print(f"Une erreur est survenue pendant le chiffrement : {e}")
        return

    # --- ÉTAPE 2: Déchiffrement ---
    print("\n" + "="*20 + " DÉBUT DU DÉCHIFFREMENT " + "="*20)
    try:
        picmix.decrypt(
            encrypted_image_path=IMAGE_CHIFFREE,
            output_path=IMAGE_DECHIFFREE,
            key=CLE_SECRETE,
            config=simulation_config
        )
        print("Déchiffrement terminé avec succès.")
    except Exception as e:
        print(f"Une erreur est survenue pendant le déchiffrement : {e}")
        return

    # --- ÉTAPE 3: Vérification ---
    print("\n" + "="*20 + " VÉRIFICATION DES RÉSULTATS " + "="*20)
    
    # Comparaison de l'image originale et de l'image chiffrée
    # Elles devraient être très différentes.
    mse_encrypt = compare_images(IMAGE_ORIGINALE, IMAGE_CHIFFREE)
    print(f"Différence (MSE) entre l'original et la chiffrée : {mse_encrypt:.2f}")
    if mse_encrypt > 100: # Seuil arbitraire pour dire que l'image a bien été modifiée
        print(" -> SUCCÈS : L'image chiffrée est bien différente de l'originale.")
    else:
        print(" -> AVERTISSEMENT : L'image chiffrée est très similaire à l'originale. Augmentez 'time_steps' ?")
        
    # Comparaison de l'image originale et de l'image déchiffrée
    # Elles devraient être quasi identiques !
    mse_decrypt = compare_images(IMAGE_ORIGINALE, IMAGE_DECHIFFREE)
    print(f"Différence (MSE) entre l'original et la déchiffrée : {mse_decrypt:.4f}")
    if mse_decrypt < 1.0: # L'erreur devrait être très faible due aux arrondis numériques
        print(" -> SUCCÈS : L'image a été restaurée avec succès !")
    else:
        print(" -> ÉCHEC : L'image restaurée est différente de l'originale. Le processus n'est pas réversible.")

if __name__ == "__main__":
    main()