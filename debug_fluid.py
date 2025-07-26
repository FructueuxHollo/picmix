import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# On importe directement le moteur depuis le package
from picmix.engine import PicMixEngine
from picmix.io import load_image_as_array

def compare_images_metrics(img1_path, img2_path):
    """Calcule le MSE et le PSNR entre deux images."""
    img1 = load_image_as_array(img1_path)
    img2 = load_image_as_array(img2_path)
    
    if img1.shape != img2.shape:
        print("Les images n'ont pas la même taille !")
        return {'mse': float('inf'), 'psnr': 0}
        
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return {'mse': 0, 'psnr': float('inf')}
        
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) # Image normalisée, donc max_val = 1.0
    return {'mse': mse, 'psnr': psnr}

def main():
    print("--- Début du test de réversibilité de la simulation fluide ---")

    IMAGE_ORIGINALE = "test_image.png"
    IMAGE_FLUIDE_CHIFFREE = "debug_fluid_encrypted.png"
    IMAGE_FLUIDE_DECHIFFREE = "debug_fluid_decrypted.png"

    # --- Paramètres de simulation FIXES et CONTRÔLÉS ---
    # C'est ici que nous allons pouvoir expérimenter.
    # Pour commencer, utilisons des valeurs simples.
    simulation_config = {
        'time_steps': 200,       # Moins d'étapes = moins de diffusion accumulée
        'dt': 0.05,
        # On impose une viscosité fixe, sans la dériver de la clé.
        'viscosity': 0.00001, 
        # On impose des fonctions de forçage simples pour commencer (sinus/cosinus)
        'force_funcs': {
            'f_x_func': lambda X, Y: 0.1 * np.sin(2 * np.pi * Y),
            'f_y_func': lambda X, Y: 0.1 * np.cos(2 * np.pi * X)
        }
    }

    # Charger l'image originale
    original_image_array = load_image_as_array(IMAGE_ORIGINALE)

    # --- Chiffrement avec la simulation fluide seule ---
    print("\n[CHIFFREMENT]")
    # On utilise une clé factice, car les paramètres importants sont surchargés par `config`
    engine_encrypt = PicMixEngine(
        image=original_image_array, 
        key="debug_key", 
        config=simulation_config
    )
    
    # Exécuter uniquement la partie "fluide" du chiffrement
    engine_encrypt._compute_velocity_history()
    
    fluid_encrypted_array = engine_encrypt._advect_pixels(engine_encrypt.rho, forward=True)
    
    # Sauvegarder l'image chiffrée par le fluide
    Image.fromarray((np.clip(fluid_encrypted_array, 0, 1) * 255).astype(np.uint8)).save(IMAGE_FLUIDE_CHIFFREE)
    
    print(f"Image chiffrée par le fluide sauvegardée dans '{IMAGE_FLUIDE_CHIFFREE}'")

    # --- Déchiffrement avec la simulation fluide seule ---
    print("\n[DÉCHIFFREMENT]")
    engine_decrypt = PicMixEngine(
        image=fluid_encrypted_array, # On part de l'image chiffrée
        key="debug_key", 
        config=simulation_config
    )
    
    # Exécuter uniquement la partie "fluide" du déchiffrement
    engine_decrypt._compute_velocity_history() # Doit redonner le même champ de vitesse
    fluid_decrypted_array = engine_decrypt._advect_pixels(engine_decrypt.rho, forward=False)
    
    Image.fromarray((np.clip(fluid_decrypted_array, 0, 1) * 255).astype(np.uint8)).save(IMAGE_FLUIDE_DECHIFFREE)
    print(f"Image déchiffrée par le fluide sauvegardée dans '{IMAGE_FLUIDE_DECHIFFREE}'")

    # --- Analyse des résultats ---
    print("\n[ANALYSE]")
    metrics = compare_images_metrics(IMAGE_ORIGINALE, IMAGE_FLUIDE_DECHIFFREE)
    print(f"  Erreur Quadratique Moyenne (MSE) : {metrics['mse']:.6f}")
    print(f"  Rapport Signal/Bruit de Crête (PSNR) : {metrics['psnr']:.2f} dB")
    
    # Affichage visuel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(load_image_as_array(IMAGE_ORIGINALE), cmap='gray'); axes[0].set_title("Originale")
    axes[1].imshow(fluid_encrypted_array, cmap='gray'); axes[1].set_title("Chiffrée (Fluide)")
    axes[2].imshow(fluid_decrypted_array, cmap='gray'); axes[2].set_title("Déchiffrée (Fluide)")
    plt.show()

if __name__ == "__main__":
    main()