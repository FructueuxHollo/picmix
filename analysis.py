# Fichier: analysis.py
# Un script pour analyser la performance, la sécurité et la réversibilité de VortexCrypt.

import vortexcrypt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns # Pour de plus jolis graphiques et KDE

# --- Configuration de l'Analyse ---

# Créer le dossier de sortie s'il n'existe pas
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paramètres du test
KEY = "a-robust-scientific-key"
IMAGE_ORIGINAL_PATH = "test_image.png"
OUTPUT_BASENAME = "encrypted_output"
STATE_ENCRYPTED_PATH = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME + ".npz")
IMAGE_ENCRYPTED_PREVIEW_PATH = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME +".png")
IMAGE_DECRYPTED_PATH = os.path.join(OUTPUT_DIR, "decrypted_image.png")

# --- Fonctions d'Aide ---

def analyze_histograms(original_arr, encrypted_arr, decrypted_arr, output_path):
    """Calcule, trace et sauvegarde les histogrammes."""
    print("Analyse des histogrammes...")
    plt.figure(figsize=(12, 7))
    
    # Utiliser seaborn pour une meilleure visualisation avec estimation de densité (KDE)
    sns.histplot(original_arr.flatten(), color='blue', label='Originale', stat='density', bins=256, kde=True)
    sns.histplot(encrypted_arr.flatten(), color='red', label='Chiffrée', stat='density', bins=256, kde=True)
    
    plt.title("Distribution des Intensités des Pixels (Histogrammes)", fontsize=16)
    plt.xlabel("Intensité du Pixel (normalisée)", fontsize=12)
    plt.ylabel("Densité de Probabilité", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique des histogrammes sauvegardé dans '{output_path}'")

def analyze_reconstruction_error(original_arr, decrypted_arr, output_path_metrics, output_path_visual):
    """Calcule les métriques d'erreur et génère une image d'erreur."""
    print("Analyse de l'erreur de reconstruction...")
    
    # Calcul des métriques
    mse = np.mean((original_arr - decrypted_arr) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    mae = np.mean(np.abs(original_arr - decrypted_arr))
    
    # Sauvegarde des métriques dans un fichier texte
    with open(output_path_metrics, "a") as f:
        f.write("\n--- Analyse de l'Erreur de Reconstruction ---\n")
        f.write(f"Erreur Quadratique Moyenne (MSE): {mse:.6e}\n")
        f.write(f"Rapport Signal/Bruit de Crête (PSNR): {psnr:.2f} dB\n")
        f.write(f"Erreur Absolue Moyenne (MAE): {mae:.6e}\n")
    print(f"Métriques d'erreur sauvegardées dans '{output_path_metrics}'")
    
    # Génération de l'image d'erreur (amplifiée pour être visible)
    error_map = np.abs(original_arr - decrypted_arr)
    # On amplifie l'erreur pour qu'elle soit visible, en clippant à 1.0
    amplified_error_map = np.clip(error_map * 10, 0.0, 1.0) 
    
    plt.figure(figsize=(8, 8))
    plt.imshow(amplified_error_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label="Erreur absolue (amplifiée 10x)")
    plt.title("Carte de l'Erreur de Reconstruction", fontsize=16)
    plt.axis('off')
    plt.savefig(output_path_visual)
    plt.close()
    print(f"Carte d'erreur sauvegardée dans '{output_path_visual}'")

def main():
    """Script principal d'analyse."""
    
    # --- Préparation ---
    # Créer une image de test si elle n'existe pas
    if not os.path.exists(IMAGE_ORIGINAL_PATH):
        print(f"Création d'une image de test '{IMAGE_ORIGINAL_PATH}'")
        img_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        img_arr[64:192, 64:128, 0] = 255
        img_arr[64:192, 128:192, 2] = 255
        Image.fromarray(img_arr).save(IMAGE_ORIGINAL_PATH)
        
    # Charger l'image originale une fois pour toutes
    original_array_color = vortexcrypt.io.load_image_as_array(IMAGE_ORIGINAL_PATH, grayscale=False)
    original_array_gray = vortexcrypt.io.load_image_as_array(IMAGE_ORIGINAL_PATH, grayscale=True)

    # Fichier de rapport
    report_file = os.path.join(OUTPUT_DIR, "results.txt")
    if os.path.exists(report_file):
        os.remove(report_file) # Effacer l'ancien rapport

    # --- Exécution et Mesure de Performance ---
    
    print("\n" + "="*20 + " EXÉCUTION DU CHIFFREMENT " + "="*20)
    start_time_encrypt = time.time()
    vortexcrypt.encrypt(
        image_path=IMAGE_ORIGINAL_PATH,
        output_path_npz=STATE_ENCRYPTED_PATH,
        key=KEY,
        save_preview=True # S'assurer que la preview est sauvegardée pour l'analyse
    )
    encrypt_duration = time.time() - start_time_encrypt
    print(f"Temps de chiffrement : {encrypt_duration:.2f} secondes")
    
    print("\n" + "="*20 + " EXÉCUTION DU DÉCHIFFREMENT " + "="*20)
    start_time_decrypt = time.time()
    vortexcrypt.decrypt(
        encrypted_state_path_npz=STATE_ENCRYPTED_PATH,
        output_path_png=IMAGE_DECRYPTED_PATH,
        key=KEY
    )
    decrypt_duration = time.time() - start_time_decrypt
    print(f"Temps de déchiffrement : {decrypt_duration:.2f} secondes")

    # Sauvegarder les temps dans le rapport
    with open(report_file, "w") as f:
        f.write("--- Analyse de Performance ---\n")
        f.write(f"Image source : {IMAGE_ORIGINAL_PATH} (Shape: {original_array_color.shape})\n")
        f.write(f"Temps d'exécution du chiffrement : {encrypt_duration:.4f} secondes\n")
        f.write(f"Temps d'exécution du déchiffrement : {decrypt_duration:.4f} secondes\n")

    # --- Lancement des Analyses ---
    
    # Charger les images chiffrée et déchiffrée
    encrypted_array_gray = vortexcrypt.io.load_image_as_array(IMAGE_ENCRYPTED_PREVIEW_PATH, grayscale=True)
    decrypted_array_gray = vortexcrypt.io.load_image_as_array(IMAGE_DECRYPTED_PATH, grayscale=True)

    # 1. Analyse des histogrammes
    analyze_histograms(
        original_array_gray, 
        encrypted_array_gray,
        decrypted_array_gray,
        os.path.join(OUTPUT_DIR, "histograms_comparison.png")
    )

    # 2. Analyse de l'erreur de reconstruction
    analyze_reconstruction_error(
        original_array_gray,
        decrypted_array_gray,
        report_file,
        os.path.join(OUTPUT_DIR, "reconstruction_error_map.png")
    )
    
    print(f"\n✅ Analyse terminée. Tous les résultats sont dans le dossier '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    # Il faut `seaborn` pour les graphiques
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn n'est pas installé. Veuillez l'installer avec : poetry add seaborn")
    else:
        main()