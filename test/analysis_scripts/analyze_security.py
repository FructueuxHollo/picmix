import vortexcrypt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import entropy

TEST_IMAGE_PATH = "test_images/lenna_test_image.png"
RESULTS_DIR = "analysis_results/security"
os.makedirs(RESULTS_DIR, exist_ok=True)
KEY = "analysis-key-12345"
CONFIG_PATH = None

# --- Fonctions d'Aide  ---
def save_metrics_to_file(filename, metrics_dict):
    """Sauvegarde un dictionnaire de métriques dans un fichier."""
    with open(filename, "w") as f:
        f.write("--- Rapport d'Analyse de Sécurité ---\n\n")
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_val in value.items():
                    f.write(f"  - {sub_key}: {sub_val:.4f}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
    print(f"Métriques sauvegardées dans '{filename}'")

def analyze_histograms(original_arr, encrypted_arr, output_path):
    """Analyse les histogrammes des versions N&B."""
    print("-> Analyse des histogrammes...")
    plt.figure(figsize=(10, 6))
    plt.hist(original_arr.flatten(), bins=256, color='blue', alpha=0.6, label='Originale', density=True)
    plt.hist(encrypted_arr.flatten(), bins=256, color='red', alpha=0.6, label='Chiffrée (Normalisée)', density=True)
    plt.title("Comparaison des Histogrammes")
    plt.xlabel("Intensité du Pixel (normalisée à [0,1])")
    plt.ylabel("Fréquence Normalisée")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    plt.close()
    print(f"   Graphique d'histogramme sauvegardé.")

def analyze_correlation(image_arr, output_path, sample_size=5000):
    """Analyse la corrélation des pixels adjacents."""
    print("-> Analyse de la corrélation...")
    if image_arr.size > sample_size:
        indices = np.random.choice(image_arr.size - 1, sample_size, replace=False)
    else:
        indices = np.arange(image_arr.size - 1)
    
    flat_arr = image_arr.flatten()
    x = flat_arr[indices]
    y_h = flat_arr[indices + 1] 
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y_h, s=1, alpha=0.5)
    plt.title("Corrélation des Pixels Adjacents")
    plt.xlabel("Intensité du Pixel p(i, j) (normalisée)")
    plt.ylabel("Intensité du Pixel p(i, j+1) (normalisée)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis([0, 1, 0, 1])
    plt.savefig(output_path)
    plt.close()
    print(f"   Graphique de corrélation sauvegardé.")

    corr_coef = np.corrcoef(x, y_h)[0, 1]
    return corr_coef

def calculate_entropy(image_arr):
    """Calcule l'entropie de Shannon de l'image."""
    print("-> Calcul de l'entropie...")
    # Pour l'entropie, on doit discrétiser en 256 niveaux
    bins = np.histogram(image_arr.flatten(), bins=256, range=(0,1))[0]
    probabilities = bins / bins.sum()
    shannon_entropy = entropy(probabilities, base=2)
    return shannon_entropy
    
def main():
    """Script principal d'analyse de sécurité."""
    print(f"--- Analyse de Sécurité sur '{TEST_IMAGE_PATH}' ---")

    original_color = vortexcrypt.io.load_image_as_array(TEST_IMAGE_PATH, grayscale=False)
    original_gray = vortexcrypt.io.load_image_as_array(TEST_IMAGE_PATH, grayscale=True)
    
    output_npz = os.path.join(RESULTS_DIR, f"{os.path.splitext(os.path.basename(TEST_IMAGE_PATH))[0]}_encrypted.npz")
    
    print("\nChiffrement de l'image pour l'analyse...")
    vortexcrypt.encrypt(
        image_path=TEST_IMAGE_PATH,
        output_path_npz=output_npz,
        key=KEY,
        config_path=CONFIG_PATH,
        save_preview=False 
    )
    
    print("Chargement des données chiffrées brutes depuis le .npz...")
    u_final_flat, _, padded_shape = vortexcrypt.io.load_state_from_npz(output_npz)
    
    # Déterminer si c'était une image couleur
    is_color_encrypted = (len(original_color.shape) == 3)
    
    if is_color_encrypted:
        # Extraire le premier canal (rouge) pour l'analyse en niveaux de gris
        encrypted_2d_padded = u_final_flat.reshape(-1, 3)[:, 0].reshape(padded_shape)
    else:
        encrypted_2d_padded = u_final_flat.reshape(padded_shape)

    # Retirer le padding
    pad = 1 # On suppose un padding de 1
    encrypted_2d_raw = encrypted_2d_padded[pad:-pad, pad:-pad]

    # --- Normaliser les données brutes pour l'analyse ---
    print("Normalisation Min-Max des données chiffrées pour l'analyse...")
    min_val, max_val = encrypted_2d_raw.min(), encrypted_2d_raw.max()
    if min_val == max_val:
        encrypted_analyzable = np.full(encrypted_2d_raw.shape, 0.5) # Cas plat
    else:
        encrypted_analyzable = (encrypted_2d_raw - min_val) / (max_val - min_val)
    
    # --- Étape 2: Lancer les analyses sur les bonnes données ---
    metrics = {}
    
    analyze_histograms(
        original_gray, 
        encrypted_analyzable, # Utiliser la version normalisée
        os.path.join(RESULTS_DIR, "histograms.png")
    )
    
    metrics['correlation'] = {}
    metrics['correlation']['original'] = analyze_correlation(
        original_gray,
        os.path.join(RESULTS_DIR, "correlation_original.png")
    )
    metrics['correlation']['encrypted'] = analyze_correlation(
        encrypted_analyzable, # Utiliser la version normalisée
        os.path.join(RESULTS_DIR, "correlation_encrypted.png")
    )
    
    metrics['entropy'] = {}
    metrics['entropy']['original'] = calculate_entropy(original_gray)
    metrics['entropy']['encrypted'] = calculate_entropy(encrypted_analyzable) # Utiliser la version normalisée

    # --- Étape 3: Sauvegarder le rapport ---
    report_path = os.path.join(RESULTS_DIR, "security_report.txt")
    save_metrics_to_file(report_path, metrics)

    print(f"\n✅ Analyse de sécurité terminée. Résultats dans '{RESULTS_DIR}'.")

if __name__ == "__main__":
    main()