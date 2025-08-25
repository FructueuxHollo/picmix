import vortexcrypt
import matplotlib.pyplot as plt
import os
import seaborn as sns

RESULTS_DIR = "test/analysis_results/histogram"
os.makedirs(RESULTS_DIR, exist_ok=True)

KEY = "NoWayThisGonnaWork"

HISTOGRAM_IMAGE = "test/test_images/moon_test_image.png"

def analyze_histograms_all(original_path, decrypted_path, output_path):
    """Trace l'histogramme des trois images (originale, chiffrée, déchiffrée)."""
    print(f"\nAnalyse des histogrammes pour '{os.path.basename(original_path)}'...")
    
    original_arr = vortexcrypt.io.load_image_as_array(original_path, grayscale=True)
    decrypted_arr = vortexcrypt.io.load_image_as_array(decrypted_path, grayscale=True)
    
    plt.figure(figsize=(12, 7))
    sns.histplot(original_arr.flatten(), color='blue', label='Originale', stat='density', bins=256, kde=True)
    sns.histplot(decrypted_arr.flatten(), color='red', label='Déchiffrée', stat='density', bins=256, kde=True,
                 line_kws={'linestyle': '--'}) # Style différent pour la déchiffrée
    
    plt.title("Comparaison des Histogrammes (Originale, Déchiffrée)")
    plt.xlabel("Intensité du Pixel (normalisée)")
    plt.ylabel("Densité de Probabilité")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique d'histogramme complet sauvegardé dans '{output_path}'")


encrypted_npz = os.path.join(RESULTS_DIR, "hist_state.npz")
decrypted_png = os.path.join(RESULTS_DIR, "hist_decrypted.png")

vortexcrypt.encrypt(
    image_path=HISTOGRAM_IMAGE,
    output_path_npz=encrypted_npz,
    key=KEY,
    save_preview=True,
    grayscale=True,
    config_path="test/analysis_scripts/perf_config.json"
)
vortexcrypt.decrypt(
    encrypted_state_path_npz=encrypted_npz,
    output_path_png=decrypted_png,
    key=KEY,
    config_path="test/analysis_scripts/perf_config.json"
)
analyze_histograms_all(
    HISTOGRAM_IMAGE,
    decrypted_png,
    os.path.join(RESULTS_DIR, "histograms_comparison.png")
)