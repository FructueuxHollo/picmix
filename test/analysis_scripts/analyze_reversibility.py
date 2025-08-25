import vortexcrypt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# --- Configuration ---
TEST_IMAGES_DIR = "test_images"
RESULTS_DIR = "analysis_results/reversibility"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configurations à tester
CONFIG_FILES = {
    "Short_Stable": "test/analysis_scripts/config_short.json",
    "Long_Stable": "test/analysis_scripts/config_long.json",
}
KEY = "reversibilitanalsiske" # La clé est moins importante ici car la config est fixe

# Image spécifique pour l'analyse d'histogramme
HISTOGRAM_IMAGE = "test/test_images/cameraman_test_image.png"

# --- Fonctions d'Aide ---

def compare_images(original_arr, decrypted_arr):
    """Calcule MSE, PSNR, et MAE."""
    if original_arr.shape != decrypted_arr.shape:
        return {'mse': float('inf'), 'psnr': 0, 'mae': float('inf')}
    
    mse = np.mean((original_arr - decrypted_arr) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    mae = np.mean(np.abs(original_arr - decrypted_arr))
    
    return {'mse': mse, 'psnr': psnr, 'mae': mae}

def run_reversibility_test(image_path, config_path):
    """Exécute un cycle complet et retourne les métriques d'erreur."""
    output_npz = os.path.join(RESULTS_DIR, "temp_state.npz")
    output_png = os.path.join(RESULTS_DIR, "temp_decrypted.png")

    original_array = vortexcrypt.io.load_image_as_array(image_path, grayscale=True)
    
    # Chiffrement
    vortexcrypt.encrypt(
        image_path=image_path,
        output_path_npz=output_npz,
        key=KEY,
        config_path=config_path,
        save_preview=True
    )
    
    # Déchiffrement
    vortexcrypt.decrypt(
        encrypted_state_path_npz=output_npz,
        output_path_png=output_png,
        key=KEY,
        config_path=config_path
    )
    
    decrypted_array = vortexcrypt.io.load_image_as_array(output_png, grayscale=True)
    
    # Si l'original était N&B, on s'assure que le déchiffré l'est aussi pour la comparaison
    if original_array.ndim == 2:
        decrypted_array = vortexcrypt.io.load_image_as_array(output_png, grayscale=True)

    return compare_images(original_array, decrypted_array)

def analyze_histograms_all(original_path, encrypted_preview_path, decrypted_path, output_path):
    """Trace l'histogramme des trois images (originale, chiffrée, déchiffrée)."""
    print(f"\nAnalyse des histogrammes pour '{os.path.basename(original_path)}'...")
    
    original_arr = vortexcrypt.io.load_image_as_array(original_path, grayscale=True)
    encrypted_arr = vortexcrypt.io.load_image_as_array(encrypted_preview_path, grayscale=True)
    decrypted_arr = vortexcrypt.io.load_image_as_array(decrypted_path, grayscale=True)
    
    plt.figure(figsize=(12, 7))
    sns.histplot(original_arr.flatten(), color='blue', label='Originale', stat='density', bins=256, kde=True)
    sns.histplot(encrypted_arr.flatten(), color='red', label='Chiffrée', stat='density', bins=256, kde=True)
    sns.histplot(decrypted_arr.flatten(), color='green', label='Déchiffrée', stat='density', bins=256, kde=True,
                 line_kws={'linestyle': '--'}) # Style différent pour la déchiffrée
    
    plt.title("Comparaison des Histogrammes (Originale, Chiffrée, Déchiffrée)")
    plt.xlabel("Intensité du Pixel (normalisée)")
    plt.ylabel("Densité de Probabilité")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique d'histogramme complet sauvegardé dans '{output_path}'")

def main():
    """Script principal d'analyse de réversibilité."""
    
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.png', '.jpg'))]
    all_results = []
    
    print("--- Démarrage de l'Analyse de Réversibilité ---")

    for config_name, config_path in CONFIG_FILES.items():
        print(f"\n" + "="*20 + f" TEST AVEC CONFIG: {config_name} " + "="*20)
        
        for image_file in image_files:
            image_path = os.path.join(TEST_IMAGES_DIR, image_file)
            shape, is_color = vortexcrypt.io.get_image_info(image_path) # Assumons que cette fonction existe dans io

            print(f"  -> Traitement de '{image_file}'...")
            metrics = run_reversibility_test(image_path, config_path)
            
            result = {
                "config": config_name,
                "image_type": "Couleur" if is_color else "N&B",
                "mse": metrics['mse'],
                "psnr": metrics['psnr'],
                "mae": metrics['mae']
            }
            all_results.append(result)
            print(f"     MSE: {metrics['mse']:.2e}, PSNR: {metrics['psnr']:.2f} dB")
            
    # --- Agrégation et Sauvegarde des Résultats ---
    df = pd.DataFrame(all_results)
    
    # Calculer les moyennes par type d'image et par config
    summary = df.groupby(['config', 'image_type']).agg(['mean', 'std']).round(4)
    
    report_path = os.path.join(RESULTS_DIR, "reversibility_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Rapport d'Analyse de Réversibilité de VortexCrypt ---\n\n")
        f.write("Métriques d'erreur moyennes (et écart-type) par configuration et type d'image.\n")
        f.write("MSE/MAE: plus c'est bas, mieux c'est. PSNR: plus c'est haut, mieux c'est.\n\n")
        f.write(summary.to_string())
        
    print(f"\n✅ Analyse de réversibilité terminée. Rapport sauvegardé dans '{report_path}'.")

    # --- Analyse d'Histogramme Spécifique ---
    # On exécute le test une dernière fois sur l'image cible avec une config représentative
    specific_config = CONFIG_FILES["Long_Stable"]
    encrypted_npz = os.path.join(RESULTS_DIR, "hist_analysis_state.npz")
    decrypted_png = os.path.join(RESULTS_DIR, "hist_analysis_decrypted.png")
    
    vortexcrypt.encrypt(
        image_path=HISTOGRAM_IMAGE,
        output_path_npz=encrypted_npz,
        key=KEY,
        config_path=specific_config,
        save_preview=True
    )
    vortexcrypt.decrypt(
        encrypted_state_path_npz=encrypted_npz,
        output_path_png=decrypted_png,
        key=KEY,
        config_path=specific_config
    )

    encrypted_preview = os.path.splitext(encrypted_npz)[0] + ".png"
    analyze_histograms_all(
        HISTOGRAM_IMAGE,
        encrypted_preview,
        decrypted_png,
        os.path.join(RESULTS_DIR, "full_histograms_comparison.png")
    )

if __name__ == "__main__":
    # Ajouter `get_image_info` à `vortexcrypt.io` si ce n'est pas déjà fait
    # (ou le définir localement)
    if not hasattr(vortexcrypt.io, 'get_image_info'):
        def get_image_info_local(image_path):
            """Récupère les informations d'une image."""
            img = Image.open(image_path).convert("RGB")  # On force en RGB
            shape = (img.height, img.width)
            arr = np.array(img)

            # Vérifie si toutes les composantes R, G, B sont égales (=> gris)
            if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
                is_color = False
            else:
                is_color = True
            return shape, is_color
        vortexcrypt.io.get_image_info = get_image_info_local

    main()