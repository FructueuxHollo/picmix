import vortexcrypt
import time
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
TEST_IMAGES_DIR = "test/test_images"
RESULTS_DIR = "test/analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ensemble de clés de test variées (longueur, caractères)
TEST_KEYS = [
    "shortkey",
    "a_longer_key_123",
    "Re@ct!on-D1ffus!on",
    "morphogenesis_is_cool_!",
    "aBcDeFgHiJkLmNoPqRsTuVwX",
]
# Fichier de configuration fixe pour isoler l'effet de la taille
FIXED_CONFIG_PATH = "test/analysis_scripts/perf_config.json"

def get_image_info(image_path):
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

def run_single_test(image_path, key, grayscale, config_path=None):
    """Exécute un cycle et retourne les temps."""
    shape, _ = get_image_info(image_path)
    output_npz = os.path.join(RESULTS_DIR, "temp_state.npz")
    output_png = os.path.join(RESULTS_DIR, "temp_decrypted.png")

    start_encrypt = time.perf_counter()
    vortexcrypt.encrypt(
        image_path=image_path,
        output_path_npz=output_npz,
        key=key,
        config_path=config_path,
        save_preview=True,
        grayscale=grayscale
    )
    encrypt_time = time.perf_counter() - start_encrypt
    
    start_decrypt = time.perf_counter()
    vortexcrypt.decrypt(
        encrypted_state_path_npz=output_npz,
        output_path_png=output_png,
        key=key,
        config_path=config_path
    )
    decrypt_time = time.perf_counter() - start_decrypt
    
    return encrypt_time, decrypt_time

def analyze_performance():
    """Analyse les temps d'exécution en faisant la moyenne sur plusieurs clés."""
    
    image_files = [
        "moon_test_image.png",
        "cameraman_test_image.png",
        "text_test_image.png",
        "bart_test_image.png",
        "house_test_image.png",
        "lenna_test_image.png",
    ]
    # image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.png', '.jpg'))]
    all_results = []
    
    # --- TEST 1 ---
    print("\n" + "="*20 + " TEST 1: Performance avec Clés Variables " + "="*20)
    for image_file in image_files:
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        shape, is_color = get_image_info(image_path)
        
        encrypt_times = []
        decrypt_times = []
        
        print(f"\nTraitement de '{image_file}' avec {len(TEST_KEYS)} clés...")
        for key in TEST_KEYS:
            enc_t, dec_t = run_single_test(image_path, key, not is_color)
            encrypt_times.append(enc_t)
            decrypt_times.append(dec_t)
            print(f"  - Clé '{key[:10]}...': Enc={enc_t:.2f}s, Dec={dec_t:.2f}s")

        result = {
            "test_type": "Clés Variables",
            "image": image_file,
            "pixels": shape[0] * shape[1],
            "is_color": is_color,
            "encrypt_time_mean": np.mean(encrypt_times),
            "encrypt_time_std": np.std(encrypt_times),
            "decrypt_time_mean": np.mean(decrypt_times),
            "decrypt_time_std": np.std(decrypt_times),
        }
        all_results.append(result)

    # --- TEST 2 ---
    print("\n" + "="*20 + " TEST 2: Performance avec Configuration Fixe " + "="*20)
    for image_file in image_files:
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        shape, is_color = get_image_info(image_path)
        
        print(f"\nTraitement de '{image_file}' avec la config fixe...")
        enc_t, dec_t = run_single_test(image_path, "fixed_config_key", not is_color, FIXED_CONFIG_PATH)
        print(f"  -> Enc={enc_t:.2f}s, Dec={dec_t:.2f}s")
        
        result = {
            "test_type": "Config Fixe",
            "image": image_file,
            "pixels": shape[0] * shape[1],
            "is_color": is_color,
            "encrypt_time_mean": enc_t,
            "encrypt_time_std": 0,
            "decrypt_time_mean": dec_t,
            "decrypt_time_std": 0,
        }
        all_results.append(result)

    # --- Save and Display ---
    df = pd.DataFrame(all_results).sort_values(by="pixels")
    report_path = os.path.join(RESULTS_DIR, "performance_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Rapport d'Analyse de Performance de VortexCrypt ---\n\n")
        f.write("TEST 1: Performance moyenne et écart-type sur plusieurs clés\n")
        f.write(df[df['test_type'] == 'Clés Variables'].to_string(index=False))
        f.write("\n\nTEST 2: Performance pour une durée de simulation fixe (T=300.0)\n")
        f.write(df[df['test_type'] == 'Config Fixe'].to_string(index=False))

    print(f"\n✅ Analyse de performance terminée. Rapport sauvegardé dans '{report_path}'.")
    
    df_fixed = df[df['test_type'] == 'Config Fixe']
    plt.figure(figsize=(12, 7))
    
    for is_color_val, marker, label_color in [(True, 'o', 'Couleur'), (False, 'x', 'N&B')]:
        subset = df_fixed[df_fixed['is_color'] == is_color_val]
        if not subset.empty:
            plt.scatter(subset['pixels'], subset['encrypt_time_mean'], 
                        c='r' if is_color_val else 'b', marker=marker, s=80, label=f'Chiffrement ({label_color})')
            plt.scatter(subset['pixels'], subset['decrypt_time_mean'], 
                        c='orange' if is_color_val else 'cyan', marker=marker, s=80, label=f'Déchiffrement ({label_color})')
            
    plt.title("Performance avec Configuration Fixe (T=30s)", fontsize=16)
    plt.xlabel("Nombre total de pixels", fontsize=12)
    plt.ylabel("Temps (secondes)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(RESULTS_DIR, "performance_fixed_config_graph.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Graphique de performance sauvegardé dans '{plot_path}'.")

if __name__ == "__main__":
    analyze_performance()