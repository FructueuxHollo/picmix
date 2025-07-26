# Fichier: debug_fluid.py (version améliorée pour l'analyse)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# On importe directement le moteur depuis le package
from picmix.engine import PicMixEngine
from picmix.io import load_image_as_array

# --- Fonctions utilitaires ---

def compare_images_metrics(img1_array, img2_array):
    """Calcule le MSE et le PSNR entre deux tableaux NumPy normalisés."""
    if img1_array.shape != img2_array.shape:
        print("Les images n'ont pas la même taille !")
        return {'mse': float('inf'), 'psnr': 0}
        
    mse = np.mean((img1_array - img2_array) ** 2)
    if mse == 0:
        return {'mse': 0, 'psnr': float('inf')}
        
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return {'mse': mse, 'psnr': psnr}

def run_single_experiment(original_image_array, key, config):
    """Exécute un cycle chiffrement/déchiffrement pour une config donnée."""
    # Chiffrement
    engine_encrypt = PicMixEngine(image=original_image_array, key=key, config=config)
    engine_encrypt._compute_velocity_history()
    fluid_encrypted_array = engine_encrypt._advect_pixels(engine_encrypt.rho, forward=True)
    
    # Déchiffrement
    # On réutilise le même moteur et le champ de vitesse déjà calculé
    fluid_decrypted_array = engine_encrypt._advect_pixels(fluid_encrypted_array, forward=False)
    
    # Calcul des métriques
    metrics = compare_images_metrics(original_image_array, fluid_decrypted_array)
    return metrics

def plot_results(param_name, param_values, mse_results, psnr_results):
    """Trace les graphiques des résultats."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Impact du paramètre "{param_name}" sur la réversibilité', fontsize=16)

    # Graphique MSE
    ax1.plot(param_values, mse_results, 'o-', color='r')
    ax1.set_ylabel("Erreur Quadratique Moyenne (MSE)")
    ax1.set_title("Erreur de reconstruction (plus c'est bas, mieux c'est)")
    ax1.grid(True)

    # Graphique PSNR
    ax2.plot(param_values, psnr_results, 'o-', color='b')
    ax2.set_xlabel(f"Valeur du paramètre : {param_name}")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Qualité de reconstruction (plus c'est haut, mieux c'est)")
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Script principal ---

def main():
    print("--- Début de l'analyse paramétrique de la simulation fluide ---")

    IMAGE_ORIGINALE = "test_image.png"
    # S'assurer que l'image de test existe
    if not os.path.exists(IMAGE_ORIGINALE):
        print(f"Création d'une image de test '{IMAGE_ORIGINALE}'")
        img_arr = np.zeros((128, 128), dtype=np.uint8)
        img_arr[32:96, 32:96] = 255
        Image.fromarray(img_arr).save(IMAGE_ORIGINALE)

    original_image_array = load_image_as_array(IMAGE_ORIGINALE)
    
    # Configuration de base (les valeurs seront surchargées dans les boucles)
    base_config = {
        'dt': 0.1,
        'viscosity': 0.00001, 
        'force_funcs': {
            'f_x_func': lambda X, Y: 0.1 * np.sin(4 * np.pi * Y),
            'f_y_func': lambda X, Y: 0.1 * np.cos(4 * np.pi * X)
        }
    }
    
    # === Expérience 1: Impact du nombre de pas de temps (time_steps) ===
    print("\n--- Expérience 1: Variation de 'time_steps' ---")
    param_name = "time_steps"
    param_values_ts = [5, 10, 20, 30, 50, 75, 100]
    mse_results_ts = []
    psnr_results_ts = []

    for steps in param_values_ts:
        start_time = time.time()
        config = base_config.copy()
        config['time_steps'] = steps
        
        metrics = run_single_experiment(original_image_array, "debug_key", config)
        
        mse_results_ts.append(metrics['mse'])
        psnr_results_ts.append(metrics['psnr'])
        
        duration = time.time() - start_time
        print(f"  {param_name}={steps:3d} -> MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB (en {duration:.2f}s)")
        
    plot_results(param_name, param_values_ts, mse_results_ts, psnr_results_ts)

    # === Expérience 2: Impact de la taille du pas de temps (dt) ===
    print("\n--- Expérience 2: Variation de 'dt' (avec T constant) ---")
    param_name = "dt"
    param_values_dt = [0.2, 0.1, 0.05, 0.02, 0.01]
    mse_results_dt = []
    psnr_results_dt = []
    T_final = 5.0 # Temps de simulation total constant

    for dt_val in param_values_dt:
        start_time = time.time()
        config = base_config.copy()
        config['dt'] = dt_val
        config['time_steps'] = int(T_final / dt_val)
        
        metrics = run_single_experiment(original_image_array, "debug_key", config)

        mse_results_dt.append(metrics['mse'])
        psnr_results_dt.append(metrics['psnr'])
        
        duration = time.time() - start_time
        print(f"  {param_name}={dt_val:.3f} (soit {config['time_steps']} steps) -> MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB (en {duration:.2f}s)")

    plot_results(param_name, param_values_dt, mse_results_dt, psnr_results_dt)

    # === Expérience 3: Impact de la viscosité (viscosity) ===
    print("\n--- Expérience 3: Variation de 'viscosity' ---")
    param_name = "viscosity"
    param_values_visc = [0.0, 0.00001, 0.0001, 0.001, 0.01]
    mse_results_visc = []
    psnr_results_visc = []
    
    for visc in param_values_visc:
        start_time = time.time()
        config = base_config.copy()
        config['time_steps'] = 30 # Fixer le nombre de pas
        config['viscosity'] = visc
        
        metrics = run_single_experiment(original_image_array, "debug_key", config)

        mse_results_visc.append(metrics['mse'])
        psnr_results_visc.append(metrics['psnr'])
        
        duration = time.time() - start_time
        print(f"  {param_name}={visc:.5f} -> MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB (en {duration:.2f}s)")

    plot_results(param_name, param_values_visc, mse_results_visc, psnr_results_visc)


if __name__ == "__main__":
    main()