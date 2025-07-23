from utils import StrangSplitting, add_padding, generate_random_2d_array, laplacian_matrix, load_image_as_array, plot_strang_splitting_results, save_array_as_image

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

img = load_image_as_array("test_image.png")  # -> array (Nx, Ny)

# ------------------------
# Initialisation
# ------------------------

# Générer un champ de vitesse aléatoire pour le test
img_u = add_padding(img, 1)
img_v = generate_random_2d_array(img_u.shape)

# Sauvegarder les images avec rembourrage
save_array_as_image(img_u, "test_image_u.png")
save_array_as_image(img_v, "test_image_v.png")

# ------------------------
# Paramètres du modèle
# ------------------------
Nx, Ny = img_u.shape
Lx, Ly = 5, 5  # Dimensions du domaine
dx, dy = Lx / Nx, Ly / Ny

# Paramètres Gray-Scott
ru, rv = 0.16, 0.08
f, k = 0.060, 0.12

# dt = np.min([dx**2 /ru, dy**2 / rv, 1]) / 4   # Pas de temps pour la simulation
# dt = dx**2 / (4 * np.max([ru, rv]))   # Pas de temps pour la simulation
dt = 0.1  # Pas de temps pour la simulation
T = .3 # Durée totale de la simulation

print(f"Pas de temps dt: {dt:.4f}, Durée totale T: {T:.2f}")

# Convertir les images en tableaux numpy 1D
u = img_u.flatten()
v = img_v.flatten()

# Laplacien 2D avec Neumann
L = laplacian_matrix(Nx, Ny, dx, dy)

strang = StrangSplitting(*img_u.shape)

# # ------------------------
# # Simulation d'un pas
# # ------------------------
# u_new, v_new = strang.forward_step(u, v, ru, rv, f, k, L, dt)


# ------------------------
# Simulation sur toute la durée T
# ------------------------
def simulate_strang_splitting(u, v, ru, rv, f, k, L, dt, T):
    """
    Simule le schéma de Strang-splitting sur une durée T.
    
    Args:
        u (np.ndarray): Champ de vitesse u initial.
        v (np.ndarray): Champ de vitesse v initial.
        ru, rv (float): Paramètres du modèle Gray-Scott.
        f, k (float): Paramètres de réaction.
        L (scipy.sparse.csr_matrix): Matrice Laplacienne.
        dt (float): Pas de temps.
        T (float): Durée totale de la simulation.
    
    Returns:
        np.ndarray: Champs de vitesse u et v après T secondes.
    """
    num_steps = int(T / dt)
    for _ in tqdm(range(num_steps)):
        u, v = strang.forward_step(u, v, ru, rv, f, k, L, dt)
    return u, v


# Backward simulation (reverse time)
def simulate_backward_strang_splitting(u, v, ru, rv, f, k, L, dt, T):
    """
    Simule le schéma de Strang-splitting à l'envers sur une durée T.
    
    Args:
        u (np.ndarray): Champ de vitesse u initial.
        v (np.ndarray): Champ de vitesse v initial.
        ru, rv (float): Paramètres du modèle Gray-Scott.
        f, k (float): Paramètres de réaction.
        L (scipy.sparse.csr_matrix): Matrice Laplacienne.
        dt (float): Pas de temps.
        T (float): Durée totale de la simulation.
    
    Returns:
        np.ndarray: Champs de vitesse u et v après T secondes en arrière.
    """
    num_steps = int(T / dt)
    for _ in tqdm(range(num_steps)):
        u, v = strang.backward_step(u, v, ru, rv, f, k, L, dt)
    return u, v


print(f"Nombre de pas de temps: {int(T / dt)}")

print("Démarrage de la simulation forward ...")
# Exécuter la simulation sur toute la durée T
u_final, v_final = simulate_strang_splitting(u, v, ru, rv, f, k, L, dt, T)
# Exécuter la simulation deux fois pour tester la stabilité
# print("Démarrage de la simulation forward (2 fois) ...")
# N = 1
# for _ in range(N):
#     u_final, v_final = simulate_strang_splitting(u_final, v_final, ru, rv, f, k, L, dt, T)

        
# ------------------------
# Affichage
# ------------------------
coef = 1.8
u_final +=  coef * v_final
plot_strang_splitting_results(u, v, u_final, v_final, img_u.shape, Lx, Ly)

# # Sauvegarder les résultats finaux
# save_array_as_image(u_final.reshape(img_u.shape), "u_final.png")
# save_array_as_image(v_final.reshape(img_v.shape), "v_final.png")



# print("Démarrage de la simulation backward ...")
# Exécuter la simulation à l'envers sur toute la durée T
print("Démarrage de la simulation backward (2 fois) ...")
# Exécuter la simulation à l'envers sur toute la durée T
# for _ in range(N+5):
#     u_backward, v_backward = simulate_backward_strang_splitting(u_final, v_final, ru, rv, f, k, L, dt, T)
#     # Pour tester la stabilité
u_backward, v_backward = simulate_backward_strang_splitting(u_final - coef * v_final, v_final, ru, rv, f, k, L, dt, T)

# Afficher les résultats de la simulation à l'envers
plot_strang_splitting_results(u_final, v_final, u_backward, v_backward, img_u.shape, Lx, Ly)