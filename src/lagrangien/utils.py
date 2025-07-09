import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image
from tqdm import tqdm


def semi_lagrangian_decrypt(rho_final, u_list, dt, dx=1.0, dy=1.0):
    """
    Décrypte une image encryptée par advection semi-Lagrangienne.

    Args:
        rho_final (np.ndarray): Image encryptée (H, W), normalisée [0,1].
        u_list (list): Liste des champs de vitesse (2, H, W), ordre chronologique.
        dt (float): Pas de temps utilisé à l'encryption.
        dx, dy (float): Pas spatial (par défaut 1.0).

    Returns:
        np.ndarray: Image décryptée (H, W).
    """
    rho = rho_final.copy()
    H, W = rho.shape

    # Grille spatiale
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # Parcours des champs de vitesse à l'envers (remonter le temps)
    for u in reversed(u_list):
        ux, uy = u[0], u[1]

        # Coordonnées d'arrivée de la particule (marche arrière)
        X_forw = X + ux * dt / dx
        Y_forw = Y + uy * dt / dy

        # Interpolation bilinéaire dans rho
        coords = np.array([Y_forw.flatten(), X_forw.flatten()])
        rho = map_coordinates(rho, coords, order=1, mode='reflect').reshape(H, W)

    return rho


def semi_lagrangian_encrypt(rho0, u_list, dt, dx=1.0, dy=1.0):
    """
    Applique l'encryption semi-Lagrangienne en utilisant le champ u(x, t).

    Args:
        rho0 (np.ndarray): image d'origine (H, W), normalisée [0,1].
        u_list (list): liste de champs de vitesse u^n (chacun de forme (2, H, W)).
        dt (float): pas de temps.
        dx, dy (float): taille de grille en x et y.

    Returns:
        np.ndarray: image encryptée.
    """
    rho = rho0.copy()
    H, W = rho.shape

    # Création des coordonnées du maillage
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    for u in u_list:
        ux, uy = u[0], u[1]

        # Position d'où vient chaque pixel (pied de caractéristique)
        X_back = X - ux * dt / dx
        Y_back = Y - uy * dt / dy

        # Interpolation bilinéaire sur rho
        coords = np.array([Y_back.flatten(), X_back.flatten()])
        rho = map_coordinates(rho, coords, order=1, mode='reflect').reshape(H, W)

    return rho

def generate_velocity_field(shape, steps=100, dt=0.1, nu=0.01):
    """
    Génère une séquence déterministe de champs de vitesse u(x, y, t) pour VortexCrypt,
    sans utiliser de clé pour l'instant.

    Args:
        shape (tuple): Dimensions de l'image (H, W).
        steps (int): Nombre de pas de temps.
        dt (float): Pas de temps.
        nu (float): Viscosité.

    Returns:
        list[np.ndarray]: Liste des champs u^n (forme (2, H, W)) pour chaque pas de temps.
    """
    H, W = shape

    # Initialisation déterministe de u(x, y, 0)
    u = np.zeros((2, H, W))
    
    # Initialisation  des forces f(x, y) (stationnaires dans le temps)
    f = np.zeros((2, H, W))
    # Champ initial : simple vortex (ou bruit léger fixe)
    for i in range(H):
        for j in range(W):
            u[0, i, j] = np.sin(2 * np.pi * i / H) 
            u[1, i, j] = np.cos(2 * np.pi * j / W) * np.sin(2 * np.pi * i / H)

            # Terme source fixe (stationnaire dans le temps)
            f[0, i, j] = np.sin(2 * np.pi * j / W) * np.cos(2 * np.pi * i / H)
            f[1, i, j] = np.cos(2 * np.pi * i / H) 

    u_list = [u.copy()]

    for _ in range(steps):
        grad_u = compute_advection(u, dx=1.0, dy=1.0)
        u = u + dt * (-grad_u + f)
        u = diffuse_fft(u, nu, dt)
        u_list.append(u.copy())

    return u_list


def compute_advection(u, dx=1.0, dy=1.0):
    """
    Approxime (u · ∇)u pour un champ u = [u_x, u_y].
    """
    ux, uy = u[0], u[1]

    dudx = np.gradient(ux, dx, axis=1)
    dudy = np.gradient(ux, dy, axis=0)
    dvdx = np.gradient(uy, dx, axis=1)
    dvdy = np.gradient(uy, dy, axis=0)

    adv_x = ux * dudx + uy * dudy
    adv_y = ux * dvdx + uy * dvdy

    return np.array([adv_x, adv_y])


def diffuse_fft(u, nu, dt):
    """
    Applique la diffusion de Burgers via transformée de Fourier.
    """
    H, W = u.shape[1:]
    kx = np.fft.fftfreq(W) * 2 * np.pi
    ky = np.fft.fftfreq(H) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    u_new = np.zeros_like(u)
    for i in range(2):  # composantes x et y
        u_hat = np.fft.fft2(u[i])
        u_hat *= np.exp(-nu * K2 * dt)
        u_new[i] = np.real(np.fft.ifft2(u_hat))
    return u_new
