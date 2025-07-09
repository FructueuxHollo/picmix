import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.optimize import root

class StrangSplitting:
    def __init__(self, Nx, Ny):
        """
        Initialise les dimensions du domaine.
        
        Args:
            Nx (int): Nombre de points en x.
            Ny (int): Nombre de points en y.
        """
        self.Nx = Nx
        self.Ny = Ny
        
    def reaction_rhs(self, u, v, f, k):
        """ Calcule les termes de réaction pour u et v """
        uv2 = u * v**2
        du = -uv2 + f * (1 - u)
        dv = +uv2 - (f - k) * v
        return du, dv


    # Étape 1 : Sous-pas réaction (demi-pas)
    def reaction_half_step(self, u, v, f, k, dt):
        """
        Effectue un demi-pas de temps pour la partie réaction seule.
        Schéma de Heun (ordre 2).
        """
        tau = 0.5 * dt

        # k1 = R(u, v)
        du1, dv1 = self.reaction_rhs(u, v, f, k)

        # estimation intermédiaire (Euler)
        u_star = u + tau * du1
        v_star = v + tau * dv1

        # k2 = R(u*, v*)
        du2, dv2 = self.reaction_rhs(u_star, v_star, f, k)

        # Heun update (moyenne des pentes)
        u_new = u + tau * 0.5 * (du1 + du2)
        v_new = v + tau * 0.5 * (dv1 + dv2)

        return u_new, v_new
    
    # Étape 2 : Sous-pas diffusion (pas entier)
    
    def diffusion_step(self, u, v, ru, rv, L, dt):
        """Applique Crank–Nicolson pour la diffusion de u et v."""
        N = u.size
        I = sp.eye(N)

        # Assemblage des matrices
        A_u = I - 0.5 * dt * ru * L
        B_u = I + 0.5 * dt * ru * L

        A_v = I - 0.5 * dt * rv * L
        B_v = I + 0.5 * dt * rv * L

        # Résolution du système linéaire
        u_vec = u.flatten()
        v_vec = v.flatten()

        rhs_u = B_u @ u_vec
        rhs_v = B_v @ v_vec

        u_new = spla.spsolve(A_u, rhs_u).reshape(u.shape)
        v_new = spla.spsolve(A_v, rhs_v).reshape(v.shape)

        return u_new, v_new
    

    def reaction_half_step_midpoint_implicit(self, u, v, f, k, dt):
        """
        Intègre la réaction sur un demi-pas de temps (tau = dt/2)
        avec la méthode du midpoint implicite.
        """
        tau = 0.5 * dt
        u_ = u.copy().reshape(self.Nx, self.Ny)
        v_ = v.copy().reshape(self.Nx, self.Ny)
        
        u_new = np.zeros((self.Nx, self.Ny))
        v_new = np.zeros((self.Nx, self.Ny))
        # print(u.shape, v.shape)
        Nx, Ny = self.Nx, self.Ny
        for i in range(Nx):
            for j in range(Ny):
                u0 = u_[i, j]
                v0 = v_[i, j]

                # Fonction F à annuler : F([u1, v1]) = [eq_u, eq_v]
                def F(w1):
                    u1, v1 = w1
                    u_bar = 0.5 * (u0 + u1)
                    v_bar = 0.5 * (v0 + v1)
                    uv2 = u_bar * v_bar**2
                    fu = -uv2 + f * (1 - u_bar)
                    fv = +uv2 - (f - k) * v_bar
                    eq_u = u1 - u0 - tau * fu
                    eq_v = v1 - v0 - tau * fv
                    return [eq_u, eq_v]

                # Résolution par Newton (ou hybr)
                sol = root(F, [u0, v0], method='hybr')

                if sol.success:
                    u_new[i, j], v_new[i, j] = sol.x
                else:
                    raise RuntimeError(f"Newton failed at ({i}, {j}): {sol.message}")

        return u_new.flatten(), v_new.flatten()
    
    # Étape 3 : Sous-pas réaction (dernier demi-pas)
    def forward_step(self, u, v, ru, rv, f, k, L, dt):
        """Effectue un pas complet forward avec Strang–splitting réversible."""
        # 1. Réaction demi-pas
        # u1, v1 = self.reaction_half_step_midpoint_implicit(u, v, f, k, dt)
        u1, v1 = self.reaction_half_step(u, v, f, k, dt)

        # 2. Diffusion pas complet
        u2, v2 = self.diffusion_step(u1, v1, ru, rv, L, dt)

        # 3. Réaction demi-pas final
        # u_new, v_new = self.reaction_half_step_midpoint_implicit(u2, v2, f, k, dt)
        u_new, v_new = self.reaction_half_step(u2, v2, f, k, dt)

        return u_new, v_new
    
    def backward_step(self, u, v, ru, rv, f, k, L, dt): 
        """Applique le schéma de Strang-splitting à l’envers (reverse time)."""
        # 1. Réaction demi-pas (temps négatif)
        # u1, v1 = self.reaction_half_step_midpoint_implicit(u, v, f, k, -dt)
        u1, v1 = self.reaction_half_step(u, v, f, k, -dt)

        # 2. Diffusion temps négatif
        u2, v2 = self.diffusion_step(u1, v1, ru, rv, L, -dt)

        # 3. Réaction demi-pas final (temps négatif)
        # u_back, v_back = self.reaction_half_step_midpoint_implicit(u2, v2, f, k, -dt)
        u_back, v_back = self.reaction_half_step(u2, v2, f, k, -dt)

        return u_back, v_back
    
    def error_metric(self, u1: np.ndarray, u2: np.ndarray) -> float:
        """
        Calcule l'erreur entre deux tableaux d'images en utilisant la MSE (Mean Squared Error).

        Args:
            u1 (np.ndarray): Premier tableau d'image.
            u2 (np.ndarray): Deuxième tableau d'image.

        Returns:
            float: La MSE entre les deux tableaux.
        """
        if u1.shape != u2.shape:
            print("Les images n'ont pas la même taille !")
            return float('inf')
        
        mse = np.mean((u1 - u2) ** 2)
        return mse
    
def laplacian_matrix(Nx, Ny, dx, dy):
        """Construit la matrice Laplacienne 2D avec conditions de Neumann."""
        Ix = sp.eye(Nx)
        Iy = sp.eye(Ny)

        # Laplacien 1D sur x
        Dx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
        Dx = Dx.tolil()
        Dx[0, 0:2] = [-2, 2]  # Neumann
        Dx[-1, -2:] = [2, -2]

        # Laplacien 1D sur y
        Dy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2
        Dy = Dy.tolil()
        Dy[0, 0:2] = [-2, 2]
        Dy[-1, -2:] = [2, -2]

        L = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)
        return L.tocsr()


def generate_random_2d_array(shape: tuple) -> np.ndarray:
    """
    Génère un tableau 2D aléatoire de la forme spécifiée.

    Args:
        shape (tuple): Les dimensions du tableau (hauteur, largeur).

    Returns:
        np.ndarray: Un tableau 2D aléatoire.
    """
    return np.random.rand(*shape)

def add_padding(image: np.ndarray, pad_width: int) -> np.ndarray:
    """
    Ajoute un rembourrage (padding) autour de l'image.

    Args:
        image (np.ndarray): L'image d'entrée.
        pad_width (int): La largeur du rembourrage à ajouter.

    Returns:
        np.ndarray: L'image avec le rembourrage ajouté.
    """
    return np.pad(image, pad_width=pad_width, mode='reflect')

def load_image_as_array(image_path: str) -> np.ndarray:
    """
    Charge une image à partir d'un chemin de fichier, la convertit en niveaux de gris
    et la retourne sous forme de tableau NumPy normalisé (valeurs entre 0.0 et 1.0).

    Args:
        image_path (str): Le chemin vers le fichier image.

    Returns:
        np.ndarray: Le tableau NumPy représentant l'image en niveaux de gris.
    """
    try:
        img = Image.open(image_path)
        # Convertir en niveaux de gris ('L' pour Luminance)
        img_gray = img.convert('L')
        # Convertir en tableau NumPy et normaliser
        img_array = np.array(img_gray, dtype=np.float64) / 255.0
        return img_array
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{image_path}' n'a pas été trouvé.")
        raise
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {e}")
        raise

def save_array_as_image(image_array: np.ndarray, output_path: str):
    """
    Sauvegarde un tableau NumPy en tant que fichier image.
    Le tableau est dé-normalisé (multiplié par 255) et converti en entiers 8 bits.

    Args:
        image_array (np.ndarray): Le tableau à sauvegarder.
        output_path (str): Le chemin où sauvegarder l'image.
    """
    try:
        # S'assurer que les valeurs sont bien entre 0 et 1 avant de convertir
        image_array = np.clip(image_array, 0.0, 1.0)
        
        # Dé-normaliser et convertir en entiers 8-bits non signés
        img_data = (image_array * 255).astype(np.uint8)
        
        # Créer une image PIL à partir du tableau
        img = Image.fromarray(img_data, 'L')
        
        # Sauvegarder l'image
        img.save(output_path)
        print(f"Image sauvegardée avec succès à l'emplacement : '{output_path}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image: {e}")
        raise
    


def plot_strang_splitting_results(u, v, u_new, v_new, img_shape, Lx, Ly):
    """
    Affiche les résultats du schéma de Strang-splitting pour u et v avant et après un pas.

    Args:
        u (np.ndarray): Tableau u initial.
        v (np.ndarray): Tableau v initial.
        u_new (np.ndarray): Tableau u après un pas.
        v_new (np.ndarray): Tableau v après un pas.
        img_shape (tuple): La forme de l'image (hauteur, largeur).
        Lx (float): Taille du domaine en x.
        Ly (float): Taille du domaine en y.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["u initial", "v initial", "u final", "v final"]
    data = [u, v, u_new, v_new]

    for ax, title, d in zip(axes.flat, titles, data):
        im = ax.imshow(d.reshape(img_shape), cmap="inferno", origin="lower", extent=[0, Lx, 0, Ly])
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()