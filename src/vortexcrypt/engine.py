import numpy as np
import hashlib
from typing import Dict, Any, Tuple
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import root
from tqdm import tqdm
from numba import jit

@jit(nopython=True, fastmath=True)
def _newton_raphson_step_2x2(u0, v0, u1, v1, f, k, tau):
    """Effectue une seule itération de Newton-Raphson pour le système 2x2."""
    # Évaluation au point milieu
    u_bar, v_bar = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    
    # Calcul des résidus (valeur de la fonction F)
    uv2 = u_bar * v_bar**2
    res_u = u1 - u0 - tau * (-uv2 + f * (1 - u_bar))
    res_v = v1 - v0 - tau * (uv2 - (f + k) * v_bar)
    
    # Calcul du Jacobien
    
    j11 = 1.0 + 0.5 * tau * (-v_bar**2 + f)
    j12 = tau * u_bar * v_bar
    j21 = -0.5 * tau * v_bar**2
    j22 = 1.0 - 0.5 * tau * (2 * u_bar * v_bar - (f + k))
    
    # Inversion de la matrice Jacobienne 2x2
    det = j11 * j22 - j12 * j21
    safe_mask = np.abs(det)> 1e-12

    # Initialiser les mises à jour à zéro
    delta_u = np.zeros_like(u0)
    delta_v = np.zeros_like(v0)

    # Appliquer le masque 1D. Numba supporte cela.
    safe_det = det[safe_mask]
    safe_j11, safe_j12 = j11[safe_mask], j12[safe_mask]
    safe_j21, safe_j22 = j21[safe_mask], j22[safe_mask]
    safe_res_u, safe_res_v = res_u[safe_mask], res_v[safe_mask]
    
    inv_det = 1.0 / safe_det
    
    # Calcul du pas de Newton uniquement sur les éléments sûrs
    safe_delta_u = -( (safe_j22 * inv_det) * safe_res_u + (-safe_j12 * inv_det) * safe_res_v )
    safe_delta_v = -( (-safe_j21 * inv_det) * safe_res_u + (safe_j11 * inv_det) * safe_res_v )
    
    # Placer les deltas calculés dans les tableaux de mise à jour complets
    delta_u[safe_mask] = safe_delta_u
    delta_v[safe_mask] = safe_delta_v
    
    # Mise à jour de la solution
    u1_new = u1 + delta_u
    v1_new = v1 + delta_v
    
    return u1_new, v1_new

@jit(nopython=True, fastmath=True)
def _reaction_solver_numba(u_flat, v_flat, f, k, tau, max_iter=5, tol=1e-6):
    """
    Solveur de Newton-Raphson vectorisé pour tableaux 1D.
    """
    u0 = u_flat
    v0 = v_flat
    
    # L'estimation initiale est simplement l'état précédent
    u1 = u0.copy()
    v1 = v0.copy()

    for _ in range(max_iter):
        u1_old, v1_old = u1.copy(), v1.copy()
        
        # L'itération de Newton est appliquée à tous les pixels en même temps
        u1, v1 = _newton_raphson_step_2x2(u0, v0, u1, v1, f, k, tau)
        
        # Vérification de la convergence
        error_u = np.abs(u1 - u1_old)
        error_v = np.abs(v1 - v1_old)
        if np.max(error_u) < tol and np.max(error_v) < tol:
            break
            
    return u1, v1

class VortexCryptEngine:
    """
    Core engine for encryption/decryption based on the Gray-Scott
    reaction-diffusion model and a time-reversible Strang-splitting integrator.
    """
    # Parameter ranges known to produce complex patterns ("mitosis")
    F_RATE_RANGE: Tuple[float, float] = (0.01, 0.1)
    K_RATE_RANGE: Tuple[float, float] = (0.045, 0.07)
    RU_RATE_RANGE: Tuple[float, float] = (0.1, 0.2)
    TIME_RANGE: Tuple[float, float] = (20.0, 100.0) # Total simulation time

    def __init__(self, key: str, image_shape: Tuple, config: Dict[str, Any] = None):
        """
        Initializes the simulation engine.

        Args:
            key (str): The secret key (8-24 characters).
            image_shape (Tuple[int, int]): The (height, width) of the original image.
            config (Dict, str, Any], optional): Dictionary to override default parameters.
        """
        if not (8 <= len(key) <= 24):
            raise ValueError("Key must be between 8 and 24 characters long.")

        self.key = key
        self.original_shape = image_shape
        self.is_color = len(image_shape) == 3
        
        # --- 1. Configuration ---
        self.config = {
            'dt': 0.2,
            'pad_width': 1
        }
        if config:
            self.config.update(config)
            
        self.padded_shape = (
            self.original_shape[0] + 2 * self.config['pad_width'],
            self.original_shape[1] + 2 * self.config['pad_width']
        )
        self.Nx, self.Ny = self.padded_shape
        
        # --- 2. Parameter & Field Initialization ---
        self.params: Dict[str, Any] = {}
        self._derive_parameters()

        self.v0 = self._synthesize_initial_catalyst()
        self.L_op = self._laplacian_matrix()
        
        print(f"VortexCrypt Engine initialized for {'COLOR' if self.is_color else 'GRAYSCALE'} image.")

    def encrypt(self, image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the forward encryption process.

        Args:
            image_array (np.ndarray): The original normalized image array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The final flattened states u(T) and v(T).
        """
        print("\n--- Running Encryption Pass ---")
        pad_width = ((self.config['pad_width'], self.config['pad_width']),
                     (self.config['pad_width'], self.config['pad_width']),
                     (0, 0)) if self.is_color else self.config['pad_width']
        u0_padded = np.pad(
            image_array.astype(np.float64),
            pad_width=pad_width,
            mode='reflect'
        )

        v_initial_flat = self.v0.flatten()
        if self.is_color:
            u_final_channels = []
            for i in range(3):
                print(f"  Encrypting channel {i+1}/3...")
                u_initial_flat = u0_padded[:, :, i].flatten()
                u_final_c, _ = self._simulate_pass(u_initial_flat, v_initial_flat, forward=True, show_progress=False)
                u_final_channels.append(u_final_c)
            print("  Finalizing catalyst field...")
            _, v_final_flat = self._simulate_pass(u_initial_flat, v_initial_flat, forward=True, show_progress=True)
            u_final_flat = np.stack(u_final_channels, axis=-1).flatten()
        else:
            u_initial_flat = u0_padded.flatten()
            u_final_flat, v_final_flat = self._simulate_pass(u_initial_flat, v_initial_flat, forward=True)
        
        return u_final_flat, v_final_flat

    def decrypt(self, u_final_flat: np.ndarray, v_final_flat: np.ndarray) -> np.ndarray:
        """
        Runs the backward decryption process.

        Args:
            u_final_flat (np.ndarray): The final encrypted u-field (flattened).
            v_final_flat (np.ndarray): The final catalyst v-field (flattened).

        Returns:
            np.ndarray: The decrypted 2D image array.
        """
        print("\n--- Running Decryption Pass ---")

        if self.is_color:
            u_final_channels_flat = u_final_flat.reshape(-1, 3)
            decrypted_channels = []
            for i in range(3):
                print(f"  Decrypting channel {i+1}/3...")
                u_final_c = u_final_channels_flat[:, i]
                u_decrypted_c, _ = self._simulate_pass(u_final_c, v_final_flat, forward=False, show_progress=False)
                decrypted_channels.append(u_decrypted_c.reshape(self.padded_shape))
            
            decrypted_padded = np.stack(decrypted_channels, axis=-1)
        else:
            u_decrypted_flat, _ = self._simulate_pass(u_final_flat, v_final_flat, forward=False)
            decrypted_padded = u_decrypted_flat.reshape(self.padded_shape)

        pad = self.config['pad_width']
        decrypted_image = decrypted_padded[pad:-pad, pad:-pad, :] if self.is_color else decrypted_padded[pad:-pad, pad:-pad]
        
        return decrypted_image

    # ======================================================================
    # Private methods for initialization and simulation
    # ======================================================================

    def _derive_parameters(self):
        """Derives Gray-Scott model parameters from the key (Paper's Algorithm 1)."""
        hash_digest = hashlib.sha256(self.key.encode()).digest()
        seed = int.from_bytes(hash_digest[:4], 'big')
        self.prng = np.random.default_rng(seed)

        map_range = lambda x, r: r[0] + x * (r[1] - r[0])

        self.params['f_rate'] = map_range(self.prng.random(), self.F_RATE_RANGE)
        self.params['k_rate'] = map_range(self.prng.random(), self.K_RATE_RANGE)
        self.params['ru_rate'] = map_range(self.prng.random(), self.RU_RATE_RANGE)
        self.params['rv_rate'] = self.params['ru_rate'] / 2.0
        self.params['T'] = map_range(self.prng.random(), self.TIME_RANGE)
        
        self.params['n_steps'] = int(self.params['T'] / self.config['dt'])

    def _synthesize_initial_catalyst(self) -> np.ndarray:
        """Generates the initial catalyst field v0 (Paper's Algorithm 2)."""
        v0 = np.zeros(self.padded_shape)
        num_kernels = self.prng.integers(3, 8)
        
        for _ in range(num_kernels):
            xc = self.prng.integers(0, self.Nx)
            yc = self.prng.integers(0, self.Ny)
            A = self.prng.uniform(0.1, 0.5)
            sigma = self.prng.uniform(5.0, 15.0)
            
            for i in range(self.Nx):
                for j in range(self.Ny):
                    dist_sq = (i - xc)**2 + (j - yc)**2
                    v0[i, j] += A * np.exp(-dist_sq / (2 * sigma**2))
            
        return v0

    def _laplacian_matrix(self) -> sp.csr_matrix:
        """Builds the 2D Laplacian matrix with Neumann boundary conditions."""
        dx2, dy2 = (1.0 / self.Nx)**2, (1.0 / self.Ny)**2 # Domain size is normalized
        
        Ix = sp.eye(self.Nx)
        Iy = sp.eye(self.Ny)
        
        Dx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(self.Nx, self.Nx)) * dx2
        Dy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(self.Ny, self.Ny)) * dy2
        
        Dx, Dy = Dx.tolil(), Dy.tolil()
        Dx[0, :2], Dx[-1, -2:] = [2, -2], [2, -2]
        Dy[0, :2], Dy[-1, -2:] = [2, -2], [2, -2]
        
        return (sp.kron(Iy, Dx) + sp.kron(Dy, Ix)).tocsr()
    
    def _simulate_pass(self, u_initial, v_initial, forward=True, show_progress=True):
        """Runs a full simulation pass (forward or backward)."""
        u, v = u_initial.copy(), v_initial.copy()
        dt = self.config['dt'] if forward else -self.config['dt']
        p = self.params
        
        desc = "Encrypting" if forward else "Decrypting"
        iterator = tqdm(range(p['n_steps']), desc=desc, ncols=80) if show_progress else range(p['n_steps'])

        for _ in iterator:
            u, v = self._strang_step(u, v, p['ru_rate'], p['rv_rate'], p['f_rate'], p['k_rate'], self.L_op, dt)
        
        return u, v
        
    # --- Strang-Splitting Numerical Scheme ---
    
    def _strang_step(self, u, v, ru, rv, f, k, L, dt):
        """Performs one full time-reversible Strang-splitting step."""
        u1, v1 = self._reaction_half_step(u, v, f, k, dt)
        u2, v2 = self._diffusion_step(u1, v1, ru, rv, L, dt)
        u_new, v_new = self._reaction_half_step(u2, v2, f, k, dt)
        return u_new, v_new
        
    def _diffusion_step(self, u, v, ru, rv, L, dt):
        """Implicit diffusion step using Crank-Nicolson."""
        N = u.size
        I = sp.eye(N)
        A_u, B_u = I - 0.5 * dt * ru * L, I + 0.5 * dt * ru * L
        A_v, B_v = I - 0.5 * dt * rv * L, I + 0.5 * dt * rv * L
        u_new = spla.spsolve(A_u, B_u @ u)
        v_new = spla.spsolve(A_v, B_v @ v)
        return u_new, v_new

    def _reaction_half_step(self, u_flat, v_flat, f, k, dt):
        """Implicit reaction step using a JIT-compiled vectorized Newton-Raphson solver."""
        tau = 0.5 * dt
        u_new_flat, v_new_flat = _reaction_solver_numba(
            u_flat, v_flat, f, k, tau
        )
        
        # clipping results to ensure stability
        np.clip(u_new_flat, 0, 2, out=u_new_flat)
        np.clip(v_new_flat, 0, 2, out=v_new_flat)

        return u_new_flat, v_new_flat