import numpy as np
import hashlib
from typing import Dict, Any, Tuple
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import root
from tqdm import tqdm

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

    def __init__(self, key: str, image_shape: Tuple[int, int], config: Dict[str, Any] = None):
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
        
        # --- 1. Configuration ---
        self.config = {
            'dt': 1.0,
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
        
        print("VortexCrypt Engine initialized.")
        print(f"  - Padded domain size: {self.Nx}x{self.Ny}")
        print(f"  - Derived params: f={self.params['f_rate']:.4f}, k={self.params['k_rate']:.4f}, "
              f"ru={self.params['ru_rate']:.4f}, T={self.params['T']:.2f} ({self.params['n_steps']} steps)")

    def encrypt(self, image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the forward encryption process.

        Args:
            image_array (np.ndarray): The original normalized image array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The final flattened states u(T) and v(T).
        """
        print("\n--- Running Encryption Pass ---")
        u0_padded = np.pad(
            image_array.astype(np.float64),
            pad_width=self.config['pad_width'],
            mode='reflect'
        )
        
        u_final_flat, v_final_flat = self._simulate_pass(
            u_initial=u0_padded.flatten(),
            v_initial=self.v0.flatten(),
            forward=True
        )
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
        u_decrypted_flat, _ = self._simulate_pass(
            u_initial=u_final_flat,
            v_initial=v_final_flat,
            forward=False
        )
        
        decrypted_padded = u_decrypted_flat.reshape(self.padded_shape)
        pad = self.config['pad_width']
        decrypted_image = decrypted_padded[pad:-pad, pad:-pad]
        
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
        
        y_coords, x_coords = np.ogrid[:self.Ny, :self.Nx]

        for _ in range(num_kernels):
            xc = self.prng.integers(0, self.Nx)
            yc = self.prng.integers(0, self.Ny)
            A = self.prng.uniform(0.1, 0.5)
            sigma = self.prng.uniform(5.0, 15.0)
            
            dist_sq = (x_coords - xc)**2 + (y_coords - yc)**2
            v0 += A * np.exp(-dist_sq / (2 * sigma**2))
            
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
        """Implicit reaction step using midpoint rule, solved per-pixel."""
        tau = 0.5 * dt
        u0_2d, v0_2d = u_flat.reshape(self.padded_shape), v_flat.reshape(self.padded_shape)
        u1_2d, v1_2d = np.zeros_like(u0_2d), np.zeros_like(v0_2d)
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                u0, v0 = u0_2d[i, j], v0_2d[i, j]

                def F(w1):
                    u1, v1 = w1
                    u_bar, v_bar = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
                    uv2 = u_bar * v_bar**2
                    eq_u = u1 - u0 - tau * (-uv2 + f * (1 - u_bar))
                    eq_v = v1 - v0 - tau * (uv2 - (f + k) * v_bar)
                    return [eq_u, eq_v]

                sol = root(F, [u0, v0], method='hybr')
                if not sol.success:
                    raise RuntimeError(f"Non-linear solver failed at pixel ({i}, {j}).")
                u1_2d[i, j], v1_2d[i, j] = sol.x

        return u1_2d.flatten(), v1_2d.flatten()