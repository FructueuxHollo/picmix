import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ------------------------
# Paramètres du modèle
# ------------------------
Nx, Ny = 64, 64
Lx, Ly = 2.5, 2.5
dx, dy = Lx / Nx, Ly / Ny
dt = 1.0

# Paramètres Gray-Scott
ru, rv = 0.16, 0.08
f, k = 0.060, 0.062

# Grille spatiale
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# ------------------------
# Initialisation
# ------------------------
def initial_conditions(Nx, Ny):
    u = np.ones((Nx, Ny))
    v = np.zeros((Nx, Ny))
    # Perturbation centrale
    r = 8
    u[Nx//2 - r:Nx//2 + r, Ny//2 - r:Ny//2 + r] = 0.50
    v[Nx//2 - r:Nx//2 + r, Ny//2 - r:Ny//2 + r] = 0.25
    return u, v

u, v = initial_conditions(Nx, Ny)

# ------------------------
# Laplacien 2D avec Neumann
# ------------------------
def laplacian_matrix(Nx, Ny, dx, dy):
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)

    Dx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
    Dx = Dx.tolil()
    Dx[0, 0:2] = [-2, 2]
    Dx[-1, -2:] = [2, -2]

    Dy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2
    Dy = Dy.tolil()
    Dy[0, 0:2] = [-2, 2]
    Dy[-1, -2:] = [2, -2]

    L = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)
    return L.tocsr()

L = laplacian_matrix(Nx, Ny, dx, dy)

# ------------------------
# Étapes élémentaires
# ------------------------
def reaction_rhs(u, v, f, k):
    uv2 = u * v**2
    du = -uv2 + f * (1 - u)
    dv = +uv2 - (f - k) * v
    return du, dv

def reaction_half_step(u, v, f, k, dt):
    tau = 0.5 * dt
    du1, dv1 = reaction_rhs(u, v, f, k)
    u_star = u + tau * du1
    v_star = v + tau * dv1
    du2, dv2 = reaction_rhs(u_star, v_star, f, k)
    u_new = u + tau * 0.5 * (du1 + du2)
    v_new = v + tau * 0.5 * (dv1 + dv2)
    return u_new, v_new

def diffusion_step(u, v, ru, rv, L, dt):
    N = u.size
    I = sp.eye(N)
    A_u = I - 0.5 * dt * ru * L
    B_u = I + 0.5 * dt * ru * L
    A_v = I - 0.5 * dt * rv * L
    B_v = I + 0.5 * dt * rv * L
    u_vec = u.flatten()
    v_vec = v.flatten()
    rhs_u = B_u @ u_vec
    rhs_v = B_v @ v_vec
    u_new = spla.spsolve(A_u, rhs_u).reshape(u.shape)
    v_new = spla.spsolve(A_v, rhs_v).reshape(v.shape)
    return u_new, v_new

def strang_forward_step(u, v, ru, rv, f, k, L, dt):
    u1, v1 = reaction_half_step(u, v, f, k, dt)
    u2, v2 = diffusion_step(u1, v1, ru, rv, L, dt)
    u_new, v_new = reaction_half_step(u2, v2, f, k, dt)
    return u_new, v_new

# ------------------------
# Simulation d'un pas
# ------------------------
u_new, v_new = strang_forward_step(u, v, ru, rv, f, k, L, dt)

# ------------------------
# Affichage
# ------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
titles = ["u initial", "v initial", "u après 1 pas", "v après 1 pas"]
data = [u, v, u_new, v_new]

for ax, title, d in zip(axes.flat, titles, data):
    im = ax.imshow(d, cmap="inferno", origin="lower", extent=[0, Lx, 0, Ly])
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
