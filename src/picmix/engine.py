import numpy as np
import hashlib
from typing import Tuple, Dict, Any

class PicMixEngine:
    """
    Classe principale pour gérer la simulation de chiffrement/déchiffrement d'images
    basée sur les équations de la dynamique des fluides (modèle de Burgers).
    """
    def __init__(self, image: np.ndarray, key: str, config: Dict[str, Any] = None):
        """
        Initialise le moteur de simulation.

        Args:
            image (np.ndarray): L'image sous forme de tableau NumPy (normalisée entre 0 et 1).
            key (str): La clé secrète fournie par l'utilisateur.
            config (Dict, optional): Dictionnaire pour surcharger les paramètres par défaut.
        """
        print("Initializing PicMix Engine...")
        
        # --- 1. Configuration par défaut ---
        # Ces valeurs peuvent être ajustées pour modifier le comportement du chiffrement.
        self.config = {
            'time_steps': 50,      # Nombre total de pas de temps (équivalent à T/dt)
            'dt': 0.1,             # Taille du pas de temps
            'default_viscosity': 0.1 # Viscosité par défaut si la clé ne permet pas d'en calculer une
        }
        if config:
            self.config.update(config)

        # --- 2. Discrétisation Spatiale ---
        # L'image d'entrée définit notre domaine spatial.
        # ρ (rho) représente la densité du "colorant", c'est-à-dire l'intensité des pixels.
        self.rho = image.astype(np.float64)
        self.height, self.width = self.rho.shape
        print(f"Domain size: {self.width}x{self.height}")

        # Grille de coordonnées (utile pour les fonctions f et g)
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Pas spatiaux (h dans les équations)
        self.dx = 1.0 / (self.width - 1)
        self.dy = 1.0 / (self.height - 1)

        # --- 3. Initialisation des champs ---
        # u est le champ de vitesse [vx, vy]. On le stocke comme un tableau 3D.
        # Le premier canal (0) est vx, le second (1) est vy.
        self.u = np.zeros((self.height, self.width, 2), dtype=np.float64)
        
        # Historique complet du champ de vitesse. C'est crucial pour le déchiffrement.
        # On stocke u(t) pour chaque pas de temps.
        self.u_history = np.zeros((self.config['time_steps'] + 1, self.height, self.width, 2), dtype=np.float64)

        # Le papier mentionne les paramètres ν, T, f, g. Ils seront générés
        # à l'étape suivante à partir de la clé.
        self.key = key
        self.parameters = {} # Sera rempli par _derive_parameters_from_key

        # Appel de la méthode de génération
        self._derive_parameters_from_key()

        print("Engine initialized.")


    def run_encryption(self) -> np.ndarray:
        """Exécute le processus de chiffrement complet."""
        print("--- Starting Encryption Process ---")
        
        # 1. (Déjà fait dans __init__) Générer les paramètres à partir de la clé
        
        # 2. Calculer l'historique du champ de vitesse u(t)
        self._compute_velocity_history()
        
        # 3. Advecter l'image rho(t) vers l'avant dans le temps.
        encrypted_rho = self._advect_pixels(forward=True)

        print("--- Encryption Process Finished ---")
        return encrypted_rho

    def run_decryption(self) -> np.ndarray:
        """Exécute le processus de déchiffrement complet."""
        print("--- Starting Decryption Process ---")

        # 1. (Déjà fait dans __init__) Générer les mêmes paramètres à partir de la clé
        
        # 2. Re-calculer le même historique du champ de vitesse u(t)
        self._compute_velocity_history()

        # 3. Advecter l'image rho(t) vers l'arrière dans le temps.
        decrypted_rho = self._advect_pixels(forward=False)
        
        print("--- Decryption Process Finished ---")
        return decrypted_rho
    
    def _derive_parameters_from_key(self):
        """
        Génère tous les paramètres de simulation de manière déterministe à partir de la clé secrète.
        Ceci implémente la logique de la section 4.1 du papier.
        """
        print(f"Deriving parameters from key: '{self.key}'")
        
        # --- 1. Génération de la graine (Seed) via SHA-256 ---
        # On transforme la clé string en un entier de haute entropie.
        hash_object = hashlib.sha256(self.key.encode())
        hash_digest = hash_object.digest() # On obtient un hash de 32 octets

        # La classe np.random.RandomState attend une graine de 32 bits (4 octets).
        # On prend les 4 premiers octets du hachage pour créer notre graine.
        seed_bytes = hash_digest[:4]
        seed = int.from_bytes(seed_bytes, 'big')
        
        # On initialise le générateur de nombres pseudo-aléatoires (PRNG) de NumPy
        # avec cette graine. Cela garantit que toutes les opérations "aléatoires"
        # qui suivent seront identiques pour la même clé.
        self.prng = np.random.RandomState(seed)
        
        # --- 2. Dérivation de la Viscosité (ν) ---
        # On implémente l'Algorithme 1 du papier.
        digits = []
        for char in self.key:
            # On traite chaque chiffre des codes ASCII des caractères de la clé.
            for digit_char in str(ord(char)):
                digit = int(digit_char)
                if digit != 0:
                    digits.append(digit)
        
        if len(set(digits)) > 1:
            min_val = min(digits)
            max_val = max(digits)
            viscosity = min_val / max_val
        else:
            # Cas par défaut si la clé est trop simple (ex: "11111111")
            viscosity = self.config['default_viscosity']
            
        self.parameters['viscosity'] = viscosity
        
        # --- 3. Synthèse des Fonctions Source (f) et de Bord (g) ---
        # On implémente l'Algorithme 2. On génère 4 fonctions : f_x, f_y, g_x, g_y.
        
        # Base de fonctions candidates (comme dans l'Algorithme 2)
        # Chaque fonction prend en entrée (X, Y, params)
        self.function_templates = [
            lambda X, Y, p: p[0] * np.sin(p[1] * X + p[2] * Y + p[3]),
            lambda X, Y, p: p[0] * np.cos(p[1] * X) * np.sin(p[2] * Y + p[3]),
            lambda X, Y, p: p[0] * np.exp(-((X - p[1])**2 + (Y - p[2])**2) / p[3]**2),
            lambda X, Y, p: p[0] * X**2 + p[1] * Y**2 + p[2] * X * Y + p[3],
            lambda X, Y, p: p[0] * np.tanh(p[1] * X + p[2] * Y + p[3])
        ]
        
        def synthesize_function():
            """Petite fonction d'aide pour synthétiser une fonction aléatoire."""
            # Choisir un template de fonction au hasard
            template_idx = self.prng.randint(0, len(self.function_templates))
            template = self.function_templates[template_idx]
            
            # Générer des paramètres aléatoires pour ce template
            # On génère 4 paramètres dans une plage raisonnable [-2, 2]
            # Le dernier paramètre pour le Gaussien doit être > 0
            params = self.prng.uniform(-2, 2, size=4)
            if template_idx == 2: # Cas du Gaussien, l'écart-type p[3] doit être positif
                params[3] = self.prng.uniform(0.1, 1.0)
                
            # Retourne une fonction "callable" qui a déjà ses paramètres fixés.
            # C'est une "closure" en programmation.
            return lambda X, Y: template(X, Y, params)

        # On génère les 4 fonctions dont on a besoin pour le modèle
        self.parameters['f_x_func'] = synthesize_function()
        self.parameters['f_y_func'] = synthesize_function()
        # Pour l'instant on ne gère pas les conditions de bord 'g', on les met à zero.
        # C'est une simplification pour commencer.
        self.parameters['g_x_func'] = lambda X, Y: 0.0
        self.parameters['g_y_func'] = lambda X, Y: 0.0

        # On évalue ces fonctions une seule fois sur notre grille pour obtenir les tableaux de forçage
        self.f_x = self.parameters['f_x_func'](self.X, self.Y)
        self.f_y = self.parameters['f_y_func'](self.X, self.Y)
        
        # On met T (temps final) directement dans les paramètres
        self.parameters['T'] = self.config['time_steps'] * self.config['dt']
        
        print("Derived parameters:")
        print(f"  - Viscosity (ν): {self.parameters['viscosity']:.4f}")
        print(f"  - Final time (T): {self.parameters['T']:.2f}")
        print("  - Forcing and boundary functions generated.")

    def _compute_velocity_history(self):
        """
        Calcule l'historique complet du champ de vitesse u(x, y, t) en résolvant
        l'équation de Burgers vectorielle. Le résultat est stocké dans self.u_history.
        """
        print("Computing velocity field history...")
        
        # Récupération des paramètres pour plus de lisibilité
        nu = self.parameters['viscosity']
        dt = self.config['dt']
        
        # Initialisation du champ de vitesse à t=0 (on suppose qu'il est nul au départ)
        self.u.fill(0)
        self.u_history[0] = self.u.copy()
        
        # Boucle temporelle
        num_steps = self.config['time_steps']
        for n in range(num_steps):
            # On utilise tqdm pour avoir une belle barre de progression
            if n % 10 == 0:
                print(f"  -> Velocity step {n}/{num_steps}")

            # Copie de l'état actuel pour les calculs
            u_prev = self.u.copy()

            # --- Operator Splitting ---
            # Note : L'ordre peut varier, mais nous allons suivre une séquence logique.
            # Idéalement, on utiliserait des schémas plus complexes, mais pour commencer :

            # 1. Diffusion (implicite pour la stabilité)
            # On résout pour la composante vx (u[:,:,0]) et vy (u[:,:,1]) séparément
            self.u[:,:,0] = self._solve_diffusion(u_prev[:,:,0], u_prev[:,:,0], nu, dt)
            self.u[:,:,1] = self._solve_diffusion(u_prev[:,:,1], u_prev[:,:,1], nu, dt)
            
            # 2. Advection (avec un schéma upwind)
            u_advected_x = self._solve_advection_upwind(self.u[:,:,0], self.u, dt)
            u_advected_y = self._solve_advection_upwind(self.u[:,:,1], self.u, dt)
            self.u = np.stack([u_advected_x, u_advected_y], axis=-1)

            # 3. Forçage (simple ajout)
            self.u[:,:,0] += dt * self.f_x
            self.u[:,:,1] += dt * self.f_y
            
            # 4. (Optionnel mais recommandé) Projection pour enlever la divergence
            # Le papier mentionne que Burgers n'a pas cette contrainte, mais en pratique
            # cela peut éviter que le champ "explose". Pour l'instant, on l'omet pour
            # rester fidèle au papier, mais on garde l'idée en tête.

            # Stockage du résultat dans l'historique
            self.u_history[n + 1] = self.u.copy()

        print("Velocity field history computed.")

    def _solve_diffusion(self, x: np.ndarray, x0: np.ndarray, k: float, dt: float, iters: int = 20) -> np.ndarray:
        """
        Résout l'équation de diffusion implicitement en utilisant un solveur de Gauss-Seidel.
        
        Args:
            x (np.ndarray): Le champ à diffuser (sera modifié en place et retourné).
            x0 (np.ndarray): L'état initial du champ avant diffusion.
            k (float): Coefficient de diffusion (ici, la viscosité ν).
            dt (float): Pas de temps.
            iters (int): Nombre d'itérations pour le solveur.

        Returns:
            np.ndarray: Le champ après diffusion.
        """
        a = dt * k * (self.width - 1) * (self.height - 1) # Normalisation des pas dx, dy
        
        for _ in range(iters):
            # Mise à jour des points intérieurs
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] + x[:-2, 1:-1] + 
                                                x[1:-1, 2:] + x[1:-1, :-2])) / (1 + 4 * a)
            # On ne gère pas les bords pour l'instant (condition de Neumann implicite = 0 flux)
            # On pourrait ajouter une fonction set_bnd() plus tard si nécessaire.

        return x
    
    def _solve_advection_upwind(self, field: np.ndarray, velocity_field: np.ndarray, dt: float) -> np.ndarray:
        """
        Advecte un champ en utilisant un schéma upwind du premier ordre.
        Ceci correspond à l'équation (9) du papier (Encryption).
        """
        vx = velocity_field[:, :, 0]
        vy = velocity_field[:, :, 1]
        
        # Création de copies paddées pour gérer les bords facilement
        padded_field = np.pad(field, 1, mode='edge')
        
        # Différences finies
        grad_x_pos = (padded_field[1:-1, 1:-1] - padded_field[1:-1, :-2]) / self.dx
        grad_x_neg = (padded_field[1:-1, 2:] - padded_field[1:-1, 1:-1]) / self.dx
        
        grad_y_pos = (padded_field[1:-1, 1:-1] - padded_field[:-2, 1:-1]) / self.dy
        grad_y_neg = (padded_field[2:, 1:-1] - padded_field[1:-1, 1:-1]) / self.dy
        
        # Le coeur du schéma UPWIND
        # Si la vitesse est positive, on prend la différence "en amont" (backward difference)
        # Si la vitesse est négative, on prend la différence "en amont" (forward difference)
        advection_term = (np.maximum(vx, 0) * grad_x_pos + np.minimum(vx, 0) * grad_x_neg +
                        np.maximum(vy, 0) * grad_y_pos + np.minimum(vy, 0) * grad_y_neg)
                        
        # Mise à jour avec Euler explicite
        return field - dt * advection_term


    def _solve_advection_downwind(self, field: np.ndarray, velocity_field: np.ndarray, dt: float) -> np.ndarray:
        """
        Advecte un champ en utilisant un schéma downwind du premier ordre.
        Ceci correspond à l'équation (10) du papier (Decryption).
        """
        vx = velocity_field[:, :, 0]
        vy = velocity_field[:, :, 1]

        # Création de copies paddées
        padded_field = np.pad(field, 1, mode='edge')
        
        # Différences finies
        # Noter que les gradients sont inversés par rapport à upwind
        grad_x_pos = (padded_field[1:-1, 2:] - padded_field[1:-1, 1:-1]) / self.dx # Forward
        grad_x_neg = (padded_field[1:-1, 1:-1] - padded_field[1:-1, :-2]) / self.dx # Backward
        
        grad_y_pos = (padded_field[2:, 1:-1] - padded_field[1:-1, 1:-1]) / self.dy # Forward
        grad_y_neg = (padded_field[1:-1, 1:-1] - padded_field[:-2, 1:-1]) / self.dy # Backward

        # Le coeur du schéma DOWNWIND
        # Si la vitesse est positive, on prend la différence "en aval" (forward difference)
        # Si la vitesse est négative, on prend la différence "en aval" (backward difference)
        advection_term = (np.maximum(vx, 0) * grad_x_pos + np.minimum(vx, 0) * grad_x_neg +
                        np.maximum(vy, 0) * grad_y_pos + np.minimum(vy, 0) * grad_y_neg)
                        
        # Mise à jour avec Euler explicite. Notez le PLUS ici.
        return field + dt * advection_term
    
    def _advect_pixels(self, forward: bool = True) -> np.ndarray:
        """
        Advecte le champ de densité en utilisant les schémas appropriés.
        """
        num_steps = self.config['time_steps']
        dt = self.config['dt']
        rho_current = self.rho.copy()
        
        if forward:
            print("Advecting pixels FORWARD (Encryption) using UPWIND scheme...")
            time_indices = range(num_steps)
            for n in time_indices:
                if n % 10 == 0: print(f"  -> Pixel advection step {n}/{num_steps}")
                velocity_field = self.u_history[n + 1]
                temp = self._solve_advection_upwind(rho_current, velocity_field, dt)
                rho_current = np.clip(temp, 0.0, 1.0)  # Assurer que les valeurs restent entre 0 et 1
                # rho_current = self._solve_advection_upwind(rho_current, velocity_field, dt)
        else:
            print("Advecting pixels BACKWARD (Decryption) using DOWNWIND scheme...")
            time_indices = range(num_steps - 1, -1, -1)
            for n in time_indices:
                if n % 10 == (num_steps - 1) % 10 or n % 10 == 0:
                    print(f"  -> Pixel advection step {n}/{num_steps}")
                velocity_field = self.u_history[n + 1]
                temp = self._solve_advection_downwind(rho_current, velocity_field, dt)
                rho_current = np.clip(temp, 0.0, 1.0)  # Assurer que les valeurs restent entre 0 et 1
                # rho_current = self._solve_advection_downwind(rho_current, velocity_field, dt)
                
        return rho_current