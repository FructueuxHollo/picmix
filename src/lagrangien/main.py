# 1. Charger image
from utils import generate_velocity_field, load_image_as_array, save_array_as_image, semi_lagrangian_decrypt, semi_lagrangian_encrypt

dt = 0.01  # Pas de temps pour l'encryption

img = load_image_as_array("test_image.png")  # -> array (Nx, Ny)

# 2. Générer champ de vitesse
u_history = generate_velocity_field(img.shape, dt=dt, steps=500)

# 3. Encryption
encrypted = semi_lagrangian_encrypt(img, u_history, dt=dt)

# 4. Decryption
decrypted = semi_lagrangian_decrypt(encrypted, u_history, dt=dt)

# 5. Sauvegarde
save_array_as_image(encrypted, "encrypted.png")
save_array_as_image(decrypted, "decrypted.png")
