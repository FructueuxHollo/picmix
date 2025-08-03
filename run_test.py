import vortexcrypt
import os

SECRET_PASSPHRASE = "mon_secret_passphrase"
output_dir = "crypto_files"
os.makedirs(output_dir, exist_ok=True)

original_image_path = "test_image.png"
# Le fichier de données est maintenant un .npz
encrypted_data_path = os.path.join(output_dir, "encrypted_data.npz")
# Ajout d'un chemin pour la preview
decrypted_image_path = os.path.join(output_dir, "decrypted_final.png")
# Chemin vers le fichier de configuration, si nécessaire
config_file = "config.json"  

# --- 1. Chiffrement ---
print(f"Chiffrement de '{original_image_path}'...")
vortexcrypt.encrypt(
    image_path=original_image_path,
    key=SECRET_PASSPHRASE,
    output_path_npz=encrypted_data_path,
    config_path=config_file,  
)
print("\n" + "="*50 + "\n")

# --- 2. Déchiffrement ---
print(f"Déchiffrement de '{encrypted_data_path}'...")
vortexcrypt.decrypt(
    encrypted_state_path_npz=encrypted_data_path,
    key=SECRET_PASSPHRASE,
    output_path_png=decrypted_image_path,
    config_path=config_file  
)

print(f"\nProcessus terminé. L'image déchiffrée est : '{decrypted_image_path}'")