import vortexcrypt
import os

SECRET_PASSPHRASE = "my_secret_passphrase"
output_dir = "crypto_files"
os.makedirs(output_dir, exist_ok=True)

original_image_path = "text_test_image.png"

encrypted_data_path = os.path.join(output_dir, "encrypted_data.npz")

decrypted_image_path = os.path.join(output_dir, "decrypted_final.png")


# --- 1. Encryption ---
print(f" Encryption '{original_image_path}'...")
vortexcrypt.encrypt(
    image_path=original_image_path,
    key=SECRET_PASSPHRASE,
    output_path_npz=encrypted_data_path,
)
print("\n" + "="*50 + "\n")

# --- 2. Decryption ---
print(f"Decryption '{encrypted_data_path}'...")
vortexcrypt.decrypt(
    encrypted_state_path_npz=encrypted_data_path,
    key=SECRET_PASSPHRASE,
    output_path_png=decrypted_image_path,
)

print(f"\nProcess Completed. The decrypted image is : '{decrypted_image_path}'")