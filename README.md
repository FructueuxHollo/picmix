# VortexCrypt 🖼️✨

**A novel, high-security image encryption package based on time-reversible reaction-diffusion systems.**

VortexCrypt provides a robust framework for encrypting and decrypting digital images using the principles of chemical morphogenesis, specifically the Gray-Scott model. By embedding the image's pixel data as the initial state of a chemical concentration and evolving it through a complex, chaotic simulation, it generates a highly complex and secure ciphertext.

The core of its security and functionality lies in a mathematically proven **time-reversible numerical integrator (Strang-Splitting)**, which guarantees that the decryption process is a true inverse of the encryption, allowing for near-perfect reconstruction of the original image (limited only by floating-point precision).

  ![VortexCrypt Workflow Diagram](https://github.com/FructueuxHollo/vortexcrypt/raw/master/docs/images/workflow.png)

## Features

-   **High Security:** Leverages the chaotic and unpredictable nature of reaction-diffusion systems.
-   **Key-Driven Dynamics:** All simulation parameters are deterministically generated from your secret key using a cryptographic hash (SHA-256), ensuring a massive key space and high sensitivity.
-   **Reversibility:** Utilizes a time-symmetric numerical scheme to ensure the encrypted state can be decrypted with the correct key.
-   **Color Image Support:** Encrypts both grayscale and RGB images.
-   **Flexible Configuration:** Allows advanced users to override simulation parameters via a JSON file.
-   **Easy-to-Use CLI:** Simple and secure command-line interface for encryption and decryption.

## Installation

You can install VortexCrypt using `pip`. It is recommended to install it in a virtual environment.

```bash
pip install vortexcrypt
```

## Quick Start: Command-Line Interface (CLI)

The easiest way to use VortexCrypt is through its command-line interface.

### Encrypt an Image

This command will encrypt `my_image.png`, prompt you securely for a password, and save the complete encrypted state to `encrypted_data.npz`. It will also save a visual preview as `encrypted_data.png`.

```bash
vortexcrypt encrypt my_image.png --output encrypted_data.npz
```

### Decrypt an Image

This command will decrypt the `encrypted_data.npz` file using the same secret key and save the result to `decrypted_image.png`.

```bash
vortexcrypt decrypt encrypted_data.npz --output decrypted_image.png
```

### Using a Configuration File

For advanced control, you can override the key-derived parameters with a `config.json` file.

```bash
# Encrypt with custom parameters
vortexcrypt encrypt my_image.png --key "my-secret" --config config.json

# You MUST use the same config file for decryption
vortexcrypt decrypt encrypted_state.npz --key "my-secret" --config config.json
```

## Usage as a Python Library (API)

You can also integrate VortexCrypt directly into your Python projects.

```python
import vortexcrypt
from PIL import Image

# Define paths and key
image_path = "path/to/my_image.png"
encrypted_path = "path/to/encrypted_state.npz"
decrypted_path = "path/to/decrypted_image.png"
secret_key = "this-is-a-very-secure-key"

# --- Encryption ---
vortexcrypt.encrypt(
    image_path=image_path,
    output_path_npz=encrypted_path,
    key=secret_key
)
print("Encryption complete.")

# --- Decryption ---
vortexcrypt.decrypt(
    encrypted_state_path_npz=encrypted_path,
    output_path_png=decrypted_path,
    key=secret_key
)
print("Decryption complete.")
```

## Development

To install the package for development purposes:

1.  Clone this repository: `git clone https://your-repo-url/vortexcrypt.git`
2.  Navigate into the directory: `cd vortexcrypt`
3.  Install dependencies with Poetry: `poetry install`
4.  Activate the virtual environment: `poetry shell`
5.  You can now run the CLI using `vortexcrypt ...` directly.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
