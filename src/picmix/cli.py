# Fichier: src/picmix/cli.py

import typer
import os
from PIL import Image
from typing_extensions import Annotated

from . import encrypt as encrypt_api
from . import decrypt as decrypt_api
from . import __version__

app = typer.Typer(
    name="picmix",
    help="🖼️ A novel image encryption tool using reaction-diffusion systems (VortexCrypt)."
)

def version_callback(value: bool):
    """Callback to display the version"""
    if value:
        print(f"PicMix CLI Version: {__version__}")
        raise typer.Exit()

@app.command()
def encrypt(
    image_path: Annotated[typer.FileText, typer.Argument(help="Path to the source image file.")],
    key: Annotated[str, typer.Option(prompt=True, confirmation_prompt=True, hide_input=True, help="Secret key (8-24 characters).")],
    output: Annotated[str, typer.Option("--output", "-o", help="Path for the output .npz file. [default: encrypted_state.npz]")] = "encrypted_state.npz",
    config: Annotated[str, typer.Option("--config", "-c", help="Path to an optional JSON configuration file.")] = None,
    no_preview: Annotated[bool, typer.Option("--no-preview", help="Disable saving a .png preview of the encrypted image.")] = False,
):
    """
    Encrypts an image file using the VortexCrypt algorithm.
    """
    typer.echo(f"Encrypting '{image_path.name}'...")
    try:
        encrypt_api(
            image_path=image_path.name,
            output_path_npz=output,
            key=key,
            config_path=config,
            save_preview=not no_preview
        )
        typer.secho(f"✅ Encryption successful! State saved to '{output}.npz'", fg=typer.colors.GREEN)
        if not no_preview:
            preview_file = os.path.splitext(output)[0] + ".png"
            typer.secho(f"🖼️ Preview saved to '{preview_file}'", fg=typer.colors.CYAN)
    except Exception as e:
        typer.secho(f"❌ Error during encryption: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.command()
def decrypt(
    state_path: Annotated[typer.FileText, typer.Argument(help="Path to the encrypted .npz state file.")],
    key: Annotated[str, typer.Option(prompt=True, hide_input=True, help="The secret key used for encryption.")],
    output: Annotated[str, typer.Option("--output", "-o", help="Path for the decrypted .png image. [default: decrypted_image.png]")] = "decrypted_image.png",
    config: Annotated[str, typer.Option("--config", "-c", help="Path to the same optional JSON config file used for encryption.")] = None,
):
    """
    Decrypts an image from a .npz state file.
    """
    typer.echo(f"Decrypting '{state_path.name}'...")
    try:
        decrypt_api(
            encrypted_state_path_npz=state_path.name,
            output_path_png=output,
            key=key,
            config_path=config
        )
        typer.secho(f"✅ Decryption successful! Image saved to '{output}'", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"❌ Error during decryption: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=version_callback, is_eager=True, help="Show the application's version and exit.")] = False
):
    """
    Main callback for the CLI application.
    """
    pass

if __name__ == "__main__":
    app()