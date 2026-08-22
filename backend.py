import time
import torch
from PIL import Image

from src.model import RealESRGAN

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
MODEL_CACHE = {}


def load_model(scale: int) -> RealESRGAN:
    if scale not in MODEL_CACHE:
        model = RealESRGAN(device=DEVICE, scale=scale)
        model.load_weights(download=True)
        MODEL_CACHE[scale] = model
    return MODEL_CACHE[scale]


def process_image(input_img: Image.Image, scale_choice: str, tile_size: int):
    if input_img is None:
        return None, "Por favor, sube una imagen antes de procesar."

    try:

        scale = int(scale_choice.replace("x", ""))
        
        start_time = time.time()
        orig_w, orig_h = input_img.size

        if scale == 1:
            model = load_model(scale=2)
            sr_image = model.predict(input_img, tile_size=tile_size)
            sr_image = sr_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        else:
            model = load_model(scale=scale)
            sr_image = model.predict(input_img, tile_size=tile_size)

        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        final_w, final_h = sr_image.size

        telemetry = (
            f"⚡ {orig_w}×{orig_h} px ➔ {final_w}×{final_h} px | "
            f"⏱️ {elapsed:.2f}s | "
            f"🖥️ {DEVICE.type.upper()}"
        )

        return sr_image, telemetry

    except Exception as e:
        print(e)
        return None, f"Error interno: {str(e)}"
    

def show_image_info(img: Image.Image):
    if img is None:
        return ""
    w, h = img.size
    pixeles_totales = (w * h) / 1_000_000
    return f"**Información:** Resolución {w}×{h} px | {pixeles_totales:.1f} Megapíxeles | Modo: {img.mode}"