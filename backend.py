import time
import logging
import torch
from PIL import Image

from src.model import RealESRGAN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(".logs/app.log"),  
        logging.StreamHandler()          
    ]
)
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
MODEL_CACHE = {}
logger.info(f"Backend inicializado. Motor de inferencia fijado en: {DEVICE.type.upper()}")

def clear_device_cache():
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


def load_model(scale: int) -> RealESRGAN:
    if scale not in MODEL_CACHE:
        logger.info(f"Cargando pesos del modelo x{scale} en memoria...")
        model = RealESRGAN(device=DEVICE, scale=scale)
        model.load_weights(download=True)
        MODEL_CACHE[scale] = model
        logger.info(f"Modelo x{scale} cargado y cacheado con éxito.")
    return MODEL_CACHE[scale]


def process_image(input_img: Image.Image, scale_choice: str, tile_size: int):
    if input_img is None:
        logger.warning("Intento de procesamiento rechazado: No hay imagen.")
        return None, "Por favor, sube una imagen antes de procesar."

    try:
        scale = int(scale_choice.replace("x", ""))
        orig_w, orig_h = input_img.size
        
        logger.info(f"Iniciando tarea: {orig_w}x{orig_h}px ➔ Objetivo: x{scale} | Tile: {tile_size}")

        start_time = time.time()

        if scale == 1:
            model = load_model(scale=2)
            sr_image = model.predict(input_img, tile_size=tile_size)
            sr_image = sr_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        else:
            model = load_model(scale=scale)
            sr_image = model.predict(input_img, tile_size=tile_size)

        clear_device_cache()

        elapsed = time.time() - start_time
        final_w, final_h = sr_image.size

        telemetry = (
            f"⚡ {orig_w}×{orig_h} px ➔ {final_w}×{final_h} px | "
            f"⏱️ {elapsed:.2f}s | "
            f"🖥️ {DEVICE.type.upper()}"
        )
        
        logger.info(f"Tarea completada: {final_w}x{final_h}px generados en {elapsed:.2f}s")

        return sr_image, telemetry

    except Exception as e:
        logger.error(f"Fallo crítico durante la inferencia: {str(e)}", exc_info=True)
        return None, f"Error interno: {str(e)}"
    

def show_image_info(img: Image.Image):
    if img is None:
        return ""
    w, h = img.size
    pixeles_totales = (w * h) / 1_000_000
    
    logger.info(f"Nueva imagen cargada en UI: {w}x{h}px ({img.mode})")

    return f"**Información:** Resolución {w}×{h} px | {pixeles_totales:.1f} Megapíxeles | Modo: {img.mode}"

def log_cancellation():
    logger.warning("🛑 El usuario ha cancelado el proceso de inferencia en curso.")