import time
import torch
import gradio as gr
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


def show_image_info(img: Image.Image):
    if img is None:
        return ""
    w, h = img.size
    pixeles_totales = (w * h) / 1_000_000
    return f"**Información:** Resolución {w}×{h} px | {pixeles_totales:.1f} Megapíxeles | Modo: {img.mode}"


# 1. Tema nativo Soft con colores Morado y Gris neutro (Slate)
theme = gr.themes.Neon(
    primary_hue="purple", 
    secondary_hue="slate",
    neutral_hue="slate"
)

# 2. Script para FORZAR siempre el modo claro (fondo blanco)
forzar_modo_claro = """
function() {
    document.body.classList.remove('dark');
}
"""

# 3. Mantenemos solo la animación del botón para darle ese toque premium
css_animacion = """
.boton-animado {
    transition: all 0.2s ease-in-out !important;
}
.boton-animado:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 12px -3px rgba(147, 51, 234, 0.3) !important;
}
.boton-animado:active {
    transform: translateY(1px) scale(0.99) !important;
    box-shadow: 0 4px 6px -2px rgba(147, 51, 234, 0.3) !important;
}
"""

with gr.Blocks(theme=theme, css=css_animacion, js=forzar_modo_claro, title="Open Upscaler") as demo:
    gr.Markdown(
        """
        # 🚀 Open Upscaler Studio
        ### Motor de mejora de resolución de imágenes potenciado con IA (Real-ESRGAN)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):

            input_image = gr.Image(
                label="Imagen Original",
                type="pil",
                sources=["upload", "clipboard"],
            )
            
            image_info = gr.Markdown("")

            scale_selector = gr.Radio(
                choices=["x1", "x2", "x4", "x8"],
                value="x1",
                label="Factor de Escala",
            )

            with gr.Accordion("Opciones Avanzadas", open=False):
                tile_slider = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=256,
                    value=1024,
                    label="Tile Size",
                )

            submit_btn = gr.Button("✨ Mejorar Resolución", variant="primary", elem_classes=["boton-animado"])

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Resultado en Alta Resolución",
                type="pil",
                interactive=False,
            )

            cancel_btn = gr.Button("🛑 Cancelar", variant="stop", elem_classes=["boton-animado"])

            telemetry_text = gr.Textbox(
                label="Métricas",
                interactive=False,
            )

    input_image.upload(fn=show_image_info, inputs=input_image, outputs=image_info)
    input_image.clear(fn=lambda: "", inputs=None, outputs=image_info)

    evento_inferencia = submit_btn.click(
        fn=process_image,
        inputs=[input_image, scale_selector, tile_slider],
        outputs=[output_image, telemetry_text],
    )
    
    cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[evento_inferencia])


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )