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

    model = load_model(scale=scale)
    sr_image = model.predict(input_img, tile_size=int(tile_size))

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


with gr.Blocks(title="Open Upscaler") as demo:
    gr.Markdown("# 🚀 Open Upscaler Studio")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Imagen Original",
                type="pil",
                sources=["upload", "clipboard"],
            )

            scale_selector = gr.Radio(
                choices=["x2", "x4", "x8"],
                value="x2",
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

            submit_btn = gr.Button("✨ Mejorar Resolución", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Resultado en Alta Resolución",
                type="pil",
                interactive=False,
            )
            telemetry_text = gr.Textbox(
                label="Métricas",
                interactive=False,
            )

    submit_btn.click(
        fn=process_image,
        inputs=[input_image, scale_selector, tile_slider],
        outputs=[output_image, telemetry_text],
    )

if __name__ == "__main__":
    # Inicia en localhost y abre automáticamente la pestaña en tu navegador
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )