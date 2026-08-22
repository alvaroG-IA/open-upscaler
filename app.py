import logging
import gradio as gr

from backend import show_image_info, process_image

theme = gr.themes.Neon(
    primary_hue="purple", 
    secondary_hue="slate",
    neutral_hue="slate"
)

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

with gr.Blocks(theme=theme, css=css_animacion, title="Open Upscaler") as demo:
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