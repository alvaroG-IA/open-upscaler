import os
import math
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .model_arch import RRDBNet


MODEL_SPECS = {
    2: {"repo_id": "ai-forever/Real-ESRGAN", "filename": "RealESRGAN_x2.pth"},
    4: {"repo_id": "ai-forever/Real-ESRGAN", "filename": "RealESRGAN_x4.pth"},
    8: {"repo_id": "ai-forever/Real-ESRGAN", "filename": "RealESRGAN_x8.pth"},
}


class RealESRGAN:
    def __init__(self, device: torch.device, scale: int = 4):
        if scale not in MODEL_SPECS:
            raise ValueError(f"Escala no soportada: {scale}. Opciones válidas: 2, 4, 8")

        self.device = device
        self.scale = scale
        self.spec = MODEL_SPECS[scale]

        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )

    def load_weights(self, model_path: str = None, download: bool = True):
        if model_path is None:
            model_path = os.path.join("weights", self.spec["filename"])

        if not os.path.exists(model_path) and download:
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            print(f"[OpenUpscaler] Descargando pesos x{self.scale} desde Hugging Face...")
            model_path = hf_hub_download(
                repo_id=self.spec["repo_id"],
                filename=self.spec["filename"],
                local_dir=os.path.dirname(model_path) or "weights",
            )
            print(f"[OpenUpscaler] Pesos descargados en: {model_path}")

        loadnet = torch.load(model_path, map_location=self.device, weights_only=True)
        
        if "params_ema" in loadnet:
            state_dict = loadnet["params_ema"]
        elif "params" in loadnet:
            state_dict = loadnet["params"]
        else:
            state_dict = loadnet

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def predict(self, lr_image: Image.Image, tile_size: int = 1024, tile_pad: int = 16) -> Image.Image:
        """
        Inferencia de super-resolución.
        Procesa de forma directa o mediante mosaicos (tiling) si la imagen excede tile_size.
        """
        # Asegurar formato RGB
        if lr_image.mode != "RGB":
            lr_image = lr_image.convert("RGB")

        img_np = np.array(lr_image).astype(np.float32) / 255.0
        # HWC -> NCHW
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        _, _, h, w = tensor.shape

        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            output = self.model(tensor)
        else:
            output = self._predict_tiled(tensor, tile_size=tile_size, tile_pad=tile_pad)

        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        sr_img = (output * 255.0).round().astype(np.uint8)
        return Image.fromarray(sr_img)

    def _predict_tiled(self, tensor: torch.Tensor, tile_size: int, tile_pad: int) -> torch.Tensor:
        batch, channel, height, width = tensor.shape
        out_h, out_w = height * self.scale, width * self.scale
        output = torch.zeros((batch, channel, out_h, out_w), device=self.device)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile_size
                ofs_y = y * tile_size

                in_sx = max(ofs_x - tile_pad, 0)
                in_ex = min(ofs_x + tile_size + tile_pad, width)
                in_sy = max(ofs_y - tile_pad, 0)
                in_ey = min(ofs_y + tile_size + tile_pad, height)

                tile = tensor[:, :, in_sy:in_ey, in_sx:in_ex]
                out_tile = self.model(tile)

                out_sx = ofs_x * self.scale
                out_ex = min((ofs_x + tile_size) * self.scale, out_w)
                out_sy = ofs_y * self.scale
                out_ey = min((ofs_y + tile_size) * self.scale, out_h)

                t_sx = (ofs_x - in_sx) * self.scale
                t_ex = t_sx + (out_ex - out_sx)
                t_sy = (ofs_y - in_sy) * self.scale
                t_ey = t_sy + (out_ey - out_sy)

                output[:, :, out_sy:out_ey, out_sx:out_ex] = out_tile[:, :, t_sy:t_ey, t_sx:t_ex]

        return output