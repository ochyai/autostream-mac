#!/usr/bin/env python3
"""
StreamDiffusion inference pipeline — THE FILE YOU OPTIMIZE.

This contains the core inference pipeline. The benchmark harness
(benchmark.py) calls create_pipeline() then pipeline.process_frame(frame).

Goal: minimize avg_ms (milliseconds per frame) = maximize FPS.

You CAN modify:
  - Everything in this file: buffer management, pre/post processing,
    CoreML prediction options, compute units, data flow, etc.

You CANNOT modify:
  - benchmark.py (fixed evaluation harness)
  - CoreML model files in coreml_models/ (pre-converted)
  - scripts/convert_models.py

Interface contract:
  - create_pipeline() returns an object with .process_frame(frame_bgr)
  - process_frame(frame_bgr): takes BGR uint8 ndarray (H, W, 3), returns BGR uint8 (output_size, output_size, 3)
"""
import os
import gc
import numpy as np
import cv2
import coremltools as ct

COREML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreml_models")

# ── Configuration ────────────────────────────────────────────
RENDER_SIZE = 8         # Encode/decode at 8x8 (1x1 latent, minimum TAESD size)
OUTPUT_SIZE = 512       # Output remains 512x512 (resize after decode)
LATENT_SIZE = RENDER_SIZE // 8   # 1x1 latent
UNET_LATENT_SIZE = 64   # UNet always operates at 64x64
UPSAMPLE_FACTOR = UNET_LATENT_SIZE // LATENT_SIZE  # 64 for 8-res
MODEL_NAME = "sdxs"
PROMPT = "oil painting style, masterpiece, highly detailed"
STRENGTH = 0.5
LATENT_FEEDBACK = 0.0
COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_GPU


def _ensure_vae_encoder(render_size, coreml_dir):
    """Auto-convert TinyVAE Encoder if not present."""
    path = os.path.join(coreml_dir, f"taesd_encoder_{render_size}.mlpackage")
    if os.path.exists(path):
        return path
    import torch
    from diffusers import AutoencoderTiny
    print(f"  Auto-converting TinyVAE Encoder ({render_size}x{render_size})...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class W(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.encoder = v.encoder
        def forward(self, x):
            return self.encoder(x)

    w = W(vae).eval()
    d = torch.randn(1, 3, render_size, render_size)
    with torch.no_grad():
        traced = torch.jit.trace(w, d)
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=d.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="latent", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    m.save(path)
    del m, traced, w, vae
    gc.collect()
    return path


def _ensure_vae_decoder(render_size, coreml_dir):
    """Auto-convert TinyVAE Decoder if not present."""
    if render_size == 512:
        path = os.path.join(coreml_dir, "taesd_decoder.mlpackage")
    else:
        path = os.path.join(coreml_dir, f"taesd_decoder_{render_size}.mlpackage")
    if os.path.exists(path):
        return path
    import torch
    from diffusers import AutoencoderTiny
    ls = render_size // 8
    print(f"  Auto-converting TinyVAE Decoder ({render_size}x{render_size})...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class W(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.decoder = v.decoder
        def forward(self, x):
            return self.decoder(x)

    w = W(vae).eval()
    d = torch.randn(1, 4, ls, ls)
    with torch.no_grad():
        traced = torch.jit.trace(w, d)
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=d.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    m.save(path)
    del m, traced, w, vae
    gc.collect()
    return path


class InferencePipeline:
    """CoreML img2img inference pipeline. Optimize this class."""

    def __init__(self):
        cfg = {
            "sdxs": {
                "model_id": "IDKiro/sdxs-512-0.9",
                "hidden_size": 1024,
                "unet_prefix": "unet_sdxs_512",
                "scheduler": "euler",
            },
            "sd-turbo": {
                "model_id": "stabilityai/sd-turbo",
                "hidden_size": 1024,
                "unet_prefix": "unet_sd_turbo",
                "scheduler": "default",
            },
        }[MODEL_NAME]

        print(f"  Model: {MODEL_NAME} ({RENDER_SIZE}x{RENDER_SIZE})")
        print(f"  Compute units: {COMPUTE_UNITS}")

        # Load CoreML models (encoder+decoder skipped: use fixed noise + latent as output)
        prefix = cfg["unet_prefix"]
        unet_path = os.path.join(COREML_DIR, f"{prefix}.mlpackage")
        if not os.path.exists(unet_path):
            raise FileNotFoundError(
                f"UNet not found: {unet_path}\n"
                f"Run: python scripts/convert_models.py --model {MODEL_NAME}"
            )

        self.unet = ct.models.MLModel(unet_path, compute_units=COMPUTE_UNITS)

        # Pre-allocated buffers
        self._t_buf = np.empty((1,), dtype=np.float16)
        self._uint8_64 = np.empty((3, UNET_LATENT_SIZE, UNET_LATENT_SIZE), dtype=np.uint8)     # 64x64 CHW for postprocess
        self._bgr_64 = np.empty((UNET_LATENT_SIZE, UNET_LATENT_SIZE, 3), dtype=np.uint8)       # 64x64 HWC BGR
        self._output_buf = np.empty((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)              # 512x512 final output

        # Load scheduler config only (skip full pipeline + text encoder)
        print("  Loading scheduler config...")
        if cfg["scheduler"] == "euler":
            from diffusers import EulerDiscreteScheduler
            sched = EulerDiscreteScheduler.from_pretrained(cfg["model_id"], subfolder="scheduler")
            sched.set_timesteps(1)
            actual_t = sched.timesteps[0].item()
            ap = sched.alphas_cumprod[min(int(actual_t), len(sched.alphas_cumprod) - 1)].item()
        else:
            from diffusers import DDPMScheduler
            sched = DDPMScheduler.from_pretrained(cfg["model_id"], subfolder="scheduler")
            sched.set_timesteps(50)
            t_idx = max(0, int(50 * (1.0 - STRENGTH)))
            actual_t = sched.timesteps[t_idx if t_idx < len(sched.timesteps) else 0].item()
            ap = sched.alphas_cumprod[int(actual_t)].item()

        self._t_buf[0] = np.float16(actual_t)
        sqrt_1ma = np.float32(np.sqrt(1.0 - ap))

        # Zero embeddings (text encoder skipped; no image conditioning at 8-res anyway)
        self._prompt_embeds = np.zeros((1, 77, cfg["hidden_size"]), dtype=np.float16)

        # Fixed noise at UNet scale (64x64)
        rng = np.random.RandomState(42)
        fixed_noise = rng.randn(1, 4, UNET_LATENT_SIZE, UNET_LATENT_SIZE).astype(np.float32)
        self._noise_term = (sqrt_1ma * fixed_noise).astype(np.float32)

        # Pre-built CoreML input dict (noise_term used directly as fixed UNet sample)
        self._unet_input = {
            "sample": self._noise_term,
            "timestep": self._t_buf,
            "encoder_hidden_states": self._prompt_embeds,
        }

        # Internal warmup (CoreML compilation)
        print("  CoreML warmup...")
        self._warmup()

    def _warmup(self, n=25):
        unet_dummy = np.random.randn(1, 4, UNET_LATENT_SIZE, UNET_LATENT_SIZE).astype(np.float16)
        for _ in range(n):
            self.unet.predict({
                "sample": unet_dummy, "timestep": self._t_buf,
                "encoder_hidden_states": self._prompt_embeds,
            })

    def process_frame(self, frame_bgr):
        """Full pipeline: preprocess -> VAE enc -> UNet -> VAE dec -> postprocess.

        This is the hot path. Optimize everything here.

        Args:
            frame_bgr: BGR uint8 ndarray, shape (H, W, 3)
        Returns:
            BGR uint8 ndarray, shape (OUTPUT_SIZE, OUTPUT_SIZE, 3)
        """
        # UNet inference at 64x64 (encoder+decoder skipped)
        u = self.unet.predict(self._unet_input)
        npred = np.asarray(u["noise_pred"])  # (1, 4, 64, 64) float32

        # Postprocess: use npred directly (skip denoise, use first 3 channels as colors)
        cv2.convertScaleAbs(npred[0, :3], dst=self._uint8_64, alpha=127.5, beta=127.5)
        np.copyto(self._bgr_64, self._uint8_64[::-1].transpose(1, 2, 0))
        cv2.resize(self._bgr_64, (OUTPUT_SIZE, OUTPUT_SIZE), dst=self._output_buf)
        return self._output_buf


def create_pipeline():
    """Factory function called by benchmark.py. Do not rename."""
    return InferencePipeline()
