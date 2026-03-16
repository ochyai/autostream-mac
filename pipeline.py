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
RENDER_SIZE = 64        # Encode/decode at 64x64 (64x fewer pixels than 512, ~64x faster VAE)
OUTPUT_SIZE = 512       # Output remains 512x512 (resize after decode)
LATENT_SIZE = RENDER_SIZE // 8   # 8x8 — minimal VAE latent
UNET_LATENT_SIZE = 64   # UNet always operates at 64x64
UPSAMPLE_FACTOR = UNET_LATENT_SIZE // LATENT_SIZE  # 8 for 64-res
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
        import torch

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

        # Load CoreML models
        enc_path = _ensure_vae_encoder(RENDER_SIZE, COREML_DIR)
        dec_path = _ensure_vae_decoder(RENDER_SIZE, COREML_DIR)
        prefix = cfg["unet_prefix"]
        unet_path = os.path.join(COREML_DIR, f"{prefix}.mlpackage")
        if not os.path.exists(unet_path):
            raise FileNotFoundError(
                f"UNet not found: {unet_path}\n"
                f"Run: python scripts/convert_models.py --model {MODEL_NAME}"
            )

        self.vae_encoder = ct.models.MLModel(enc_path, compute_units=COMPUTE_UNITS)
        self.vae_decoder = ct.models.MLModel(dec_path, compute_units=COMPUTE_UNITS)
        self.unet = ct.models.MLModel(unet_path, compute_units=COMPUTE_UNITS)

        # Pre-allocated buffers
        self._img_buf = np.empty((1, 3, RENDER_SIZE, RENDER_SIZE), dtype=np.float32)     # 256x256 encoder input
        self._unet_sample = np.empty((1, 4, UNET_LATENT_SIZE, UNET_LATENT_SIZE), dtype=np.float32)  # 64x64 UNet input
        self._out_buf = np.empty((1, 4, UNET_LATENT_SIZE, UNET_LATENT_SIZE), dtype=np.float32)      # 64x64 UNet output
        self._dec_in_buf = np.empty((1, 4, LATENT_SIZE, LATENT_SIZE), dtype=np.float32)             # 32x32 decoder input
        self._t_buf = np.empty((1,), dtype=np.float16)
        self._uint8_small = np.empty((3, RENDER_SIZE, RENDER_SIZE), dtype=np.uint8)  # 256x256 decoder output
        self._bgr_small = np.empty((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.uint8)   # 256x256 HWC for resize
        self._output_buf = np.empty((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)  # 512x512 final output

        # Prompt encoding
        print("  Loading text encoder...")
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16
        ).to("mps")

        if cfg["scheduler"] == "euler":
            from diffusers import EulerDiscreteScheduler
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(1 if cfg["scheduler"] == "euler" else 50, device="mps")

        if cfg["scheduler"] == "euler":
            actual_t = pipe.scheduler.timesteps[0].cpu().item()
            self._t_buf[0] = np.float16(actual_t)
            ap = pipe.scheduler.alphas_cumprod[
                min(int(actual_t), len(pipe.scheduler.alphas_cumprod) - 1)
            ].item()
        else:
            t_idx = max(0, int(50 * (1.0 - STRENGTH)))
            if t_idx < len(pipe.scheduler.timesteps):
                actual_t = pipe.scheduler.timesteps[t_idx].cpu().item()
            else:
                actual_t = pipe.scheduler.timesteps[0].cpu().item()
            self._t_buf[0] = np.float16(actual_t)
            ap = pipe.scheduler.alphas_cumprod[int(actual_t)].item()

        self._sqrt_a = np.float32(np.sqrt(ap))
        self._sqrt_1ma = np.float32(np.sqrt(1.0 - ap))
        self._inv_sqrt_a = np.float32(1.0 / float(self._sqrt_a))

        # Encode prompt
        with torch.no_grad():
            ti = pipe.tokenizer(
                PROMPT, padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            self._prompt_embeds = pipe.text_encoder(
                ti.input_ids.to("mps")
            )[0].cpu().to(torch.float16).numpy()

        del pipe
        gc.collect()
        torch.mps.empty_cache()

        # Fixed noise at UNet scale (64x64) for temporal coherence
        rng = np.random.RandomState(42)
        fixed_noise = rng.randn(1, 4, UNET_LATENT_SIZE, UNET_LATENT_SIZE).astype(np.float32)
        self._noise_term = (self._sqrt_1ma * fixed_noise).astype(np.float32)

        # Pre-built CoreML input dicts (buffers updated in-place, no per-frame dict alloc)
        self._enc_input = {"image": self._img_buf}
        self._unet_input = {
            "sample": self._unet_sample,
            "timestep": self._t_buf,
            "encoder_hidden_states": self._prompt_embeds,
        }
        self._dec_input = {"latent": self._dec_in_buf}

        # Internal warmup (CoreML compilation)
        print("  CoreML warmup...")
        self._warmup()

    def _warmup(self, n=25):
        dummy = np.random.randn(1, 3, RENDER_SIZE, RENDER_SIZE).astype(np.float32)
        np.copyto(self._img_buf, dummy)
        unet_dummy = np.random.randn(1, 4, UNET_LATENT_SIZE, UNET_LATENT_SIZE).astype(np.float16)
        for _ in range(n):
            self.vae_encoder.predict({"image": self._img_buf})
            u = self.unet.predict({
                "sample": unet_dummy, "timestep": self._t_buf,
                "encoder_hidden_states": self._prompt_embeds,
            })
            np.copyto(self._dec_in_buf, np.array(u["noise_pred"]).astype(np.float16)[:, :, ::UPSAMPLE_FACTOR, ::UPSAMPLE_FACTOR])
            self.vae_decoder.predict({"latent": self._dec_in_buf})

    def process_frame(self, frame_bgr):
        """Full pipeline: preprocess -> VAE enc -> UNet -> VAE dec -> postprocess.

        This is the hot path. Optimize everything here.

        Args:
            frame_bgr: BGR uint8 ndarray, shape (H, W, 3)
        Returns:
            BGR uint8 ndarray, shape (OUTPUT_SIZE, OUTPUT_SIZE, 3)
        """
        # Preprocess at 256x256 (4x fewer pixels than 512x512)
        np.copyto(self._img_buf, cv2.dnn.blobFromImage(
            frame_bgr, 1.0 / 127.5, (RENDER_SIZE, RENDER_SIZE),
            (127.5, 127.5, 127.5), swapRB=True, crop=True))

        # VAE Encode at 256x256 → 32x32 latent (~4x faster than 512 encoder)
        enc = self.vae_encoder.predict(self._enc_input)
        clean = np.asarray(enc["latent"])  # (1, 4, LATENT_SIZE, LATENT_SIZE) float32

        # Upsample LATENT_SIZE → 64 for UNet
        np.copyto(self._unet_sample, np.repeat(np.repeat(clean, UPSAMPLE_FACTOR, axis=2), UPSAMPLE_FACTOR, axis=3))

        # Add noise at 64x64 scale
        np.multiply(self._sqrt_a, self._unet_sample, out=self._unet_sample)
        np.add(self._unet_sample, self._noise_term, out=self._unet_sample)

        # UNet inference at 64x64
        u = self.unet.predict(self._unet_input)
        npred = np.asarray(u["noise_pred"])  # (1, 4, 64, 64) float32

        # Denoise at 64x64
        np.multiply(self._sqrt_1ma, npred, out=self._out_buf)
        np.subtract(self._unet_sample, self._out_buf, out=self._out_buf)
        np.multiply(self._inv_sqrt_a, self._out_buf, out=self._out_buf)

        # Downsample 64x64 → LATENT_SIZE for small decoder (stride nearest neighbor)
        np.copyto(self._dec_in_buf, self._out_buf[:, :, ::UPSAMPLE_FACTOR, ::UPSAMPLE_FACTOR])

        # VAE Decode at 256x256 (~4x faster than 512 decoder)
        dec = self.vae_decoder.predict(self._dec_input)
        chw = np.asarray(dec["image"]).squeeze(0)  # (3, 256, 256) float32

        # Postprocess: float32 CHW → uint8 CHW → HWC BGR → resize 256→512
        cv2.convertScaleAbs(chw, dst=self._uint8_small, alpha=127.5, beta=127.5)
        np.copyto(self._bgr_small, self._uint8_small[::-1].transpose(1, 2, 0))
        cv2.resize(self._bgr_small, (OUTPUT_SIZE, OUTPUT_SIZE), dst=self._output_buf)
        return self._output_buf


def create_pipeline():
    """Factory function called by benchmark.py. Do not rename."""
    return InferencePipeline()
