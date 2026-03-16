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
RENDER_SIZE = 512
OUTPUT_SIZE = 512
LATENT_SIZE = RENDER_SIZE // 8
MODEL_NAME = "sdxs"
PROMPT = "oil painting style, masterpiece, highly detailed"
STRENGTH = 0.5
LATENT_FEEDBACK = 0.3
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
        self._img_buf = np.empty((1, 3, RENDER_SIZE, RENDER_SIZE), dtype=np.float16)
        self._lat_buf = np.empty((1, 4, LATENT_SIZE, LATENT_SIZE), dtype=np.float16)
        self._out_buf = np.empty((1, 4, LATENT_SIZE, LATENT_SIZE), dtype=np.float16)
        self._t_buf = np.empty((1,), dtype=np.float16)

        # Normalization LUT: pixel [0,255] -> [-1, 1] in float16
        self._norm_lut = (np.arange(256, dtype=np.float32) / 127.5 - 1.0).astype(np.float16)

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

        self._sqrt_a = np.float16(np.sqrt(ap))
        self._sqrt_1ma = np.float16(np.sqrt(1.0 - ap))
        self._inv_sqrt_a = np.float16(1.0 / float(self._sqrt_a))

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

        # Fixed noise for temporal coherence
        rng = np.random.RandomState(42)
        self._fixed_noise = rng.randn(1, 4, LATENT_SIZE, LATENT_SIZE).astype(np.float16)
        self._noise_term = (self._sqrt_1ma * self._fixed_noise).astype(np.float16)
        self._prev_denoised = np.zeros((1, 4, LATENT_SIZE, LATENT_SIZE), dtype=np.float16)
        self._has_prev = False
        self._fb = np.float16(LATENT_FEEDBACK)
        self._fb_inv = np.float16(1.0 - LATENT_FEEDBACK)

        # Pre-built CoreML input dicts (buffers updated in-place, no per-frame dict alloc)
        self._enc_input = {"image": self._img_buf}
        self._unet_input = {
            "sample": self._lat_buf,
            "timestep": self._t_buf,
            "encoder_hidden_states": self._prompt_embeds,
        }
        self._dec_input = {"latent": self._out_buf}

        # Internal warmup (CoreML compilation)
        print("  CoreML warmup...")
        self._warmup()

    def _warmup(self, n=25):
        dummy = np.random.randn(1, 3, RENDER_SIZE, RENDER_SIZE).astype(np.float16)
        np.copyto(self._img_buf, dummy)
        for _ in range(n):
            e = self.vae_encoder.predict({"image": self._img_buf})
            np.copyto(self._lat_buf, np.array(e["latent"]).astype(np.float16))
            u = self.unet.predict({
                "sample": self._lat_buf, "timestep": self._t_buf,
                "encoder_hidden_states": self._prompt_embeds,
            })
            np.copyto(self._out_buf, np.array(u["noise_pred"]).astype(np.float16))
            self.vae_decoder.predict({"latent": self._out_buf})

    def process_frame(self, frame_bgr):
        """Full pipeline: preprocess -> VAE enc -> UNet -> VAE dec -> postprocess.

        This is the hot path. Optimize everything here.

        Args:
            frame_bgr: BGR uint8 ndarray, shape (H, W, 3)
        Returns:
            BGR uint8 ndarray, shape (OUTPUT_SIZE, OUTPUT_SIZE, 3)
        """
        h, w = frame_bgr.shape[:2]

        # Center crop to square
        if w > h:
            off = (w - h) // 2
            frame_bgr = frame_bgr[:, off:off + h]
        elif h > w:
            off = (h - w) // 2
            frame_bgr = frame_bgr[off:off + w, :]

        # Resize + normalize
        resized = cv2.resize(frame_bgr, (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_NEAREST)
        rgb = resized[:, :, ::-1]
        np.copyto(self._img_buf, self._norm_lut[rgb].transpose(2, 0, 1)[np.newaxis])

        # VAE Encode
        enc = self.vae_encoder.predict(self._enc_input)
        clean = np.asarray(enc["latent"], dtype=np.float16)

        # Latent feedback from previous frame
        if self._has_prev and LATENT_FEEDBACK > 0:
            clean = self._fb_inv * clean + self._fb * self._prev_denoised

        # Compute noisy directly into _lat_buf (no temp allocation)
        np.multiply(self._sqrt_a, clean, out=self._lat_buf)
        np.add(self._lat_buf, self._noise_term, out=self._lat_buf)

        # UNet inference
        u = self.unet.predict(self._unet_input)
        npred = np.asarray(u["noise_pred"], dtype=np.float16)

        # Compute denoised directly into _out_buf (no temp allocations)
        np.multiply(self._sqrt_1ma, npred, out=self._out_buf)
        np.subtract(self._lat_buf, self._out_buf, out=self._out_buf)
        np.multiply(self._inv_sqrt_a, self._out_buf, out=self._out_buf)
        np.copyto(self._prev_denoised, self._out_buf)
        self._has_prev = True

        # VAE Decode
        dec = self.vae_decoder.predict(self._dec_input)
        r = np.asarray(dec["image"], dtype=np.float32).squeeze(0).transpose(1, 2, 0)
        r = ((r + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        if r.shape[0] != OUTPUT_SIZE:
            r = cv2.resize(r, (OUTPUT_SIZE, OUTPUT_SIZE))

        return cv2.cvtColor(r, cv2.COLOR_RGB2BGR)


def create_pipeline():
    """Factory function called by benchmark.py. Do not rename."""
    return InferencePipeline()
