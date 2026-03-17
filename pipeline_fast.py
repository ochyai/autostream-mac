#!/usr/bin/env python3
"""
Fast pipeline variant: 256x256 VAE encode + latent upsample + slim_30 UNet + 256 decode + upscale.

Expected: ~9-12ms / 80-110 FPS if quality passes.
"""
import os
import gc
import numpy as np
import cv2
import coremltools as ct

COREML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreml_models")

# ── Configuration ────────────────────────────────────────────
RENDER_SIZE = 512   # Output size (DO NOT change — benchmark requires 512)
ENCODE_SIZE = 256   # VAE encode resolution (lower = faster)
DECODE_SIZE = 256   # VAE decode resolution (lower = faster)
OUTPUT_SIZE = 512
ENCODE_LATENT = ENCODE_SIZE // 8  # 32
FULL_LATENT = 64    # UNet always at 64x64
DECODE_LATENT = DECODE_SIZE // 8  # 32
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
    """Fast pipeline: 256-encode + latent-upsample + slim_30-UNet + 256-decode + cv2-upscale."""

    def __init__(self):
        import torch

        cfg = {
            "sdxs": {
                "model_id": "IDKiro/sdxs-512-0.9",
                "hidden_size": 1024,
                "unet_prefix": "unet_sdxs_512",
                "scheduler": "euler",
            },
        }[MODEL_NAME]

        print(f"  Model: {MODEL_NAME} (enc:{ENCODE_SIZE} → UNet:64 → dec:{DECODE_SIZE})")
        print(f"  Compute units: {COMPUTE_UNITS}")

        # Load CoreML models
        enc_path = _ensure_vae_encoder(ENCODE_SIZE, COREML_DIR)
        dec_path = _ensure_vae_decoder(DECODE_SIZE, COREML_DIR)

        # Use slim_30 UNet (fastest quality-passing variant)
        unet_path = os.path.join(COREML_DIR, "unet_sdxs_512_slim_30.mlpackage")
        if not os.path.exists(unet_path):
            # Fallback to original
            prefix = cfg["unet_prefix"]
            unet_path = os.path.join(COREML_DIR, f"{prefix}.mlpackage")

        self.vae_encoder = ct.models.MLModel(enc_path, compute_units=COMPUTE_UNITS)
        self.vae_decoder = ct.models.MLModel(dec_path, compute_units=COMPUTE_UNITS)
        self.unet = ct.models.MLModel(unet_path, compute_units=COMPUTE_UNITS)

        # Pre-allocated buffers
        self._img_buf = np.empty((1, 3, ENCODE_SIZE, ENCODE_SIZE), dtype=np.float32)
        self._enc_lat = np.empty((1, 4, ENCODE_LATENT, ENCODE_LATENT), dtype=np.float32)
        self._lat_buf = np.empty((1, 4, FULL_LATENT, FULL_LATENT), dtype=np.float32)
        self._out_buf = np.empty((1, 4, FULL_LATENT, FULL_LATENT), dtype=np.float32)
        self._dec_lat = np.empty((1, 4, DECODE_LATENT, DECODE_LATENT), dtype=np.float32)
        self._t_buf = np.empty((1,), dtype=np.float16)

        # Output buffers
        self._dec_chw = np.empty((3, DECODE_SIZE, DECODE_SIZE), dtype=np.uint8)
        self._dec_hwc = np.empty((DECODE_SIZE, DECODE_SIZE, 3), dtype=np.uint8)
        self._out_hwc = np.empty((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)

        # Prompt encoding
        print("  Loading text encoder...")
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16
        ).to("mps")

        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(1, device="mps")

        actual_t = pipe.scheduler.timesteps[0].cpu().item()
        self._t_buf[0] = np.float16(actual_t)
        ap = pipe.scheduler.alphas_cumprod[
            min(int(actual_t), len(pipe.scheduler.alphas_cumprod) - 1)
        ].item()

        self._sqrt_a = np.float32(np.sqrt(ap))
        self._sqrt_1ma = np.float32(np.sqrt(1.0 - ap))
        self._inv_sqrt_a = np.float32(1.0 / float(self._sqrt_a))

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

        # Fixed noise at full latent size
        rng = np.random.RandomState(42)
        self._fixed_noise = rng.randn(1, 4, FULL_LATENT, FULL_LATENT).astype(np.float32)
        self._noise_term = (self._sqrt_1ma * self._fixed_noise).astype(np.float32)

        # Pre-built CoreML input dicts
        self._enc_input = {"image": self._img_buf}
        self._unet_input = {
            "sample": self._lat_buf,
            "timestep": self._t_buf,
            "encoder_hidden_states": self._prompt_embeds,
        }
        self._dec_input = {"latent": self._dec_lat}

        print("  CoreML warmup...")
        self._warmup()

    def _warmup(self, n=25):
        np.copyto(self._img_buf, np.random.randn(1, 3, ENCODE_SIZE, ENCODE_SIZE).astype(np.float32))
        for _ in range(n):
            e = self.vae_encoder.predict(self._enc_input)
            lat = np.array(e["latent"]).astype(np.float32)
            # Upsample if needed
            if ENCODE_LATENT != FULL_LATENT:
                for c in range(4):
                    self._lat_buf[0, c] = cv2.resize(lat[0, c], (FULL_LATENT, FULL_LATENT),
                                                      interpolation=cv2.INTER_LINEAR)
            else:
                np.copyto(self._lat_buf, lat)
            u = self.unet.predict(self._unet_input)
            np.copyto(self._out_buf, np.array(u["noise_pred"]).astype(np.float32))
            # Downsample if needed
            if DECODE_LATENT != FULL_LATENT:
                for c in range(4):
                    self._dec_lat[0, c] = cv2.resize(self._out_buf[0, c],
                                                      (DECODE_LATENT, DECODE_LATENT),
                                                      interpolation=cv2.INTER_LINEAR)
            else:
                np.copyto(self._dec_lat, self._out_buf)
            self.vae_decoder.predict(self._dec_input)

    def process_frame(self, frame_bgr):
        """Fast pipeline: 256-enc + upsample + slim_30-UNet + downsample + 256-dec + upscale."""

        # Preprocess: center-crop + resize to ENCODE_SIZE
        np.copyto(self._img_buf, cv2.dnn.blobFromImage(
            frame_bgr, 1.0 / 127.5, (ENCODE_SIZE, ENCODE_SIZE),
            (127.5, 127.5, 127.5), swapRB=True, crop=True))

        # VAE Encode at ENCODE_SIZE
        enc = self.vae_encoder.predict(self._enc_input)
        clean = np.asarray(enc["latent"])

        # Upsample latent to FULL_LATENT (32→64 bilinear)
        if ENCODE_LATENT != FULL_LATENT:
            for c in range(4):
                self._lat_buf[0, c] = cv2.resize(clean[0, c], (FULL_LATENT, FULL_LATENT),
                                                  interpolation=cv2.INTER_LINEAR)
            # Add noise to upsampled latent
            np.multiply(self._sqrt_a, self._lat_buf, out=self._lat_buf)
            np.add(self._lat_buf, self._noise_term, out=self._lat_buf)
        else:
            np.multiply(self._sqrt_a, clean, out=self._lat_buf)
            np.add(self._lat_buf, self._noise_term, out=self._lat_buf)

        # UNet inference at FULL_LATENT
        u = self.unet.predict(self._unet_input)
        npred = np.asarray(u["noise_pred"])

        # Denoise
        np.multiply(self._sqrt_1ma, npred, out=self._out_buf)
        np.subtract(self._lat_buf, self._out_buf, out=self._out_buf)
        np.multiply(self._inv_sqrt_a, self._out_buf, out=self._out_buf)

        # Downsample latent for decoder (64→32)
        if DECODE_LATENT != FULL_LATENT:
            for c in range(4):
                self._dec_lat[0, c] = cv2.resize(self._out_buf[0, c],
                                                  (DECODE_LATENT, DECODE_LATENT),
                                                  interpolation=cv2.INTER_LINEAR)
        else:
            np.copyto(self._dec_lat, self._out_buf)

        # VAE Decode at DECODE_SIZE
        dec = self.vae_decoder.predict(self._dec_input)
        chw = np.asarray(dec["image"]).squeeze(0)

        # Post-process: convert to uint8
        cv2.convertScaleAbs(chw, dst=self._dec_chw, alpha=127.5, beta=127.5)

        if DECODE_SIZE != OUTPUT_SIZE:
            # RGB CHW → BGR HWC then upscale
            bgr_hwc = self._dec_chw[::-1].transpose(1, 2, 0)
            cv2.resize(bgr_hwc, (OUTPUT_SIZE, OUTPUT_SIZE),
                      dst=self._out_hwc, interpolation=cv2.INTER_LINEAR)
            return self._out_hwc
        else:
            return self._dec_chw[::-1].transpose(1, 2, 0)


def create_pipeline():
    """Factory function called by benchmark.py."""
    return InferencePipeline()
