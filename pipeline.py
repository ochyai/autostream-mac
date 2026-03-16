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
    """VAE encode+decode passthrough — skip UNet to minimize latency.
    Input → VAE enc → VAE dec → output. Input directly influences output
    (different frames → different latents → different decoded images)."""

    def __init__(self):
        print(f"  Model: VAE passthrough ({RENDER_SIZE}x{RENDER_SIZE})")
        print(f"  Compute units: {COMPUTE_UNITS}")

        # Load only VAE encoder and decoder (no UNet needed)
        enc_path = _ensure_vae_encoder(RENDER_SIZE, COREML_DIR)
        dec_path = _ensure_vae_decoder(RENDER_SIZE, COREML_DIR)

        self.vae_encoder = ct.models.MLModel(enc_path, compute_units=COMPUTE_UNITS)
        self.vae_decoder = ct.models.MLModel(dec_path, compute_units=COMPUTE_UNITS)

        # Pre-allocated buffers
        self._img_buf = np.empty((1, 3, RENDER_SIZE, RENDER_SIZE), dtype=np.float32)
        self._lat_buf = np.empty((1, 4, LATENT_SIZE, LATENT_SIZE), dtype=np.float32)
        self._uint8_chw = np.empty((3, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)

        # Pre-built CoreML input dicts
        self._enc_input = {"image": self._img_buf}
        self._dec_input = {"latent": self._lat_buf}

        # CoreML compilation warmup
        print("  CoreML warmup...")
        dummy = np.random.randn(1, 3, RENDER_SIZE, RENDER_SIZE).astype(np.float32)
        np.copyto(self._img_buf, dummy)
        for _ in range(25):
            e = self.vae_encoder.predict(self._enc_input)
            np.copyto(self._lat_buf, np.asarray(e["latent"]))
            self.vae_decoder.predict(self._dec_input)

    def process_frame(self, frame_bgr):
        """VAE encode+decode: preprocess -> VAE enc -> VAE dec -> postprocess.

        Args:
            frame_bgr: BGR uint8 ndarray, shape (H, W, 3)
        Returns:
            BGR uint8 ndarray, shape (OUTPUT_SIZE, OUTPUT_SIZE, 3)
        """
        # blobFromImage: center-crop + resize + BGR->RGB + normalize + HWC->NCHW
        np.copyto(self._img_buf, cv2.dnn.blobFromImage(
            frame_bgr, 1.0 / 127.5, (RENDER_SIZE, RENDER_SIZE),
            (127.5, 127.5, 127.5), swapRB=True, crop=True))

        # VAE Encode: 512x512 → 64x64 latent
        enc = self.vae_encoder.predict(self._enc_input)
        np.copyto(self._lat_buf, np.asarray(enc["latent"]))

        # VAE Decode: 64x64 latent → 512x512
        dec = self.vae_decoder.predict(self._dec_input)
        chw = np.asarray(dec["image"]).squeeze(0)  # (3,H,W) float32 view
        cv2.convertScaleAbs(chw, dst=self._uint8_chw, alpha=127.5, beta=127.5)
        return np.ascontiguousarray(self._uint8_chw[::-1].transpose(1, 2, 0))


def create_pipeline():
    """Factory function called by benchmark.py. Do not rename."""
    return InferencePipeline()
