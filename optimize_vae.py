#!/usr/bin/env python3
"""
VAE Optimization for the autostream pipeline.

The UNet is no longer the bottleneck (slim_30: 8ms).
VAE encode (5.9ms) + decode (7.0ms) = 12.9ms = 61% of total pipeline.

Optimization strategies:
1. Test different compute units for VAE (ANE, CPU, GPU combos)
2. Lower-res encoding (256x256) + latent upsampling
3. Channel-slimmed VAE (like we did for UNet)
4. VAE with different resolution decoders

Run: .venv/bin/python optimize_vae.py
"""
import os
import sys
import json
import time
import gc
import shutil
import numpy as np
import cv2

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import coremltools as ct

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
COREML_DIR = os.path.join(WORK_DIR, "coreml_models")
LOG_FILE = os.path.join(WORK_DIR, "vae_optimize.log")
RESULTS_FILE = os.path.join(WORK_DIR, "vae_optimize_results.json")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def profile_model(model, inputs, n=50, warmup=10):
    """Profile a CoreML model, return avg ms."""
    for _ in range(warmup):
        model.predict(inputs)
    t0 = time.time()
    for _ in range(n):
        model.predict(inputs)
    return (time.time() - t0) / n * 1000


def convert_vae_encoder(resolution):
    """Convert TinyVAE encoder at given resolution."""
    from diffusers import AutoencoderTiny

    path = os.path.join(COREML_DIR, f"taesd_encoder_{resolution}.mlpackage")
    if os.path.exists(path):
        return path

    log(f"  Converting VAE encoder at {resolution}x{resolution}...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class W(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.encoder = v.encoder
        def forward(self, x):
            return self.encoder(x)

    w = W(vae).eval()
    d = torch.randn(1, 3, resolution, resolution)
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
    del m, traced, w, vae; gc.collect()
    return path


def convert_vae_decoder(resolution):
    """Convert TinyVAE decoder at given resolution."""
    from diffusers import AutoencoderTiny

    if resolution == 512:
        path = os.path.join(COREML_DIR, "taesd_decoder.mlpackage")
    else:
        path = os.path.join(COREML_DIR, f"taesd_decoder_{resolution}.mlpackage")
    if os.path.exists(path):
        return path

    log(f"  Converting VAE decoder at {resolution}x{resolution}...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()
    ls = resolution // 8

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
    del m, traced, w, vae; gc.collect()
    return path


def test_compute_units():
    """Test different compute unit combos for VAE."""
    log("\n=== Test 1: Compute Units for VAE ===")

    results = []
    units_to_test = [
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("ALL", ct.ComputeUnit.ALL),
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
    ]

    enc_path = os.path.join(COREML_DIR, "taesd_encoder_512.mlpackage")
    dec_path = os.path.join(COREML_DIR, "taesd_decoder.mlpackage")

    dummy_img = np.random.randn(1, 3, 512, 512).astype(np.float32)
    dummy_lat = np.random.randn(1, 4, 64, 64).astype(np.float32)

    for name, unit in units_to_test:
        log(f"  Testing {name}...")
        try:
            enc = ct.models.MLModel(enc_path, compute_units=unit)
            dec = ct.models.MLModel(dec_path, compute_units=unit)

            enc_ms = profile_model(enc, {"image": dummy_img})
            dec_ms = profile_model(dec, {"latent": dummy_lat})

            log(f"    Encode: {enc_ms:.2f}ms, Decode: {dec_ms:.2f}ms, Total: {enc_ms+dec_ms:.2f}ms")
            results.append({
                "test": "compute_units",
                "config": name,
                "enc_ms": enc_ms,
                "dec_ms": dec_ms,
                "total_ms": enc_ms + dec_ms,
            })

            del enc, dec
            gc.collect()
        except Exception as e:
            log(f"    Failed: {e}")
            results.append({"test": "compute_units", "config": name, "error": str(e)})

    return results


def test_lower_res_encoding():
    """Test encoding at lower resolution + latent upsampling."""
    log("\n=== Test 2: Lower-Res Encoding + Latent Upsample ===")

    results = []
    resolutions = [256, 384]  # Original is 512

    dec_path = os.path.join(COREML_DIR, "taesd_decoder.mlpackage")
    dec = ct.models.MLModel(dec_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    for res in resolutions:
        log(f"  Testing encode {res}x{res} → upsample to 64x64...")

        # Ensure encoder exists
        enc_path = convert_vae_encoder(res)
        enc = ct.models.MLModel(enc_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

        dummy_img = np.random.randn(1, 3, res, res).astype(np.float32)
        latent_size = res // 8

        # Profile encode
        enc_ms = profile_model(enc, {"image": dummy_img})

        # Profile latent upsample (numpy bilinear)
        dummy_lat = np.random.randn(1, 4, latent_size, latent_size).astype(np.float32)
        t0 = time.time()
        for _ in range(50):
            # Bilinear upsample latent to 64x64
            upsampled = np.zeros((1, 4, 64, 64), dtype=np.float32)
            for c in range(4):
                upsampled[0, c] = cv2.resize(
                    dummy_lat[0, c], (64, 64), interpolation=cv2.INTER_LINEAR
                )
        upsample_ms = (time.time() - t0) / 50 * 1000

        # Profile decode (always at 512x512)
        dummy_lat_64 = np.random.randn(1, 4, 64, 64).astype(np.float32)
        dec_ms = profile_model(dec, {"latent": dummy_lat_64})

        # Also profile preprocess (resize to lower res)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        t0 = time.time()
        for _ in range(50):
            blob = cv2.dnn.blobFromImage(frame, 1.0/127.5, (res, res),
                                         (127.5, 127.5, 127.5), swapRB=True, crop=True)
        preprocess_ms = (time.time() - t0) / 50 * 1000

        total = preprocess_ms + enc_ms + upsample_ms + dec_ms
        log(f"    preprocess({res}): {preprocess_ms:.2f}ms")
        log(f"    encode({res}): {enc_ms:.2f}ms")
        log(f"    upsample({latent_size}→64): {upsample_ms:.2f}ms")
        log(f"    decode(512): {dec_ms:.2f}ms")
        log(f"    total: {total:.2f}ms (vs 512 baseline ~13ms)")

        results.append({
            "test": "lower_res_encode",
            "config": f"{res}x{res}",
            "preprocess_ms": preprocess_ms,
            "enc_ms": enc_ms,
            "upsample_ms": upsample_ms,
            "dec_ms": dec_ms,
            "total_ms": total,
        })

        del enc; gc.collect()

    del dec; gc.collect()
    return results


def test_mixed_compute_units():
    """Test different compute units for encoder vs decoder."""
    log("\n=== Test 3: Mixed Compute Units (enc ≠ dec) ===")

    results = []
    combos = [
        ("enc:CPU_AND_NE dec:CPU_AND_GPU", ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU),
        ("enc:CPU_AND_GPU dec:CPU_AND_NE", ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_NE),
        ("enc:ALL dec:CPU_AND_GPU", ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_GPU),
        ("enc:CPU_AND_GPU dec:ALL", ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.ALL),
    ]

    enc_path = os.path.join(COREML_DIR, "taesd_encoder_512.mlpackage")
    dec_path = os.path.join(COREML_DIR, "taesd_decoder.mlpackage")

    dummy_img = np.random.randn(1, 3, 512, 512).astype(np.float32)
    dummy_lat = np.random.randn(1, 4, 64, 64).astype(np.float32)

    for name, enc_unit, dec_unit in combos:
        log(f"  Testing {name}...")
        try:
            enc = ct.models.MLModel(enc_path, compute_units=enc_unit)
            dec = ct.models.MLModel(dec_path, compute_units=dec_unit)

            enc_ms = profile_model(enc, {"image": dummy_img})
            dec_ms = profile_model(dec, {"latent": dummy_lat})

            log(f"    Encode: {enc_ms:.2f}ms, Decode: {dec_ms:.2f}ms, Total: {enc_ms+dec_ms:.2f}ms")
            results.append({
                "test": "mixed_compute_units",
                "config": name,
                "enc_ms": enc_ms,
                "dec_ms": dec_ms,
                "total_ms": enc_ms + dec_ms,
            })

            del enc, dec; gc.collect()
        except Exception as e:
            log(f"    Failed: {e}")
            results.append({"test": "mixed_compute_units", "config": name, "error": str(e)})

    return results


def test_lower_res_decoder():
    """Test decoding at lower resolution + cv2 upscale."""
    log("\n=== Test 4: Lower-Res Decode + cv2 Upscale ===")

    results = []
    decode_resolutions = [256, 384]

    enc_path = os.path.join(COREML_DIR, "taesd_encoder_512.mlpackage")
    enc = ct.models.MLModel(enc_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    dummy_img = np.random.randn(1, 3, 512, 512).astype(np.float32)

    for res in decode_resolutions:
        log(f"  Testing decode at {res}x{res} + upscale to 512...")

        # Need decoder at this resolution
        dec_path = convert_vae_decoder(res)
        dec = ct.models.MLModel(dec_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

        latent_size = res // 8
        dummy_lat = np.random.randn(1, 4, latent_size, latent_size).astype(np.float32)

        # Profile encode (full 512)
        enc_ms = profile_model(enc, {"image": dummy_img})

        # Profile latent downsample (64→latent_size)
        full_lat = np.random.randn(1, 4, 64, 64).astype(np.float32)
        t0 = time.time()
        for _ in range(50):
            small_lat = np.zeros((1, 4, latent_size, latent_size), dtype=np.float32)
            for c in range(4):
                small_lat[0, c] = cv2.resize(
                    full_lat[0, c], (latent_size, latent_size),
                    interpolation=cv2.INTER_LINEAR
                )
        downsample_ms = (time.time() - t0) / 50 * 1000

        # Profile decode at lower res
        dec_ms = profile_model(dec, {"latent": dummy_lat})

        # Profile upscale to 512
        small_img = np.random.randn(3, res, res).astype(np.float32)
        t0 = time.time()
        for _ in range(50):
            # HWC for cv2
            hwc = small_img.transpose(1, 2, 0)
            upscaled = cv2.resize(hwc, (512, 512), interpolation=cv2.INTER_LINEAR)
        upscale_ms = (time.time() - t0) / 50 * 1000

        total = enc_ms + downsample_ms + dec_ms + upscale_ms
        log(f"    encode(512): {enc_ms:.2f}ms")
        log(f"    downsample(64→{latent_size}): {downsample_ms:.2f}ms")
        log(f"    decode({res}): {dec_ms:.2f}ms")
        log(f"    upscale({res}→512): {upscale_ms:.2f}ms")
        log(f"    total VAE: {total:.2f}ms")

        results.append({
            "test": "lower_res_decode",
            "config": f"decode_{res}",
            "enc_ms": enc_ms,
            "downsample_ms": downsample_ms,
            "dec_ms": dec_ms,
            "upscale_ms": upscale_ms,
            "total_ms": total,
        })

        del dec; gc.collect()

    del enc; gc.collect()
    return results


def main():
    with open(LOG_FILE, 'w') as f: f.write("")

    log("=" * 60)
    log("  VAE Optimization Pipeline")
    log("=" * 60)

    all_results = []

    # Test 1: Compute units
    try:
        results = test_compute_units()
        all_results.extend(results)
    except Exception as e:
        log(f"Test 1 failed: {e}")
        import traceback; traceback.print_exc()

    # Test 2: Lower-res encoding
    try:
        results = test_lower_res_encoding()
        all_results.extend(results)
    except Exception as e:
        log(f"Test 2 failed: {e}")
        import traceback; traceback.print_exc()

    # Test 3: Mixed compute units
    try:
        results = test_mixed_compute_units()
        all_results.extend(results)
    except Exception as e:
        log(f"Test 3 failed: {e}")
        import traceback; traceback.print_exc()

    # Test 4: Lower-res decoding
    try:
        results = test_lower_res_decoder()
        all_results.extend(results)
    except Exception as e:
        log(f"Test 4 failed: {e}")
        import traceback; traceback.print_exc()

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    log(f"\n{'='*60}")
    log("  SUMMARY")
    log(f"{'='*60}")
    log(f"  Baseline (CPU_AND_GPU, 512x512): enc=5.87ms + dec=7.03ms = 12.90ms")
    log("")

    for r in all_results:
        if "error" not in r:
            log(f"  [{r['test']}] {r['config']}: total={r['total_ms']:.2f}ms")
        else:
            log(f"  [{r['test']}] {r['config']}: FAILED — {r['error'][:50]}")

    log(f"\nResults: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
