#!/usr/bin/env python3
"""
Benchmark ALL model variants (original + optimized).
Tests each UNet/VAE combination and records results.

Run: .venv/bin/python benchmark_all_variants.py
"""
import os
import sys
import time
import glob
import numpy as np
import cv2
import coremltools as ct

COREML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreml_models")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_variants_results.tsv")

# Benchmark params
N_WARMUP = 30
N_BENCHMARK = 100
FRAME_H, FRAME_W = 480, 640
SEED = 42
RENDER_SIZE = 512
OUTPUT_SIZE = 512
LATENT_SIZE = 64

# Quality thresholds
MIN_UNIQUE_COLORS = 10000
MIN_TRANSFORM_DIFF = 20.0
MIN_EDGE_STRENGTH = 5.0


def generate_frames(n, seed):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(n)]


def get_memory_mb():
    try:
        import subprocess
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                           capture_output=True, text=True)
        return int(r.stdout.strip()) / 1024
    except:
        return 0.0


def build_pipeline(unet_path, enc_path, dec_path):
    """Build a minimal pipeline for benchmarking a specific model combo."""
    import gc

    cu = ct.ComputeUnit.CPU_AND_GPU
    vae_encoder = ct.models.MLModel(enc_path, compute_units=cu)
    vae_decoder = ct.models.MLModel(dec_path, compute_units=cu)
    unet = ct.models.MLModel(unet_path, compute_units=cu)

    # Scheduler setup (minimal, same as pipeline.py)
    from diffusers import EulerDiscreteScheduler
    sched = EulerDiscreteScheduler.from_pretrained("IDKiro/sdxs-512-0.9", subfolder="scheduler")
    sched.set_timesteps(1)
    actual_t = sched.timesteps[0].item()
    ap = sched.alphas_cumprod[min(int(actual_t), len(sched.alphas_cumprod) - 1)].item()

    t_buf = np.array([actual_t], dtype=np.float16)
    sqrt_a = np.float32(np.sqrt(ap))
    sqrt_1ma = np.float32(np.sqrt(1.0 - ap))
    inv_sqrt_a = np.float32(1.0 / sqrt_a)

    # Prompt embeddings (encode once)
    import torch
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "IDKiro/sdxs-512-0.9", torch_dtype=torch.float16).to("mps")
    with torch.no_grad():
        ti = pipe.tokenizer("oil painting style, masterpiece, highly detailed",
                            padding="max_length", max_length=pipe.tokenizer.model_max_length,
                            truncation=True, return_tensors="pt")
        prompt_embeds = pipe.text_encoder(ti.input_ids.to("mps"))[0].cpu().to(torch.float16).numpy()
    del pipe
    gc.collect()
    torch.mps.empty_cache()

    # Fixed noise
    rng = np.random.RandomState(42)
    fixed_noise = rng.randn(1, 4, LATENT_SIZE, LATENT_SIZE).astype(np.float32)
    noise_term = (sqrt_1ma * fixed_noise).astype(np.float32)

    # Buffers
    img_buf = np.empty((1, 3, RENDER_SIZE, RENDER_SIZE), dtype=np.float32)
    lat_buf = np.empty((1, 4, LATENT_SIZE, LATENT_SIZE), dtype=np.float32)
    out_buf = np.empty((1, 4, LATENT_SIZE, LATENT_SIZE), dtype=np.float32)
    uint8_chw = np.empty((3, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)

    enc_input = {"image": img_buf}
    unet_input = {"sample": lat_buf, "timestep": t_buf, "encoder_hidden_states": prompt_embeds}
    dec_input = {"latent": out_buf}

    def process_frame(frame_bgr):
        np.copyto(img_buf, cv2.dnn.blobFromImage(
            frame_bgr, 1.0/127.5, (RENDER_SIZE, RENDER_SIZE),
            (127.5, 127.5, 127.5), swapRB=True, crop=True))
        enc = vae_encoder.predict(enc_input)
        clean = np.asarray(enc["latent"])
        np.multiply(sqrt_a, clean, out=lat_buf)
        np.add(lat_buf, noise_term, out=lat_buf)
        u = unet.predict(unet_input)
        npred = np.asarray(u["noise_pred"])
        np.multiply(sqrt_1ma, npred, out=out_buf)
        np.subtract(lat_buf, out_buf, out=out_buf)
        np.multiply(inv_sqrt_a, out_buf, out=out_buf)
        dec = vae_decoder.predict(dec_input)
        chw = np.asarray(dec["image"]).squeeze(0)
        cv2.convertScaleAbs(chw, dst=uint8_chw, alpha=127.5, beta=127.5)
        return uint8_chw[::-1].transpose(1, 2, 0)

    return process_frame


def benchmark_variant(process_fn, label):
    """Benchmark a single pipeline variant."""
    frames = generate_frames(N_WARMUP + N_BENCHMARK, SEED)

    # Warmup
    for f in frames[:N_WARMUP]:
        process_fn(f)

    # Benchmark
    times = []
    for f in frames[N_WARMUP:]:
        t0 = time.perf_counter()
        process_fn(f)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)

    avg_ms = sum(times) / len(times)
    fps = 1000.0 / avg_ms

    # Quick quality check
    test_frames = generate_frames(3, SEED + 9999)
    outputs = [np.array(process_fn(f), dtype=np.uint8, copy=True) for f in test_frames]

    # Unique colors
    pixels = outputs[0].reshape(-1, 3)
    packed = (pixels[:, 0].astype(np.uint32) << 16 |
              pixels[:, 1].astype(np.uint32) << 8 |
              pixels[:, 2].astype(np.uint32))
    unique_colors = len(np.unique(packed))

    # Transform diff
    h, w = test_frames[0].shape[:2]
    off = (w - h) // 2
    sq = test_frames[0][:, off:off+h]
    inp = cv2.resize(sq, (OUTPUT_SIZE, OUTPUT_SIZE))
    transform_diff = float(np.mean(np.abs(outputs[0].astype(np.float32) - inp.astype(np.float32))))

    # Edge strength
    gray = cv2.cvtColor(outputs[0], cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = float(np.mean(np.sqrt(sx**2 + sy**2)))

    quality_pass = (unique_colors >= MIN_UNIQUE_COLORS and
                    transform_diff >= MIN_TRANSFORM_DIFF and
                    edge_strength >= MIN_EDGE_STRENGTH)

    mem = get_memory_mb()

    return {
        "label": label,
        "avg_ms": avg_ms,
        "fps": fps,
        "memory_mb": mem,
        "unique_colors": unique_colors,
        "transform_diff": transform_diff,
        "edge_strength": edge_strength,
        "quality_pass": quality_pass,
    }


def find_unet_variants():
    """Find all UNet model variants."""
    base = os.path.join(COREML_DIR, "unet_sdxs_512.mlpackage")
    variants = [("original", base)]

    for suffix in ["pal2", "pal4", "pal6", "int8", "pruned", "pal4_pruned"]:
        path = base.replace(".mlpackage", f"_{suffix}.mlpackage")
        if os.path.exists(path):
            variants.append((suffix, path))
    return variants


def main():
    print("=" * 60)
    print("Benchmarking ALL Model Variants")
    print("=" * 60)

    enc_path = os.path.join(COREML_DIR, "taesd_encoder_512.mlpackage")
    dec_path = os.path.join(COREML_DIR, "taesd_decoder.mlpackage")

    # Also check for optimized VAE variants
    enc_pal4 = enc_path.replace(".mlpackage", "_pal4.mlpackage")
    dec_pal4 = dec_path.replace(".mlpackage", "_pal4.mlpackage")

    unet_variants = find_unet_variants()
    print(f"\nFound {len(unet_variants)} UNet variants:")
    for name, path in unet_variants:
        print(f"  {name}: {path}")

    results = []

    for unet_name, unet_path in unet_variants:
        # Test with original VAE
        label = f"unet_{unet_name}"
        print(f"\n{'─' * 40}")
        print(f"Testing: {label}")
        print(f"{'─' * 40}")

        try:
            process_fn = build_pipeline(unet_path, enc_path, dec_path)
            result = benchmark_variant(process_fn, label)
            results.append(result)
            status = "PASS" if result["quality_pass"] else "FAIL"
            print(f"  → {result['avg_ms']:.1f}ms / {result['fps']:.1f} FPS "
                  f"| colors={result['unique_colors']} edge={result['edge_strength']:.1f} "
                  f"| {status}")
        except Exception as e:
            print(f"  → ERROR: {e}")
            results.append({"label": label, "avg_ms": 0, "fps": 0, "memory_mb": 0,
                           "unique_colors": 0, "transform_diff": 0, "edge_strength": 0,
                           "quality_pass": False, "error": str(e)})

        # Test with palettized VAE if available
        if os.path.exists(enc_pal4) and os.path.exists(dec_pal4):
            label_vae = f"unet_{unet_name}+vae_pal4"
            print(f"\n  Also testing with palettized VAE: {label_vae}")
            try:
                process_fn = build_pipeline(unet_path, enc_pal4, dec_pal4)
                result = benchmark_variant(process_fn, label_vae)
                results.append(result)
                status = "PASS" if result["quality_pass"] else "FAIL"
                print(f"  → {result['avg_ms']:.1f}ms / {result['fps']:.1f} FPS "
                      f"| colors={result['unique_colors']} edge={result['edge_strength']:.1f} "
                      f"| {status}")
            except Exception as e:
                print(f"  → ERROR: {e}")

    # Write results
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    with open(RESULTS_FILE, "w") as f:
        f.write("variant\tavg_ms\tfps\tmemory_mb\tunique_colors\ttransform_diff\tedge_strength\tquality_pass\n")
        for r in sorted(results, key=lambda x: x.get("avg_ms", 999)):
            qp = "true" if r.get("quality_pass") else "false"
            line = (f"{r['label']}\t{r.get('avg_ms',0):.3f}\t{r.get('fps',0):.2f}\t"
                    f"{r.get('memory_mb',0):.0f}\t{r.get('unique_colors',0)}\t"
                    f"{r.get('transform_diff',0):.2f}\t{r.get('edge_strength',0):.2f}\t{qp}")
            f.write(line + "\n")
            marker = "✓" if r.get("quality_pass") else "✗"
            print(f"  {marker} {r['label']:30s}  {r.get('avg_ms',0):7.1f}ms  {r.get('fps',0):7.1f} FPS  "
                  f"colors={r.get('unique_colors',0):6d}  edge={r.get('edge_strength',0):.1f}")

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
