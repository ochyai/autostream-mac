#!/usr/bin/env python3
"""
Model optimization script for autostream-mac.
Applies coremltools optimizations to CoreML models:
  1. Palettization (2, 4, 6 bit variants)
  2. Linear quantization (INT8)
  3. Weight pruning (magnitude-based)
  4. Combined optimizations

Run: .venv/bin/python optimize_models.py
"""
import os
import sys
import time
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OpLinearQuantizerConfig,
    OpMagnitudePrunerConfig,
    OptimizationConfig,
    palettize_weights,
    linear_quantize_weights,
    prune_weights,
)

COREML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreml_models")

# Models to optimize
MODELS = {
    "unet": os.path.join(COREML_DIR, "unet_sdxs_512.mlpackage"),
    "vae_enc": os.path.join(COREML_DIR, "taesd_encoder_512.mlpackage"),
    "vae_dec": os.path.join(COREML_DIR, "taesd_decoder.mlpackage"),
}


def get_model_size_mb(path):
    """Get total size of mlpackage in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def optimize_palettize(model_path, n_bits, output_suffix):
    """Apply palettization (weight clustering) to reduce model size."""
    out_path = model_path.replace(".mlpackage", f"_{output_suffix}.mlpackage")
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return out_path

    print(f"  Loading {os.path.basename(model_path)}...")
    model = ct.models.MLModel(model_path)

    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            mode="kmeans",
            nbits=n_bits,
        )
    )

    print(f"  Palettizing to {n_bits}-bit...")
    t0 = time.time()
    optimized = palettize_weights(model, config)
    elapsed = time.time() - t0

    optimized.save(out_path)
    orig_mb = get_model_size_mb(model_path)
    opt_mb = get_model_size_mb(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Size: {orig_mb:.1f}MB → {opt_mb:.1f}MB ({opt_mb/orig_mb*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s")
    return out_path


def optimize_quantize(model_path, output_suffix="int8"):
    """Apply INT8 linear quantization."""
    out_path = model_path.replace(".mlpackage", f"_{output_suffix}.mlpackage")
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return out_path

    print(f"  Loading {os.path.basename(model_path)}...")
    model = ct.models.MLModel(model_path)

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
        )
    )

    print(f"  Quantizing to INT8...")
    t0 = time.time()
    optimized = linear_quantize_weights(model, config)
    elapsed = time.time() - t0

    optimized.save(out_path)
    orig_mb = get_model_size_mb(model_path)
    opt_mb = get_model_size_mb(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Size: {orig_mb:.1f}MB → {opt_mb:.1f}MB ({opt_mb/orig_mb*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s")
    return out_path


def optimize_prune(model_path, threshold=1e-3, output_suffix="pruned"):
    """Apply magnitude-based weight pruning."""
    out_path = model_path.replace(".mlpackage", f"_{output_suffix}.mlpackage")
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return out_path

    print(f"  Loading {os.path.basename(model_path)}...")
    model = ct.models.MLModel(model_path)

    config = OptimizationConfig(
        global_config=OpMagnitudePrunerConfig(
            threshold=threshold,
        )
    )

    print(f"  Pruning (threshold={threshold})...")
    t0 = time.time()
    optimized = prune_weights(model, config)
    elapsed = time.time() - t0

    optimized.save(out_path)
    orig_mb = get_model_size_mb(model_path)
    opt_mb = get_model_size_mb(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Size: {orig_mb:.1f}MB → {opt_mb:.1f}MB ({opt_mb/orig_mb*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s")
    return out_path


def optimize_combined(model_path, n_bits=4, threshold=1e-3, output_suffix="pal4_pruned"):
    """Apply pruning + palettization together."""
    out_path = model_path.replace(".mlpackage", f"_{output_suffix}.mlpackage")
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return out_path

    print(f"  Loading {os.path.basename(model_path)}...")
    model = ct.models.MLModel(model_path)

    # First prune
    prune_config = OptimizationConfig(
        global_config=OpMagnitudePrunerConfig(threshold=threshold)
    )
    print(f"  Step 1: Pruning (threshold={threshold})...")
    model = prune_weights(model, prune_config)

    # Then palettize
    pal_config = OptimizationConfig(
        global_config=OpPalettizerConfig(mode="kmeans", nbits=n_bits)
    )
    print(f"  Step 2: Palettizing to {n_bits}-bit...")
    t0 = time.time()
    optimized = palettize_weights(model, pal_config)
    elapsed = time.time() - t0

    optimized.save(out_path)
    orig_mb = get_model_size_mb(model_path)
    opt_mb = get_model_size_mb(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Size: {orig_mb:.1f}MB → {opt_mb:.1f}MB ({opt_mb/orig_mb*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s")
    return out_path


def main():
    print("=" * 60)
    print("CoreML Model Optimization for autostream-mac")
    print("=" * 60)

    unet_path = MODELS["unet"]
    enc_path = MODELS["vae_enc"]
    dec_path = MODELS["vae_dec"]

    results = []

    # ── UNet optimizations (the bottleneck) ──────────────────
    print("\n" + "─" * 60)
    print("UNet Optimizations")
    print("─" * 60)

    # Palettization variants
    for nbits in [6, 4, 2]:
        print(f"\n[UNet] {nbits}-bit palettization:")
        try:
            p = optimize_palettize(unet_path, nbits, f"pal{nbits}")
            results.append(("unet", f"pal{nbits}", p))
        except Exception as e:
            print(f"  ERROR: {e}")

    # INT8 quantization
    print(f"\n[UNet] INT8 quantization:")
    try:
        p = optimize_quantize(unet_path)
        results.append(("unet", "int8", p))
    except Exception as e:
        print(f"  ERROR: {e}")

    # Magnitude pruning
    print(f"\n[UNet] Magnitude pruning:")
    try:
        p = optimize_prune(unet_path)
        results.append(("unet", "pruned", p))
    except Exception as e:
        print(f"  ERROR: {e}")

    # Combined: prune + 4-bit palettize
    print(f"\n[UNet] Combined prune + 4-bit palettize:")
    try:
        p = optimize_combined(unet_path, n_bits=4)
        results.append(("unet", "pal4_pruned", p))
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── VAE optimizations (smaller gains but worth trying) ───
    print("\n" + "─" * 60)
    print("VAE Encoder Optimizations")
    print("─" * 60)

    for nbits in [4]:
        print(f"\n[VAE Enc] {nbits}-bit palettization:")
        try:
            p = optimize_palettize(enc_path, nbits, f"pal{nbits}")
            results.append(("vae_enc", f"pal{nbits}", p))
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "─" * 60)
    print("VAE Decoder Optimizations")
    print("─" * 60)

    for nbits in [4]:
        print(f"\n[VAE Dec] {nbits}-bit palettization:")
        try:
            p = optimize_palettize(dec_path, nbits, f"pal{nbits}")
            results.append(("vae_dec", f"pal{nbits}", p))
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)
    for model_name, opt_type, path in results:
        mb = get_model_size_mb(path)
        print(f"  {model_name:10s} {opt_type:15s} → {mb:7.1f} MB  {os.path.basename(path)}")

    print(f"\nOriginal sizes:")
    for name, path in MODELS.items():
        print(f"  {name:10s} {get_model_size_mb(path):7.1f} MB")

    print("\nDone! Now benchmark each variant with pipeline.py")


if __name__ == "__main__":
    main()
