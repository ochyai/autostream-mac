#!/usr/bin/env python3
"""
Fixed benchmark harness for StreamDiffusion pipeline optimization.
DO NOT MODIFY THIS FILE. Modify pipeline.py instead.

Generates synthetic frames, runs the pipeline, measures latency.
Primary metric: avg_ms (lower is better) = faster inference.

Quality gates prevent degenerate optimizations:
  - Every benchmark frame is UNIQUE (no caching possible)
  - Output must vary with input (no fixed output)
  - Output must have color diversity and spatial structure
  - Output must differ meaningfully from input (no VAE passthrough)

Usage: python benchmark.py
"""
import os
import sys
import time
import subprocess
import numpy as np
import cv2

# ── Fixed constants ──────────────────────────────────────────
N_WARMUP = 50       # warmup frames (not timed)
N_BENCHMARK = 200   # timed frames
FRAME_H = 480
FRAME_W = 640
SEED = 42

# ── Quality thresholds ───────────────────────────────────────
MIN_INPUT_VARIANCE = 0.5     # mean abs pixel diff between outputs for different inputs
MIN_UNIQUE_COLORS = 1000     # minimum unique colors per output frame (512x512 has 262144 pixels)
MIN_SPATIAL_STDDEV = 10.0    # minimum std dev of pixel values (spatial structure)
MIN_TRANSFORM_DIFF = 20.0    # minimum diff between input and output (prevents VAE passthrough)


def generate_unique_frames(n, seed):
    """Generate n unique deterministic synthetic input frames."""
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
            for _ in range(n)]


def get_memory_mb():
    """Get current process RSS in MB."""
    try:
        pid = os.getpid()
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                           capture_output=True, text=True)
        return int(r.stdout.strip()) / 1024
    except Exception:
        return 0.0


def run_benchmark(pipeline):
    """Run speed benchmark. Every frame is unique — caching is impossible."""
    # Generate ALL unique frames upfront (no frame repeats)
    all_frames = generate_unique_frames(N_WARMUP + N_BENCHMARK, SEED)
    warmup_frames = all_frames[:N_WARMUP]
    bench_frames = all_frames[N_WARMUP:]

    # Warmup
    print(f"  Warming up ({N_WARMUP} unique frames)...")
    for frame in warmup_frames:
        pipeline.process_frame(frame)

    # Benchmark
    print(f"  Benchmarking ({N_BENCHMARK} unique frames)...")
    times = []
    for frame in bench_frames:
        t0 = time.perf_counter()
        pipeline.process_frame(frame)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]
    avg_ms = sum(times_ms) / len(times_ms)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    times_ms_sorted = sorted(times_ms)
    p50_ms = times_ms_sorted[len(times_ms_sorted) // 2]
    p99_ms = times_ms_sorted[int(len(times_ms_sorted) * 0.99)]
    min_ms = times_ms_sorted[0]
    max_ms = times_ms_sorted[-1]
    memory_mb = get_memory_mb()

    return {
        "avg_ms": avg_ms,
        "fps": fps,
        "p50_ms": p50_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "memory_mb": memory_mb,
        "n_frames": N_BENCHMARK,
    }


def check_quality(pipeline):
    """
    Quality checks using completely fresh frames (never seen during benchmark).
    Catches: caching, fixed output, VAE passthrough, solid colors.
    """
    # Different seed — these frames are completely new to the pipeline
    test_frames = generate_unique_frames(5, SEED + 9999)

    outputs = []
    for frame in test_frames:
        out = pipeline.process_frame(frame)
        # copy=True: pipeline may return a view into a reused buffer
        outputs.append(np.array(out, dtype=np.uint8, copy=True))

    # ── Check 1: Input sensitivity ───────────────────────────
    # Outputs for different inputs must differ from each other.
    pair_diffs = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            diff = np.mean(np.abs(
                outputs[i].astype(np.float32) - outputs[j].astype(np.float32)))
            pair_diffs.append(diff)
    input_variance = float(np.mean(pair_diffs)) if pair_diffs else 0.0
    input_pass = input_variance >= MIN_INPUT_VARIANCE

    # ── Check 2: Color diversity ─────────────────────────────
    # Each output must have rich color content (not blocky/flat).
    min_unique = float("inf")
    for out in outputs:
        pixels = out.reshape(-1, 3)
        packed = (pixels[:, 0].astype(np.uint32) << 16 |
                  pixels[:, 1].astype(np.uint32) << 8 |
                  pixels[:, 2].astype(np.uint32))
        min_unique = min(min_unique, len(np.unique(packed)))
    unique_colors = int(min_unique)
    color_pass = unique_colors >= MIN_UNIQUE_COLORS

    # ── Check 3: Spatial structure ───────────────────────────
    # Output must have texture, not flat/uniform regions.
    min_stddev = float("inf")
    for out in outputs:
        min_stddev = min(min_stddev, float(np.std(out.astype(np.float32))))
    spatial_stddev = min_stddev
    struct_pass = spatial_stddev >= MIN_SPATIAL_STDDEV

    # ── Check 4: Transformation magnitude ────────────────────
    # Output must differ meaningfully from input.
    # Catches VAE passthrough (encode→decode without UNet = reconstruction ≈ input).
    transform_diffs = []
    for frame, out in zip(test_frames, outputs):
        # Center-crop input to square, resize to output dimensions
        h, w = frame.shape[:2]
        if w > h:
            off = (w - h) // 2
            sq = frame[:, off:off + h]
        elif h > w:
            off = (h - w) // 2
            sq = frame[off:off + w, :]
        else:
            sq = frame
        input_resized = cv2.resize(sq, (out.shape[1], out.shape[0]))
        diff = np.mean(np.abs(
            out.astype(np.float32) - input_resized.astype(np.float32)))
        transform_diffs.append(diff)
    transform_diff = float(np.mean(transform_diffs))
    transform_pass = transform_diff >= MIN_TRANSFORM_DIFF

    # ── Overall ──────────────────────────────────────────────
    quality_pass = all([input_pass, color_pass, struct_pass, transform_pass])

    return {
        "quality_pass": quality_pass,
        "unique_colors": unique_colors,
        "input_variance": input_variance,
        "spatial_stddev": spatial_stddev,
        "transform_diff": transform_diff,
        "input_pass": input_pass,
        "color_pass": color_pass,
        "struct_pass": struct_pass,
        "transform_pass": transform_pass,
    }


def main():
    from pipeline import create_pipeline

    print("=" * 60)
    print("StreamDiffusion-Mac Benchmark (Headless)")
    print("=" * 60)

    print("\nLoading pipeline...")
    t0 = time.perf_counter()
    pipeline = create_pipeline()
    load_time = time.perf_counter() - t0
    print(f"  Pipeline loaded in {load_time:.1f}s")

    metrics = run_benchmark(pipeline)

    print(f"\n  Running quality checks...")
    quality = check_quality(pipeline)

    # Output in autoresearch-compatible format
    print("\n---")
    print(f"avg_ms:         {metrics['avg_ms']:.3f}")
    print(f"fps:            {metrics['fps']:.2f}")
    print(f"p50_ms:         {metrics['p50_ms']:.3f}")
    print(f"p99_ms:         {metrics['p99_ms']:.3f}")
    print(f"min_ms:         {metrics['min_ms']:.3f}")
    print(f"max_ms:         {metrics['max_ms']:.3f}")
    print(f"memory_mb:      {metrics['memory_mb']:.1f}")
    print(f"n_frames:       {metrics['n_frames']}")
    print(f"load_secs:      {load_time:.1f}")
    print(f"quality_pass:   {str(quality['quality_pass']).lower()}")
    print(f"unique_colors:  {quality['unique_colors']}")
    print(f"input_variance: {quality['input_variance']:.2f}")
    print(f"spatial_stddev: {quality['spatial_stddev']:.2f}")
    print(f"transform_diff: {quality['transform_diff']:.2f}")

    if not quality["quality_pass"]:
        print("QUALITY_FAIL")
        if not quality["input_pass"]:
            print(f"  FAIL: input_variance {quality['input_variance']:.2f} < {MIN_INPUT_VARIANCE} (outputs identical for different inputs)")
        if not quality["color_pass"]:
            print(f"  FAIL: unique_colors {quality['unique_colors']} < {MIN_UNIQUE_COLORS} (output lacks color richness)")
        if not quality["struct_pass"]:
            print(f"  FAIL: spatial_stddev {quality['spatial_stddev']:.2f} < {MIN_SPATIAL_STDDEV} (output is flat)")
        if not quality["transform_pass"]:
            print(f"  FAIL: transform_diff {quality['transform_diff']:.2f} < {MIN_TRANSFORM_DIFF} (output too similar to input — is UNet being used?)")


if __name__ == "__main__":
    main()
