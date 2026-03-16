#!/usr/bin/env python3
"""
Fixed benchmark harness for StreamDiffusion pipeline optimization.
DO NOT MODIFY THIS FILE. Modify pipeline.py instead.

Generates synthetic frames, runs the pipeline, measures latency.
The single metric is avg_ms (lower is better) = faster inference.

Quality gates prevent degenerate optimizations (e.g. ignoring input,
outputting solid colors). A pipeline must pass ALL quality checks.

Usage: python benchmark.py
"""
import os
import sys
import time
import subprocess
import numpy as np

# ── Fixed constants ──────────────────────────────────────────
N_WARMUP = 50       # warmup frames (not timed)
N_BENCHMARK = 200   # timed frames
N_SYNTHETIC = 10    # number of unique synthetic frames to cycle through
FRAME_H = 480
FRAME_W = 640
SEED = 42

# ── Quality thresholds ───────────────────────────────────────
MIN_INPUT_VARIANCE = 2.0    # mean absolute pixel diff between outputs for different inputs
MIN_UNIQUE_COLORS = 100     # minimum unique colors in each output frame
MIN_SPATIAL_STDDEV = 10.0   # minimum std dev of pixel values (spatial structure)


def generate_synthetic_frames():
    """Generate deterministic synthetic input frames."""
    rng = np.random.RandomState(SEED)
    return [rng.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
            for _ in range(N_SYNTHETIC)]


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
    """Run the benchmark. Returns metrics dict."""
    frames = generate_synthetic_frames()

    # Warmup
    print(f"  Warming up ({N_WARMUP} frames)...")
    for i in range(N_WARMUP):
        pipeline.process_frame(frames[i % N_SYNTHETIC])

    # Benchmark
    print(f"  Benchmarking ({N_BENCHMARK} frames)...")
    times = []
    for i in range(N_BENCHMARK):
        t0 = time.perf_counter()
        pipeline.process_frame(frames[i % N_SYNTHETIC])
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]
    avg_ms = sum(times_ms) / len(times_ms)
    fps = 1000.0 / avg_ms
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
    Run quality checks AFTER the speed benchmark.
    This ensures the pipeline is not a degenerate shortcut
    (e.g. solid color output, ignoring input).

    Returns a dict with quality metrics and pass/fail status.
    """
    rng = np.random.RandomState(SEED + 1000)  # different seed from benchmark frames

    # Generate 3 visually distinct synthetic frames
    test_frames = []
    for i in range(3):
        frame = rng.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        test_frames.append(frame)

    # Process each frame and collect outputs
    outputs = []
    for frame in test_frames:
        out = pipeline.process_frame(frame)
        # Ensure output is a numpy array
        out = np.asarray(out, dtype=np.uint8)
        outputs.append(out)

    # ── Check 1: Input sensitivity ───────────────────────────
    # Outputs for different inputs must differ from each other.
    # Compute mean absolute pixel difference between all pairs.
    pair_diffs = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            diff = np.mean(np.abs(outputs[i].astype(np.float32) - outputs[j].astype(np.float32)))
            pair_diffs.append(diff)
    input_variance = np.mean(pair_diffs) if pair_diffs else 0.0
    input_sensitivity_pass = input_variance >= MIN_INPUT_VARIANCE

    # ── Check 2: Color diversity ─────────────────────────────
    # Each output must have at least MIN_UNIQUE_COLORS unique colors.
    min_unique = float("inf")
    for out in outputs:
        # Reshape to (N, 3) and find unique rows
        pixels = out.reshape(-1, 3)
        # Pack RGB into single uint32 for fast unique counting
        packed = (pixels[:, 0].astype(np.uint32) << 16 |
                  pixels[:, 1].astype(np.uint32) << 8 |
                  pixels[:, 2].astype(np.uint32))
        n_unique = len(np.unique(packed))
        min_unique = min(min_unique, n_unique)
    unique_colors = int(min_unique)
    color_diversity_pass = unique_colors >= MIN_UNIQUE_COLORS

    # ── Check 3: Structural content ──────────────────────────
    # Output must have spatial structure (not flat/near-flat).
    min_stddev = float("inf")
    for out in outputs:
        stddev = np.std(out.astype(np.float32))
        min_stddev = min(min_stddev, stddev)
    spatial_stddev = float(min_stddev)
    structural_pass = spatial_stddev >= MIN_SPATIAL_STDDEV

    # ── Overall ──────────────────────────────────────────────
    quality_pass = input_sensitivity_pass and color_diversity_pass and structural_pass

    return {
        "quality_pass": quality_pass,
        "unique_colors": unique_colors,
        "input_variance": input_variance,
        "spatial_stddev": spatial_stddev,
        "input_sensitivity_pass": input_sensitivity_pass,
        "color_diversity_pass": color_diversity_pass,
        "structural_pass": structural_pass,
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

    # Quality checks (after speed benchmark, does not affect timing)
    print(f"\n  Running quality checks...")
    quality = check_quality(pipeline)

    # Output in autoresearch-compatible format
    print("\n---")
    print(f"avg_ms:     {metrics['avg_ms']:.3f}")
    print(f"fps:        {metrics['fps']:.2f}")
    print(f"p50_ms:     {metrics['p50_ms']:.3f}")
    print(f"p99_ms:     {metrics['p99_ms']:.3f}")
    print(f"min_ms:     {metrics['min_ms']:.3f}")
    print(f"max_ms:     {metrics['max_ms']:.3f}")
    print(f"memory_mb:  {metrics['memory_mb']:.1f}")
    print(f"n_frames:   {metrics['n_frames']}")
    print(f"load_secs:  {load_time:.1f}")
    print(f"quality_pass: {str(quality['quality_pass']).lower()}")
    print(f"unique_colors: {quality['unique_colors']}")
    print(f"input_variance: {quality['input_variance']:.2f}")
    print(f"spatial_stddev: {quality['spatial_stddev']:.2f}")

    if not quality["quality_pass"]:
        print("QUALITY_FAIL")
        # Print which checks failed for debugging
        if not quality["input_sensitivity_pass"]:
            print(f"  FAIL: input_variance {quality['input_variance']:.2f} < {MIN_INPUT_VARIANCE} (outputs do not vary with input)")
        if not quality["color_diversity_pass"]:
            print(f"  FAIL: unique_colors {quality['unique_colors']} < {MIN_UNIQUE_COLORS} (output lacks color diversity)")
        if not quality["structural_pass"]:
            print(f"  FAIL: spatial_stddev {quality['spatial_stddev']:.2f} < {MIN_SPATIAL_STDDEV} (output is flat/near-flat)")


if __name__ == "__main__":
    main()
