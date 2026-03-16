#!/usr/bin/env python3
"""
Fixed benchmark harness for StreamDiffusion pipeline optimization.
DO NOT MODIFY THIS FILE. Modify pipeline.py instead.

Generates synthetic frames, runs the pipeline, measures latency.
The single metric is avg_ms (lower is better) = faster inference.

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


if __name__ == "__main__":
    main()
