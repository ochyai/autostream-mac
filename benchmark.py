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
  - Sample frames are saved to output_samples/ for visual inspection
  - RENDER_SIZE must be >= 512 (no resolution downgrade tricks)

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
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_samples")

# ── Quality thresholds ───────────────────────────────────────
MIN_INPUT_VARIANCE = 0.5     # mean abs pixel diff between outputs for different inputs
MIN_UNIQUE_COLORS = 10000    # minimum unique colors per 512x512 output (real diffusion: 50k+)
MIN_SPATIAL_STDDEV = 10.0    # minimum std dev of pixel values (spatial structure)
MIN_TRANSFORM_DIFF = 20.0    # minimum diff between input and output (prevents VAE passthrough)
MIN_RENDER_SIZE = 512        # pipeline must process at full resolution (no downscale tricks)
MIN_EDGE_STRENGTH = 5.0      # minimum edge content (Sobel magnitude, prevents blurry mush)


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


def save_sample_grid(test_frames, outputs):
    """Save input/output comparison grid as PNG for visual inspection."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save individual pairs and a comparison grid
    n = len(test_frames)
    out_h, out_w = outputs[0].shape[:2]

    for i, (frame, out) in enumerate(zip(test_frames, outputs)):
        # Center-crop and resize input to match output
        h, w = frame.shape[:2]
        if w > h:
            off = (w - h) // 2
            sq = frame[:, off:off + h]
        elif h > w:
            off = (h - w) // 2
            sq = frame[off:off + w, :]
        else:
            sq = frame
        inp_resized = cv2.resize(sq, (out_w, out_h))

        # Save individual output
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"output_{i}.png"), out)

        # Save side-by-side comparison
        pair = np.hstack([inp_resized, out])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"compare_{i}.png"), pair)

    # Save full grid: all inputs on top, all outputs on bottom
    row_in = np.hstack([cv2.resize(
        f[:, (f.shape[1]-f.shape[0])//2:(f.shape[1]-f.shape[0])//2+f.shape[0]]
        if f.shape[1] > f.shape[0] else f, (out_w, out_h))
        for f in test_frames])
    row_out = np.hstack(outputs)
    grid = np.vstack([row_in, row_out])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grid.png"), grid)
    print(f"  Saved {n} sample pairs to {OUTPUT_DIR}/")


def check_quality(pipeline):
    """
    Quality checks using completely fresh frames (never seen during benchmark).
    Saves visual samples to output_samples/ for human review.
    """
    # Different seed — these frames are completely new to the pipeline
    test_frames = generate_unique_frames(5, SEED + 9999)

    outputs = []
    for frame in test_frames:
        out = pipeline.process_frame(frame)
        # copy=True: pipeline may return a view into a reused buffer
        outputs.append(np.array(out, dtype=np.uint8, copy=True))

    # Save visual samples for human inspection
    save_sample_grid(test_frames, outputs)

    # ── Check 0: Resolution gate ─────────────────────────────
    # Pipeline must not use RENDER_SIZE < 512. Check via pipeline config.
    render_size = getattr(pipeline, 'render_size', None)
    if render_size is None:
        # Try to read from module
        try:
            from pipeline import RENDER_SIZE
            render_size = RENDER_SIZE
        except (ImportError, AttributeError):
            render_size = 512  # assume compliant if we can't detect
    render_pass = render_size >= MIN_RENDER_SIZE

    # ── Check 1: Input sensitivity ───────────────────────────
    pair_diffs = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            diff = np.mean(np.abs(
                outputs[i].astype(np.float32) - outputs[j].astype(np.float32)))
            pair_diffs.append(diff)
    input_variance = float(np.mean(pair_diffs)) if pair_diffs else 0.0
    input_pass = input_variance >= MIN_INPUT_VARIANCE

    # ── Check 2: Color diversity ─────────────────────────────
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
    min_stddev = float("inf")
    for out in outputs:
        min_stddev = min(min_stddev, float(np.std(out.astype(np.float32))))
    spatial_stddev = min_stddev
    struct_pass = spatial_stddev >= MIN_SPATIAL_STDDEV

    # ── Check 4: Transformation magnitude ────────────────────
    transform_diffs = []
    for frame, out in zip(test_frames, outputs):
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

    # ── Check 5: Edge content (anti-blur) ────────────────────
    # Output must have sharp edges, not blurry mush from extreme downscale+upscale
    min_edge = float("inf")
    for out in outputs:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        min_edge = min(min_edge, float(np.mean(edge_mag)))
    edge_strength = min_edge
    edge_pass = edge_strength >= MIN_EDGE_STRENGTH

    # ── Overall ──────────────────────────────────────────────
    quality_pass = all([render_pass, input_pass, color_pass,
                        struct_pass, transform_pass, edge_pass])

    return {
        "quality_pass": quality_pass,
        "render_size": render_size,
        "unique_colors": unique_colors,
        "input_variance": input_variance,
        "spatial_stddev": spatial_stddev,
        "transform_diff": transform_diff,
        "edge_strength": edge_strength,
        "render_pass": render_pass,
        "input_pass": input_pass,
        "color_pass": color_pass,
        "struct_pass": struct_pass,
        "transform_pass": transform_pass,
        "edge_pass": edge_pass,
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
    print(f"render_size:    {quality['render_size']}")
    print(f"unique_colors:  {quality['unique_colors']}")
    print(f"input_variance: {quality['input_variance']:.2f}")
    print(f"spatial_stddev: {quality['spatial_stddev']:.2f}")
    print(f"transform_diff: {quality['transform_diff']:.2f}")
    print(f"edge_strength:  {quality['edge_strength']:.2f}")

    if not quality["quality_pass"]:
        print("QUALITY_FAIL")
        if not quality["render_pass"]:
            print(f"  FAIL: render_size {quality['render_size']} < {MIN_RENDER_SIZE} (no resolution downgrade — optimize the model, not the resolution)")
        if not quality["input_pass"]:
            print(f"  FAIL: input_variance {quality['input_variance']:.2f} < {MIN_INPUT_VARIANCE} (outputs identical for different inputs)")
        if not quality["color_pass"]:
            print(f"  FAIL: unique_colors {quality['unique_colors']} < {MIN_UNIQUE_COLORS} (output lacks color richness)")
        if not quality["struct_pass"]:
            print(f"  FAIL: spatial_stddev {quality['spatial_stddev']:.2f} < {MIN_SPATIAL_STDDEV} (output is flat)")
        if not quality["transform_pass"]:
            print(f"  FAIL: transform_diff {quality['transform_diff']:.2f} < {MIN_TRANSFORM_DIFF} (output too similar to input — is UNet being used?)")
        if not quality["edge_pass"]:
            print(f"  FAIL: edge_strength {quality['edge_strength']:.2f} < {MIN_EDGE_STRENGTH} (output is blurry — no sharp detail)")


if __name__ == "__main__":
    main()
