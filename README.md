# autostream-mac

Autonomous optimization of real-time diffusion model inference on Apple Silicon.

Applies the [autoresearch](https://github.com/karpathy/autoresearch) methodology — autonomous AI-driven experiment loops — to optimizing [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) inference speed on Mac.

## What this does

An AI coding agent (Claude Code, etc.) runs an infinite loop:

1. Edit `pipeline.py` with an optimization idea
2. Run `benchmark.py` to measure avg_ms per frame
3. If faster → **keep** (commit). If slower → **discard** (revert).
4. Repeat forever.

You go to sleep. You wake up to a log of 100+ experiments and a faster pipeline.

## The metric

**avg_ms** — average milliseconds per inference frame. Lower is better.

The benchmark runs 200 frames through the full CoreML pipeline (preprocess → VAE encode → UNet → VAE decode → postprocess) and reports the average.

```
---
avg_ms:     44.123
fps:        22.66
p50_ms:     43.890
p99_ms:     48.234
memory_mb:  4200.0
```

## What matters

| File | Role | Editable? |
|------|------|-----------|
| `pipeline.py` | Inference pipeline. **The file the agent edits.** | Yes |
| `benchmark.py` | Fixed evaluation harness. Measures avg_ms. | No |
| `program.md` | Autonomous experiment protocol. | No |
| `camera.py` | Reference camera app (for manual testing). | No |
| `results.tsv` | Experiment history log. | Append only |
| `scripts/convert_models.py` | One-time CoreML model conversion. | No |

## Quick start

Requirements: Apple Silicon Mac (M1+), Python 3.9-3.12, [uv](https://docs.astral.sh/uv/) or pip.

```bash
# Clone
git clone https://github.com/ochyai/autostream-mac.git
cd autostream-mac

# Setup
chmod +x setup.sh && ./setup.sh

# Activate
source .venv/bin/activate

# Convert models to CoreML (one-time, ~5 min)
python scripts/convert_models.py

# Run headless benchmark
python benchmark.py

# Or run interactive camera
python camera.py --prompt "oil painting style, masterpiece"
```

## Running the autonomous loop

Point your coding agent at `program.md` and let it run:

```bash
# With Claude Code
claude -p "Read program.md in /path/to/autostream-mac and follow the protocol. Run tag: mar16"
```

The agent will:
1. Create branch `autostream/mar16`
2. Establish baseline on your hardware
3. Start the infinite experiment loop
4. Log all results to `results.tsv`

Each experiment takes ~3-4 minutes. Expect ~15-20 per hour, ~120-160 overnight.

## Results format

`results.tsv` tracks every experiment:

```
commit	avg_ms	fps	memory_mb	status	description
baseline	44.123	22.66	4200	keep	baseline (SDXS-512 CoreML)
a1b2c3d	43.500	22.99	4200	keep	remove redundant astype in VAE decode
e4f5g6h	45.200	22.12	4200	discard	COMPUTE_UNIT.ALL (ANE slower)
```

## Why Apple Silicon optimization is interesting

CUDA optimization wisdom does not transfer to Apple Silicon's unified memory architecture:

- **Quantization is ineffective** — compute-bound, not memory-bandwidth-bound
- **Parallel inference is impossible** — CoreML serializes Metal GPU commands
- **The software ecosystem is immature** — no torch.compile, no Flash Attention on MPS

This means the optimization landscape is largely unexplored. Unlike NVIDIA GPUs where decades of tooling (cuDNN, TensorRT, xformers) have squeezed out most gains, Apple Silicon still has low-hanging fruit waiting to be found through systematic search.

## Hardware context

Results vary by hardware. The baseline was established on:

| Machine | Chip | Memory | Baseline FPS |
|---------|------|--------|-------------|
| Mac Studio | M3 Ultra | 512 GB | ~22.7 |
| MacBook Pro | M4 Max | 128 GB | ~15.4 |

The autonomous loop finds hardware-specific optima — what works on M3 Ultra may not help on M4 Max, and vice versa.

## Architecture

The inference pipeline (single frame):

```
Input BGR frame (480x640)
  ↓ center crop + resize (512x512)
  ↓ normalize via LUT [-1, 1]
  ↓ VAE Encode (CoreML TAESD, ~5ms)
  ↓ noise addition (fixed seed)
  ↓ UNet inference (CoreML SDXS, ~24ms)
  ↓ denoise
  ↓ VAE Decode (CoreML TAESD, ~5ms)
  ↓ denormalize + color convert
Output BGR frame (512x512)
```

The UNet dominates at ~24ms. But pre/post processing, buffer management, and data copies add up. The experiment loop systematically attacks every stage.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch methodology
- [trevin-creator](https://github.com/trevin-creator) — autoresearch-mlx Apple Silicon port
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) — original pipeline
- [SDXS](https://github.com/IDKiro/sdxs) — distilled diffusion model
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT
