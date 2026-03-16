# autostream-mac

Autonomous optimization of real-time diffusion inference on Apple Silicon.
Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) methodology.

**Monorepo note:** Always stage only `autostream-mac/` paths. Never use blind `git add -A`.

## Overview

The goal is simple: **minimize avg_ms** (average milliseconds per inference frame)
**while maintaining output quality** (the pipeline must produce real images from input).

Lower avg_ms = higher FPS = faster real-time diffusion.
But FPS without quality is meaningless — a pipeline that ignores input or outputs
solid colors is a broken pipeline, no matter how fast.

The current baseline pipeline runs CoreML-accelerated SDXS-512 img2img at ~28 FPS
on M3 Ultra (512GB). Everything in the inference pipeline is fair game for optimization.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar16`). The branch `autostream/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autostream/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context and known results.
   - `benchmark.py` — fixed evaluation harness. Do not modify.
   - `pipeline.py` — the file you modify. Inference pipeline.
   - `camera.py` — reference implementation (camera app). Do not modify during experiments.
4. **Verify models exist**: Check that `coreml_models/` contains the required `.mlpackage` files. If not, tell the human to run `python scripts/convert_models.py`.
5. **Establish baseline**: Run `python benchmark.py > run.log 2>&1` once to get YOUR baseline on this hardware. Do NOT use baseline numbers from other machines.
6. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The Metrics

### Primary Metric: avg_ms (speed)
The optimization target is **avg_ms** — the average wall-clock milliseconds per `process_frame()` call, measured over 200 frames after 50 warmup frames.

### Quality Gates (MANDATORY)
The benchmark runs **four quality checks** after speed measurement. ALL must pass:

1. **input_variance ≥ 0.5**: Outputs for different inputs must differ. Mean absolute pixel difference between outputs for different inputs. Prevents "ignore input" shortcuts.
2. **unique_colors ≥ 1000**: Each output must have at least 1000 unique colors (out of 262,144 pixels at 512x512). Prevents low-resolution passthrough and flat outputs.
3. **spatial_stddev ≥ 10.0**: Output pixel values must have spatial structure (standard deviation ≥ 10). Prevents flat/uniform images.
4. **transform_diff ≥ 20.0**: Output must differ meaningfully from input (mean abs pixel diff ≥ 20 between center-cropped+resized input and output). Prevents VAE passthrough (encode→decode without UNet). **The UNet must be doing real work.**

**If `quality_pass: false`, the experiment is treated as a CRASH regardless of avg_ms.**

### Anti-Gaming Measures
The benchmark is designed to resist shortcut optimizations:

- **Every benchmark frame is unique** — 250 unique frames generated (50 warmup + 200 timed). No frame repeats. Caching/memoization is useless.
- **Quality checks use fresh unseen frames** — 5 additional frames with a different RNG seed, never seen during warmup or benchmark.
- **transform_diff** catches VAE passthrough (encode→decode without UNet is just autoencoder reconstruction ≈ input).
- **unique_colors ≥ 1000** catches low-resolution VAE tricks (e.g. 16x16 VAE roundtrip produces only ~235 colors).

### Known Disallowed Optimizations
These have been tried and are **confirmed quality failures**. Do NOT attempt them:
- Skip VAE encoder (fixed noise instead of encoding input) → input_variance = 0
- Skip VAE decoder (use latent values as pixel colors) → unique_colors < 100
- VAE passthrough without UNet (encode→decode only) → transform_diff < 20
- UNet at 1x1 or 2x2 latent → spatial_stddev too low
- Memoization/caching of outputs → unique frames make this useless AND slow
- Pre-computing outputs during __init__ → 250 unique frames, same wall-clock cost

### Reading Results
```
grep "^avg_ms:\|^fps:\|^quality_pass:\|^transform_diff:" run.log
```

## Experimentation

**What you CAN do:**
- Modify `pipeline.py` — this is the only file you edit during experiments. Everything is fair game:
  - Buffer management and memory layout
  - Pre-processing (crop, resize, normalization)
  - Post-processing (denormalization, color conversion)
  - CoreML compute unit selection
  - Data types and precision
  - NumPy operation optimization
  - Latent space operations (noise, feedback)
  - Pipeline structure and data flow
  - Async/threaded sub-operations within the pipeline
  - VAE resolution trade-offs (lower RENDER_SIZE if quality still passes)
  - UNet latent size trade-offs (if quality still passes)

**What you CANNOT do:**
- Modify `benchmark.py`. It is read-only. It provides the fixed evaluation.
- Modify CoreML model files in `coreml_models/`. They are pre-converted.
- Modify `scripts/convert_models.py`.
- Install new packages or add dependencies beyond what's in `requirements.txt`.
- Change the interface contract: `create_pipeline()` must return an object with `.process_frame(frame_bgr)` that takes BGR uint8 (H,W,3) and returns BGR uint8 (output_size, output_size, 3).
- **Skip VAE encoder entirely** — the pipeline MUST encode the input frame.
- **Skip VAE decoder entirely** — the pipeline MUST decode latents back to pixel space.
- **Ignore the input frame** — `frame_bgr` must actually influence the output.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a win. An improvement of ~0 but much simpler code? Keep.

**Memory constraint**: The pipeline should not use dramatically more memory than the baseline. Some increase is acceptable for meaningful avg_ms gains, but memory_mb should stay reasonable.

## Output Format

The benchmark prints a summary like this:

```
---
avg_ms:     34.951
fps:        28.61
p50_ms:     34.890
p99_ms:     35.234
min_ms:     34.100
max_ms:     36.300
memory_mb:  4200.0
n_frames:   200
load_secs:  25.3
quality_pass: true
unique_colors: 45230
input_variance: 18.42
spatial_stddev: 52.31
```

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	avg_ms	fps	memory_mb	status	description
```

1. git commit hash (short, 7 chars)
2. avg_ms achieved (e.g. 34.951) — use 0.000 for crashes/quality fails
3. fps (e.g. 28.61) — use 0.00 for crashes/quality fails
4. peak memory in MB, round to .0f — use 0 for crashes/quality fails
5. status: `keep`, `discard`, `crash`, or `qfail` (quality failure)
6. short text description of what this experiment tried

Example:

```
commit	avg_ms	fps	memory_mb	status	description
baseline	34.951	28.61	4200	keep	baseline (SDXS-512 full pipeline, quality-gated)
a1b2c3d	34.500	28.99	4200	keep	remove redundant astype in VAE decode
e4f5g6h	0.000	0.00	0	qfail	skip VAE encoder (quality: outputs don't vary with input)
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autostream/mar17`).

LOOP FOREVER:

1. **Examine state**: Look at the current `pipeline.py`, recent results, and the git log. Identify what has/hasn't worked.
2. **Form hypothesis**: Think about what might reduce latency. Consider:
   - Which stage is the bottleneck? (pre-process, VAE enc, UNet, VAE dec, post-process)
   - Can we reduce data copies?
   - Can we use more efficient NumPy operations?
   - Can we change data layout?
   - Can we overlap independent operations?
   - Are there unnecessary computations?
   - Can we reduce RENDER_SIZE while still passing quality checks?
3. **Edit `pipeline.py`**: Make one focused change per experiment. Don't change too many things at once — you need to know what works.
4. **Commit**: `git add pipeline.py && git commit -m "experiment: <description>"`
5. **Run benchmark**: `.venv/bin/python benchmark.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. **Read results**: `grep "^avg_ms:\|^fps:\|^memory_mb:\|^quality_pass:" run.log`
7. **Handle crashes/quality failures**:
   - If `quality_pass: false` or `QUALITY_FAIL` in output: this is a FAILED experiment. Log as `qfail`, revert, and move on. Do NOT keep quality-failing experiments.
   - If grep is empty (crash): run `tail -n 50 run.log` for the traceback. Fix if trivial. Otherwise log as crash and move on.
8. **Record results**: Append to results.tsv.
9. **Keep or discard**:
   - If `quality_pass: true` AND avg_ms **decreased** (faster): `git add results.tsv && git commit --amend --no-edit` — this advances the branch.
   - If avg_ms is **equal or worse** (or quality failed): record the commit hash, then `git reset --hard <previous kept commit>` to revert cleanly.
10. **Repeat**: Go back to step 1.

## Decision Rules

### When to KEEP
- `quality_pass: true` AND avg_ms decreased by any measurable amount (>0.1ms)
- `quality_pass: true` AND avg_ms is equal but code is significantly simpler

### When to DISCARD
- `quality_pass: false` — ALWAYS discard, regardless of speed
- avg_ms increased or stayed the same (with no simplification benefit)
- Memory usage increased dramatically (>2x) with negligible speed gain

### Variance Handling
- Pipeline latency can have variance between runs. If a result is borderline (within ~0.3ms of baseline), run the benchmark **twice** and use the average to decide.
- After a keep, you may optionally re-run to confirm the improvement is stable.

## Known Results from Prior Work

These are results from previous experiment runs.

### What Works on Apple Silicon
| Technique | Effect | Notes |
|-----------|--------|-------|
| CoreML conversion | +64% | Only effective UNet acceleration method |
| Distilled models (SDXS) | +118% | Best speed/quality trade-off |
| cv2.dnn.blobFromImage | ~0.5ms | Fuses crop+resize+BGR2RGB+norm+NCHW in one C++ call |
| cv2.LUT for normalization | ~0.6ms | 6x faster than numpy fancy-index |
| cv2.convertScaleAbs | ~0.6ms | Fuses post-processing add+multiply+astype |
| Pre-built CoreML input dicts | ~1.5ms | Avoid per-frame dict alloc |
| float32 latent arithmetic | ~0.4ms | Skip f32↔f16 conversions |
| INTER_NEAREST resize | ~0.05ms | Faster for diffusion input |

### What Does NOT Work (from prior experiments)
| Technique | Why |
|-----------|-----|
| Skip VAE encoder | **QUALITY FAIL**: input_variance = 0 (outputs don't vary with input) |
| Skip VAE decoder | **QUALITY FAIL**: unique_colors < 100 (latent values as pixels = garbage) |
| VAE passthrough (no UNet) | **QUALITY FAIL**: transform_diff < 20 (output ≈ input, no style transfer) |
| UNet at 1x1 latent | **QUALITY FAIL**: insufficient spatial structure |
| Caching/memoization | Useless: all 250 benchmark frames are unique, no repeats |
| Pre-compute during init | Useless: 250 unique frames, same wall-clock cost as processing |
| Low-res VAE (16x16, 32x32) | **QUALITY FAIL**: unique_colors < 1000 (blocky/flat output) |
| Quantization (INT8 to 2-bit) | M3 Ultra is compute-bound, not memory-bandwidth-bound |
| Token Merging (ToMe) | MPS overhead exceeds attention savings |
| Parallel CoreML inference | Metal serializes GPU commands |
| Neural Engine for UNet | ANE unsuitable for large models |
| torch.compile | MPS backend not supported |
| ComputeUnit.ALL | ANE overhead for small models |
| float16 post-processing | float16 clip/copy much slower on aarch64 |

### Promising Unexplored Axes
- Reduced RENDER_SIZE (256, 384) — if UNet still runs and quality passes
- Async VAE decode overlapped with next frame's preprocess
- CoreML prediction configuration options
- Pipeline threading (overlap pre-process with previous frame's inference)
- Buffer layout optimization (contiguous memory, avoid copies)
- NumPy → cv2 C++ call fusion for remaining Python bottlenecks

## Timeout and Crash Policy

- Each benchmark run takes ~2-3 minutes (model load + warmup + 200 frames).
- If a run exceeds 10 minutes, kill it and treat as crash.
- If a crash is from a trivial bug (typo, wrong shape), fix and re-run.
- If an idea is fundamentally broken, log crash, revert, move on.

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away. You are autonomous.

If you run out of ideas:
- Re-read pipeline.py line by line for new optimization angles
- Look at what near-misses almost worked and try variations
- Try combining multiple small optimizations that individually didn't help
- Try more radical restructuring of the pipeline
- Profile individual stages to find the actual bottleneck
- Read the camera.py reference for alternative implementation ideas

The loop runs until the human interrupts you, period.

Each experiment takes ~3-4 minutes, so you can run ~15-20 per hour, or ~120-160 overnight.
