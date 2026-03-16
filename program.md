# autostream-mac

Autonomous optimization of real-time diffusion inference on Apple Silicon.
Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) methodology.

**Monorepo note:** Always stage only `autostream-mac/` paths. Never use blind `git add -A`.

## Overview

The goal is simple: **minimize avg_ms** (average milliseconds per inference frame).
Lower avg_ms = higher FPS = faster real-time diffusion.

The current baseline pipeline runs CoreML-accelerated SDXS-512 img2img at ~22 FPS
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

## The Metric

The single optimization target is **avg_ms** — the average wall-clock milliseconds per `process_frame()` call, measured over 200 frames after 50 warmup frames.

The benchmark also reports supporting metrics (p50, p99, memory) but avg_ms is what determines keep/discard.

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

**What you CANNOT do:**
- Modify `benchmark.py`. It is read-only. It provides the fixed evaluation.
- Modify CoreML model files in `coreml_models/`. They are pre-converted.
- Modify `scripts/convert_models.py`.
- Install new packages or add dependencies beyond what's in `requirements.txt`.
- Change the interface contract: `create_pipeline()` must return an object with `.process_frame(frame_bgr)` that takes BGR uint8 (H,W,3) and returns BGR uint8 (output_size, output_size, 3).

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a win. An improvement of ~0 but much simpler code? Keep.

**Memory constraint**: The pipeline should not use dramatically more memory than the baseline. Some increase is acceptable for meaningful avg_ms gains, but memory_mb should stay reasonable.

## Output Format

The benchmark prints a summary like this:

```
---
avg_ms:     44.123
fps:        22.66
p50_ms:     43.890
p99_ms:     48.234
min_ms:     42.100
max_ms:     51.300
memory_mb:  4200.0
n_frames:   200
load_secs:  25.3
```

Read the key metric with:
```
grep "^avg_ms:" run.log
```

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	avg_ms	fps	memory_mb	status	description
```

1. git commit hash (short, 7 chars)
2. avg_ms achieved (e.g. 44.123) — use 0.000 for crashes
3. fps (e.g. 22.66) — use 0.00 for crashes
4. peak memory in MB, round to .0f — use 0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	avg_ms	fps	memory_mb	status	description
baseline	44.123	22.66	4200	keep	baseline (SDXS-512 CoreML CPU_AND_GPU)
a1b2c3d	43.500	22.99	4200	keep	remove redundant astype in VAE decode
e4f5g6h	45.200	22.12	4200	discard	use COMPUTE_UNIT.ALL (ANE slower)
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autostream/mar16`).

LOOP FOREVER:

1. **Examine state**: Look at the current `pipeline.py`, recent results, and the git log. Identify what has/hasn't worked.
2. **Form hypothesis**: Think about what might reduce latency. Consider:
   - Which stage is the bottleneck? (pre-process, VAE enc, UNet, VAE dec, post-process)
   - Can we reduce data copies?
   - Can we use more efficient NumPy operations?
   - Can we change data layout?
   - Can we overlap independent operations?
   - Are there unnecessary computations?
3. **Edit `pipeline.py`**: Make one focused change per experiment. Don't change too many things at once — you need to know what works.
4. **Commit**: `git add pipeline.py && git commit -m "experiment: <description>"`
5. **Run benchmark**: `python benchmark.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. **Read results**: `grep "^avg_ms:\|^fps:\|^memory_mb:" run.log`
7. **Handle crashes**: If grep is empty, run `tail -n 50 run.log` for the traceback. Fix if it's trivial (typo, import). If the idea is broken, log as crash and move on.
8. **Record results**: Append to results.tsv.
9. **Keep or discard**:
   - If avg_ms **decreased** (faster): `git add results.tsv && git commit --amend --no-edit` — this advances the branch.
   - If avg_ms is **equal or worse**: record the discard commit hash, then `git reset --hard <previous kept commit>` to revert cleanly.
10. **Repeat**: Go back to step 1.

## Decision Rules

### When to KEEP
- avg_ms decreased by any measurable amount (>0.1ms)
- avg_ms is equal but code is significantly simpler (fewer lines, clearer logic)

### When to DISCARD
- avg_ms increased or stayed the same (with no simplification benefit)
- Memory usage increased dramatically (>2x) with negligible speed gain

### Variance Handling
- Pipeline latency can have variance between runs. If a result is borderline (within ~0.3ms of baseline), run the benchmark **twice** and use the average to decide.
- After a keep, you may optionally re-run to confirm the improvement is stable.

## Known Results from Prior Work

These are results from the original streamdiffusion-mac experiment report. They inform your search but **do not replace running your own experiments** — hardware and software versions differ.

### What Works on Apple Silicon
| Technique | Effect | Notes |
|-----------|--------|-------|
| CoreML conversion | +64% | Only effective UNet acceleration method |
| Distilled models (SDXS) | +118% | Best speed/quality trade-off |

### What Does NOT Work (from prior experiments)
| Technique | Why |
|-----------|-----|
| Quantization (INT8 to 2-bit) | M3 Ultra is compute-bound, not memory-bandwidth-bound |
| Token Merging (ToMe) | MPS overhead exceeds attention savings |
| Parallel CoreML inference | Metal serializes GPU commands |
| Neural Engine for UNet | ANE unsuitable for large models |
| torch.compile | MPS backend not supported |
| Attention Slicing | MPS memory management overhead |

### Unexplored Optimization Axes
These have NOT been systematically tested in the current pipeline code:
- Pre/post processing optimization (NumPy → more efficient alternatives)
- Buffer management (avoiding copies, pre-allocation patterns)
- CoreML prediction configuration options
- Data type minimization in non-CoreML stages
- Latent space operation simplification
- Resolution/quality trade-offs (render_size 384 vs 512)
- Async patterns within single-frame pipeline

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
