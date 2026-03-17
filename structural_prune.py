#!/usr/bin/env python3
"""
Structural Pruning + Knowledge Distillation for SDXS-512 UNet.

Physically removes channels from the UNet to reduce FLOPs (not just zero weights).
Then fine-tunes via knowledge distillation to recover quality.
Finally converts to CoreML and benchmarks.

This is the correct path: M3 Ultra is compute-bound, not memory-bound.
Only reducing FLOPs (actual operations) will speed up inference.

Run: .venv/bin/python structural_prune.py
"""
import os
import sys
import json
import time
import gc
import copy
import shutil
import numpy as np

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch_pruning as tp

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
COREML_DIR = os.path.join(WORK_DIR, "coreml_models")
RESULTS_FILE = os.path.join(WORK_DIR, "pruning_results.json")

MODEL_ID = "IDKiro/sdxs-512-0.9"
RENDER_SIZE = 512
LATENT_SIZE = RENDER_SIZE // 8  # 64
HIDDEN_SIZE = 1024

# Pruning configurations to try
PRUNING_CONFIGS = [
    {"name": "prune_20pct", "ratio": 0.20, "finetune_steps": 500},
    {"name": "prune_30pct", "ratio": 0.30, "finetune_steps": 800},
    {"name": "prune_40pct", "ratio": 0.40, "finetune_steps": 1200},
    {"name": "prune_50pct", "ratio": 0.50, "finetune_steps": 2000},
]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_ID}")
print(f"Pruning configs: {len(PRUNING_CONFIGS)}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, sample_shape, timestep_val, hidden_states_shape):
    """Estimate FLOPs by running a forward pass with profiling."""
    model.eval()
    device = next(model.parameters()).device
    sample = torch.randn(*sample_shape, device=device)
    timestep = torch.tensor([timestep_val], device=device, dtype=torch.long)
    hidden_states = torch.randn(*hidden_states_shape, device=device)

    # Use torch_pruning's built-in FLOPs counter if available
    try:
        from torch_pruning.utils import count_ops
        # Simple MACs estimation based on parameter count and typical UNet compute patterns
        # Actual FLOPs ≈ 2 * MACs
        pass
    except:
        pass

    # Fallback: use parameter count as proxy (roughly proportional for conv-heavy models)
    return count_params(model)


def load_teacher():
    """Load the full SDXS-512 UNet as teacher model."""
    from diffusers import UNet2DConditionModel
    print("Loading teacher UNet...")
    teacher = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32
    ).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = count_params(teacher)
    print(f"  Teacher params: {teacher_params:,}")
    return teacher, teacher_params


def get_prompt_embeddings():
    """Get text encoder embeddings for the prompt."""
    from diffusers import StableDiffusionPipeline
    print("Loading text encoder for prompt embeddings...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(DEVICE)

    prompt = "oil painting style, masterpiece, highly detailed"
    with torch.no_grad():
        ti = pipe.tokenizer(
            prompt, padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        embeds = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]

    # Get timestep from scheduler
    from diffusers import EulerDiscreteScheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(1, device=DEVICE)
    timestep = pipe.scheduler.timesteps[0].item()

    del pipe
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    print(f"  Prompt embeddings: {embeds.shape}")
    print(f"  Timestep: {timestep}")
    return embeds, timestep


def structural_prune(teacher, ratio, verbose=True):
    """Apply structural pruning using DepGraph."""
    from diffusers import UNet2DConditionModel

    if verbose:
        print(f"\n  Pruning at {ratio*100:.0f}% channel reduction...")

    # Create a fresh copy to prune
    model = copy.deepcopy(teacher).to(DEVICE)
    model.train()

    original_params = count_params(model)

    # Create example inputs for dependency graph analysis
    example_inputs = {
        "sample": torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE),
        "timestep": torch.tensor([800], device=DEVICE, dtype=torch.long),
        "encoder_hidden_states": torch.randn(1, 77, HIDDEN_SIZE, device=DEVICE),
    }

    # Identify layers to prune (conv and linear layers, skip normalization and embeddings)
    ignored_layers = []
    # Don't prune the input/output convolutions (they have fixed channel counts)
    ignored_layers.append(model.conv_in)
    ignored_layers.append(model.conv_out)
    # Don't prune time embedding layers (small, critical)
    for m in model.time_embedding.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            ignored_layers.append(m)

    # Build importance scorer (L1 norm of channels)
    importance = tp.importance.MagnitudeImportance(p=1)

    # Build pruner with DepGraph
    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
        isomorphic=False,  # Allow different pruning rates per layer
    )

    # Execute pruning
    for g in pruner.step(interactive=True):
        g.prune()

    pruned_params = count_params(model)
    reduction = 1.0 - pruned_params / original_params

    if verbose:
        print(f"  Original: {original_params:,} params")
        print(f"  Pruned:   {pruned_params:,} params ({reduction*100:.1f}% reduction)")

    model.eval()

    # Verify forward pass works
    with torch.no_grad():
        try:
            out = model(**example_inputs).sample
            if verbose:
                print(f"  Forward pass OK: output shape {out.shape}")
        except Exception as e:
            print(f"  ERROR: Forward pass failed: {e}")
            return None, 0

    return model, reduction


def finetune_distillation(teacher, student, prompt_embeds, timestep, num_steps, verbose=True):
    """Fine-tune pruned model via knowledge distillation."""
    if verbose:
        print(f"\n  Fine-tuning with KD for {num_steps} steps...")

    student.train()
    teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    loss_fn = torch.nn.MSELoss()

    losses = []
    t0 = time.time()

    timestep_tensor = torch.tensor([timestep], device=DEVICE, dtype=torch.long)

    for step in range(num_steps):
        # Random latent input (simulates real inference conditions)
        sample = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)

        # Teacher prediction (target)
        with torch.no_grad():
            teacher_out = teacher(
                sample=sample,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
            ).sample

        # Student prediction
        student_out = student(
            sample=sample,
            timestep=timestep_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample

        # MSE loss between student and teacher outputs
        loss = loss_fn(student_out, teacher_out)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if verbose and (step + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            eta = (num_steps - step - 1) / steps_per_sec
            print(f"    Step {step+1}/{num_steps}: loss={avg_loss:.6f} "
                  f"({steps_per_sec:.1f} steps/s, ETA {eta:.0f}s)")

    final_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
    elapsed = time.time() - t0

    if verbose:
        print(f"  Fine-tuning complete: final_loss={final_loss:.6f}, "
              f"total={elapsed:.1f}s ({num_steps/elapsed:.1f} steps/s)")

    student.eval()
    return student, final_loss


def convert_to_coreml(model, name, verbose=True):
    """Convert pruned PyTorch UNet to CoreML."""
    import coremltools as ct

    if verbose:
        print(f"\n  Converting {name} to CoreML...")

    model = model.cpu().eval().float()

    # Get the actual channel dimensions from the pruned model
    in_channels = model.conv_in.in_channels  # should be 4

    # Trace the model
    sample = torch.randn(1, in_channels, LATENT_SIZE, LATENT_SIZE)
    timestep = torch.tensor([800], dtype=torch.long)
    hidden_states = torch.randn(1, 77, HIDDEN_SIZE)

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

    wrapper = UNetWrapper(model).eval()

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (sample, timestep, hidden_states),
            strict=False
        )

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=(1, in_channels, LATENT_SIZE, LATENT_SIZE), dtype=np.float16),
            ct.TensorType(name="timestep", shape=(1,), dtype=np.float16),
            ct.TensorType(name="encoder_hidden_states", shape=(1, 77, HIDDEN_SIZE), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="noise_pred", dtype=np.float16),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )

    output_path = os.path.join(COREML_DIR, f"unet_sdxs_512_{name}.mlpackage")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    mlmodel.save(output_path)

    # Check file size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    if verbose:
        print(f"  Saved: {output_path}")
        print(f"  Size: {total_size / 1024**2:.1f} MB")

    del mlmodel, traced, wrapper
    gc.collect()

    return output_path, total_size


def benchmark_variant(model_path, name, verbose=True):
    """Benchmark a CoreML model variant using the standard benchmark."""
    if verbose:
        print(f"\n  Benchmarking {name}...")

    # Modify pipeline.py temporarily to use this model
    pipeline_path = os.path.join(WORK_DIR, "pipeline.py")
    benchmark_path = os.path.join(WORK_DIR, "benchmark.py")

    # Read current pipeline
    with open(pipeline_path, 'r') as f:
        original_pipeline = f.read()

    # Create a temporary pipeline that uses the pruned model
    model_basename = os.path.basename(model_path).replace(".mlpackage", "")
    modified = original_pipeline.replace(
        'unet_path = os.path.join(COREML_DIR, f"{prefix}.mlpackage")',
        f'unet_path = os.path.join(COREML_DIR, "{model_basename}.mlpackage")'
    )

    with open(pipeline_path, 'w') as f:
        f.write(modified)

    try:
        # Run benchmark
        import subprocess
        result = subprocess.run(
            [os.path.join(WORK_DIR, ".venv/bin/python"), benchmark_path],
            capture_output=True, text=True, timeout=600,
            cwd=WORK_DIR,
        )

        output = result.stdout + result.stderr

        # Parse results
        metrics = {}
        for line in output.split('\n'):
            line = line.strip()
            for key in ['avg_ms', 'fps', 'memory_mb', 'quality_pass', 'unique_colors',
                       'input_variance', 'spatial_stddev', 'transform_diff', 'edge_strength']:
                if line.startswith(f'{key}:'):
                    val = line.split(':', 1)[1].strip()
                    if val in ('true', 'false'):
                        metrics[key] = val == 'true'
                    else:
                        try:
                            metrics[key] = float(val)
                        except ValueError:
                            metrics[key] = val

        if verbose:
            if 'avg_ms' in metrics:
                print(f"  Result: {metrics.get('avg_ms', '?')}ms / "
                      f"{metrics.get('fps', '?')} FPS | "
                      f"quality={'PASS' if metrics.get('quality_pass', False) else 'FAIL'}")
            else:
                print(f"  Benchmark failed or no output parsed")
                print(f"  Last 20 lines of output:")
                for line in output.split('\n')[-20:]:
                    print(f"    {line}")

        return metrics

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  Benchmark timed out (600s)")
        return {"error": "timeout"}
    except Exception as e:
        if verbose:
            print(f"  Benchmark error: {e}")
        return {"error": str(e)}
    finally:
        # Restore original pipeline
        with open(pipeline_path, 'w') as f:
            f.write(original_pipeline)


def main():
    results = []

    print("=" * 60)
    print("  Structural Pruning Pipeline for SDXS-512")
    print("=" * 60)

    # Step 1: Load teacher and get embeddings
    teacher, teacher_params = load_teacher()
    prompt_embeds, timestep = get_prompt_embeddings()

    # Step 2: Run each pruning configuration
    for config in PRUNING_CONFIGS:
        name = config["name"]
        ratio = config["ratio"]
        steps = config["finetune_steps"]

        print(f"\n{'='*60}")
        print(f"  Config: {name} (ratio={ratio}, finetune={steps} steps)")
        print(f"{'='*60}")

        result = {"name": name, "ratio": ratio, "steps": steps}

        # Prune
        try:
            pruned_model, param_reduction = structural_prune(teacher, ratio)
            if pruned_model is None:
                result["status"] = "prune_failed"
                results.append(result)
                continue
            result["param_reduction"] = param_reduction
            result["pruned_params"] = count_params(pruned_model)
        except Exception as e:
            print(f"  Pruning failed: {e}")
            import traceback
            traceback.print_exc()
            result["status"] = "prune_failed"
            result["error"] = str(e)
            results.append(result)
            continue

        # Fine-tune with knowledge distillation
        try:
            pruned_model, final_loss = finetune_distillation(
                teacher, pruned_model, prompt_embeds, timestep, steps
            )
            result["final_loss"] = final_loss
        except Exception as e:
            print(f"  Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            result["status"] = "finetune_failed"
            result["error"] = str(e)
            results.append(result)
            continue

        # Convert to CoreML
        try:
            model_path, model_size = convert_to_coreml(pruned_model, name)
            result["model_size_mb"] = model_size / 1024**2
            result["coreml_path"] = model_path
        except Exception as e:
            print(f"  CoreML conversion failed: {e}")
            import traceback
            traceback.print_exc()
            result["status"] = "convert_failed"
            result["error"] = str(e)
            results.append(result)
            # Clean up
            del pruned_model
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            continue

        # Free GPU memory before benchmark
        del pruned_model
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

        # Benchmark
        try:
            metrics = benchmark_variant(model_path, name)
            result.update(metrics)
            result["status"] = "success"
        except Exception as e:
            print(f"  Benchmark failed: {e}")
            result["status"] = "benchmark_failed"
            result["error"] = str(e)

        results.append(result)

        # Save intermediate results
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {RESULTS_FILE}")

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  {'Name':<20} {'Params':<12} {'Size MB':<10} {'avg_ms':<10} {'FPS':<8} {'Quality':<8}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    # Baseline
    print(f"  {'original':<20} {teacher_params:>11,} {'626.0':>9} {'35.0':>9} {'28.6':>7} {'PASS':>7}")

    for r in results:
        if r.get("status") == "success":
            quality = "PASS" if r.get("quality_pass", False) else "FAIL"
            print(f"  {r['name']:<20} {r.get('pruned_params', 0):>11,} "
                  f"{r.get('model_size_mb', 0):>9.1f} "
                  f"{r.get('avg_ms', 0):>9.1f} "
                  f"{r.get('fps', 0):>7.1f} "
                  f"{quality:>7}")
        else:
            print(f"  {r['name']:<20} {'FAILED':>11} — {r.get('status', 'unknown')}: {r.get('error', '')[:40]}")

    # Save final results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Full results: {RESULTS_FILE}")

    # Clean up
    del teacher
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
