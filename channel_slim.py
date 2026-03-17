#!/usr/bin/env python3
"""
Channel-Slimmed UNet: Create physically smaller UNet models by reducing channel dimensions.

Instead of pruning (which failed with torch-pruning on diffusers UNet),
we create new UNet instances with reduced block_out_channels and initialize
them from the teacher's weights by slicing channels.

This physically reduces FLOPs (the only way to speed up on compute-bound M3 Ultra).

Variants:
  slim_80: [256, 512, 1024]  ~213M params  (35% reduction)
  slim_50: [160, 320, 640]   ~82M params   (75% reduction)
  slim_30: [96, 192, 384]    ~30M params   (91% reduction)

Run: .venv/bin/python channel_slim.py
"""
import os
import sys
import json
import time
import gc
import shutil
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, EulerDiscreteScheduler
import coremltools as ct

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
COREML_DIR = os.path.join(WORK_DIR, "coreml_models")
RESULTS_FILE = os.path.join(WORK_DIR, "slim_results.json")
LOG_FILE = os.path.join(WORK_DIR, "slim.log")

MODEL_ID = "IDKiro/sdxs-512-0.9"
RENDER_SIZE = 512
LATENT_SIZE = 64
HIDDEN_SIZE = 1024
PROMPT = "oil painting style, masterpiece, highly detailed"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# All channel dims must be divisible by norm_num_groups=32 AND attention_head_dim=8
SLIM_CONFIGS = [
    {
        "name": "slim_80",
        "channels": [256, 512, 1024],
        "finetune_steps": 300,
        "lr": 2e-4,
    },
    {
        "name": "slim_50",
        "channels": [160, 320, 640],
        "finetune_steps": 800,
        "lr": 3e-4,
    },
    {
        "name": "slim_30",
        "channels": [96, 192, 384],
        "finetune_steps": 1500,
        "lr": 5e-4,
    },
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def create_slim_unet(channels):
    """Create a new UNet with reduced channels."""
    config = {
        "sample_size": 64,
        "in_channels": 4,
        "out_channels": 4,
        "center_input_sample": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "mid_block_type": None,
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
        "only_cross_attention": [True, False, False],
        "block_out_channels": channels,
        "layers_per_block": 1,
        "downsample_padding": 1,
        "mid_block_scale_factor": 1,
        "dropout": 0.0,
        "act_fn": "silu",
        "norm_num_groups": 32,
        "norm_eps": 1e-05,
        "cross_attention_dim": 1024,
        "transformer_layers_per_block": 1,
        "attention_head_dim": [8, 8, 8],
        "use_linear_projection": True,
        "resnet_time_scale_shift": "default",
        "conv_in_kernel": 3,
        "conv_out_kernel": 3,
    }
    return UNet2DConditionModel(**config)


def init_from_teacher(student, teacher):
    """Initialize student from teacher by slicing channels where dimensions match."""
    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()

    initialized = 0
    skipped = 0

    for key in student_sd:
        if key not in teacher_sd:
            skipped += 1
            continue

        t_tensor = teacher_sd[key]
        s_tensor = student_sd[key]

        if t_tensor.shape == s_tensor.shape:
            student_sd[key] = t_tensor.clone()
            initialized += 1
        else:
            # Try to slice matching dimensions
            try:
                slices = []
                for dim in range(len(s_tensor.shape)):
                    if s_tensor.shape[dim] <= t_tensor.shape[dim]:
                        slices.append(slice(0, s_tensor.shape[dim]))
                    else:
                        # Student is bigger than teacher in this dim — skip
                        slices = None
                        break

                if slices is not None:
                    student_sd[key] = t_tensor[tuple(slices)].clone()
                    initialized += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    student.load_state_dict(student_sd)
    return initialized, skipped


def get_prompt_embeddings(device):
    """Get text encoder embeddings."""
    log("Loading text encoder...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(1, device=device)
    timestep = pipe.scheduler.timesteps[0].item()

    with torch.no_grad():
        ti = pipe.tokenizer(PROMPT, padding="max_length",
                           max_length=pipe.tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
        embeds = pipe.text_encoder(ti.input_ids.to(device))[0]

    del pipe
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    log(f"  Embeddings: {embeds.shape}, timestep: {timestep}")
    return embeds, timestep


def finetune_kd(teacher, student, prompt_embeds, timestep, config):
    """Knowledge distillation fine-tuning."""
    steps = config["finetune_steps"]
    lr = config["lr"]
    log(f"  Fine-tuning {steps} steps, lr={lr}...")

    student.train()
    teacher.eval()

    # Only train student parameters
    for p in student.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    ts_tensor = torch.tensor([timestep], device=DEVICE, dtype=torch.long)
    loss_fn = torch.nn.MSELoss()

    losses = []
    t0 = time.time()

    for step in range(steps):
        sample = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)

        with torch.no_grad():
            target = teacher(sample=sample, timestep=ts_tensor,
                           encoder_hidden_states=prompt_embeds).sample

        pred = student(sample=sample, timestep=ts_tensor,
                      encoder_hidden_states=prompt_embeds).sample

        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg = np.mean(losses[-100:])
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            eta = (steps - step - 1) / sps
            log(f"    Step {step+1}/{steps}: loss={avg:.6f} ({sps:.1f} s/s, ETA {eta:.0f}s)")

    final_loss = np.mean(losses[-min(50, len(losses)):])
    elapsed = time.time() - t0
    log(f"  Done: loss={final_loss:.6f}, {elapsed:.1f}s total")

    student.eval()
    return final_loss


def convert_to_coreml(model, name):
    """Convert to CoreML mlpackage."""
    log(f"  Converting {name} to CoreML...")

    model = model.cpu().eval().float()

    class Wrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(sample=sample, timestep=timestep,
                           encoder_hidden_states=encoder_hidden_states).sample

    wrapper = Wrapper(model).eval()

    sample = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE)
    ts = torch.tensor([800], dtype=torch.long)
    hs = torch.randn(1, 77, HIDDEN_SIZE)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample, ts, hs), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=(1, 4, LATENT_SIZE, LATENT_SIZE), dtype=np.float16),
            ct.TensorType(name="timestep", shape=(1,), dtype=np.float16),
            ct.TensorType(name="encoder_hidden_states", shape=(1, 77, HIDDEN_SIZE), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="noise_pred", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )

    out_path = os.path.join(COREML_DIR, f"unet_sdxs_512_{name}.mlpackage")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_path) for f in fns
    )

    log(f"  Saved: {out_path} ({total_size/1024**2:.1f} MB)")
    del mlmodel, traced, wrapper
    gc.collect()
    return out_path, total_size


def benchmark_model(model_name):
    """Run standard benchmark with a specific UNet model."""
    log(f"  Benchmarking {model_name}...")
    import subprocess

    pipeline_path = os.path.join(WORK_DIR, "pipeline.py")

    with open(pipeline_path, 'r') as f:
        original = f.read()

    # Patch pipeline to use our model
    patched = original.replace(
        'unet_path = os.path.join(COREML_DIR, f"{prefix}.mlpackage")',
        f'unet_path = os.path.join(COREML_DIR, "unet_sdxs_512_{model_name}.mlpackage")'
    )

    with open(pipeline_path, 'w') as f:
        f.write(patched)

    try:
        result = subprocess.run(
            [os.path.join(WORK_DIR, ".venv/bin/python"), os.path.join(WORK_DIR, "benchmark.py")],
            capture_output=True, text=True, timeout=600, cwd=WORK_DIR,
        )
        output = result.stdout + result.stderr

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

        if 'avg_ms' in metrics:
            qp = 'PASS' if metrics.get('quality_pass', False) else 'FAIL'
            log(f"  → {metrics['avg_ms']:.1f}ms / {metrics.get('fps', 0):.1f} FPS | quality={qp}")
        else:
            log(f"  → Benchmark failed, last lines:")
            for l in output.split('\n')[-10:]:
                log(f"    {l}")

        return metrics

    except subprocess.TimeoutExpired:
        log("  → Benchmark timeout (600s)")
        return {"error": "timeout"}
    except Exception as e:
        log(f"  → Benchmark error: {e}")
        return {"error": str(e)}
    finally:
        with open(pipeline_path, 'w') as f:
            f.write(original)


def main():
    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write("")

    log("=" * 60)
    log("  Channel-Slimmed UNet Pipeline")
    log(f"  Device: {DEVICE}")
    log("=" * 60)

    # Load teacher
    log("Loading teacher model...")
    teacher = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32
    ).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = count_params(teacher)
    log(f"  Teacher: {teacher_params:,} params")

    # Get embeddings
    prompt_embeds, timestep = get_prompt_embeddings(DEVICE)

    results = []

    for config in SLIM_CONFIGS:
        name = config["name"]
        channels = config["channels"]

        log(f"\n{'='*60}")
        log(f"  {name}: channels={channels}")
        log(f"{'='*60}")

        result = {"name": name, "channels": channels}

        # Create slim model
        try:
            log("  Creating slim UNet...")
            student = create_slim_unet(channels).to(DEVICE)
            student_params = count_params(student)
            reduction = 1.0 - student_params / teacher_params
            log(f"  Params: {student_params:,} ({reduction*100:.1f}% reduction)")
            result["params"] = student_params
            result["param_reduction"] = reduction
        except Exception as e:
            log(f"  Creation failed: {e}")
            import traceback; traceback.print_exc()
            result["status"] = "create_failed"
            result["error"] = str(e)
            results.append(result)
            continue

        # Initialize from teacher
        try:
            log("  Initializing from teacher (channel slicing)...")
            init_count, skip_count = init_from_teacher(student, teacher)
            log(f"  Initialized {init_count} tensors, skipped {skip_count}")
            result["init_count"] = init_count
        except Exception as e:
            log(f"  Init failed: {e}")
            result["status"] = "init_failed"
            results.append(result)
            del student; gc.collect()
            continue

        # Verify forward pass
        with torch.no_grad():
            try:
                test_sample = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)
                test_ts = torch.tensor([timestep], device=DEVICE, dtype=torch.long)
                out = student(sample=test_sample, timestep=test_ts,
                             encoder_hidden_states=prompt_embeds).sample
                log(f"  Forward pass OK: {out.shape}")
                del test_sample, test_ts, out
            except Exception as e:
                log(f"  Forward pass FAILED: {e}")
                result["status"] = "forward_failed"
                result["error"] = str(e)
                results.append(result)
                del student; gc.collect()
                continue

        # Fine-tune
        try:
            final_loss = finetune_kd(teacher, student, prompt_embeds, timestep, config)
            result["final_loss"] = final_loss
        except Exception as e:
            log(f"  Fine-tuning failed: {e}")
            import traceback; traceback.print_exc()
            result["status"] = "finetune_failed"
            result["error"] = str(e)
            results.append(result)
            del student; gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            continue

        # Convert to CoreML
        try:
            student_cpu = student.cpu()
            del student; gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

            model_path, model_size = convert_to_coreml(student_cpu, name)
            result["model_size_mb"] = model_size / 1024**2
            del student_cpu; gc.collect()
        except Exception as e:
            log(f"  CoreML conversion failed: {e}")
            import traceback; traceback.print_exc()
            result["status"] = "convert_failed"
            result["error"] = str(e)
            results.append(result)
            gc.collect()
            continue

        # Benchmark
        try:
            metrics = benchmark_model(name)
            result.update(metrics)
            result["status"] = "success"
        except Exception as e:
            log(f"  Benchmark failed: {e}")
            result["status"] = "benchmark_failed"
            result["error"] = str(e)

        results.append(result)

        # Save intermediate
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    log(f"\n{'='*60}")
    log("  FINAL RESULTS")
    log(f"{'='*60}")
    log(f"  {'Name':<12} {'Channels':<18} {'Params':<12} {'Size MB':<9} {'avg_ms':<9} {'FPS':<8} {'Quality'}")
    log(f"  {'-'*12} {'-'*18} {'-'*12} {'-'*9} {'-'*9} {'-'*8} {'-'*7}")
    log(f"  {'original':<12} {'[320,640,1280]':<18} {teacher_params:>11,} {'626.0':>8} {'35.0':>8} {'28.6':>7} {'PASS':>7}")

    for r in results:
        if r.get("status") == "success":
            ch = str(r['channels'])
            qp = "PASS" if r.get('quality_pass', False) else "FAIL"
            log(f"  {r['name']:<12} {ch:<18} {r.get('params',0):>11,} "
                f"{r.get('model_size_mb',0):>8.1f} {r.get('avg_ms',0):>8.1f} "
                f"{r.get('fps',0):>7.1f} {qp:>7}")
        else:
            log(f"  {r['name']:<12} FAILED: {r.get('status','?')} — {r.get('error','')[:50]}")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nDone. Results: {RESULTS_FILE}")

    del teacher; gc.collect()


if __name__ == "__main__":
    main()
