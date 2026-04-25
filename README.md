# CLR-Routing

Uncertainty-Aware Prototype-Guided Routing for Task-Agnostic Continual Learning.

## Setup

```bash
uv venv && source .venv/bin/activate
uv sync && uv sync --group dev
```

### PyTorch backend

The default `uv sync` installs the CPU build of PyTorch. Install the right
accelerator build for your machine:

**Linux / Windows with NVIDIA GPU (CUDA):**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# adjust cu124 to match your driver: nvidia-smi to check
```

**macOS with Apple Silicon (M1–M4):**
```bash
uv pip install torch torchvision  # default Mac wheels include MPS
```

**CPU-only:**
```bash
# already installed by uv sync
```

Then:
```bash
wandb login
```

### Device selection

`device: auto` in `configs/base.yaml` picks the best available backend in the
order CUDA > MPS > CPU. Override at the command line if needed:

```bash
python scripts/train.py device=mps   # force MPS
python scripts/train.py device=cpu   # force CPU
```

### Apple Silicon notes

On a 24GB M-series Mac, ViT-small + 5 LoRA experts + batch size 64 fits
comfortably. A few performance notes:

- MPS is meaningfully slower than a discrete CUDA GPU but workable for
  Split CIFAR-100 with 10 tasks.
- `pin_memory` is automatically disabled on MPS (only valid for CUDA).
- If you see warnings about ops falling back to CPU, the `PYTORCH_ENABLE_MPS_FALLBACK=1`
  env var is set automatically — these ops will run, just on CPU rather than the GPU.
- DataLoader workers default to 4. If you see any multiprocessing weirdness
  on macOS, try `data.num_workers=0` to debug.

## Quick start

```bash
# Train with prototype-guided routing on Split CIFAR-100
python scripts/train.py experiment=proto_routing

# Baseline: fixed top-1 routing
python scripts/train.py experiment=baseline_top1

# Ablation: global replay instead of per-expert
python scripts/train.py experiment=proto_routing replay.global_replay=true
```

## Architecture

The codebase is organized around four swappable interfaces:

| Interface | Location | Implementations |
|-----------|----------|-----------------|
| `BackboneBase` | `models/backbone.py` | `ViTBackbone` (extensible to text encoders) |
| `RoutingStrategy` | `models/router.py` | `PrototypeRouter`, `FixedTopKRouter` |
| `ReplayBuffer` | `continual/buffer.py` | `PerExpertReplayBuffer`, `GlobalReplayBuffer` |
| `MetricsTracker` | `continual/metrics.py` | `ContinualMetrics` |

The `ContinualLearner` (in `models/__init__.py`) composes a backbone, LoRA expert bank,
router, and prototype memory into a single `nn.Module`. The `ContinualTrainer`
(in `continual/trainer.py`) orchestrates the task loop given any combination of these.

This design lets you swap routing strategies or buffers at the config level without
touching trainer code — the recommended way to run ablations.

## Tests

```bash
pytest tests/ -v
```

The most important tests are in `tests/test_router.py` (routing math) and
`tests/test_metrics.py` (forgetting computation). Run these whenever you touch
either module.
