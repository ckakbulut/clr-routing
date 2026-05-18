# CLMM — Continual Learning with Multi-Expert LoRA Routing

**Uncertainty-Aware Prototype-Guided Routing with LoRA Experts for Task-Agnostic Continual Learning**

Group project for **COMS6998: Continual Learning and Memory Models**

**Authors:** Berkay Baris Turan, Can Kerem Akbulut

---

## Introduction

Modern neural networks struggle with *catastrophic forgetting*: when trained sequentially on a stream of tasks, they tend to overwrite knowledge acquired on earlier tasks as they learn new ones. This project addresses that problem in a *task-agnostic* setting in which the model receives no task identity at training or inference time.

Our approach combines the following ideas:

- **Frozen ViT backbone + LoRA expert adapters.** A pretrained Vision Transformer (ViT-Tiny) is kept frozen. A bank of lightweight LoRA adapters is injected into every MLP layer. Each adapter specialises on a different region of the input distribution.
- **Prototype-guided routing.** Per-class and per-expert prototype vectors are maintained as exponential moving averages. At each forward pass, the routing distribution is computed as a cosine-similarity softmax between the input embedding and the expert prototypes.
- **Entropy-adaptive expert selection.** Instead of always activating a fixed number of experts, the system uses the entropy of the routing distribution to decide how many experts to use per sample: low-entropy (confident) inputs activate a single expert; high-entropy (ambiguous) inputs activate up to three. This reduces compute on easy samples while retaining capacity for hard ones.
- **Expert-specific replay buffers.** A class-balanced replay buffer is maintained per expert. Samples are stored in the buffer of their top-1 routed expert, so replay draws from experts that actually processed each class, improving backward transfer and reducing interference.

Baselines compare static top-1, top-2, and top-3 routing and global (non-expert-specific) replay.

---

## File Structure

```
clr-routing/
│
├── CLMM_PROPEL_Notebook.ipynb   # Self-contained Colab notebook (see below)
│
├── src/clmm/                    # Main Python package
│   ├── config.py                # Config dataclass — all hyperparameters in one place
│   ├── data.py                  # CIFAR-100 task split, TaskSubset, transforms
│   ├── lora.py                  # ExpertLoRALinear, ViT MLP replacement, layer iterator
│   ├── model.py                 # PrototypeManager, CLLoRAViT (main model), RouterMLP
│   ├── routing.py               # select_experts, routing_entropy, KL loss, load-balance loss
│   ├── replay.py                # ClassBalancedBuffer, ReplayManager (global + per-expert)
│   ├── trainer.py               # train_one_task, evaluate_task, compute_metrics, reinit helpers
│   ├── experiment.py            # method_config, run_experiment (ties everything together)
│   ├── plotting.py              # Publication-style figures (accuracy, forgetting, heatmaps, …)
│   └── utils.py                 # set_seed, get_device
│
├── scripts/
│   └── run_clmm.py              # CLI entrypoint — runs experiments and saves figures
│
├── tests/
│   ├── test_clmm_lora.py        # ExpertLoRALinear forward-pass correctness
│   ├── test_clmm_routing.py     # select_experts, entropy, KL loss
│   ├── test_clmm_replay.py      # ClassBalancedBuffer, ReplayManager
│   ├── test_clmm_model.py       # PrototypeManager assignment and EMA updates
│   └── test_clmm_metrics.py     # compute_metrics: avg accuracy, forgetting, BWT
│
├── notebooks/
│   └── run_experiments.ipynb    # Hydra-based multi-run notebook (older driver)
│
├── pyproject.toml
└── requirements.txt
```

---

## Running on Google Colab (no local GPU required)

**`CLMM_PROPEL_Notebook.ipynb` is a complete, top-to-bottom runnable instance of the entire project.** If you do not have access to local GPU compute, open it in Google Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com) and upload or open the notebook from your Google Drive.
2. Enable a GPU through Runtime → Change runtime type → T4 GPU (or A100 if available).
3. Run the cells from top to bottom. No other setup is needed. The notebook installs its own dependencies, downloads CIFAR-100, and saves all outputs (figures, accuracy matrices) to your Google Drive.
4. To run a quick smoke test instead of the full 7-task experiment, set `debug=True` in the configuration cell before running.

The notebook is fully self-contained and does not require cloning the repository. It mirrors the `src/clmm/` package exactly.

---

## Local Setup

### 1. Create the virtual environment and install dependencies

```bash
uv venv && source .venv/bin/activate
uv sync && uv sync --group dev
```

### 2. Install the right PyTorch build for your hardware

**NVIDIA GPU (CUDA 12.4):**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# adjust cu124 to match your driver version (check with: nvidia-smi)
```

**Apple Silicon (M1–M4, MPS):**
```bash
uv pip install torch torchvision  # default Mac wheels include MPS
```

**CPU only:**
```bash
# already installed by uv sync — no extra step needed
```

### 3. Install the clmm package in editable mode

```bash
pip install -e .
```

---

## Running Experiments

### Run the full experiment (all methods)

```bash
python scripts/run_clmm.py --methods full static_top1 static_top2 static_top3 global_buffer \
    --out_dir ./outputs/clmm \
    --data_root ./data
```

### Run only the proposed method

```bash
python scripts/run_clmm.py --methods full --out_dir ./outputs/clmm --data_root ./data
```

### Quick smoke test (debug mode — 2 tasks, fewer samples)

```bash
python scripts/run_clmm.py --debug --methods full --out_dir ./outputs/clmm --data_root ./data
```

### Available methods

| Flag | Description |
|------|-------------|
| `full` | Proposed method: entropy-adaptive routing + expert replay + KL loss |
| `static_top1` | Baseline: always activate 1 expert |
| `static_top2` | Baseline: always activate 2 experts |
| `static_top3` | Baseline: always activate 3 experts |
| `global_buffer` | Ablation: entropy-adaptive routing + single global replay buffer |

All figures (accuracy curves, forgetting curves, routing heatmaps, expert utilization, entropy histograms, class-expert similarity maps) are saved as PNG files to `--out_dir`.

---

## Running Tests

```bash
pytest tests/ -v
```

All 51 tests should pass. The test suite covers LoRA forward-pass correctness, routing math, buffer behavior, prototype manager updates, and continual learning metrics.

---

## Key Hyperparameters

All hyperparameters live in `src/clmm/config.py` as a single `Config` dataclass. The most important ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 3 | Number of LoRA expert adapters |
| `lora_rank` | 32 | LoRA rank per expert per layer |
| `routing_mode` | `entropy_adaptive` | Routing strategy |
| `entropy_threshold_low` | 0.774 | Normalized entropy below which 1 expert is used |
| `entropy_threshold_high` | 0.910 | Normalized entropy above which 3 experts are used |
| `tau` | 0.1 | Softmax temperature for routing distribution |
| `buffer_size_per_expert` | 400 | Replay buffer capacity per expert |
| `epochs_per_task` | 3 | Training epochs per task |
| `lambda_replay` | 2.0 | Replay loss weight |
| `num_tasks` | 7 | Number of continual learning tasks |
| `classes_per_task` | 6 | CIFAR-100 classes per task |
