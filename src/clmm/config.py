from dataclasses import dataclass, asdict


@dataclass
class Config:
    dataset: str = "cifar100"
    num_classes: int = 42
    num_tasks: int = 7
    classes_per_task: int = 6
    random_task_order: bool = True
    seed: int = 42

    model_name: str = "vit_tiny_patch16_224"
    pretrained: bool = True
    input_size: int = 224

    num_experts: int = 3
    router_hidden_dim: int = 384
    router_embed_dim: int = 384
    lora_rank: int = 32
    lora_alpha: int = 8

    buffer_size_per_expert: int = 400
    global_buffer_size: int = 1000
    batch_size: int = 32
    replay_batch_size: int = 32
    eval_batch_size: int = 64
    epochs_per_task: int = 3
    warmup_tasks: int = 1

    tau: float = 0.1
    beta_proto: float = 0.05
    gamma_task_proto: float = 0.05
    lambda_replay: float = 2.0
    lambda_route: float = 0.5
    lambda_balance: float = 0.01

    routing_mode: str = "entropy_adaptive"  # entropy_adaptive, top1, top2, top3
    replay_mode: str = "expert"             # expert, global, none
    use_routing_kl: bool = True
    use_load_balance: bool = False

    entropy_threshold_low: float = 0.774
    entropy_threshold_high: float = 0.910

    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    use_amp: bool = True
    max_grad_norm: float = 1.0

    use_normalized_entropy: bool = True

    debug: bool = False
    debug_num_tasks: int = 2
    debug_train_samples_per_task: int = 300
    debug_test_samples_per_task: int = 200

    out_dir: str = "/content/drive/MyDrive/HomeworksBT/CLMM/clmm_outputs"
