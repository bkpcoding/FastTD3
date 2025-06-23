import os
from dataclasses import dataclass
import tyro


@dataclass
class BaseArgs:
    """Default hyperparameters for GRPO training."""

    env_name: str = "T1JoystickFlatTerrain"
    env_type: str = "mujoco_playground"
    total_timesteps: int = 2000000
    num_envs: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    ent_coef: float = 0.0
    update_epochs: int = 10
    batch_size: int = 64
    hidden_dim: int = 256
    group_size: int = 8
    log_interval: int = 1000
    seed: int = 1
    project: str = "rl_scratch"
    exp_name: str = "grpo"
    use_wandb: bool = False
    output_dir: str = "logs"
    compile: bool = False
    amp: bool = False
    amp_dtype: str = "bf16"
    max_grad_norm: float = 1.0


def get_args():
    return tyro.cli(BaseArgs)
