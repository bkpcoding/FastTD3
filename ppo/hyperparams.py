import os
from dataclasses import dataclass
import tyro


@dataclass
class BaseArgs:
    """Default hyperparameters for PPO training."""

    env_name: str = "T1JoystickFlatTerrain"
    """Identifier of the environment."""
    env_type: str = "mujoco_playground"
    """Type of the environment. Currently only MuJoCo Playground is supported."""
    total_timesteps: int = 100_000
    """Total timesteps to train for."""
    num_envs: int = 32768
    """Number of parallel environments."""
    learning_rate: float = 1e-4
    """Learning rate for the optimizer."""
    gamma: float = 0.98
    """Discount factor."""
    gae_lambda: float = 0.95
    """Lambda for GAE."""
    clip_eps: float = 0.2
    """Clipping range for the policy loss."""
    ent_coef: float = 0.0
    """Entropy regularization coefficient."""
    vf_coef: float = 0.5
    """Value function loss coefficient."""
    update_epochs: int = 5
    """Number of optimization epochs per update."""
    batch_size: int = 1024
    """Mini-batch size."""
    rollout_length: int = 32
    """Number of steps to collect before each update."""
    hidden_dim: int = 512
    """Hidden dimension of policy and value networks."""
    log_interval: int = 5000
    """Interval (in steps) between logging to stdout."""
    eval_interval: int = 5000
    """Evaluation interval in environment steps."""
    num_eval_envs: int = 1024
    """Number of parallel evaluation environments."""
    render_interval: int = 5000
    """Interval (in steps) between video rendering. If 0, disabled."""
    use_wandb: bool = True
    """Enable logging to Weights & Biases."""
    project: str = "MJX"
    """wandb project name."""
    exp_name: str = "ppo_brax_hyperparams"
    """Experiment name used for logging and checkpointing."""
    seed: int = 1
    """Random seed."""
    max_grad_norm: float = 0.0
    """Maximum gradient norm for clipping."""
    output_dir: str = "logs"
    """Directory to store checkpoints and logs."""
    save_interval: int = 0
    """How often to save model checkpoints. If 0, disabled."""
    compile: bool = True
    """Use torch.compile on key functions."""
    device_rank: int = 0
    """Device rank for multi-GPU training."""
    amp: bool = False
    """Enable automatic mixed precision."""
    amp_dtype: str = "bf16"
    """Precision for AMP (bf16 or fp16)."""
    enable_asymmetric_obs: bool = False
    """Enable asymmetric actor-critic observations (if environment supports it)."""
    use_simbav2: bool = False
    """Use SimbaV2 architecture instead of standard MLP networks."""
    use_domain_randomization: bool = False
    """Use domain randomization."""
    use_push_randomization: bool = False
    """Use push randomization."""
    use_tuned_reward: bool = False
    """Use tuned reward."""
    weight_decay: float = 0.0
    """Weight decay coefficient for optimizer."""
    use_layer_norm: bool = False
    """Use layer normalization in actor/critic networks."""
    reward_normalization: bool = False
    """Enable reward normalization using running statistics."""
    


# Placeholder dataclasses for other environment types. These will be filled out
# with environment specific defaults in the future.
@dataclass
class MuJoCoPlaygroundArgs(BaseArgs):
    pass


@dataclass
class HumanoidBenchArgs(BaseArgs):
    pass


@dataclass
class IsaacLabArgs(BaseArgs):
    pass


def get_args():
    """Parse command line arguments using tyro and return an Args instance."""
    return tyro.cli(BaseArgs)
