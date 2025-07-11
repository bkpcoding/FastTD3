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
    total_timesteps: int = 200000000
    """Total timesteps to train for."""
    num_envs: int = 4
    """Number of parallel environments."""
    learning_rate: float = 3e-4
    """Learning rate for the optimizer."""
    gamma: float = 0.99
    """Discount factor."""
    gae_lambda: float = 0.95
    """Lambda for GAE."""
    clip_coeff: float = 0.2
    """Clipping range for the policy loss."""
    ent_coeff: float = 0.01
    """Entropy regularization coefficient."""
    vf_coeff: float = 0.5
    """Value function loss coefficient."""
    update_epochs: int = 4
    """Number of optimization epochs per update."""
    batch_size: int = 32
    """Mini-batch size."""
    rollout_length: int = 64
    """Number of steps to collect before each update."""
    hidden_dim: int = 256
    """Hidden dimension of policy and value networks."""
    log_interval: int = 10000
    """Interval (in steps) between logging to stdout."""
    eval_interval: int = 100000
    """Evaluation interval in environment steps."""
    num_eval_envs: int = 10
    """Number of parallel evaluation environments."""
    use_wandb: bool = False
    """Enable logging to Weights & Biases."""
    project: str = "rl_scratch"
    """wandb project name."""
    exp_name: str = "ppo"
    """Experiment name used for logging and checkpointing."""
    seed: int = 1
    """Random seed."""
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""
    output_dir: str = "logs"
    """Directory to store checkpoints and logs."""
    save_interval: int = 0
    """How often to save model checkpoints. If 0, disabled."""
    compile: bool = True
    """Use torch.compile on key functions."""
    amp: bool = False
    """Enable automatic mixed precision."""
    amp_dtype: str = "bf16"
    """Precision for AMP (bf16 or fp16)."""
    
    # PBT specific parameters (added for compatibility)
    pbt_enabled: bool = False
    """whether to enable PBT"""
    pbt_policy_idx: int = 0
    """the index of this policy in the population"""
    pbt_num_policies: int = 8
    """the number of policies in the population"""
    pbt_workspace: str = "pbt_workspace"
    """the workspace directory for PBT"""
    pbt_interval_steps: int = 100000
    """the interval in steps between PBT operations"""
    pbt_start_after: int = 50000
    """start PBT operations after this many steps"""
    pbt_initial_delay: int = 10000
    """initial delay before any PBT operations"""
    pbt_replace_fraction_best: float = 0.25
    """fraction of best policies to use for replacement"""
    pbt_replace_fraction_worst: float = 0.25
    """fraction of worst policies to replace"""
    pbt_replace_threshold_frac_std: float = 0.3
    """threshold for replacement as fraction of std"""
    pbt_replace_threshold_frac_absolute: float = 0.1
    """absolute threshold for replacement as fraction of performance"""
    pbt_mutation_rate: float = 0.8
    """probability of mutating each parameter"""
    pbt_change_min: float = 1.1
    """minimum change factor for mutations"""
    pbt_change_max: float = 1.5
    """maximum change factor for mutations"""
    pbt_dbg_mode: bool = False
    """enable PBT debug mode"""
    pbt_restart: bool = False
    """internal flag for PBT restart detection"""
    checkpoint_path: str = None
    """path to checkpoint file for loading"""


def get_args():
    """Parse command line arguments using tyro and return an Args instance."""
    return tyro.cli(BaseArgs)


# PBT mutation configuration
PPO_MUTATION_CONFIG = {
    "learning_rate": "mutate_learning_rate",
    "gamma": "mutate_discount", 
    "gae_lambda": "mutate_gae_lambda",
    "clip_coeff": "mutate_clip_coeff",
    "ent_coeff": "mutate_entropy_coeff",
    "vf_coeff": "mutate_float",
    "update_epochs": "mutate_mini_epochs",
    "hidden_dim": "mutate_hidden_dim",
}