"""Configuration for multi-agent PPO training with independent agents."""

from dataclasses import dataclass
from typing import Optional
import tyro


@dataclass
class AgentConfig:
    """Configuration for a single independent agent."""
    
    # Environment settings
    num_envs: int = 64
    """Number of parallel environments for this agent."""
    
    # PPO Hyperparameters
    rollout_length: int = 32
    """Number of steps to collect before each update."""
    learning_rate: float = 3e-4
    """Learning rate for the optimizer."""
    gamma: float = 0.99
    """Discount factor."""
    gae_lambda: float = 0.95
    """Lambda for GAE."""
    
    # Training parameters
    update_epochs: int = 4
    """Number of optimization epochs per update."""
    batch_size: int = 256
    """Mini-batch size."""
    clip_eps: float = 0.2
    """Clipping range for the policy loss."""
    ent_coef: float = 0.01
    """Entropy regularization coefficient."""
    vf_coef: float = 0.5
    """Value function loss coefficient."""
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""
    
    # Network architecture
    hidden_dim: int = 256
    """Hidden dimension of policy and value networks."""


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent PPO training."""
    
    # Environment settings
    env_name: str = "T1JoystickFlatTerrain"
    """Environment name."""
    env_type: str = "mujoco_playground"
    """Type of environment."""
    
    # Training settings
    total_timesteps: int = 1000000
    """Total timesteps to train for."""
    seed: int = 42
    """Random seed."""
    
    # Agent configurations
    agent_1_config: AgentConfig = AgentConfig(
        num_envs=64,
        rollout_length=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        batch_size=256,
        hidden_dim=256
    )
    """Configuration for agent 1."""
    
    agent_2_config: AgentConfig = AgentConfig(
        num_envs=64,
        rollout_length=64,
        learning_rate=1e-4,
        gamma=0.95,
        gae_lambda=0.98,
        batch_size=256,
        hidden_dim=256
    )
    """Configuration for agent 2."""
    
    # Trajectory filtering settings
    trajectory_filter_timestep: int = 50000
    """Timestep after which to start filtering trajectories."""
    top_k_trajectories: int = 10
    """Number of top trajectories to keep for state indexing."""
    
    # FAISS settings
    faiss_index_type: str = "Flat"
    """Type of FAISS index to use."""
    faiss_use_gpu: bool = True
    """Whether to use GPU for FAISS operations."""
    
    # Logging and evaluation
    log_interval: int = 10000
    """Interval for logging."""
    eval_interval: int = 50000
    """Interval for evaluation."""
    num_eval_envs: int = 5
    """Number of evaluation environments per agent."""
    
    # Output settings
    output_dir: str = "multi_agent_logs"
    """Directory for saving logs and models."""
    save_interval: int = 100000
    """Interval for saving models."""
    
    # Wandb settings
    use_wandb: bool = False
    """Whether to use Weights & Biases logging."""
    project: str = "multi_agent_ppo"
    """Wandb project name."""
    
    # Performance settings
    compile: bool = False
    """Use torch.compile for optimization. Disabled for multi-agent to avoid threading conflicts."""
    amp: bool = False
    """Use automatic mixed precision."""
    amp_dtype: str = "bf16"
    """AMP dtype (bf16 or fp16)."""


def get_multi_agent_config() -> MultiAgentConfig:
    """Parse command line arguments and return MultiAgentConfig."""
    return tyro.cli(MultiAgentConfig)


def validate_config(config: MultiAgentConfig) -> None:
    """Validate the multi-agent configuration."""
    
    # Validate timestep settings
    if config.trajectory_filter_timestep >= config.total_timesteps:
        raise ValueError(
            f"trajectory_filter_timestep ({config.trajectory_filter_timestep}) "
            f"must be less than total_timesteps ({config.total_timesteps})"
        )
    
    # Validate top_k setting
    if config.top_k_trajectories <= 0:
        raise ValueError(f"top_k_trajectories must be positive, got {config.top_k_trajectories}")
    
    # Validate agent configurations
    for agent_name, agent_config in [("agent_1", config.agent_1_config), ("agent_2", config.agent_2_config)]:
        if agent_config.num_envs <= 0:
            raise ValueError(f"{agent_name} num_envs must be positive, got {agent_config.num_envs}")
        if agent_config.rollout_length <= 0:
            raise ValueError(f"{agent_name} rollout_length must be positive, got {agent_config.rollout_length}")
        if agent_config.learning_rate <= 0:
            raise ValueError(f"{agent_name} learning_rate must be positive, got {agent_config.learning_rate}")
        if not 0 <= agent_config.gamma <= 1:
            raise ValueError(f"{agent_name} gamma must be in [0, 1], got {agent_config.gamma}")
        if not 0 <= agent_config.gae_lambda <= 1:
            raise ValueError(f"{agent_name} gae_lambda must be in [0, 1], got {agent_config.gae_lambda}")
    
    print("Multi-agent configuration validated successfully.")


if __name__ == "__main__":
    config = get_multi_agent_config()
    validate_config(config)
    print("Configuration:", config)