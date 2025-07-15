"""Configuration for multi-agent PPO training with independent agents."""

from dataclasses import dataclass
from typing import Optional, List
import tyro
import os
from datetime import datetime


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
    device_rank: int = 0
    """Device rank for multi-GPU training."""


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent PPO training."""
    
    # Environment settings
    env_name: str = "h1hand-hurdle-v0"
    """Environment name."""
    env_type: str = "mujoco_playground"
    """Type of environment."""
    
    # Training settings
    total_timesteps: int = 10_000_000
    """Total timesteps to train for sub-agents."""
    main_agent_timesteps: int = 10_000_000
    """Total timesteps to train for main agent."""
    seed: int = 42
    """Random seed."""
    
    # Agent configurations
    num_agents: int = 1
    """Number of sub-agents (excluding main agent)."""
    use_tuned_reward: bool = False
    """Whether to use tuned reward."""
    use_domain_randomization: bool = False
    """Whether to use domain randomization."""
    use_push_randomization: bool = False
    """Whether to use push randomization."""
    agent_configs: List[AgentConfig] = None
    """List of configurations for each sub-agent."""
    
    main_agent_config: AgentConfig = AgentConfig(
        num_envs=1024,
        rollout_length=64,
        learning_rate=2e-4,
        gamma=0.99,
        gae_lambda=0.95,
        batch_size=256,
        hidden_dim=256
    )
    """Configuration for main agent with FAISS state initialization."""
    
    def __post_init__(self):
        """Initialize default agent configs if not provided."""
        if self.agent_configs is None:
            # Create default configs for N agents with different hyperparameters
            self.agent_configs = []
            for i in range(self.num_agents):
                # Vary hyperparameters slightly for each agent
                lr_base = 3e-4
                lr_factor = 0.5 ** i  # Decrease learning rate for each agent
                gamma_base = 0.99 - (i * 0.02)  # Slightly different discount factors
                
                config = AgentConfig(
                    num_envs=512,
                    rollout_length=64,
                    learning_rate=lr_base * lr_factor,
                    gamma=max(0.90, gamma_base),  # Keep gamma reasonable
                    gae_lambda=0.95 + (i * 0.01),  # Slightly different GAE lambda
                    batch_size=256,
                    hidden_dim=256,
                    # device_rank=0   # current support for only 1 GPU
                )
                self.agent_configs.append(config)
    
    # Trajectory filtering settings
    trajectory_filter_timestep: int = 10_000
    """Timestep after which to start filtering trajectories."""
    top_k_trajectories: int = 512
    """Number of top trajectories to keep for state indexing."""
    
    # FAISS settings
    faiss_index_type: str = "Flat"
    """Type of FAISS index to use."""
    faiss_use_gpu: bool = False
    """Whether to use GPU for FAISS operations."""
    
    # Logging and evaluation
    log_interval: int = 10000
    """Interval for logging."""
    eval_interval: int = 100000
    """Interval for evaluation."""
    num_eval_envs: int = 32
    """Number of evaluation environments per agent."""
    
    # Output settings
    output_dir: str = "multi_agent_logs"
    """Directory for saving logs and models."""
    save_interval: int = 100000
    """Interval for saving models."""
    
    def get_experiment_dir(self) -> str:
        """Generate timestamped experiment directory."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        experiment_dir = os.path.join("experiments", date_str, time_str)
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    # Wandb settings
    use_wandb: bool = False
    """Whether to use Weights & Biases logging."""
    project: str = "rl_scratch"
    """Wandb project name."""
    
    # Logging settings
    verbose: bool = False
    """Enable verbose logging output."""
    
    # Performance settings
    compile: bool = False
    """Use torch.compile for optimization. Disabled for multi-agent to avoid threading conflicts."""
    amp: bool = False
    """Use automatic mixed precision."""
    amp_dtype: str = "bf16"
    """AMP dtype (bf16 or fp16)."""


def get_multi_agent_config() -> MultiAgentConfig:
    """Parse command line arguments and return MultiAgentConfig."""
    base_config = tyro.cli(MultiAgentConfig)
    if base_config.env_name.startswith("h1hand-") or base_config.env_name.startswith("h1-"):
        return tyro.cli(HumanoidBenchMultiAgentConfig)
    elif base_config.env_name.startswith("Isaac-"):
        return tyro.cli(IsaacLabMultiAgentConfig)
    elif base_config.env_name.startswith(("G1", "T1", "Go1", "Leap")):
        return tyro.cli(MuJoCoPlaygroundMultiAgentConfig)
    else:
        # Return base config for unknown environments
        return base_config

@dataclass
class HumanoidBenchMultiAgentConfig(MultiAgentConfig):
    """Default configuration for HumanoidBench environments."""
    total_timesteps: int = 1000_000
    main_agent_timesteps: int = 1000_000
    trajectory_filter_timestep: int = 10_000
    top_k_trajectories: int = 32
    num_agents: int = 1
    
    def __post_init__(self):
        # Create HumanoidBench-specific agent configs
        if self.agent_configs is None:
            self.agent_configs = []
            for i in range(self.num_agents):
                config = AgentConfig(
                    num_envs=32,
                    rollout_length=32,
                    learning_rate=3e-4 * (0.8 ** i),
                    gamma=0.99,
                    gae_lambda=0.95,
                    batch_size=128,
                    hidden_dim=256,
                )
                self.agent_configs.append(config)
        
        # Update main agent config for HumanoidBench
        self.main_agent_config = AgentConfig(
            num_envs=128,
            rollout_length=64,
            learning_rate=2e-4,
            gamma=0.99,
            gae_lambda=0.95,
            batch_size=256,
            hidden_dim=256
        )


@dataclass
class MuJoCoPlaygroundMultiAgentConfig(MultiAgentConfig):
    """Default configuration for MuJoCo Playground environments."""
    total_timesteps: int = 10_000_000
    main_agent_timesteps: int = 10_000_000
    trajectory_filter_timestep: int = 10_000
    top_k_trajectories: int = 128
    num_agents:int = 4
    
    def __post_init__(self):
        if self.agent_configs is None:
            self.agent_configs = []
            for i in range(self.num_agents):
                config = AgentConfig(
                    num_envs=1024,
                    rollout_length=32,
                    learning_rate=3e-4 * (0.7 ** i),
                    gamma=0.97,  # Lower gamma for MJX environments
                    gae_lambda=0.95,
                    batch_size=512,
                    hidden_dim=256,
                )
                self.agent_configs.append(config)
        
        self.main_agent_config = AgentConfig(
            num_envs=2048,
            rollout_length=64,
            learning_rate=2e-4,
            gamma=0.97,
            gae_lambda=0.95,
            batch_size=512,
            hidden_dim=256
        )




@dataclass
class IsaacLabMultiAgentConfig(MultiAgentConfig):
    """Default configuration for IsaacLab environments."""
    total_timesteps: int = 100_000
    main_agent_timesteps: int = 100_000
    trajectory_filter_timestep: int = 20_000
    top_k_trajectories: int = 256
    
    def __post_init__(self):
        if self.agent_configs is None:
            self.agent_configs = []
            for i in range(self.num_agents):
                config = AgentConfig(
                    num_envs=2048,
                    rollout_length=32,
                    learning_rate=3e-4 * (0.8 ** i),
                    gamma=0.99,
                    gae_lambda=0.95,
                    batch_size=512,
                    hidden_dim=256,
                )
                self.agent_configs.append(config)
        
        self.main_agent_config = AgentConfig(
            num_envs=4096,
            rollout_length=32,
            learning_rate=2e-4,
            gamma=0.99,
            gae_lambda=0.95,
            batch_size=512,
            hidden_dim=256
        )


def validate_config(config: MultiAgentConfig) -> None:
    """Validate the multi-agent configuration."""
    
    # Validate timestep settings
    if config.trajectory_filter_timestep >= config.total_timesteps:
        raise ValueError(
            f"trajectory_filter_timestep ({config.trajectory_filter_timestep}) "
            f"must be less than total_timesteps ({config.total_timesteps})"
        )
    
    if config.trajectory_filter_timestep >= config.main_agent_timesteps:
        raise ValueError(
            f"trajectory_filter_timestep ({config.trajectory_filter_timestep}) "
            f"must be less than main_agent_timesteps ({config.main_agent_timesteps})"
        )
    
    if config.main_agent_timesteps <= 0:
        raise ValueError(f"main_agent_timesteps must be positive, got {config.main_agent_timesteps}")
    
    if config.total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive, got {config.total_timesteps}")
    
    # Validate top_k setting
    if config.top_k_trajectories <= 0:
        raise ValueError(f"top_k_trajectories must be positive, got {config.top_k_trajectories}")
    
    # Validate number of agents
    if config.num_agents <= 0:
        raise ValueError(f"num_agents must be positive, got {config.num_agents}")
    
    if len(config.agent_configs) != config.num_agents:
        raise ValueError(f"Number of agent_configs ({len(config.agent_configs)}) must match num_agents ({config.num_agents})")
    
    # Validate agent configurations
    all_agent_configs = [(f"agent_{i+1}", config.agent_configs[i]) for i in range(config.num_agents)]
    all_agent_configs.append(("main_agent", config.main_agent_config))
    
    for agent_name, agent_config in all_agent_configs:
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


def save_config_to_experiment_dir(config: MultiAgentConfig, experiment_dir: str) -> str:
    """Save configuration to experiment directory as JSON."""
    import json
    from dataclasses import asdict
    
    config_path = os.path.join(experiment_dir, "config.json")
    config_dict = asdict(config)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return config_path


if __name__ == "__main__":
    config = get_multi_agent_config()
    validate_config(config)
    print("Configuration:", config)
