"""Multi-agent PPO training with independent agents and FAISS state storage."""

import os
import sys
import time
import threading
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
import logging

from .multi_agent_config import MultiAgentConfig, AgentConfig, get_multi_agent_config, validate_config, save_config_to_experiment_dir
from .spatial_coverage import visualize_faiss_states_by_agent
from .faiss_state_storage import FAISSStateStorage
from fast_td3.environments.mujoco_playground_env import make_env
from fast_td3.fast_td3_utils import EmpiricalNormalization
from .ppo import ActorCritic
from .ppo_utils import RolloutBuffer

# Environment setup
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import torch._dynamo
torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


@dataclass
class TrajectoryInfo:
    """Information about a trajectory for filtering."""
    episode_id: int
    total_reward: float
    final_state: np.ndarray
    episode_length: int
    timestamp: float


class IndependentAgent:
    """Independent PPO agent that shares FAISS state storage."""
    
    def __init__(self, 
                 agent_id: str,
                 agent_config: AgentConfig,
                 global_config: MultiAgentConfig,
                 device: torch.device,
                 shared_faiss_storage: FAISSStateStorage,
                 total_timesteps: Optional[int] = None):
        self.agent_id = agent_id
        self.config = agent_config
        self.global_config = global_config
        self.device = device
        self.total_timesteps = total_timesteps if total_timesteps is not None else global_config.total_timesteps
        
        # Set up logging for this agent
        self.logger = logging.getLogger(f"Agent_{agent_id}")
        
        # Set up random seed for this agent
        seed = global_config.seed + hash(agent_id) % 1000
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.logger.info(f"Initializing with seed {seed}")
        
        # Create environments
        self.envs, _, _ = make_env(
            global_config.env_name,
            seed=seed,
            num_envs=agent_config.num_envs,
            num_eval_envs=1,
            device_rank=0,
        )
        
        # Create evaluation environments
        self.eval_envs, _, _ = make_env(
            global_config.env_name,
            seed=seed + 1000,
            num_envs=global_config.num_eval_envs,
            num_eval_envs=1,
            device_rank=0,
        )
        
        # Get environment dimensions
        self.envs.reset()
        self.n_obs = self.envs.num_obs if isinstance(self.envs.num_obs, int) else self.envs.num_obs[0]
        self.n_act = self.envs.num_actions
        
        # Initialize networks
        self.policy = ActorCritic(self.n_obs, self.n_act, agent_config.hidden_dim, device=device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=agent_config.learning_rate)
        self.normalizer = EmpiricalNormalization(shape=self.n_obs, device=device)
        
        # Use shared FAISS storage
        self.faiss_storage = shared_faiss_storage
        
        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            agent_config.rollout_length, 
            agent_config.num_envs, 
            self.n_obs, 
            self.n_act, 
            device=device
        )
        
        # AMP setup
        self.amp_enabled = global_config.amp and torch.cuda.is_available()
        self.amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp_dtype = torch.bfloat16 if global_config.amp_dtype == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        
        # Compiled functions
        if global_config.compile:
            self.policy_act = torch.compile(self.policy.act)
            self.policy_value = torch.compile(self.policy.value)
            self.policy_get_dist = torch.compile(self.policy.get_dist)
            self.normalize_obs = torch.compile(self.normalizer.forward)
        else:
            self.policy_act = self.policy.act
            self.policy_value = self.policy.value
            self.policy_get_dist = self.policy.get_dist
            self.normalize_obs = self.normalizer.forward
        
        # Training state
        self.global_step = 0
        self.obs = self.envs.reset()
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.current_episode_reward = torch.zeros(agent_config.num_envs, device=device)
        self.current_episode_length = torch.zeros(agent_config.num_envs, device=device)
        self.num_episodes = 0
        
        # Trajectory filtering
        self.trajectory_buffer: List[TrajectoryInfo] = []
        self.filtering_active = False
        self.last_states_buffer: List[np.ndarray] = []
        
        # Thread control and synchronization
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.faiss_lock = threading.Lock()  # For thread-safe FAISS operations
        
        # Metrics
        self.total_policy_loss = 0.0
        self.total_value_loss = 0.0
        self.total_entropy = 0.0
        self.num_updates = 0
        
        # Evaluation metrics
        self.latest_eval_reward = 0.0
        self.latest_eval_length = 0.0
        self.eval_count = 0
        
        self.logger.info(f"Initialized with {agent_config.num_envs} environments")
        self.logger.debug(f"Config: rollout_length={agent_config.rollout_length}, "
                         f"lr={agent_config.learning_rate}, gamma={agent_config.gamma}, "
                         f"gae_lambda={agent_config.gae_lambda}")
    
    def _collect_rollout(self) -> Dict[str, Any]:
        """Collect a single rollout."""
        rollout_start_time = time.time()
        
        # Clear buffer
        self.buffer.clear()
        
        # Store states for potential trajectory filtering
        episode_states = {i: [] for i in range(self.config.num_envs)}
        episode_rewards_temp = {i: 0.0 for i in range(self.config.num_envs)}
        
        # Collect rollout
        for _ in range(self.config.rollout_length):
            with torch.no_grad(), autocast(
                device_type=self.amp_device_type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                norm_obs = self.normalize_obs(self.obs)
                action, logp, value = self.policy_act(norm_obs)
            
            # Store current states
            if isinstance(self.obs, torch.Tensor):
                current_states = self.obs.cpu().numpy()
            else:
                current_states = np.array(self.obs)
            
            for env_idx in range(self.config.num_envs):
                episode_states[env_idx].append(current_states[env_idx])
            
            next_obs, reward, done, _ = self.envs.step(action)
            self.buffer.add(self.obs, action, logp, reward, done, value)
            
            self.obs = next_obs
            self.global_step += self.config.num_envs
            
            # Track episode statistics
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            # Update temporary episode rewards
            for env_idx in range(self.config.num_envs):
                episode_rewards_temp[env_idx] += reward[env_idx].item()
            
            # Check for episode completions
            if done.any():
                for env_idx in range(self.config.num_envs):
                    if done[env_idx]:
                        # Record episode completion
                        episode_reward = self.current_episode_reward[env_idx].item()
                        episode_length = self.current_episode_length[env_idx].item()
                        
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.num_episodes += 1
                        
                        # Store trajectory info for filtering
                        if self.filtering_active and len(episode_states[env_idx]) > 0:
                            final_state = episode_states[env_idx][-1]  # Last state
                            trajectory_info = TrajectoryInfo(
                                episode_id=self.num_episodes,
                                total_reward=episode_reward,
                                final_state=final_state,
                                episode_length=int(episode_length),
                                timestamp=time.time()
                            )
                            self.trajectory_buffer.append(trajectory_info)
                        
                        # Reset episode tracking
                        self.current_episode_reward[env_idx] = 0
                        self.current_episode_length[env_idx] = 0
                        episode_states[env_idx] = []
                        episode_rewards_temp[env_idx] = 0.0
        
        # Compute advantages
        with torch.no_grad(), autocast(
            device_type=self.amp_device_type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            last_values = self.policy_value(self.normalize_obs(self.obs))
        
        self.buffer.compute_returns_and_advantage(
            last_values, self.config.gamma, self.config.gae_lambda, self.config.num_envs
        )
        
        rollout_time = time.time() - rollout_start_time
        
        return {
            "rollout_time": rollout_time,
            "steps_collected": self.config.num_envs * self.config.rollout_length,
        }
    
    def _update_policy(self) -> Dict[str, float]:
        """Update policy using collected rollout."""
        update_start_time = time.time()
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_updates = 0
        
        for _ in range(self.config.update_epochs):
            for b_obs, b_actions, b_logp, b_returns, b_adv in self.buffer.get_batches(self.config.batch_size):
                with autocast(
                    device_type=self.amp_device_type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    dist = self.policy_get_dist(self.normalize_obs(b_obs))
                    new_logp = dist.log_prob(b_actions).sum(-1)
                    
                    ratio = (new_logp - b_logp).exp()
                    pg_loss1 = -b_adv * ratio
                    pg_loss2 = -b_adv * torch.clamp(
                        ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps
                    )
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    value = self.policy_value(self.normalize_obs(b_obs))
                    value_loss = F.mse_loss(value, b_returns)
                    entropy = dist.entropy().sum(-1).mean()
                    
                    loss = (
                        policy_loss
                        + self.config.vf_coef * value_loss
                        - self.config.ent_coef * entropy
                    )
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=self.config.max_grad_norm if self.config.max_grad_norm > 0 else float("inf"),
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                epoch_updates += 1
        
        update_time = time.time() - update_start_time
        
        # Update global metrics
        if epoch_updates > 0:
            avg_policy_loss = epoch_policy_loss / epoch_updates
            avg_value_loss = epoch_value_loss / epoch_updates
            avg_entropy = epoch_entropy / epoch_updates
            
            self.total_policy_loss += avg_policy_loss
            self.total_value_loss += avg_value_loss
            self.total_entropy += avg_entropy
            self.num_updates += 1
        
        return {
            "policy_loss": avg_policy_loss if epoch_updates > 0 else 0.0,
            "value_loss": avg_value_loss if epoch_updates > 0 else 0.0,
            "entropy": avg_entropy if epoch_updates > 0 else 0.0,
            "update_time": update_time,
            "grad_norm": grad_norm.item() if epoch_updates > 0 else 0.0,
        }
    
    def _process_trajectory_filtering(self):
        """Process trajectory filtering and add top-k states to FAISS (only for sub-agents)."""
        if len(self.trajectory_buffer) < self.global_config.top_k_trajectories:
            return
        
        # Only add states from sub-agents (Agent 1 and Agent 2), not main agent
        if self.agent_id == "main":
            # Main agent doesn't add states to FAISS, only consumes them
            self.trajectory_buffer.clear()
            return
        
        # Sort trajectories by reward (descending)
        self.trajectory_buffer.sort(key=lambda x: x.total_reward, reverse=True)
        
        # Take top-k trajectories
        top_k_trajectories = self.trajectory_buffer[:self.global_config.top_k_trajectories]
        
        # Add final states to shared FAISS storage (thread-safe)
        with self.faiss_lock:
            for traj in top_k_trajectories:
                self.faiss_storage.add_state(
                    state=traj.final_state,
                    episode_id=traj.episode_id,
                    step_id=traj.episode_length,
                    reward=traj.total_reward,
                    done=True,
                    metadata={
                        "agent_id": self.agent_id,
                        "timestamp": traj.timestamp,
                        "episode_length": traj.episode_length
                    }
                )
        
        # Clear trajectory buffer
        self.trajectory_buffer.clear()
        
        reward_list = [f'{t.total_reward:.2f}' for t in top_k_trajectories[:3]]  # Show first 3
        self.logger.debug(f"Added {len(top_k_trajectories)} top trajectories to FAISS "
                         f"(sample rewards: {reward_list})")
    
    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate the current policy."""
        self.normalizer.eval()
        
        episode_returns = torch.zeros(self.global_config.num_eval_envs, device=self.device)
        episode_lengths = torch.zeros(self.global_config.num_eval_envs, device=self.device)
        done_masks = torch.zeros(self.global_config.num_eval_envs, dtype=torch.bool, device=self.device)
        
        obs = self.eval_envs.reset()
        
        max_steps = 1000
        for _ in range(max_steps):
            with torch.no_grad(), autocast(
                device_type=self.amp_device_type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                norm_obs = self.normalize_obs(obs)
                action, _, _ = self.policy_act(norm_obs)
            
            next_obs, rewards, dones, _ = self.eval_envs.step(action)
            episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = torch.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = torch.logical_or(done_masks, dones)
            
            if done_masks.all():
                break
            obs = next_obs
        
        self.normalizer.train()
        return episode_returns.mean().item(), episode_lengths.mean().item()
    
    def train(self):
        """Main training loop for this agent."""
        self.logger.info("Starting training loop")
        
        last_log_time = time.time()
        last_eval_time = time.time()
        
        while self.running and self.global_step < self.total_timesteps:
            # Check if we should start filtering
            if not self.filtering_active and self.global_step >= self.global_config.trajectory_filter_timestep:
                self.filtering_active = True
                self.logger.info(f"Started trajectory filtering at step {self.global_step}")
            
            # Collect rollout
            self._collect_rollout()
            
            # Update policy
            update_metrics = self._update_policy()
            
            # Process trajectory filtering
            if self.filtering_active:
                self._process_trajectory_filtering()
            
            # Logging
            current_time = time.time()
            if current_time - last_log_time >= 30.0:  # Log every 30 seconds (less frequent)
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
                avg_length = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0
                progress = (self.global_step / self.total_timesteps) * 100
                
                self.logger.debug(f"Step {self.global_step:,} ({progress:.1f}%) | "
                                 f"Reward: {avg_reward:.2f} | Episodes: {self.num_episodes} | "
                                 f"Policy Loss: {update_metrics['policy_loss']:.6f}")
                
                last_log_time = current_time
            
            # Evaluation
            if current_time - last_eval_time >= 30.0:  # Evaluate every 30 seconds
                eval_return, eval_length = self._evaluate()
                self.latest_eval_reward = eval_return
                self.latest_eval_length = eval_length
                self.eval_count += 1
                self.logger.debug(f"Evaluation #{self.eval_count}: Return {eval_return:.2f}, Length {eval_length:.1f}")
                last_eval_time = current_time
        
        self.logger.info(f"Training completed at step {self.global_step}")
        
        # Log final FAISS statistics (storage is shared, so don't save individually)
        stats = self.faiss_storage.get_performance_stats()
        self.logger.debug(f"Final shared FAISS stats: {stats['total_states']} states, "
                         f"avg search time: {stats.get('avg_search_time', 0)*1000:.2f}ms")
    
    def start_training(self):
        """Start training in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.train, daemon=True)
        self.thread.start()
    
    def stop_training(self):
        """Stop training."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            "agent_id": self.agent_id,
            "global_step": self.global_step,
            "num_episodes": self.num_episodes,
            "avg_reward": self.latest_eval_reward,  # Use evaluation reward instead of training
            "training_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
            "eval_reward": self.latest_eval_reward,
            "eval_length": self.latest_eval_length,
            "eval_count": self.eval_count,
            "avg_length": sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0,
            "filtering_active": self.filtering_active,
            "faiss_states": len(self.faiss_storage.state_info),
            "trajectories_in_buffer": len(self.trajectory_buffer),
            "total_policy_loss": self.total_policy_loss / max(1, self.num_updates),
            "total_value_loss": self.total_value_loss / max(1, self.num_updates),
            "total_entropy": self.total_entropy / max(1, self.num_updates),
        }


class MainAgent(IndependentAgent):
    """Main agent that initializes half of its environments from FAISS states."""
    
    def __init__(self, 
                 agent_config: AgentConfig,
                 global_config: MultiAgentConfig,
                 device: torch.device,
                 shared_faiss_storage: FAISSStateStorage):
        super().__init__("main", agent_config, global_config, device, shared_faiss_storage, global_config.main_agent_timesteps)
        
        # Additional initialization for state sampling
        self.faiss_init_ratio = 0.5  # Use 50% FAISS states, 50% random
        self.min_faiss_states_required = max(1, int(agent_config.num_envs * self.faiss_init_ratio))
        self.state_initialization_active = False
        
        # Track sampled FAISS state IDs for removal (using sets for O(1) operations)
        self.sampled_state_ids = set()
        self.removal_batch_size = 50  # Remove states in batches
        
        # Mapping from FAISS index to actual state info index (for efficient removal)
        self.faiss_to_state_mapping = {}  # faiss_id -> state_info_index
        self.next_faiss_id = 0  # Track next available FAISS ID
        
        self.logger.info(f"Initialized with {agent_config.num_envs} environments")
        self.logger.info(f"Will use {self.faiss_init_ratio*100:.0f}% FAISS states for initialization when available")
    
    def _initialize_environments_with_faiss_states(self) -> bool:
        """Initialize environments using mix of random and FAISS states."""
        # Check if we have enough FAISS states
        num_faiss_states = len(self.faiss_storage.state_info)
        if num_faiss_states < self.min_faiss_states_required:
            return False
        
        self.logger.debug(f"Initializing environments with {num_faiss_states} FAISS states available")
        
        # Reset environments normally first
        self.obs = self.envs.reset()
        
        # Determine how many environments to initialize from FAISS
        num_faiss_envs = min(
            int(self.config.num_envs * self.faiss_init_ratio),
            num_faiss_states
        )
        
        if num_faiss_envs == 0:
            return False
        
        # Randomly select which environments to initialize from FAISS
        faiss_env_indices = random.sample(range(self.config.num_envs), num_faiss_envs)
        
        # Randomly select FAISS states to use
        with self.faiss_lock:
            available_state_indices = np.arange(len(self.faiss_storage.state_info))
            selected_faiss_indices = np.random.choice(available_state_indices, size=num_faiss_envs, replace=False)
        
        # Apply FAISS states to selected environments (vectorized operations)
        try:
            # Get selected state vectors in batch
            selected_states = [self.faiss_storage.state_info[idx].state_vector for idx in selected_faiss_indices]
            state_array = np.array(selected_states)
            
            # Convert to tensor batch
            states_tensor = torch.from_numpy(state_array).to(self.device)
            
            # Set observations for selected environments (vectorized assignment)
            faiss_env_indices = np.array(faiss_env_indices)
            self.obs[faiss_env_indices] = states_tensor
            
            # Track sampled state IDs for removal (set operations are O(1))
            self.sampled_state_ids.update(selected_faiss_indices)
            
            # Get sample rewards for logging (vectorized)
            sample_rewards = [self.faiss_storage.state_info[idx].reward for idx in selected_faiss_indices[:3]]
            reward_sample = [f"{r:.2f}" for r in sample_rewards]
            
            self.logger.debug(f"Initialized {num_faiss_envs}/{self.config.num_envs} environments from FAISS states")
            self.logger.debug(f"Used FAISS states with sample rewards: {reward_sample}")
            
            # Remove sampled states if we have enough for a batch
            self._remove_sampled_states_if_needed()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize from FAISS states: {e}")
            # Fall back to normal reset
            self.obs = self.envs.reset()
            return False
    
    def _remove_sampled_states_if_needed(self):
        """Remove sampled states from FAISS storage efficiently when batch size is reached."""
        if len(self.sampled_state_ids) >= self.removal_batch_size:
            try:
                # Convert set to numpy array for efficient operations
                ids_to_remove = np.array(list(self.sampled_state_ids), dtype=np.int64)
                
                if len(ids_to_remove) > 0:
                    # Remove from FAISS index using remove_ids (vectorized operation)
                    self.faiss_storage.index.remove_ids(ids_to_remove)
                    
                    # Create boolean mask for state_info removal (vectorized)
                    state_info_len = len(self.faiss_storage.state_info)
                    keep_mask = np.ones(state_info_len, dtype=bool)
                    valid_ids = ids_to_remove[ids_to_remove < state_info_len]
                    keep_mask[valid_ids] = False
                    
                    # Filter state_info using list comprehension (faster than for loop)
                    self.faiss_storage.state_info = [
                        state for i, state in enumerate(self.faiss_storage.state_info) 
                        if keep_mask[i]
                    ]
                    
                    # Update episode_states mapping efficiently
                    episodes_to_remove = set()
                    for episode_id, state_list in self.faiss_storage.episode_states.items():
                        # Remove using set difference (O(n) instead of O(n^2))
                        remaining_states = set(state_list) - self.sampled_state_ids
                        if remaining_states:
                            self.faiss_storage.episode_states[episode_id] = list(remaining_states)
                        else:
                            episodes_to_remove.add(episode_id)
                    
                    # Remove empty episodes
                    for episode_id in episodes_to_remove:
                        del self.faiss_storage.episode_states[episode_id]
                    
                    self.logger.debug(f"Removed {len(ids_to_remove)} sampled states from FAISS storage")
                
                # Clear the sampled states set
                self.sampled_state_ids.clear()
                
            except Exception as e:
                self.logger.warning(f"Failed to remove sampled states: {e}")
                # Clear the set anyway to prevent memory issues
                self.sampled_state_ids.clear()
    
    def _collect_rollout(self) -> Dict[str, Any]:
        """Collect rollout with potential FAISS state initialization."""
        
        # Check if we should use FAISS initialization based on timestep threshold
        if not self.state_initialization_active:
            # Only start using FAISS after trajectory_filter_timestep
            if self.global_step >= self.global_config.trajectory_filter_timestep:
                # Check if we have enough FAISS states to start using them
                num_faiss_states = len(self.faiss_storage.state_info)
                if num_faiss_states >= self.min_faiss_states_required:
                    self.state_initialization_active = True
                    self.logger.info(f"Activating FAISS state initialization at step {self.global_step} "
                                    f"with {num_faiss_states} states (filter threshold: {self.global_config.trajectory_filter_timestep})")
                else:
                    # Log that we're waiting for FAISS states
                    if self.global_step % 50000 == 0:  # Log occasionally to avoid spam
                        self.logger.debug(f"Waiting for FAISS states: have {num_faiss_states}, "
                                         f"need {self.min_faiss_states_required} (step {self.global_step})")
            else:
                # Before filter timestep, just use random initialization
                if self.global_step % 100000 == 0:  # Log less frequently
                    remaining_steps = self.global_config.trajectory_filter_timestep - self.global_step
                    self.logger.debug(f"Using random initialization until step {self.global_config.trajectory_filter_timestep} "
                                     f"({remaining_steps} steps remaining)")
        
        # Initialize environments (potentially with FAISS states)
        if self.state_initialization_active:
            # Randomly decide whether to use FAISS initialization this rollout
            use_faiss_init = random.random() < 0.3  # 30% chance per rollout
            if use_faiss_init:
                success = self._initialize_environments_with_faiss_states()
                if success:
                    self.logger.debug("Using FAISS initialization for this rollout")
        
        # Proceed with normal rollout collection
        return super()._collect_rollout()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics including FAISS initialization status."""
        stats = super().get_stats()
        stats.update({
            "state_initialization_active": self.state_initialization_active,
            "min_faiss_states_required": self.min_faiss_states_required,
            "faiss_init_ratio": self.faiss_init_ratio,
            "past_filter_timestep": self.global_step >= self.global_config.trajectory_filter_timestep,
            "filter_timestep_threshold": self.global_config.trajectory_filter_timestep,
        })
        return stats


def main():
    """Main function to run multi-agent training."""
    config = get_multi_agent_config()
    validate_config(config)
    
    # Create timestamped experiment directory
    experiment_dir = config.get_experiment_dir()
    
    # Save configuration to experiment directory
    config_path = save_config_to_experiment_dir(config, experiment_dir)
    
    # Setup logging with experiment directory
    log_file = f"{experiment_dir}/training.log"
    logger = setup_logging(verbose=config.verbose, log_file=log_file)
    
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Configuration saved to: {config_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Training {config.env_name}")
    logger.info(f"  Agents 1 & 2: {config.total_timesteps:,} timesteps")
    logger.info(f"  Main Agent: {config.main_agent_timesteps:,} timesteps")
    logger.info(f"Trajectory filtering starts at step {config.trajectory_filter_timestep}")
    logger.info(f"Top-k trajectories: {config.top_k_trajectories}")
    logger.info(f"Verbose logging: {config.verbose}")
    
    # Experiment directory already created by config.get_experiment_dir()
    
    # Create shared FAISS storage
    # Use observation dimension from a temporary environment to initialize FAISS
    temp_env, _, _ = make_env(config.env_name, seed=config.seed, num_envs=1, num_eval_envs=1, device_rank=0)
    temp_env.reset()
    obs_dim = temp_env.num_obs if isinstance(temp_env.num_obs, int) else temp_env.num_obs[0]
    
    shared_faiss_storage = FAISSStateStorage(
        state_dim=obs_dim,
        index_type=config.faiss_index_type,
        use_gpu=config.faiss_use_gpu and torch.cuda.is_available()
    )
    logger.info(f"Created shared FAISS storage with state dimension {obs_dim}")
    
    # Create agents with shared FAISS storage and specific timestep limits
    agent_1 = IndependentAgent("1", config.agent_1_config, config, device, shared_faiss_storage, config.total_timesteps)
    agent_2 = IndependentAgent("2", config.agent_2_config, config, device, shared_faiss_storage, config.total_timesteps)
    main_agent = MainAgent(config.main_agent_config, config, device, shared_faiss_storage)
    
    # Initialize wandb if requested
    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.project,
            name=f"multi_agent_{config.env_name}_{config.seed}",
            config=config.__dict__,
            save_code=True,
        )
    
    # Start training
    logger.info("Starting multi-agent training...")
    start_time = time.time()
    
    agent_1.start_training()
    agent_2.start_training()
    main_agent.start_training()
    
    try:
        # Monitor training
        while agent_1.running or agent_2.running or main_agent.running:
            time.sleep(30)  # Check every 30 seconds
            
            # Get stats from all agents
            stats_1 = agent_1.get_stats()
            stats_2 = agent_2.get_stats()
            stats_main = main_agent.get_stats()
            
            # Check if all agents are done (different timestep limits)
            agent_1_done = stats_1["global_step"] >= config.total_timesteps
            agent_2_done = stats_2["global_step"] >= config.total_timesteps
            main_agent_done = stats_main["global_step"] >= config.main_agent_timesteps
            
            if agent_1_done and agent_2_done and main_agent_done:
                break
            
            # Get shared FAISS stats
            shared_faiss_stats = shared_faiss_storage.get_performance_stats()
            total_faiss_states = shared_faiss_stats['total_states']
            
            # Calculate progress for each agent
            agent_1_progress = (stats_1['global_step'] / config.total_timesteps) * 100
            agent_2_progress = (stats_2['global_step'] / config.total_timesteps) * 100  
            main_agent_progress = (stats_main['global_step'] / config.main_agent_timesteps) * 100
            
            # Simple progress display (non-verbose) - showing evaluation reward
            print(f"Main Agent: {stats_main['global_step']:,}/{config.main_agent_timesteps:,} ({main_agent_progress:.1f}%) | "
                  f"Eval Reward: {stats_main['eval_reward']:.2f} ({stats_main['eval_count']} evals) | FAISS: {total_faiss_states} states")
            
            # Verbose detailed stats
            if config.verbose:
                logger.info("="*80)
                logger.info("MULTI-AGENT TRAINING PROGRESS")
                logger.info("="*80)
                logger.info(f"Agent 1: Step {stats_1['global_step']:,}/{config.total_timesteps:,} ({agent_1_progress:.1f}%) | "
                           f"Eval: {stats_1['eval_reward']:.2f} | Train: {stats_1['training_reward']:.2f} | Episodes: {stats_1['num_episodes']} | "
                           f"Filtering: {stats_1['filtering_active']}")
                logger.info(f"Agent 2: Step {stats_2['global_step']:,}/{config.total_timesteps:,} ({agent_2_progress:.1f}%) | "
                           f"Eval: {stats_2['eval_reward']:.2f} | Train: {stats_2['training_reward']:.2f} | Episodes: {stats_2['num_episodes']} | "
                           f"Filtering: {stats_2['filtering_active']}")
                logger.info(f"Main Agent: Step {stats_main['global_step']:,}/{config.main_agent_timesteps:,} ({main_agent_progress:.1f}%) | "
                           f"Eval: {stats_main['eval_reward']:.2f} | Train: {stats_main['training_reward']:.2f} | Episodes: {stats_main['num_episodes']} | "
                           f"FAISS Init: {stats_main['state_initialization_active']}")
                logger.info(f"Shared FAISS: {total_faiss_states} total states stored")
            
            # Log to wandb (always log, regardless of verbose setting)
            if config.use_wandb:
                wandb_metrics = {
                    # Agent 1 metrics
                    "agent_1/eval_reward": stats_1['eval_reward'],
                    "agent_1/training_reward": stats_1['training_reward'],
                    "agent_1/eval_count": stats_1['eval_count'],
                    "agent_1/episodes": stats_1['num_episodes'],
                    "agent_1/global_step": stats_1['global_step'],
                    "agent_1/progress": agent_1_progress,
                    "agent_1/filtering_active": stats_1['filtering_active'],
                    "agent_1/policy_loss": stats_1['total_policy_loss'],
                    "agent_1/value_loss": stats_1['total_value_loss'],
                    "agent_1/entropy": stats_1['total_entropy'],
                    
                    # Agent 2 metrics
                    "agent_2/eval_reward": stats_2['eval_reward'],
                    "agent_2/training_reward": stats_2['training_reward'],
                    "agent_2/eval_count": stats_2['eval_count'],
                    "agent_2/episodes": stats_2['num_episodes'],
                    "agent_2/global_step": stats_2['global_step'],
                    "agent_2/progress": agent_2_progress,
                    "agent_2/filtering_active": stats_2['filtering_active'],
                    "agent_2/policy_loss": stats_2['total_policy_loss'],
                    "agent_2/value_loss": stats_2['total_value_loss'],
                    "agent_2/entropy": stats_2['total_entropy'],
                    
                    # Main agent metrics
                    "main_agent/eval_reward": stats_main['eval_reward'],
                    "main_agent/training_reward": stats_main['training_reward'],
                    "main_agent/eval_count": stats_main['eval_count'],
                    "main_agent/episodes": stats_main['num_episodes'],
                    "main_agent/global_step": stats_main['global_step'],
                    "main_agent/progress": main_agent_progress,
                    "main_agent/faiss_init_active": stats_main['state_initialization_active'],
                    "main_agent/policy_loss": stats_main['total_policy_loss'],
                    "main_agent/value_loss": stats_main['total_value_loss'],
                    "main_agent/entropy": stats_main['total_entropy'],
                    "main_agent/past_filter_timestep": stats_main.get('past_filter_timestep', False),
                    
                    # Shared FAISS metrics
                    "shared/faiss_states": total_faiss_states,
                    "shared/faiss_episodes": shared_faiss_stats.get('total_episodes', 0),
                    
                    # Combined metrics
                    "combined/total_steps": stats_1['global_step'] + stats_2['global_step'] + stats_main['global_step'],
                    "combined/total_episodes": stats_1['num_episodes'] + stats_2['num_episodes'] + stats_main['num_episodes'],
                    "combined/avg_eval_reward": (stats_1['eval_reward'] + stats_2['eval_reward'] + stats_main['eval_reward']) / 3,
                    "combined/avg_training_reward": (stats_1['training_reward'] + stats_2['training_reward'] + stats_main['training_reward']) / 3,
                }
                
                wandb.log(wandb_metrics)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    finally:
        # Stop agents
        logger.info("Stopping agents...")
        agent_1.stop_training()
        agent_2.stop_training()
        main_agent.stop_training()
        
        # Final statistics
        total_time = time.time() - start_time
        final_stats_1 = agent_1.get_stats()
        final_stats_2 = agent_2.get_stats()
        final_stats_main = main_agent.get_stats()
        final_shared_stats = shared_faiss_storage.get_performance_stats()
        
        # Simple final summary
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Main Agent final eval reward: {final_stats_main['eval_reward']:.2f} ({final_stats_main['eval_count']} evaluations)")
        print(f"Total FAISS states: {final_shared_stats['total_states']}")
        
        # Verbose final statistics
        if config.verbose:
            logger.info("="*80)
            logger.info("FINAL MULTI-AGENT TRAINING RESULTS")
            logger.info("="*80)
            logger.info(f"Total training time: {total_time:.1f}s")
            logger.info(f"Agent 1 final stats: {final_stats_1}")
            logger.info(f"Agent 2 final stats: {final_stats_2}")
            logger.info(f"Main Agent final stats: {final_stats_main}")
            logger.info(f"Shared FAISS final stats: {final_shared_stats['total_states']} total states")
        
        # Save shared FAISS storage to experiment directory
        faiss_save_path = f"{experiment_dir}/shared_faiss_storage"
        shared_faiss_storage.save(faiss_save_path)
        logger.info(f"Saved shared FAISS storage to {faiss_save_path}")
        
        # Generate spatial coverage visualizations
        try:
            coverage_plot_paths = visualize_faiss_states_by_agent(
                shared_faiss_storage, 
                experiment_dir
            )
            if coverage_plot_paths:
                logger.info(f"Generated {len(coverage_plot_paths)} spatial coverage visualizations:")
                for path in coverage_plot_paths:
                    logger.info(f"  - {os.path.basename(path)}")
            else:
                logger.warning("No FAISS states available for spatial coverage visualization")
        except Exception as e:
            logger.error(f"Failed to generate spatial coverage visualization: {e}")
        
        if config.use_wandb:
            wandb.log({
                "final/total_time": total_time,
                "final/agent_1_steps": final_stats_1['global_step'],
                "final/agent_2_steps": final_stats_2['global_step'],
                "final/main_agent_steps": final_stats_main['global_step'],
                "final/main_agent_faiss_init_used": final_stats_main['state_initialization_active'],
                "final/shared_faiss_states": final_shared_stats['total_states'],
                "final/shared_faiss_episodes": final_shared_stats['total_episodes'],
            })
            wandb.finish()
        
        logger.info("Multi-agent training completed!")


if __name__ == "__main__":
    main()