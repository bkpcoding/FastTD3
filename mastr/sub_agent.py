
"""Multi-agent PPO training with independent agents and FAISS state storage."""

import os
import sys
import time
import threading
import multiprocessing as mp
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

from .hyperparameters import MultiAgentConfig, AgentConfig, get_multi_agent_config, validate_config, save_config_to_experiment_dir
from ppo.spatial_coverage import visualize_faiss_states_by_agent
from ppo.faiss_state_storage import FAISSStateStorage
from fast_td3.fast_td3_utils import EmpiricalNormalization
from ppo.ppo import ActorCritic
from ppo.ppo_utils import RolloutBuffer
from mastr.mastr_utils import TrajectoryInfo


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



class IndependentAgent:
    """Independent PPO agent that shares FAISS state storage."""
    
    def __init__(self, 
                 agent_id: str,
                 agent_config: AgentConfig,
                 global_config: MultiAgentConfig,
                 device: torch.device,
                 shared_faiss_storage: FAISSStateStorage,
                 total_timesteps: Optional[int] = None,
                 mjx_state_mapping: Optional[Dict[Any, Any]] = None):
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
        
        if self.global_config.env_name.startswith("h1hand-") or self.global_config.env_name.startswith("h1-") or self.global_config.env_name.startswith("h1hand-hurdle-"):
            from fast_td3.environments.humanoid_bench_env import HumanoidBenchEnv

            env_type = "humanoid_bench"
            envs = HumanoidBenchEnv(self.global_config.env_name, self.config.num_envs, device=device)
            eval_envs = HumanoidBenchEnv(
                self.global_config.env_name, self.global_config.num_eval_envs, device=device
            )
            render_env = HumanoidBenchEnv(
                self.global_config.env_name, 1, render_mode="rgb_array", device=device
            )
        elif self.global_config.env_name.startswith("Isaac-"):
            from fast_td3.environments.isaaclab_env import IsaacLabEnv

            env_type = "isaaclab"
            envs = IsaacLabEnv(
                self.global_config.env_name,
                device.type,
                self.config.num_envs,
                self.global_config.seed,
                action_bounds=self.global_config.action_bounds,
            )
            eval_envs = IsaacLabEnv(
                self.global_config.env_name,
                device.type,
                self.global_config.num_eval_envs,
                self.global_config.seed,
                action_bounds=self.global_config.action_bounds,
            )
            render_env = envs
        elif self.global_config.env_name.startswith("MTBench-"):
            from fast_td3.environments.mtbench_env import MTBenchEnv

            env_name = "-".join(self.global_config.env_name.split("-")[1:])
            env_type = "mtbench"
            envs = MTBenchEnv(env_name, self.config.device_rank, self.config.num_envs, self.global_config.seed)
            eval_envs = MTBenchEnv(env_name, self.config.device_rank, self.global_config.num_eval_envs, self.global_config.seed)
            render_env = envs
        else:
            from fast_td3.environments.mujoco_playground_env import make_env

            # TODO: Check if re-using same envs for eval could reduce memory usage
            env_type = "mujoco_playground"
            envs, eval_envs, render_env = make_env(
                self.global_config.env_name,
                self.global_config.seed,
                self.config.num_envs,
                self.global_config.num_eval_envs,
                self.config.device_rank,
                use_tuned_reward=self.global_config.use_tuned_reward,
                use_domain_randomization=self.global_config.use_domain_randomization,
                use_push_randomization=self.global_config.use_push_randomization,
            )
        self.envs = envs
        self.eval_envs = eval_envs
        self.render_env = render_env
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
        
        # MJX state mapping (observation -> full MJX state)
        self.mjx_state_mapping = mjx_state_mapping
        
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
        self.latest_eval_step = 0
        self.eval_interval = global_config.eval_interval
        self.eval_count = 0
        
        self.logger.info(f"Initialized with {agent_config.num_envs} environments")
        self.logger.debug(f"Config: rollout_length={agent_config.rollout_length}, "
                         f"lr={agent_config.learning_rate}, gamma={agent_config.gamma}, "
                         f"gae_lambda={agent_config.gae_lambda}")
    
    
    def _capture_end_of_rollout_states(self):
        """Capture states at the end of rollout for all environments."""
        # Get final states based on environment type
        if self.global_config.env_name.startswith(("h1hand-", "h1-", "h1hand-hurdle-")):
            # For humanoidbench environments, get MuJoCo states
            mj_states = self.envs.envs.env_method("get_mj_state")
            for env_idx in range(self.config.num_envs):
                final_state = mj_states[env_idx]
                trajectory_info = TrajectoryInfo(
                    hash_key=None,
                    episode_id=self.num_episodes + env_idx,  # Use unique episode ID
                    total_reward=self.current_episode_reward[env_idx].item(),
                    final_state=final_state,
                    episode_length=int(self.current_episode_length[env_idx].item()),
                    timestamp=time.time()
                )
                self.trajectory_buffer.append(trajectory_info)
                
        elif self.global_config.env_name.startswith(("T1", "G1")):
            # For MJX environments, get current observation and save state mapping
            saved_states = self.envs.save_state()
            
            for env_idx in range(self.config.num_envs):
                final_state = saved_states[env_idx]
                final_state_obs = final_state['env_state'].obs['state']
                hash_key = str(hash(final_state_obs.tobytes()))
                
                # Create hash of observation and store mapping to full MJX state
                if env_idx < len(saved_states):
                    # obs_hash = hash(final_state_obs.tobytes())
                    try:
                        self.mjx_state_mapping[hash_key] = final_state
                    except TypeError:
                        import ipdb; ipdb.set_trace()
                trajectory_info = TrajectoryInfo(
                    episode_id=self.num_episodes + env_idx,  # Use unique episode ID
                    total_reward=self.current_episode_reward[env_idx].item(),
                    final_state=np.array(final_state_obs),
                    hash_key=hash_key,
                    episode_length=int(self.current_episode_length[env_idx].item()),
                    timestamp=time.time()
                )
                self.trajectory_buffer.append(trajectory_info)
                
        else:
            # For other environments, use current observations
            if isinstance(self.obs, torch.Tensor):
                current_obs = self.obs.cpu().numpy()
            else:
                current_obs = np.array(self.obs)
                
            for env_idx in range(self.config.num_envs):
                final_state = current_obs[env_idx]
                trajectory_info = TrajectoryInfo(
                    episode_id=self.num_episodes + env_idx,  # Use unique episode ID
                    total_reward=self.current_episode_reward[env_idx].item(),
                    final_state=final_state,
                    hash_key=hash_key,
                    episode_length=int(self.current_episode_length[env_idx].item()),
                    timestamp=time.time()
                )
                self.trajectory_buffer.append(trajectory_info)
                        
    def _collect_rollout(self) -> Dict[str, Any]:
        """Collect a single rollout."""
        rollout_start_time = time.time()
        
        # Clear buffer
        self.buffer.clear()
        
        # Track episode rewards (states are now captured only at episode completion)
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
            
            # State saving moved to episode completion for performance optimization
            
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
                        
                        # State capture now handled at end of rollout for all environments
                        
                        # Reset episode tracking
                        self.current_episode_reward[env_idx] = 0
                        self.current_episode_length[env_idx] = 0
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
        
        # Capture states at the end of rollout (regardless of episode completion)
        if self.filtering_active:
            self._capture_end_of_rollout_states()
        
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
                    hash_key=traj.hash_key if traj.hash_key else None,
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
        # TODO: check if this is correct value for evaluation
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
            if current_time - last_log_time >= 60*2:  # Log every 2 minutes
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
                avg_length = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0
                progress = (self.global_step / self.total_timesteps) * 100
                
                self.logger.debug(f"Step {self.global_step:,} ({progress:.1f}%) | "
                                 f"Reward: {avg_reward:.2f} | Episodes: {self.num_episodes} | "
                                 f"Policy Loss: {update_metrics['policy_loss']:.6f}")
                
                last_log_time = current_time
            
            # Evaluation every 100000 steps or if the step is higher th
            if self.eval_interval > 0 and abs(self.global_step - self.latest_eval_step) >= self.eval_interval:
                eval_return, eval_length = self._evaluate()
                self.latest_eval_reward = eval_return
                self.latest_eval_length = eval_length
                self.latest_eval_step = self.global_step
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
