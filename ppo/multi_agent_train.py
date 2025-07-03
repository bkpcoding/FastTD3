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

from .multi_agent_config import MultiAgentConfig, AgentConfig, get_multi_agent_config, validate_config
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
                 shared_faiss_storage: FAISSStateStorage):
        self.agent_id = agent_id
        self.config = agent_config
        self.global_config = global_config
        self.device = device
        
        # Set up random seed for this agent
        seed = global_config.seed + hash(agent_id) % 1000
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"[Agent {agent_id}] Initializing with seed {seed}")
        
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
        
        print(f"[Agent {agent_id}] Initialized with {agent_config.num_envs} environments")
        print(f"[Agent {agent_id}] Config: rollout_length={agent_config.rollout_length}, "
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
        """Process trajectory filtering and add top-k states to FAISS."""
        if len(self.trajectory_buffer) < self.global_config.top_k_trajectories:
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
        
        print(f"[Agent {self.agent_id}] Added {len(top_k_trajectories)} top trajectories to FAISS "
              f"(rewards: {[f'{t.total_reward:.2f}' for t in top_k_trajectories]})")
    
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
        print(f"[Agent {self.agent_id}] Starting training loop...")
        
        last_log_time = time.time()
        last_eval_time = time.time()
        
        while self.running and self.global_step < self.global_config.total_timesteps:
            # Check if we should start filtering
            if not self.filtering_active and self.global_step >= self.global_config.trajectory_filter_timestep:
                self.filtering_active = True
                print(f"[Agent {self.agent_id}] Started trajectory filtering at step {self.global_step}")
            
            # Collect rollout
            self._collect_rollout()
            
            # Update policy
            update_metrics = self._update_policy()
            
            # Process trajectory filtering
            if self.filtering_active:
                self._process_trajectory_filtering()
            
            # Logging
            current_time = time.time()
            if current_time - last_log_time >= 10.0:  # Log every 10 seconds
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
                avg_length = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0
                progress = (self.global_step / self.global_config.total_timesteps) * 100
                
                print(f"[Agent {self.agent_id}] Step {self.global_step:,} ({progress:.1f}%) | "
                      f"Reward: {avg_reward:.2f} | Length: {avg_length:.1f} | "
                      f"Episodes: {self.num_episodes} | Policy Loss: {update_metrics['policy_loss']:.6f} | "
                      f"FAISS States: {len(self.faiss_storage.state_info)}")
                
                last_log_time = current_time
            
            # Evaluation
            if current_time - last_eval_time >= 30.0:  # Evaluate every 30 seconds
                eval_return, eval_length = self._evaluate()
                print(f"[Agent {self.agent_id}] Evaluation: Return {eval_return:.2f}, Length {eval_length:.1f}")
                last_eval_time = current_time
        
        print(f"[Agent {self.agent_id}] Training completed at step {self.global_step}")
        
        # Print current FAISS statistics (storage is shared, so don't save individually)
        stats = self.faiss_storage.get_performance_stats()
        print(f"[Agent {self.agent_id}] Current shared FAISS stats: {stats['total_states']} states, "
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
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
            "avg_length": sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0,
            "filtering_active": self.filtering_active,
            "faiss_states": len(self.faiss_storage.state_info),
            "trajectories_in_buffer": len(self.trajectory_buffer),
            "total_policy_loss": self.total_policy_loss / max(1, self.num_updates),
            "total_value_loss": self.total_value_loss / max(1, self.num_updates),
            "total_entropy": self.total_entropy / max(1, self.num_updates),
        }


def main():
    """Main function to run multi-agent training."""
    config = get_multi_agent_config()
    validate_config(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training {config.env_name} for {config.total_timesteps:,} timesteps")
    print(f"Trajectory filtering starts at step {config.trajectory_filter_timestep}")
    print(f"Top-k trajectories: {config.top_k_trajectories}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
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
    print(f"Created shared FAISS storage with state dimension {obs_dim}")
    
    # Create agents with shared FAISS storage
    agent_1 = IndependentAgent("1", config.agent_1_config, config, device, shared_faiss_storage)
    agent_2 = IndependentAgent("2", config.agent_2_config, config, device, shared_faiss_storage)
    
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
    print("Starting multi-agent training...")
    start_time = time.time()
    
    agent_1.start_training()
    agent_2.start_training()
    
    try:
        # Monitor training
        while agent_1.running or agent_2.running:
            time.sleep(30)  # Check every 30 seconds
            
            # Get stats from both agents
            stats_1 = agent_1.get_stats()
            stats_2 = agent_2.get_stats()
            
            # Check if both agents are done
            if (stats_1["global_step"] >= config.total_timesteps and 
                stats_2["global_step"] >= config.total_timesteps):
                break
            
            # Print combined stats
            print(f"\n{'='*80}")
            print(f"MULTI-AGENT TRAINING PROGRESS")
            print(f"{'='*80}")
            # Get shared FAISS stats
            shared_faiss_stats = shared_faiss_storage.get_performance_stats()
            total_faiss_states = shared_faiss_stats['total_states']
            
            print(f"Agent 1: Step {stats_1['global_step']:,} | Reward: {stats_1['avg_reward']:.2f} | "
                  f"Episodes: {stats_1['num_episodes']} | Filtering: {stats_1['filtering_active']}")
            print(f"Agent 2: Step {stats_2['global_step']:,} | Reward: {stats_2['avg_reward']:.2f} | "
                  f"Episodes: {stats_2['num_episodes']} | Filtering: {stats_2['filtering_active']}")
            print(f"Shared FAISS: {total_faiss_states} total states stored")
            
            # Log to wandb
            if config.use_wandb:
                wandb.log({
                    "agent_1/reward": stats_1['avg_reward'],
                    "agent_1/episodes": stats_1['num_episodes'],
                    "agent_1/filtering_active": stats_1['filtering_active'],
                    "agent_1/policy_loss": stats_1['total_policy_loss'],
                    "agent_2/reward": stats_2['avg_reward'],
                    "agent_2/episodes": stats_2['num_episodes'],
                    "agent_2/filtering_active": stats_2['filtering_active'],
                    "agent_2/policy_loss": stats_2['total_policy_loss'],
                    "shared/faiss_states": total_faiss_states,
                    "combined/total_steps": stats_1['global_step'] + stats_2['global_step'],
                    "combined/total_episodes": stats_1['num_episodes'] + stats_2['num_episodes'],
                })
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Stop agents
        print("Stopping agents...")
        agent_1.stop_training()
        agent_2.stop_training()
        
        # Final statistics
        total_time = time.time() - start_time
        final_stats_1 = agent_1.get_stats()
        final_stats_2 = agent_2.get_stats()
        final_shared_stats = shared_faiss_storage.get_performance_stats()
        
        print(f"\n{'='*80}")
        print(f"FINAL MULTI-AGENT TRAINING RESULTS")
        print(f"{'='*80}")
        print(f"Total training time: {total_time:.1f}s")
        print(f"Agent 1 final stats: {final_stats_1}")
        print(f"Agent 2 final stats: {final_stats_2}")
        print(f"Shared FAISS final stats: {final_shared_stats['total_states']} total states")
        
        # Save shared FAISS storage
        faiss_save_path = f"{config.output_dir}/shared_faiss_storage"
        shared_faiss_storage.save(faiss_save_path)
        print(f"Saved shared FAISS storage to {faiss_save_path}")
        
        if config.use_wandb:
            wandb.log({
                "final/total_time": total_time,
                "final/agent_1_steps": final_stats_1['global_step'],
                "final/agent_2_steps": final_stats_2['global_step'],
                "final/shared_faiss_states": final_shared_stats['total_states'],
                "final/shared_faiss_episodes": final_shared_stats['total_episodes'],
            })
            wandb.finish()
        
        print("Multi-agent training completed!")


if __name__ == "__main__":
    main()