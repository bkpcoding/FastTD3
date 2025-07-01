"""GPU-based Asynchronous PPO training with double buffering for concurrent data generation and weight updates."""

from .hyperparams import get_args
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
import time
from collections import deque
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import threading
from typing import Optional, Tuple, Dict, Any
import copy

from fast_td3.environments.mujoco_playground_env import make_env
from fast_td3.fast_td3_utils import EmpiricalNormalization
from .ppo import ActorCritic, calculate_network_norms
from .ppo_utils import RolloutBuffer, save_ppo_params
from tensordict import TensorDict

import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"  # Enable triton gemm

import torch._dynamo

torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True


class AsyncPolicyUpdater:
    """Handles asynchronous policy updates in a separate thread."""

    def __init__(self, policy, optimizer, scaler, args, device):
        self.policy = policy
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args
        self.device = device

        # Thread management
        self.update_thread = None
        self.update_lock = threading.Lock()
        self.buffer_ready = threading.Event()
        self.update_complete = threading.Event()
        self.stop_updating = threading.Event()

        # Double buffering for async updates
        self.current_buffer = None
        self.next_buffer = None
        self.buffer_swap_lock = threading.Lock()

        # Metrics
        self.epoch_policy_loss = 0
        self.epoch_value_loss = 0
        self.epoch_entropy = 0
        self.epoch_updates = 0
        self.update_time = 0
        self.grad_norm = 0

        # AMP setup
        self.amp_enabled = args.amp and torch.cuda.is_available()
        self.amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

        # Compiled functions
        if args.compile:
            self.policy_get_dist = torch.compile(policy.get_dist)
            self.policy_value = torch.compile(policy.value)
        else:
            self.policy_get_dist = policy.get_dist
            self.policy_value = policy.value

    def start_async_updates(self):
        """Start the background update thread."""
        if self.update_thread and self.update_thread.is_alive():
            return

        self.stop_updating.clear()
        self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
        self.update_thread.start()

    def stop_async_updates(self):
        """Stop the background update thread."""
        self.stop_updating.set()
        self.buffer_ready.set()  # Wake up worker if waiting
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)

    def submit_buffer_for_update(self, buffer, normalizer):
        """Submit a buffer for asynchronous policy updates."""
        with self.buffer_swap_lock:
            self.next_buffer = (buffer, normalizer)
            self.buffer_ready.set()

    def is_update_complete(self):
        """Check if the current update is complete."""
        return self.update_complete.is_set()

    def wait_for_update_completion(self, timeout=None):
        """Wait for the current update to complete."""
        return self.update_complete.wait(timeout=timeout)

    def get_update_metrics(self):
        """Get metrics from the last update."""
        with self.update_lock:
            return {
                "policy_loss": self.epoch_policy_loss / max(1, self.epoch_updates),
                "value_loss": self.epoch_value_loss / max(1, self.epoch_updates),
                "entropy": self.epoch_entropy / max(1, self.epoch_updates),
                "update_time": self.update_time,
                "grad_norm": self.grad_norm,
                "num_updates": self.epoch_updates,
            }

    def _update_worker(self):
        """Background worker for policy updates."""
        while not self.stop_updating.is_set():
            # Wait for buffer to be ready
            if not self.buffer_ready.wait(timeout=1.0):
                continue

            if self.stop_updating.is_set():
                break

            # Get buffer for update
            with self.buffer_swap_lock:
                if self.next_buffer is None:
                    self.buffer_ready.clear()
                    continue

                buffer, normalizer = self.next_buffer
                self.next_buffer = None
                self.buffer_ready.clear()
                self.update_complete.clear()

            # Perform policy updates
            try:
                self._perform_updates(buffer, normalizer)
            except Exception as e:
                print(f"Error in async policy update: {e}")
                import traceback

                traceback.print_exc()
            finally:
                self.update_complete.set()

    def _perform_updates(self, buffer, normalizer):
        """Perform the actual policy updates."""
        update_start_time = time.time()

        with self.update_lock:
            self.epoch_policy_loss = 0
            self.epoch_value_loss = 0
            self.epoch_entropy = 0
            self.epoch_updates = 0

        normalize_obs = normalizer.forward
        if self.args.compile:
            normalize_obs = torch.compile(normalize_obs)

        for epoch in range(self.args.update_epochs):
            for b_obs, b_actions, b_logp, b_returns, b_adv in buffer.get_batches(
                self.args.batch_size
            ):
                with autocast(
                    device_type=self.amp_device_type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    dist = self.policy_get_dist(normalize_obs(b_obs))
                    new_logp = dist.log_prob(b_actions).sum(-1)

                    ratio = (new_logp - b_logp).exp()
                    pg_loss1 = -b_adv * ratio
                    pg_loss2 = -b_adv * torch.clamp(
                        ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps
                    )
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    value = self.policy_value(normalize_obs(b_obs))
                    value_loss = F.mse_loss(value, b_returns)
                    entropy = dist.entropy().sum(-1).mean()

                    loss = (
                        policy_loss
                        + self.args.vf_coef * value_loss
                        - self.args.ent_coef * entropy
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=(
                        self.args.max_grad_norm
                        if self.args.max_grad_norm > 0
                        else float("inf")
                    ),
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                with self.update_lock:
                    self.epoch_policy_loss += policy_loss.item()
                    self.epoch_value_loss += value_loss.item()
                    self.epoch_entropy += entropy.item()
                    self.epoch_updates += 1
                    self.grad_norm = grad_norm.item()

        with self.update_lock:
            self.update_time = time.time() - update_start_time


class AsyncRolloutManager:
    """Manages double-buffered rollout collection on GPU."""

    def __init__(self, args, envs, policy, normalizer, device):
        self.args = args
        self.envs = envs
        self.policy = policy
        self.normalizer = normalizer
        self.device = device

        # Get environment dimensions
        self.n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
        self.n_act = envs.num_actions

        # Double buffering for concurrent data collection and updates
        self.buffer_a = RolloutBuffer(
            args.rollout_length, args.num_envs, self.n_obs, self.n_act, device=device
        )
        self.buffer_b = RolloutBuffer(
            args.rollout_length, args.num_envs, self.n_obs, self.n_act, device=device
        )
        self.active_buffer = self.buffer_a
        self.ready_buffer = None

        # Compiled functions
        if args.compile:
            self.policy_act = torch.compile(policy.act)
            self.policy_value = torch.compile(policy.value)
            self.normalize_obs = torch.compile(normalizer.forward)
        else:
            self.policy_act = policy.act
            self.policy_value = policy.value
            self.normalize_obs = normalizer.forward

        # AMP setup
        self.amp_enabled = args.amp and torch.cuda.is_available()
        self.amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

        # Episode tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.current_episode_reward = torch.zeros(args.num_envs, device=device)
        self.current_episode_length = torch.zeros(args.num_envs, device=device)
        self.num_episodes = 0

        # Environment state
        self.obs = envs.reset()

    def collect_rollout(self):
        """Collect a full rollout using the active buffer."""
        rollout_start_time = time.time()

        # Clear active buffer
        self.active_buffer.clear()

        # Collect rollout data
        for step in range(self.args.rollout_length):
            with torch.no_grad(), autocast(
                device_type=self.amp_device_type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                norm_obs = self.normalize_obs(self.obs)
                action, logp, value = self.policy_act(norm_obs)

            next_obs, reward, done, _ = self.envs.step(action)
            self.active_buffer.add(self.obs, action, logp, reward, done, value)

            self.obs = next_obs

            # Track episode statistics
            self.current_episode_reward += reward
            self.current_episode_length += 1

            # Check for episode completions
            if done.any():
                for env_idx in range(self.args.num_envs):
                    if done[env_idx]:
                        self.episode_rewards.append(
                            self.current_episode_reward[env_idx].item()
                        )
                        self.episode_lengths.append(
                            self.current_episode_length[env_idx].item()
                        )
                        self.num_episodes += 1
                        self.current_episode_reward[env_idx] = 0
                        self.current_episode_length[env_idx] = 0

        # Compute advantages
        with torch.no_grad(), autocast(
            device_type=self.amp_device_type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            last_values = self.policy_value(self.normalize_obs(self.obs))

        self.active_buffer.compute_returns_and_advantage(
            last_values, self.args.gamma, self.args.gae_lambda, self.args.num_envs
        )

        rollout_time = time.time() - rollout_start_time

        # Swap buffers: current active becomes ready, and we get a new active buffer
        self.ready_buffer = self.active_buffer
        self.active_buffer = (
            self.buffer_a if self.active_buffer is self.buffer_b else self.buffer_b
        )

        return {
            "buffer": self.ready_buffer,
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
            "num_episodes": self.num_episodes,
            "rollout_time": rollout_time,
            "steps_collected": self.args.num_envs * self.args.rollout_length,
        }

    def get_episode_stats(self):
        """Get current episode statistics."""
        return {
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
            "num_episodes": self.num_episodes,
        }


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled = args.amp and torch.cuda.is_available()
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb if requested
    if args.use_wandb:
        import wandb

        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    print("Starting GPU-based Asynchronous PPO training")
    print(f"Device: {device}")
    print(f"Environment: {args.env_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Rollout length: {args.rollout_length}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Double buffering: Enabled for async data/update overlap")
    print("-" * 60)

    # Create training environments
    envs, _, _ = make_env(
        args.env_name,
        seed=args.seed,
        num_envs=args.num_envs,
        num_eval_envs=1,
        device_rank=0,
    )

    # Create evaluation environments
    eval_envs, _, render_env = make_env(
        args.env_name,
        seed=args.seed + 42,
        num_envs=args.num_eval_envs,
        num_eval_envs=1,
        device_rank=0,
    )

    # Get environment dimensions
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    n_act = envs.num_actions

    # Initialize policy, optimizer, and normalizer
    policy = ActorCritic(n_obs, n_act, args.hidden_dim, device=device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    # Create async components
    policy_updater = AsyncPolicyUpdater(policy, optimizer, scaler, args, device)
    rollout_manager = AsyncRolloutManager(args, envs, policy, normalizer, device)

    # Start async policy updater
    policy_updater.start_async_updates()

    # Progress tracking variables
    global_step = 0
    start_time = time.time()
    last_log_time = start_time

    # Training metrics
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    num_updates = 0

    # Evaluation metrics
    eval_returns = []
    eval_lengths = []
    last_eval_step = (
        -args.eval_interval
    )  # Initialize to trigger first evaluation at step 0
    last_save_step = -args.save_interval  # Initialize to trigger first save if needed

    def evaluate():
        """Evaluate the current policy on separate evaluation environments."""
        normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        obs = eval_envs.reset()

        # Compiled evaluation functions
        if args.compile:
            eval_policy_act = torch.compile(policy.act)
            # Use non-compiled version for normalization to avoid FX tracing conflicts
            eval_normalize_obs = normalizer.forward
        else:
            eval_policy_act = policy.act
            eval_normalize_obs = normalizer.forward

        max_steps = getattr(
            eval_envs,
            "max_episode_length",
            getattr(eval_envs, "max_episode_steps", 1000),
        )
        for _ in range(max_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                norm_obs = eval_normalize_obs(obs)
                action, _, _ = eval_policy_act(norm_obs)

            next_obs, rewards, dones, _ = eval_envs.step(action)
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        normalizer.train()
        return episode_returns.mean().item(), episode_lengths.mean().item()

    print("Starting GPU-based asynchronous training loop...")

    try:
        with tqdm(
            total=args.total_timesteps, desc="Async Training Progress", unit="steps"
        ) as pbar:
            while global_step < args.total_timesteps:
                logs_dict = TensorDict()

                # Evaluation phase
                eval_avg_return = None
                eval_avg_length = None
                if (
                    args.eval_interval > 0
                    and global_step - last_eval_step >= args.eval_interval
                ):
                    print(f"\nEvaluating at global step {global_step}")
                    eval_avg_return, eval_avg_length = evaluate()
                    eval_returns.append(eval_avg_return)
                    eval_lengths.append(eval_avg_length)
                    last_eval_step = global_step  # Update last evaluation step
                    print(
                        f"*** Evaluation - Avg Return: {eval_avg_return:.3f}, Avg Length: {eval_avg_length:.1f}****"
                    )

                    if args.use_wandb:
                        wandb.log(
                            {
                                "eval_avg_return": eval_avg_return,
                                "eval_avg_length": eval_avg_length,
                            },
                            step=global_step,
                        )

                # Collect rollout data (this happens on GPU with all environments)
                print(f"\nCollecting rollout with {args.num_envs} environments...")
                rollout_data = rollout_manager.collect_rollout()

                # Update global step
                global_step += rollout_data["steps_collected"]
                pbar.update(rollout_data["steps_collected"])

                # Submit buffer for async policy updates
                policy_updater.submit_buffer_for_update(
                    rollout_data["buffer"], normalizer
                )

                print(
                    f"Rollout collected in {rollout_data['rollout_time']:.3f}s, submitted for async updates"
                )

                # Wait for previous update to complete (if any) before proceeding with logging
                if num_updates > 0:
                    print(f"Waiting for policy update to complete...")
                    policy_updater.wait_for_update_completion(timeout=30.0)
                    update_metrics = policy_updater.get_update_metrics()
                    print(
                        f"Policy update completed in {update_metrics['update_time']:.3f}s"
                    )
                else:
                    update_metrics = {
                        "policy_loss": 0,
                        "value_loss": 0,
                        "entropy": 0,
                        "update_time": 0,
                        "grad_norm": 0,
                        "num_updates": 0,
                    }

                # Update global metrics if we have update results
                if update_metrics["num_updates"] > 0:
                    total_policy_loss += update_metrics["policy_loss"]
                    total_value_loss += update_metrics["value_loss"]
                    total_entropy += update_metrics["entropy"]
                    num_updates += 1

                # Get episode statistics from rollout manager
                episode_stats = rollout_manager.get_episode_stats()
                episode_rewards = episode_stats["episode_rewards"]
                episode_lengths = episode_stats["episode_lengths"]
                num_episodes = episode_stats["num_episodes"]

                # Progress tracking and logging
                avg_reward = (
                    sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
                    if episode_rewards
                    else 0
                )
                avg_policy_loss = (
                    total_policy_loss / num_updates if num_updates > 0 else 0
                )
                progress = (global_step / args.total_timesteps) * 100

                eval_info = (
                    f" | Eval: {eval_avg_return:.3f}"
                    if eval_avg_return is not None
                    else ""
                )
                pbar.set_description(
                    f"Async GPU Training ({progress:.1f}%) | Reward: {avg_reward:.3f}{eval_info} | Policy Loss: {avg_policy_loss:.6f} | Episodes: {num_episodes}"
                )

                # Detailed logging
                current_time = time.time()
                elapsed_time = current_time - start_time
                time_since_last_log = current_time - last_log_time

                avg_length = (
                    sum(episode_lengths[-100:]) / len(episode_lengths[-100:])
                    if episode_lengths
                    else 0
                )
                avg_value_loss = (
                    total_value_loss / num_updates if num_updates > 0 else 0
                )
                avg_entropy = total_entropy / num_updates if num_updates > 0 else 0

                steps_per_second = (
                    rollout_data["steps_collected"] / rollout_data["rollout_time"]
                    if rollout_data["rollout_time"] > 0
                    else 0
                )

                # Calculate async efficiency
                rollout_time = rollout_data["rollout_time"]
                update_time = update_metrics["update_time"]
                if rollout_time > 0 and update_time > 0:
                    async_efficiency = (
                        min(rollout_time, update_time)
                        / max(rollout_time, update_time)
                        * 100
                    )
                    overlap_time = max(0, min(rollout_time, update_time))
                else:
                    async_efficiency = 100
                    overlap_time = 0

                print(
                    f"\nAsync GPU Training Progress - Step {global_step:,}/{args.total_timesteps:,} ({progress:.1f}%)"
                )
                print(
                    f"Elapsed: {elapsed_time:.1f}s | Steps/sec: {steps_per_second:.0f} | Environments: {args.num_envs}"
                )
                print(
                    f"Episode Stats: Avg Reward: {avg_reward:.3f} | Avg Length: {avg_length:.1f} | Episodes: {num_episodes}"
                )
                print(
                    f"Loss Stats: Policy: {avg_policy_loss:.6f} | Value: {avg_value_loss:.6f} | Entropy: {avg_entropy:.6f}"
                )
                print(
                    f"Timing: Rollout: {rollout_time:.3f}s | Update: {update_time:.3f}s | Async Efficiency: {async_efficiency:.1f}%"
                )
                print(f"Async Overlap: {overlap_time:.3f}s | Double Buffering: Active")

                if eval_avg_return is not None:
                    print(
                        f"Evaluation: Avg Return: {eval_avg_return:.3f} | Avg Length: {eval_avg_length:.1f}"
                    )

                print("-" * 60)

                # Log to wandb
                if args.use_wandb:
                    logs_dict["avg_reward"] = avg_reward
                    logs_dict["avg_length"] = avg_length
                    logs_dict["policy_loss"] = avg_policy_loss
                    logs_dict["value_loss"] = avg_value_loss
                    logs_dict["entropy"] = avg_entropy
                    logs_dict["num_episodes"] = num_episodes
                    logs_dict["rollout_time"] = rollout_time
                    logs_dict["update_time"] = update_time
                    logs_dict["async_efficiency"] = async_efficiency
                    logs_dict["steps_per_second"] = steps_per_second
                    logs_dict["grad_norm"] = update_metrics["grad_norm"]

                    policy_norms = calculate_network_norms(policy, "policy")
                    logs_dict.update(policy_norms)

                    if eval_avg_return is not None:
                        logs_dict["eval_avg_return"] = eval_avg_return
                        logs_dict["eval_avg_length"] = eval_avg_length

                    wandb.log(
                        {"speed": steps_per_second, "frame": global_step, **logs_dict},
                        step=global_step,
                    )

                # Save model
                if (
                    args.save_interval > 0
                    and global_step > 0
                    and global_step - last_save_step >= args.save_interval
                ):
                    print(f"Saving model at global step {global_step}")
                    # Wait for updates to complete before saving
                    policy_updater.wait_for_update_completion(timeout=10.0)
                    last_save_step = global_step  # Update last save step
                    save_ppo_params(
                        global_step,
                        policy,
                        normalizer,
                        args,
                        f"{args.output_dir}/{run_name}_{global_step}.pt",
                    )

                last_log_time = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup: stop async updater
        print("Shutting down async policy updater...")
        policy_updater.stop_async_updates()

    # Wait for final update to complete
    print("Waiting for final policy update to complete...")
    policy_updater.wait_for_update_completion(timeout=30.0)

    total_time = time.time() - start_time
    final_episode_stats = rollout_manager.get_episode_stats()
    final_num_episodes = final_episode_stats["num_episodes"]

    print(f"\nGPU-based Asynchronous training completed!")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Final stats: {final_num_episodes} episodes, {global_step:,} timesteps")
    print(f"Average steps/sec: {global_step/total_time:.1f}")
    print(f"GPU utilization: Maximized with {args.num_envs} parallel environments")
    print(f"Async efficiency: Double buffering enabled for concurrent data/updates")

    if eval_returns:
        print(f"Final evaluation return: {eval_returns[-1]:.3f}")
        print(f"Best evaluation return: {max(eval_returns):.3f}")

    if args.use_wandb:
        wandb.log(
            {
                "final_eval_return": eval_returns[-1] if eval_returns else 0,
                "best_eval_return": max(eval_returns) if eval_returns else 0,
                "total_training_time": total_time,
                "final_steps_per_second": global_step / total_time,
            },
            step=global_step,
        )
        wandb.finish()

    save_ppo_params(
        global_step,
        policy,
        normalizer,
        args,
        f"{args.output_dir}/{run_name}_final.pt",
    )


if __name__ == "__main__":
    main()
