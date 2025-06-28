"""Minimal GRPO training loop."""

from .hyperparams import get_args
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from fast_td3.environments.mujoco_playground_env import make_env
from fast_td3.fast_td3_utils import EmpiricalNormalization
from .grpo import Actor, calculate_network_norms
from .grpo_utils import GroupRolloutBuffer, save_grpo_params
import os
import time
from collections import deque
import numpy as np
from tqdm import tqdm


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
    
    print("Starting GRPO training")
    print(f"Device: {device}")
    print(f"Environment: {args.env_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Group size: {args.group_size}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Evaluation interval: {args.eval_interval}")
    print(f"Number of eval environments: {args.num_eval_envs}")
    print(f"Using wandb: {args.use_wandb}")
    print("-" * 60)

    # Create training environments
    envs, _, _ = make_env(
        args.env_name,
        seed=args.seed,
        num_envs=args.num_envs,
        num_eval_envs=1,
        device_rank=0,
    )
    
    # Create separate evaluation environments
    eval_envs, _, render_env = make_env(
        args.env_name, 
        seed=args.seed+42, 
        num_envs=args.num_eval_envs, 
        num_eval_envs=1, 
        device_rank=0
    )
    
    obs = envs.reset()
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    n_act = envs.num_actions

    actor = Actor(n_obs, n_act, args.hidden_dim, device=device)
    actor_ref = Actor(n_obs, n_act, args.hidden_dim, device=device)
    actor_ref.load_state_dict(actor.state_dict())
    optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    buffer = GroupRolloutBuffer(args.group_size, device=device)

    # Progress tracking variables
    global_step = 0
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    num_episodes = 0
    start_time = time.time()
    last_log_time = start_time
    
    # Training metrics
    total_policy_loss = 0
    total_kl_loss = 0
    total_entropy = 0
    num_updates = 0
    
    # Evaluation metrics
    eval_returns = []
    eval_lengths = []
    last_eval_step = 0  # Track the last step we performed evaluation
    
    def evaluate():
        """Evaluate the current policy on separate evaluation environments."""
        normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        obs = eval_envs.reset()

        # Run for a fixed number of steps
        max_steps = getattr(eval_envs, 'max_episode_length', getattr(eval_envs, 'max_episode_steps', 1000))
        for _ in range(max_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                norm_obs = normalizer(obs)
                action, _ = actor.act(norm_obs)

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

    # Vectorized episode data storage
    max_episode_length = getattr(envs, 'max_episode_length', getattr(envs, 'max_episode_steps', 1000))
    
    # Check for single observation space in case of heterogeneous envs
    if isinstance(n_obs, (list, tuple)):
        if len(n_obs) > 1:
            print("Warning: Heterogeneous observation spaces detected. Using the first one.")
        n_obs = n_obs[0]

    # Check for single action space
    if isinstance(n_act, (list, tuple)):
        if len(n_act) > 1:
            print("Warning: Heterogeneous action spaces detected. Using the first one.")
        n_act = n_act[0]

    obs_buf = torch.zeros((max_episode_length, args.num_envs, n_obs), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((max_episode_length, args.num_envs, n_act), dtype=torch.float32, device=device)
    logps_buf = torch.zeros((max_episode_length, args.num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((max_episode_length, args.num_envs), dtype=torch.float32, device=device)
    episode_step_counter = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    env_indices = torch.arange(args.num_envs, device=device)

    print("Starting training loop...")
    pbar = tqdm(total=args.total_timesteps, desc="Training Progress", unit="steps")
    
    while global_step < args.total_timesteps:
        # Evaluation phase - check if we've passed the evaluation threshold
        eval_avg_return = None
        eval_avg_length = None
        if args.eval_interval > 0 and global_step >= last_eval_step + args.eval_interval:
            print(f"\nEvaluating at global step {global_step}")
            eval_avg_return, eval_avg_length = evaluate()
            eval_returns.append(eval_avg_return)
            eval_lengths.append(eval_avg_length)
            last_eval_step = global_step  # Update last evaluation step
            print(f"*** Evaluation - Avg Return: {eval_avg_return:.3f}, Avg Length: {eval_avg_length:.1f} ***")
            
            if args.use_wandb:
                wandb.log(
                    {
                        "eval_avg_return": eval_avg_return,
                        "eval_avg_length": eval_avg_length,
                    },
                    step=global_step,
                )
        
        # Data collection and training
        step_start_time = time.time()
        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalizer(obs)
            action, logp = actor.act(norm_obs)
        
        next_obs, reward, done, _ = envs.step(action)

        # Add data to buffers
        obs_buf[episode_step_counter, env_indices] = obs
        actions_buf[episode_step_counter, env_indices] = action
        logps_buf[episode_step_counter, env_indices] = logp
        rewards_buf[episode_step_counter, env_indices] = reward.to(device)
        episode_step_counter += 1
        
        global_step += args.num_envs
        pbar.update(args.num_envs)
        
        # Track episode completions
        done_indices = torch.where(done)[0]
        if len(done_indices) > 0:
            for i in done_indices:
                ep_len = episode_step_counter[i].item()
                
                # Extract episode data
                ep_obs = obs_buf[:ep_len, i]
                ep_actions = actions_buf[:ep_len, i]
                ep_logps = logps_buf[:ep_len, i]
                ep_rewards = rewards_buf[:ep_len, i]
                
                episode_reward = ep_rewards.sum().item()
                episode_length = ep_len
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                num_episodes += 1
                
                buffer.add_episode(
                    ep_obs,
                    ep_actions,
                    ep_logps,
                    ep_rewards,
                    args.gamma,
                )
            
            # Reset counters for done episodes
            episode_step_counter[done_indices] = 0
        
        obs = next_obs
        
        # Update progress bar with current metrics
        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        progress = (global_step / args.total_timesteps) * 100
        eval_info = f" | Eval: {eval_avg_return:.3f}" if eval_avg_return is not None else ""
        pbar.set_description(
            f"Training ({progress:.1f}%) | Reward: {avg_reward:.3f}{eval_info} | Episodes: {num_episodes}"
        )
        
        if buffer.num_episodes >= args.group_size:
            buffer.compute_advantages()
            actor_ref.load_state_dict(actor.state_dict())
            
            update_start_time = time.time()
            epoch_policy_loss = 0
            epoch_kl_loss = 0 
            epoch_entropy = 0
            epoch_updates = 0
            grad_norm = torch.tensor(0.0)
            
            for epoch in range(args.update_epochs):
                for b_obs, b_actions, b_logp, b_adv in buffer.get_batches(
                    args.batch_size
                ):
                    with autocast(
                        device_type=amp_device_type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        dist = actor.get_dist(normalizer(b_obs))
                        raw_actions = torch.atanh(torch.clamp(b_actions, -0.999, 0.999))
                        new_logp = dist.log_prob(raw_actions).sum(-1)
                        new_logp = new_logp - (
                            2
                            * (
                                torch.log(torch.tensor(2.0))
                                - raw_actions
                                - F.softplus(-2 * raw_actions)
                            )
                        ).sum(-1)
                        ratio = (new_logp - b_logp).exp()
                        pg_loss = -torch.min(
                            ratio * b_adv,
                            torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                            * b_adv,
                        ).mean()
                        ref_dist = actor_ref.get_dist(normalizer(b_obs))
                        kl = (
                            torch.distributions.kl.kl_divergence(dist, ref_dist)
                            .sum(-1)
                            .mean()
                        )
                        entropy = dist.entropy().sum(-1).mean()
                        loss = pg_loss + args.kl_coef * kl - args.ent_coef * entropy
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), args.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Accumulate metrics
                    epoch_policy_loss += pg_loss.item()
                    epoch_kl_loss += kl.item()
                    epoch_entropy += entropy.item()
                    epoch_updates += 1
            
            update_time = time.time() - update_start_time
            
            # Update global metrics
            total_policy_loss += epoch_policy_loss
            total_kl_loss += epoch_kl_loss
            total_entropy += epoch_entropy
            num_updates += 1
            
            buffer.clear()
            
            # Logging - similar to PPO
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_since_last_log = current_time - last_log_time
            
            # Calculate metrics
            avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
            avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
            avg_kl_loss = total_kl_loss / num_updates if num_updates > 0 else 0
            avg_entropy = total_entropy / num_updates if num_updates > 0 else 0
            
            # Calculate FPS
            fps = args.log_interval / time_since_last_log if time_since_last_log > 0 else 0
            
            # Console logging
            if global_step % args.log_interval == 0 or global_step >= args.total_timesteps:
                print(f"\nTraining Progress - Step {global_step:,}/{args.total_timesteps:,} ({progress:.1f}%)")
                print(f"Elapsed: {elapsed_time:.1f}s | FPS: {fps:.1f} | Time since last log: {time_since_last_log:.1f}s")
                print(f"Episode Stats: Avg Reward: {avg_reward:.3f} | Avg Length: {avg_length:.1f} | Episodes: {num_episodes}")
                print(f"Loss Stats: Policy: {avg_policy_loss:.6f} | KL: {avg_kl_loss:.6f} | Entropy: {avg_entropy:.6f}")
                print(f"Update Time: {update_time:.3f}s | Epochs: {args.update_epochs} | Updates this cycle: {epoch_updates}")
                
                # Add evaluation results to logging if available
                if eval_avg_return is not None:
                    print(f"Evaluation: Avg Return: {eval_avg_return:.3f} | Avg Length: {eval_avg_length:.1f}")
                
                print("-" * 60)
                last_log_time = current_time
            
            # Log to wandb if enabled - comprehensive logging like PPO
            if args.use_wandb:
                logs_dict = {
                    "avg_reward": avg_reward,
                    "avg_length": avg_length,
                    "policy_loss": avg_policy_loss,
                    "kl_loss": avg_kl_loss,
                    "entropy": avg_entropy,
                    "num_episodes": num_episodes,
                    "update_time": update_time,
                    "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    "buffer_episodes": buffer.num_episodes,
                    "total_updates": num_updates,
                    "env_rewards": reward.mean().item(),  # Current step reward
                }

                # Add network parameter norms
                actor_norms = calculate_network_norms(actor, "actor")
                logs_dict.update(actor_norms);

                # Add evaluation metrics if available
                if eval_avg_return is not None:
                    logs_dict["eval_avg_return"] = eval_avg_return
                    logs_dict["eval_avg_length"] = eval_avg_length

                # Log speed metrics like PPO
                wandb.log({"speed": fps, "frame": global_step, **logs_dict}, step=global_step)

    pbar.close()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Final stats: {num_episodes} episodes, {global_step:,} timesteps")
    print(f"Average FPS: {global_step/total_time:.1f}")
    
    # Print final evaluation results
    if eval_returns:
        print(f"Final evaluation return: {eval_returns[-1]:.3f}")
        print(f"Best evaluation return: {max(eval_returns):.3f}")
    
    # Log final results to wandb
    if args.use_wandb:
        wandb.log({
            "final_eval_return": eval_returns[-1] if eval_returns else 0,
            "best_eval_return": max(eval_returns) if eval_returns else 0,
            "total_training_time": total_time,
            "final_fps": global_step/total_time,
        }, step=global_step)
        wandb.finish()

    save_path = os.path.join(args.output_dir, f"{run_name}_final.pt")
    save_grpo_params(global_step, actor, normalizer, args, save_path)


if __name__ == "__main__":
    main()
