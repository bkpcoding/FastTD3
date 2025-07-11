"""Environment utilities for PBT baseline."""

import torch
import sys
import os

# Add FastTD3 directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fast_td3'))

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

try:
    import jax.numpy as jnp
except ImportError:
    pass


def create_env(args, device):
    """
    Create environment based on environment name.
    
    Args:
        args: Arguments containing environment configuration
        device: PyTorch device
    
    Returns:
        tuple: (envs, eval_envs, render_env, env_type, n_obs, n_act, action_bounds)
    """
    
    if args.env_name.startswith("h1hand-") or args.env_name.startswith("h1-"):
        from environments.humanoid_bench_env import HumanoidBenchEnv
        
        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name, 1, render_mode="rgb_array", device=device
        )
        
    elif args.env_name.startswith("Isaac-"):
        from environments.isaaclab_env import IsaacLabEnv
        
        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds,
        )
        eval_envs = envs
        render_env = envs
        
    elif args.env_name.startswith("MTBench-"):
        from environments.mtbench_env import MTBenchEnv
        
        env_name = "-".join(args.env_name.split("-")[1:])
        env_type = "mtbench"
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
        eval_envs = envs
        render_env = envs
        
    else:
        from environments.mujoco_playground_env import make_env
        
        env_type = "mujoco_playground"
        envs, eval_envs, render_env = make_env(
            args.env_name,
            args.seed,
            args.num_envs,
            args.num_eval_envs,
            args.device_rank,
            use_tuned_reward=args.use_tuned_reward,
            use_domain_randomization=args.use_domain_randomization,
            use_push_randomization=args.use_push_randomization,
        )
    
    # Get observation and action dimensions
    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    action_bounds = (-1.0, 1.0)
    
    # Handle asymmetric observations
    if hasattr(envs, 'asymmetric_obs') and envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
        envs.asymmetric_obs = False
    
    # Handle multi-task environments
    if env_type == "mtbench":
        # Adjust observation dimension for task embedding
        n_obs_adjusted = n_obs - envs.num_tasks + args.task_embedding_dim
        n_critic_obs_adjusted = n_critic_obs - envs.num_tasks + args.task_embedding_dim
    else:
        n_obs_adjusted = n_obs
        n_critic_obs_adjusted = n_critic_obs
    
    return (
        envs, eval_envs, render_env, env_type, 
        n_obs_adjusted, n_critic_obs_adjusted, n_act, action_bounds
    )


def evaluate_policy(actor, envs, env_type, obs_normalizer, device, max_steps=None):
    """
    Evaluate a policy on the given environments.
    
    Args:
        actor: Policy network
        envs: Environment to evaluate on
        env_type: Type of environment
        obs_normalizer: Observation normalizer
        device: PyTorch device
        max_steps: Maximum number of steps (uses env default if None)
    
    Returns:
        tuple: (mean_return, mean_length)
    """
    num_eval_envs = envs.num_envs
    episode_returns = torch.zeros(num_eval_envs, device=device)
    episode_lengths = torch.zeros(num_eval_envs, device=device)
    done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
    
    if env_type == "isaaclab":
        obs = envs.reset(random_start_init=False)
    else:
        obs = envs.reset()
    
    max_episode_steps = max_steps or envs.max_episode_steps
    
    for i in range(max_episode_steps):
        with torch.no_grad():
            # Normalize observations
            if obs_normalizer is not None:
                obs_norm = obs_normalizer(obs, update=False)
            else:
                obs_norm = obs
            
            # Get action from policy
            action = actor(obs_norm)
        
        next_obs, rewards, dones, infos = envs.step(action)
        
        if env_type == "mtbench":
            # For MTBench, use success rate as reward
            rewards = (
                infos["episode"]["success"].float() if "episode" in infos else torch.zeros_like(rewards)
            )
        
        episode_returns = torch.where(
            ~done_masks, episode_returns + rewards, episode_returns
        )
        episode_lengths = torch.where(
            ~done_masks, episode_lengths + 1, episode_lengths
        )
        
        if env_type == "mtbench" and "episode" in infos:
            dones = dones | infos["episode"]["success"]
        
        done_masks = torch.logical_or(done_masks, dones)
        
        if done_masks.all():
            break
            
        obs = next_obs
    
    return episode_returns.mean().item(), episode_lengths.mean().item()


def render_rollout(actor, render_env, env_type, obs_normalizer, device):
    """
    Perform a rollout and collect renders.
    
    Args:
        actor: Policy network
        render_env: Environment for rendering
        env_type: Type of environment
        obs_normalizer: Observation normalizer
        device: PyTorch device
    
    Returns:
        List of rendered frames
    """
    if env_type == "humanoid_bench":
        obs = render_env.reset()
        renders = [render_env.render()]
    elif env_type in ["isaaclab", "mtbench"]:
        raise NotImplementedError(
            "Rendering not supported for IsaacLab and MTBench environments"
        )
    else:
        obs = render_env.reset()
        render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
        renders = [render_env.state]
    
    for i in range(render_env.max_episode_steps):
        with torch.no_grad():
            if obs_normalizer is not None:
                obs_norm = obs_normalizer(obs, update=False)
            else:
                obs_norm = obs
            
            action = actor(obs_norm)
        
        next_obs, _, done, _ = render_env.step(action)
        
        if env_type == "mujoco_playground":
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
        
        if i % 2 == 0:  # Collect every other frame
            if env_type == "humanoid_bench":
                renders.append(render_env.render())
            else:
                renders.append(render_env.state)
        
        if done.any():
            break
            
        obs = next_obs
    
    if env_type == "mujoco_playground":
        renders = render_env.render_trajectory(renders)
    
    return renders