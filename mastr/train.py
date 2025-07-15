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

from mastr.hyperparameters import MultiAgentConfig, AgentConfig, get_multi_agent_config, validate_config, save_config_to_experiment_dir
from ppo.spatial_coverage import visualize_faiss_states_by_agent
from ppo.faiss_state_storage import FAISSStateStorage
from fast_td3.fast_td3_utils import EmpiricalNormalization
from ppo.ppo import ActorCritic
from ppo.ppo_utils import RolloutBuffer
from mastr.main_agent import MainAgent
from mastr.sub_agent import IndependentAgent
from mastr.mastr_utils import setup_logging

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



def create_agent_in_thread(agent_id: str, agent_config: AgentConfig, global_config: MultiAgentConfig, 
                          device: torch.device, shared_faiss_storage: FAISSStateStorage, 
                          total_timesteps: int, mjx_state_mapping: Dict[str, Any], 
                          results: List, index: int) -> None:
    """Create an agent in a separate thread for parallel initialization."""
    try:
        # Create agent
        agent = IndependentAgent(agent_id, agent_config, global_config, device, shared_faiss_storage, total_timesteps, mjx_state_mapping)
        
            
        results[index] = agent
    except Exception as e:
        logging.error(f"Failed to create agent {agent_id}: {e}")
        results[index] = None


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
    logger.info(f"  Sub-agents ({config.num_agents}): {config.total_timesteps:,} timesteps each")
    logger.info(f"  Main Agent: {config.main_agent_timesteps:,} timesteps")
    logger.info(f"Trajectory filtering starts at step {config.trajectory_filter_timestep}")
    logger.info(f"Top-k trajectories: {config.top_k_trajectories}")
    logger.info(f"Verbose logging: {config.verbose}")
    
    # Experiment directory already created by config.get_experiment_dir()
    
    # Create shared FAISS storage
    # Use observation dimension from a temporary environment to initialize FAISS
    if config.env_name.startswith("h1hand-") or config.env_name.startswith("h1-") or config.env_name.startswith("h1hand-hurdle-"):
        from fast_td3.environments.humanoid_bench_env import HumanoidBenchEnv
        env_type = "humanoid_bench"
        temp_env = HumanoidBenchEnv(config.env_name, 1, device=device)
        eval_envs = temp_env
        render_env = HumanoidBenchEnv(
            config.env_name, 1, render_mode="rgb_array", device=device
        )
    elif config.env_name.startswith("Isaac-"):
        from fast_td3.environments.isaaclab_env import IsaacLabEnv
        env_type = "isaaclab"
        temp_env = IsaacLabEnv(config.env_name, 1, device=device)
        eval_envs = temp_env
        render_env = temp_env
    elif config.env_name.startswith("MTBench-"):
        from fast_td3.environments.mtbench_env import MTBenchEnv
        env_type = "mtbench"
        env_name = "-".join(config.env_name.split("-")[1:])
        temp_env = MTBenchEnv(env_name, 1, device=device)
        eval_envs = temp_env
        render_env = temp_env
    else:
        env_type = "mujoco_playground"
        # import make_env
        from fast_td3.environments.mujoco_playground_env import make_env
        envs, eval_envs, render_env = make_env(
            config.env_name,
            config.seed,
            1,
            1,
            0,
            use_tuned_reward=config.use_tuned_reward,
            use_domain_randomization=config.use_domain_randomization,
            use_push_randomization=config.use_push_randomization,
        )
        temp_env = envs
    temp_env.reset()
    obs_dim = temp_env.num_obs if isinstance(temp_env.num_obs, int) else temp_env.num_obs[0]
    
    # Determine state dimension based on environment type
    if config.env_name.startswith(("h1hand-", "h1-", "h1hand-hurdle-")):
        # For humanoidbench environments, use MuJoCo state dimension
        mj_state = temp_env.envs.env_method("get_mj_state")[0]
        state_dim = len(mj_state.flatten())
        logger.info(f"Using MuJoCo state dimension {state_dim} for humanoidbench environment")
    else:
        # For other environments, use observation dimension
        state_dim = obs_dim
        logger.info(f"Using observation dimension {state_dim} for non-humanoidbench environment")
    
    shared_faiss_storage = FAISSStateStorage(
        state_dim=state_dim,
        index_type=config.faiss_index_type,
        use_gpu=config.faiss_use_gpu and torch.cuda.is_available()
    )
    logger.info(f"Created shared FAISS storage with state dimension {state_dim}")
    
    # Create shared MJX state mapping for MJX environments
    mjx_state_mapping = {} if config.env_name.startswith(("T1", "G1")) else None
    # Create agents with parallel initialization for faster startup
    logger.info(f"Creating {config.num_agents} sub-agents in parallel...")
    start_time_agents = time.time()
    
    # Use threading for parallel agent creation
    agents = []
    if config.num_agents > 1:
        # Create agents in parallel using threading
        results = [None] * config.num_agents
        threads = []
        
        for i in range(config.num_agents):
            agent_id = str(i + 1)
            agent_config = config.agent_configs[i]
            
            thread = threading.Thread(
                target=create_agent_in_thread,
                args=(agent_id, agent_config, config, device, shared_faiss_storage, 
                      config.total_timesteps, mjx_state_mapping, results, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check for any failed agent creations
        agents = [agent for agent in results if agent is not None]
        if len(agents) != config.num_agents:
            raise RuntimeError(f"Failed to create all agents: got {len(agents)}, expected {config.num_agents}")
    else:
        # Single agent case - no need for threading
        agent = IndependentAgent("1", config.agent_configs[0], config, device, \
            shared_faiss_storage, config.total_timesteps, mjx_state_mapping=mjx_state_mapping)
        agents = [agent]
    
    agent_creation_time = time.time() - start_time_agents
    logger.info(f"Created {len(agents)} agents in {agent_creation_time:.2f}s")
    
    # Create main agent
    main_agent = MainAgent(config.main_agent_config, config, device, shared_faiss_storage, mjx_state_mapping)
    
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
    logger.info(f"Starting multi-agent training with {len(agents)} sub-agents...")
    start_time = time.time()
    
    # Start all sub-agents
    for agent in agents:
        agent.start_training()
    
    # Start main agent
    main_agent.start_training()
    
    try:
        # Monitor training
        while any(agent.running for agent in agents) or main_agent.running:
            time.sleep(30)  # Check every 30 seconds
            
            # Get stats from all agents
            agent_stats = [agent.get_stats() for agent in agents]
            stats_main = main_agent.get_stats()
            
            # Check if all agents are done (different timestep limits)
            agents_done = all(stats["global_step"] >= config.total_timesteps for stats in agent_stats)
            main_agent_done = stats_main["global_step"] >= config.main_agent_timesteps
            
            if agents_done and main_agent_done:
                break
            
            # Get shared FAISS stats
            shared_faiss_stats = shared_faiss_storage.get_performance_stats()
            total_faiss_states = shared_faiss_stats['total_states']
            
            # Calculate progress for each agent
            agent_progresses = [(stats['global_step'] / config.total_timesteps) * 100 for stats in agent_stats]
            main_agent_progress = (stats_main['global_step'] / config.main_agent_timesteps) * 100
            
            # Simple progress display (non-verbose) - showing evaluation reward
            avg_agent_progress = sum(agent_progresses) / len(agent_progresses) if agent_progresses else 0
            print(f"Sub-Agents: {avg_agent_progress:.1f}% avg | "
                  f"Main Agent: {stats_main['global_step']:,}/{config.main_agent_timesteps:,} ({main_agent_progress:.1f}%) | "
                  f"Eval Reward: {stats_main['eval_reward']:.2f} ({stats_main['eval_count']} evals) | FAISS: {total_faiss_states} states")
            
            # Verbose detailed stats
            if config.verbose:
                logger.info("="*80)
                logger.info(f"MULTI-AGENT TRAINING PROGRESS ({len(agents)} sub-agents)")
                logger.info("="*80)
                
                # Log each sub-agent
                for i, stats in enumerate(agent_stats):
                    agent_id = i + 1
                    progress = agent_progresses[i]
                    logger.info(f"Agent {agent_id}: Step {stats['global_step']:,}/{config.total_timesteps:,} ({progress:.1f}%) | "
                               f"Eval: {stats['eval_reward']:.2f} | Train: {stats['training_reward']:.2f} | Episodes: {stats['num_episodes']} | "
                               f"Filtering: {stats['filtering_active']}")
                
                # Log main agent
                logger.info(f"Main Agent: Step {stats_main['global_step']:,}/{config.main_agent_timesteps:,} ({main_agent_progress:.1f}%) | "
                           f"Eval: {stats_main['eval_reward']:.2f} | Train: {stats_main['training_reward']:.2f} | Episodes: {stats_main['num_episodes']} | "
                           f"FAISS Init: {stats_main['state_initialization_active']}")
                logger.info(f"Shared FAISS: {total_faiss_states} total states stored")
            
            # Log to wandb (always log, regardless of verbose setting)
            if config.use_wandb:
                wandb_metrics = {}
                
                # Log metrics for each sub-agent
                for i, stats in enumerate(agent_stats):
                    agent_id = i + 1
                    progress = agent_progresses[i]
                    prefix = f"agent_{agent_id}"
                    
                    wandb_metrics.update({
                        f"{prefix}/eval_reward": stats['eval_reward'],
                        f"{prefix}/training_reward": stats['training_reward'],
                        f"{prefix}/eval_count": stats['eval_count'],
                        f"{prefix}/episodes": stats['num_episodes'],
                        f"{prefix}/global_step": stats['global_step'],
                        f"{prefix}/progress": progress,
                        f"{prefix}/filtering_active": stats['filtering_active'],
                        f"{prefix}/policy_loss": stats['total_policy_loss'],
                        f"{prefix}/value_loss": stats['total_value_loss'],
                        f"{prefix}/entropy": stats['total_entropy'],
                    })
                
                # Main agent metrics
                wandb_metrics.update({
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
                })
                
                # Shared FAISS metrics
                wandb_metrics.update({
                    "shared/faiss_states": total_faiss_states,
                    "shared/faiss_episodes": shared_faiss_stats.get('total_episodes', 0),
                })
                
                # Combined metrics
                total_steps = sum(stats['global_step'] for stats in agent_stats) + stats_main['global_step']
                total_episodes = sum(stats['num_episodes'] for stats in agent_stats) + stats_main['num_episodes']
                avg_eval_reward = (sum(stats['eval_reward'] for stats in agent_stats) + stats_main['eval_reward']) / (len(agent_stats) + 1)
                avg_training_reward = (sum(stats['training_reward'] for stats in agent_stats) + stats_main['training_reward']) / (len(agent_stats) + 1)
                
                wandb_metrics.update({
                    "combined/total_steps": total_steps,
                    "combined/total_episodes": total_episodes,
                    "combined/avg_eval_reward": avg_eval_reward,
                    "combined/avg_training_reward": avg_training_reward,
                    "combined/num_agents": len(agents),
                })
                
                wandb.log(wandb_metrics)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    finally:
        # Stop agents
        logger.info(f"Stopping {len(agents)} sub-agents and main agent...")
        for agent in agents:
            agent.stop_training()
        main_agent.stop_training()
        
        # Final statistics
        total_time = time.time() - start_time
        final_agent_stats = [agent.get_stats() for agent in agents]
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
            
            # Log final stats for each sub-agent
            for i, stats in enumerate(final_agent_stats):
                agent_id = i + 1
                logger.info(f"Agent {agent_id} final stats: {stats}")
            
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
            final_wandb_metrics = {
                "final/total_time": total_time,
                "final/main_agent_steps": final_stats_main['global_step'],
                "final/main_agent_faiss_init_used": final_stats_main['state_initialization_active'],
                "final/shared_faiss_states": final_shared_stats['total_states'],
                "final/shared_faiss_episodes": final_shared_stats['total_episodes'],
                "final/num_agents": len(agents),
            }
            
            # Add final stats for each sub-agent
            for i, stats in enumerate(final_agent_stats):
                agent_id = i + 1
                final_wandb_metrics[f"final/agent_{agent_id}_steps"] = stats['global_step']
                final_wandb_metrics[f"final/agent_{agent_id}_eval_reward"] = stats['eval_reward']
            
            wandb.log(final_wandb_metrics)
            wandb.finish()
        
        logger.info("Multi-agent training completed!")


if __name__ == "__main__":
    main()
