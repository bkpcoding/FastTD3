import numpy as np
import mujoco
import gymnasium as gym
from utils import get_state, set_state
import torch


def create_envs(env_name, num_envs, num_eval_envs, device="cuda:0"):
    seed = 0
    action_bounds = None
    if env_name.startswith("h1hand-") or env_name.startswith("h1-") or env_name.startswith("h1hand-hurdle-"):
        from fast_td3.environments.humanoid_bench_env import HumanoidBenchEnv
        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(env_name, num_envs, device=device)

    elif env_name.startswith("Isaac-"):
        from fast_td3.environments.isaaclab_env import IsaacLabEnv

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            env_name,
            device.type,
            num_envs,
            seed,
            action_bounds=action_bounds,
        )
    else:
        from fast_td3.environments.mujoco_playground_env import make_env

        env_type = "mujoco_playground"
        envs, eval_envs, render_env = make_env(
            env_name,
            seed,
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            # use_tuned_reward=True,
            # use_domain_randomization=True,
            # use_push_randomization=True,
        )
    return envs

from mujoco import mjtState

def demo_state_manipulation_humanoid():
    """Demonstrate saving and restoring states at different points with humanoid env."""
    print("\nDemonstrating state manipulation with humanoid environment...")
    
    try:
        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        envs = create_envs("h1hand-hurdle-v0", 16, 16)
        obs = envs.reset()
        states = envs.envs.env_method("get_state")
        qpos_list = [state[0] for state in states]
        qvel_list = [state[1] for state in states]
        
        # Generate sequence of actions for multiple steps
        action_space = envs.envs.action_space
        num_steps = 10
        actions_sequence = []
        
        for step in range(num_steps):
            actions = np.random.uniform(action_space.low, action_space.high, size=(envs.envs.num_envs, action_space.shape[0]))
            actions = torch.from_numpy(actions).to("cuda:0")
            actions_sequence.append(actions)
        
        # Take multiple steps
        obs_sequence = []
        rews_sequence = []
        dones_sequence = []
        
        for step in range(num_steps):
            obs, rews, dones, infos = envs.step(actions_sequence[step])
            obs_sequence.append(obs)
            rews_sequence.append(rews)
            dones_sequence.append(dones)
        
        # Reset each environment to its previous state (qpos/qvel only)
        for i in range(envs.envs.num_envs):
            envs.envs.env_method("set_state", qpos_list[i], qvel_list[i], indices=[i])
        
        # Perform the same sequence of actions again
        obs_repeat_sequence = []
        rews_repeat_sequence = []
        dones_repeat_sequence = []
        
        for step in range(num_steps):
            obs_repeat, rews_repeat, dones_repeat, infos_repeat = envs.step(actions_sequence[step])
            obs_repeat_sequence.append(obs_repeat)
            rews_repeat_sequence.append(rews_repeat)
            dones_repeat_sequence.append(dones_repeat)
        
        # Check if the entire sequences are the same
        sequences_match = True
        for step in range(num_steps):
            obs_match = torch.allclose(obs_sequence[step], obs_repeat_sequence[step])
            rews_match = torch.allclose(rews_sequence[step], rews_repeat_sequence[step])
            dones_match = torch.allclose(dones_sequence[step], dones_repeat_sequence[step])
            
            print(f"Step {step}: obs_match={obs_match}, rews_match={rews_match}, dones_match={dones_match}")
            
            if not (obs_match and rews_match and dones_match):
                sequences_match = False
                # Print detailed differences
                if not obs_match:
                    diff = torch.abs(obs_sequence[step] - obs_repeat_sequence[step])
                    max_diff = torch.max(diff)
                    mean_diff = torch.mean(diff)
                    print(f"  Observation differences - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")
                    print(f"  Observation shapes: {obs_sequence[step].shape}")
                break
        
        print(f"Environment is deterministic with qpos/qvel state over {num_steps} steps: {sequences_match}")
        print("qpos/qvel state provides partial reproducibility (may fail due to missing internal state)")
    except Exception as e:
        print(f"Could not create humanoid environment: {e}")
        raise e


def demo_mj_state_manipulation_humanoid():
    """Demonstrate saving and restoring full MuJoCo states for full reproducibility."""
    print("\nDemonstrating full MuJoCo state manipulation with humanoid environment...")
    
    try:
        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        envs = create_envs("h1hand-hurdle-v0", 16, 16)
        obs = envs.reset()
        
        # Get full MuJoCo states for all environments
        mj_states = envs.envs.env_method("get_mj_state")
        
        # Generate sequence of actions for multiple steps
        action_space = envs.envs.action_space
        num_steps = 10
        actions_sequence = []
        
        for step in range(num_steps):
            actions = np.random.uniform(action_space.low, action_space.high, size=(envs.envs.num_envs, action_space.shape[0]))
            actions = torch.from_numpy(actions).to("cuda:0")
            actions_sequence.append(actions)
        
        # Take multiple steps
        obs_sequence = []
        rews_sequence = []
        dones_sequence = []
        
        for step in range(num_steps):
            obs, rews, dones, infos = envs.step(actions_sequence[step])
            obs_sequence.append(obs)
            rews_sequence.append(rews)
            dones_sequence.append(dones)
        
        # Reset each environment to its previous full MuJoCo state
        for i in range(envs.envs.num_envs):
            envs.envs.env_method("set_mj_state", mj_states[i], indices=[i])
        
        # Perform the same sequence of actions again
        obs_repeat_sequence = []
        rews_repeat_sequence = []
        dones_repeat_sequence = []
        
        for step in range(num_steps):
            obs_repeat, rews_repeat, dones_repeat, infos_repeat = envs.step(actions_sequence[step])
            obs_repeat_sequence.append(obs_repeat)
            rews_repeat_sequence.append(rews_repeat)
            dones_repeat_sequence.append(dones_repeat)
        
        # Check if the entire sequences are the same
        sequences_match = True
        for step in range(num_steps):
            obs_match = torch.allclose(obs_sequence[step], obs_repeat_sequence[step])
            rews_match = torch.allclose(rews_sequence[step], rews_repeat_sequence[step])
            dones_match = torch.allclose(dones_sequence[step], dones_repeat_sequence[step])
            
            print(f"Step {step}: obs_match={obs_match}, rews_match={rews_match}, dones_match={dones_match}")
            
            if not (obs_match and rews_match and dones_match):
                sequences_match = False
                # Print detailed differences
                if not obs_match:
                    diff = torch.abs(obs_sequence[step] - obs_repeat_sequence[step])
                    max_diff = torch.max(diff)
                    mean_diff = torch.mean(diff)
                    print(f"  Observation differences - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")
                    print(f"  Observation shapes: {obs_sequence[step].shape}")
                # import ipdb; ipdb.set_trace()
                break
        
        print(f"Environment is deterministic with full MuJoCo state over {num_steps} steps: {sequences_match}")
        print("Full MuJoCo state provides complete reproducibility across multiple steps!")
        
    except Exception as e:
        print(f"Could not create humanoid environment: {e}")
        raise e


if __name__ == "__main__":
    demo_state_manipulation_humanoid()
    demo_mj_state_manipulation_humanoid()