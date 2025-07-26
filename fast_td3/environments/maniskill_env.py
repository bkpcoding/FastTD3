import mani_skill.envs
import gymnasium as gym
import torch
from typing import Optional
from mani_skill.utils.wrappers.record import RecordEpisode
from fast_td3.fast_td3_utils import PrivilegedStateBuffer
from fast_td3.fast_td3_utils import tensor_to_state_dict
class ManiskillEnv:
    def __init__(self, env_name: str, num_envs: int, seed: int, action_bounds: Optional[float] = None,
                privileged_buffer: Optional[PrivilegedStateBuffer] = None):
        self.env = gym.make(env_name, num_envs=num_envs)
        # TODO: see if we can set the action bounds
        self.num_envs = num_envs
        self.action_bounds = action_bounds
        if num_envs == 1:
            self.num_actions = self.env.action_space.shape[0]
            self.num_obs = self.env.observation_space.shape[0]
        else:
            self.num_actions = self.env.action_space.shape[1] # first dim is num_envs
            self.num_obs = self.env.observation_space.shape[1]

        self.max_episode_steps = self.env._max_episode_steps
        # TODO: check asymmetric obs is available
        self.asymmetric_obs = False
        self.privileged_buffer = privileged_buffer


    def reset(self, env_indices: Optional[torch.Tensor] = None):
        if env_indices is None:
            env_indices = torch.arange(self.num_envs)
        if self.privileged_buffer is not None:
            obs, info_ret = self.env.reset(options={"env_idx": env_indices})
            # reset the env based on the privliged state
            # set the privileged state to random 50% of env_indices
            mask = torch.rand(len(env_indices)) < 0.5
            env_indices = env_indices[mask]
            if len(env_indices) > 0:
                privileged_state = self.privileged_buffer.sample(len(env_indices), use_prioritized=True) 
                # convert to state_dict
                # state_dict = tensor_to_state_dict(privileged_state, self.privileged_buffer.state_dict_metadata)
                # self.env.set_state_dict(state_dict)
                self.env.set_state(privileged_state, env_indices)
                obs = self.env.get_obs()
        else:
            obs, info_ret = self.env.reset(options={"env_idx": env_indices})
        return obs, info_ret

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        
        # Auto-reset functionality: reset any environments that are done
        if torch.any(dones):
            # Get which environments are done
            done_mask = dones.bool()
            
            # Reset done environments
            # if self.privileged_buffer is not None:
            #     # Reset using privileged states for done environments
            #     num_done = done_mask.sum().item()
            #     if num_done > 0:
            #         privileged_states = self.privileged_buffer.sample(num_done, use_prioritized=True)
            #         state_dicts = [tensor_to_state_dict(privileged_states[i:i+1], self.privileged_buffer.state_dict_metadata) 
            #                      for i in range(num_done)]
                    
            #         # Reset environments and set states
            #         done_indices = torch.where(done_mask)[0]
            #         for i, env_idx in enumerate(done_indices):
            #             # Reset single environment
            #             single_obs, _ = self.env.reset(seed=None, options={"env_idx": env_idx.item()})
            #             # Set the privileged state
            #             self.env.set_state_dict(state_dicts[i], env_idx.item())
            #             # Get new observation
            #             single_new_obs = self.env.get_obs(env_idx.item())
            #             obs[env_idx] = single_new_obs
            # else:
            # Standard reset for done environments
            done_indices = torch.where(done_mask)[0]
            # for env_idx in done_indices:
            if len(done_indices) > 0:
                obs, info_ret = self.reset(env_indices=done_indices)
            
            # Clear done flags after reset
            dones = torch.zeros_like(dones)
        
        info_ret = {"time_outs": truncated, "observations": {"raw": {"obs": obs}}}
        # TODO: see if rewards also need to be reset
        return obs, rew, dones, info_ret

    def render(self):
        raise NotImplementedError("We don't support rendering for Maniskill environments")
