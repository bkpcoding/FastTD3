import mani_skill.envs
import gymnasium as gym
import torch
from typing import Optional
from mani_skill.utils.wrappers.record import RecordEpisode

class ManiskillEnv:
    def __init__(self, env_name: str, num_envs: int, seed: int, action_bounds: Optional[float] = None):
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


    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        info_ret = {"time_outs": truncated, "observations": {"raw": {"obs": obs}}}
        return obs, rew, dones, info_ret

    def render(self):
        raise NotImplementedError("We don't support rendering for Maniskill environments")
