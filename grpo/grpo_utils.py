import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class GroupRolloutBuffer:
    """Buffer storing a group of complete episodes for GRPO."""

    def __init__(self, group_size: int, device=None):
        self.group_size = group_size
        self.device = device
        self.episodes = []

    @property
    def num_episodes(self):
        return len(self.episodes)

    def add_episode(self, obs, actions, logprobs, rewards, gamma):
        """Add a single episode to the buffer.

        Parameters
        ----------
        obs, actions, logprobs, rewards : list[Tensor] or Tensor
            Episode data collected from one environment. Each input can be
            provided either as a list of tensors (one per step) or already
            stacked into a tensor of shape ``(T, *)``.
        gamma : float
            Discount factor used to compute the episode return.
        """

        # Convert lists to tensors if necessary. This also allows passing
        # pre-stacked tensors for minor speed benefits when episodes are
        # constructed in a vectorised manner.
        obs = torch.stack(obs) if isinstance(obs, list) else obs
        actions = torch.stack(actions) if isinstance(actions, list) else actions
        logprobs = (
            torch.stack(logprobs) if isinstance(logprobs, list) else logprobs
        )
        rewards = torch.tensor(rewards, device=self.device) if isinstance(rewards, list) else rewards

        # Vectorised discounted return computation.
        # Create discount factors: [gamma^0, gamma^1, gamma^2, ..., gamma^(T-1)]
        T = rewards.shape[0]
        discount_powers = torch.arange(T, device=self.device, dtype=torch.float32)
        discount_factors = torch.pow(gamma, discount_powers)
        
        # Compute discounted return by reversing rewards, applying discounts, and summing
        R = (rewards * discount_factors).sum()

        self.episodes.append(
            {
                "obs": obs,
                "actions": actions,
                "logprobs": logprobs,
                "return": torch.as_tensor(R, device=self.device, dtype=torch.float32),
            }
        )

    def compute_advantages(self):
        returns = torch.stack([ep["return"] for ep in self.episodes])
        mean = returns.mean()
        std = returns.std() + 1e-8
        for ep in self.episodes:
            adv = (ep["return"] - mean) / std
            ep_len = len(ep["obs"])
            ep["adv"] = adv.expand(ep_len)
        self.obs = torch.cat([ep["obs"] for ep in self.episodes])
        self.actions = torch.cat([ep["actions"] for ep in self.episodes])
        self.logprobs = torch.cat([ep["logprobs"] for ep in self.episodes])
        self.advantages = torch.cat([ep["adv"] for ep in self.episodes])

    def get_batches(self, batch_size):
        total = len(self.obs)
        sampler = BatchSampler(
            SubsetRandomSampler(range(total)), batch_size, drop_last=True
        )
        for idx in sampler:
            yield (
                self.obs[idx],
                self.actions[idx],
                self.logprobs[idx],
                self.advantages[idx],
            )

    def clear(self):
        self.episodes = []


def save_grpo_params(global_step, actor, obs_normalizer, args, save_path):
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": {
            k: v.detach().cpu() for k, v in actor.state_dict().items()
        },
        "obs_normalizer_state": (
            obs_normalizer.state_dict()
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "args": vars(args),
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")
