import torch
import torch.nn as nn
from fast_td3.fast_td3_utils import EmpiricalNormalization


class RewardNormalizer(nn.Module):
    """Reward normalizer based on FastTD3 implementation."""
    
    def __init__(
        self,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer(
            "G", torch.zeros(1, device=device)
        )  # running estimate of the discounted return
        self.register_buffer("G_r_max", torch.zeros(1, device=device))  # running-max
        self.G_rms = EmpiricalNormalization(shape=1, device=device)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def _scale_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms.std[0] + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)
        
        return rewards / denominator

    def update_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Update running statistics for reward normalization."""
        self.G = self.gamma * (1 - dones.float()) * self.G + rewards
        local_max = torch.max(torch.abs(self.G))
        self.G_r_max = max(self.G_r_max, local_max)
        self.G_rms(self.G.unsqueeze(0))

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics."""
        return self._scale_reward(rewards)

    def __call__(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Update stats and normalize rewards."""
        self.update_stats(rewards, dones)
        return self.forward(rewards)