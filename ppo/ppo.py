import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def calculate_network_norms(network: nn.Module, prefix: str = ""):
    """Return norm metrics of network parameters."""
    metrics = {}
    total_norm = 0.0
    param_count = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2).item()
            metrics[f"{prefix}_{name}_norm"] = param_norm
            total_norm += param_norm**2
            param_count += param.numel()
    total_norm = total_norm**0.5
    metrics[f"{prefix}_total_param_norm"] = total_norm
    metrics[f"{prefix}_param_count"] = param_count
    return metrics


class Actor(nn.Module):
    """Actor network for PPO with asymmetric observations support."""

    def __init__(self, n_obs: int, n_act: int, hidden_dim: int, device=None):
        super().__init__()
        self.device = device

        self.actor_net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
        )
        self.mu = nn.Linear(hidden_dim // 4, n_act, device=device)
        self.log_std = nn.Parameter(torch.zeros(n_act, device=device) - 0.5)

    def get_dist(self, obs: torch.Tensor) -> Normal:
        x = self.actor_net(obs)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.get_dist(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        # Compute log probability directly (no transformation needed)
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Critic network for PPO with asymmetric observations support."""

    def __init__(self, n_critic_obs: int, hidden_dim: int, device=None):
        super().__init__()
        self.device = device

        self.critic_net = nn.Sequential(
            nn.Linear(n_critic_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1, device=device),
        )

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic_net(critic_obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined actor-critic network with asymmetric observations support."""

    def __init__(self, n_obs: int, n_act: int, hidden_dim: int, device=None, n_critic_obs: int = None):
        super().__init__()
        self.device = device
        self.asymmetric_obs = n_critic_obs is not None and n_critic_obs != n_obs
        
        # Actor uses regular observations
        self.actor = Actor(n_obs, n_act, hidden_dim, device)
        
        # Critic uses privileged observations if available, otherwise regular observations
        critic_obs_size = n_critic_obs if self.asymmetric_obs else n_obs
        self.critic = Critic(critic_obs_size, hidden_dim, device)

    def get_dist(self, obs: torch.Tensor) -> Normal:
        return self.actor.get_dist(obs)

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor = None, deterministic: bool = False):
        action, log_prob = self.actor.act(obs, deterministic)
        
        # Use critic_obs if provided (asymmetric), otherwise use regular obs
        value_input = critic_obs if critic_obs is not None else obs
        value = self.critic.value(value_input)
        
        return action, log_prob, value

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic.value(critic_obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, critic_obs: torch.Tensor = None):
        log_prob, entropy = self.actor.evaluate_actions(obs, actions)
        
        # Use critic_obs if provided (asymmetric), otherwise use regular obs
        value_input = critic_obs if critic_obs is not None else obs
        values = self.critic.value(value_input)
        
        return log_prob, entropy, values
