import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math


def l2normalize(
    tensor: torch.Tensor, axis: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """Computes L2 normalization of a tensor."""
    return tensor / (torch.linalg.norm(tensor, ord=2, dim=axis, keepdim=True) + eps)


class Scaler(nn.Module):
    """
    A learnable scaling layer.
    """

    def __init__(
        self,
        dim: int,
        init: float = 1.0,
        scale: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), init * scale, device=device))
        self.forward_scaler = init / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler.to(x.dtype) * self.forward_scaler * x


class HyperDense(nn.Module):
    """
    A dense layer without bias and with orthogonal initialization.
    """

    def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = None):
        super().__init__()
        self.w = nn.Linear(in_dim, hidden_dim, bias=False, device=device)
        nn.init.orthogonal_(self.w.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class HyperMLP(nn.Module):
    """
    A small MLP with a specific architecture using HyperDense and Scaler.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = 1e-8,
        device: torch.device = None,
    ):
        super().__init__()
        self.w1 = HyperDense(in_dim, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.w2 = HyperDense(hidden_dim, out_dim, device=device)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        # `eps` is required to prevent zero vector.
        x = F.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperEmbedder(nn.Module):
    """
    Embeds input by concatenating a constant, normalizing, and applying layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float,
        device: torch.device = None,
    ):
        super().__init__()
        # The input dimension to the dense layer is in_dim + 1
        self.w = HyperDense(in_dim + 1, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.c_shift = c_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_axis = torch.full(
            (*x.shape[:-1], 1), self.c_shift, device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, new_axis], dim=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperLERPBlock(nn.Module):
    """
    A residual block using Linear Interpolation (LERP).
    """

    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int = 4,
        device: torch.device = None,
    ):
        super().__init__()
        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * expansion,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
            device=device,
        )
        self.alpha_scaler = Scaler(
            dim=hidden_dim,
            init=alpha_init,
            scale=alpha_scale,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        mlp_out = self.mlp(x)
        # The original paper uses (x - residual) but x is the residual here.
        # This is interpreted as alpha * (mlp_output - residual_input)
        x = residual + self.alpha_scaler(mlp_out - residual)
        x = l2normalize(x, axis=-1)
        return x


class HyperPolicy(nn.Module):
    """
    A policy that outputs mean and log_std for PPO.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        device: torch.device = None,
    ):
        super().__init__()
        # Mean head
        self.mean_w1 = HyperDense(hidden_dim, hidden_dim, device=device)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.mean_w2 = HyperDense(hidden_dim, action_dim, device=device)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim, device=device))
        
        # Log std parameter (shared across all actions)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=device) - 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mean path
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias.to(mean.dtype)
        
        # Log std (learnable parameter)
        log_std = self.log_std.expand_as(mean)
        
        return mean, log_std


class HyperValue(nn.Module):
    """
    A value function head for PPO.
    """

    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.w1 = HyperDense(hidden_dim, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.w2 = HyperDense(hidden_dim, 1, device=device)
        self.bias = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.w1(x)
        value = self.scaler(value)
        value = self.w2(value) + self.bias.to(value.dtype)
        return value.squeeze(-1)


class Actor(nn.Module):
    """Actor network using SimbaV2 architecture for PPO."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        alpha_init: float = 1.0,
        alpha_scale: float = 1.0,
        expansion: int = 4,
        c_shift: float = 1.0,
        num_blocks: int = 2,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device

        self.embedder = HyperEmbedder(
            in_dim=n_obs,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            device=device,
        )
        
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    expansion=expansion,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )
        
        self.policy_head = HyperPolicy(
            hidden_dim=hidden_dim,
            action_dim=n_act,
            scaler_init=1.0,
            scaler_scale=1.0,
            device=device,
        )

    def get_dist(self, obs: torch.Tensor) -> Normal:
        x = self.embedder(obs)
        x = self.encoder(x)
        mean, log_std = self.policy_head(x)
        std = log_std.exp()
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.get_dist(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        # Compute log probability
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Critic network using SimbaV2 architecture for PPO."""

    def __init__(
        self,
        n_critic_obs: int,
        hidden_dim: int,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        alpha_init: float = 1.0,
        alpha_scale: float = 1.0,
        expansion: int = 4,
        c_shift: float = 1.0,
        num_blocks: int = 2,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device

        self.embedder = HyperEmbedder(
            in_dim=n_critic_obs,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            device=device,
        )
        
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    expansion=expansion,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )
        
        self.value_head = HyperValue(
            hidden_dim=hidden_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
            device=device,
        )

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        x = self.embedder(critic_obs)
        x = self.encoder(x)
        return self.value_head(x)


class ActorCritic(nn.Module):
    """Combined actor-critic network using SimbaV2 architecture with asymmetric observations support."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        alpha_init: float = 1.0,
        alpha_scale: float = 1.0,
        expansion: int = 4,
        c_shift: float = 1.0,
        num_blocks: int = 2,
        device: torch.device = None,
        n_critic_obs: int = None,
    ):
        super().__init__()
        self.device = device
        self.asymmetric_obs = n_critic_obs is not None and n_critic_obs != n_obs
        
        # Actor uses regular observations
        self.actor = Actor(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            expansion=expansion,
            c_shift=c_shift,
            num_blocks=num_blocks,
            device=device,
        )
        
        # Critic uses privileged observations if available, otherwise regular observations
        critic_obs_size = n_critic_obs if self.asymmetric_obs else n_obs
        self.critic = Critic(
            n_critic_obs=critic_obs_size,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            expansion=expansion,
            c_shift=c_shift,
            num_blocks=num_blocks,
            device=device,
        )

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