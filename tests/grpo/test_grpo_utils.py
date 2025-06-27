import torch
import pytest
from grpo.grpo_utils import GroupRolloutBuffer

def test_compute_advantages_simple():
    """Test advantage calculation with a simple case of two episodes."""
    buffer = GroupRolloutBuffer(group_size=2, device='cpu')
    gamma = 0.99

    # Episode 1: len=2
    obs1 = [torch.randn(4), torch.randn(4)]
    actions1 = [torch.randn(1), torch.randn(1)]
    logprobs1 = [torch.tensor(-0.1), torch.tensor(-0.1)]
    rewards1 = [1.0, 1.0]
    buffer.add_episode(obs1, actions1, logprobs1, rewards1, gamma)
    # Return = 1.0 + 0.99 * 1.0 = 1.99

    # Episode 2: len=3
    obs2 = [torch.randn(4)] * 3
    actions2 = [torch.randn(1)] * 3
    logprobs2 = [torch.tensor(-0.2)] * 3
    rewards2 = [2.0, 2.0, 2.0]
    buffer.add_episode(obs2, actions2, logprobs2, rewards2, gamma)
    # Return = 2.0 + 0.99 * (2.0 + 0.99 * 2.0) = 5.9402

    buffer.compute_advantages()

    returns = torch.tensor([1.99, 5.9402])
    mean = returns.mean()
    std = returns.std() + 1e-8
    
    adv1 = (1.99 - mean) / std
    adv2 = (5.9402 - mean) / std

    assert buffer.advantages.shape == (len(obs1) + len(obs2),)
    assert torch.allclose(buffer.advantages[:len(obs1)], adv1)
    assert torch.allclose(buffer.advantages[len(obs1):], adv2)

def test_compute_advantages_varying_rewards():
    """Test advantage calculation with varying rewards and episode lengths."""
    buffer = GroupRolloutBuffer(group_size=3, device='cpu')
    gamma = 0.9

    # Episode 1: len=2, rewards=[1, 2]
    buffer.add_episode([torch.randn(1)]*2, [torch.randn(1)]*2, [torch.tensor(-0.1)]*2, [1, 2], gamma)
    # Return = 1 + 0.9 * 2 = 2.8

    # Episode 2: len=3, rewards=[0.5, 0.5, 0.5]
    buffer.add_episode([torch.randn(1)]*3, [torch.randn(1)]*3, [torch.tensor(-0.1)]*3, [0.5, 0.5, 0.5], gamma)
    # Return = 0.5 + 0.9 * (0.5 + 0.9 * 0.5) = 1.355

    # Episode 3: len=1, rewards=[10]
    buffer.add_episode([torch.randn(1)], [torch.randn(1)], [torch.tensor(-0.1)], [10], gamma)
    # Return = 10

    buffer.compute_advantages()

    returns = torch.tensor([2.8, 1.355, 10.0])
    mean = returns.mean()
    std = returns.std() + 1e-8

    adv1 = (2.8 - mean) / std
    adv2 = (1.355 - mean) / std
    adv3 = (10.0 - mean) / std

    assert buffer.advantages.shape == (2 + 3 + 1,)
    assert torch.allclose(buffer.advantages[0:2], adv1)
    assert torch.allclose(buffer.advantages[2:5], adv2)
    assert torch.allclose(buffer.advantages[5:6], adv3)


def test_compute_advantages_zero_std():
    """Test advantage calculation when all episodes have the same return."""
    buffer = GroupRolloutBuffer(group_size=2, device='cpu')
    gamma = 0.5

    # Episode 1: len=1, rewards=[2]
    buffer.add_episode([torch.randn(1)], [torch.randn(1)], [torch.tensor(-0.1)], [2], gamma)
    # Return = 2

    # Episode 2: len=2, rewards=[1, 2]
    buffer.add_episode([torch.randn(1)]*2, [torch.randn(1)]*2, [torch.tensor(-0.1)]*2, [1, 2], gamma)
    # Return = 1 + 0.5 * 2 = 2

    buffer.compute_advantages()

    # Returns are [2.0, 2.0], so std is 0. Advantage should be 0.
    expected_advantages = torch.zeros(3)

    assert buffer.advantages.shape == (1 + 2,)
    assert torch.allclose(buffer.advantages, expected_advantages, atol=1e-7)
