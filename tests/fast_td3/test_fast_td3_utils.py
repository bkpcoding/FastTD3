import torch
import pytest
from tensordict import TensorDict

from fast_td3.fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer

def test_empirical_normalization():
    shape = (3,)
    device = torch.device("cpu")
    normalizer = EmpiricalNormalization(shape, device)

    # Test with a single batch
    data = torch.randn(10, 3)
    normalized_data = normalizer(data)
    assert normalized_data.shape == (10, 3)
    assert torch.allclose(normalizer.mean, data.mean(dim=0), atol=1e-6)
    assert torch.allclose(normalizer.std, data.std(dim=0, unbiased=False), atol=1e-6)

    # Test with multiple batches
    data2 = torch.randn(15, 3)
    _ = normalizer(data2)
    combined_data = torch.cat([data, data2], dim=0)
    assert torch.allclose(normalizer.mean, combined_data.mean(dim=0), atol=1e-6)
    assert torch.allclose(normalizer.std, combined_data.std(dim=0, unbiased=False), atol=1e-6)

    # Test inverse
    y = torch.randn(5, 3)
    x = normalizer.inverse(y)
    assert torch.allclose(x, y * (normalizer.std + normalizer.eps) + normalizer.mean, atol=1e-6)
