"""Algebraic tests for the fixed-support LOFT linear layer."""

import torch
from torch import nn
from torch.nn import functional as F

from experiments.loft.loft_layer import LOFTLinear


def test_identity_and_dense_forward_equivalence() -> None:
    torch.manual_seed(3)
    base = nn.Linear(8, 5)
    support = torch.eye(8)[:4]
    layer = LOFTLinear(base, support, dropout=0.0)
    inputs = torch.randn(2, 3, 8)

    assert torch.equal(layer(inputs), base(inputs))
    with torch.no_grad():
        layer.skew_values.normal_(mean=0.0, std=0.1)

    actual = layer(inputs)
    expected = F.linear(inputs, layer.effective_weight(), layer.base.bias)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_cayley_and_embedded_transform_are_orthogonal() -> None:
    torch.manual_seed(5)
    layer = LOFTLinear(nn.Linear(9, 6), torch.eye(9)[:4], dropout=0.0)
    with torch.no_grad():
        layer.skew_values.normal_(mean=0.0, std=0.2)

    small = layer.orthogonal_matrix()
    dense = layer.dense_orthogonal_matrix()
    assert torch.allclose(small.T @ small, torch.eye(4), atol=2e-5, rtol=2e-5)
    assert torch.allclose(dense.T @ dense, torch.eye(9), atol=3e-5, rtol=3e-5)

    loss = layer(torch.randn(2, 9)).square().mean()
    loss.backward()
    assert layer.skew_values.grad is not None


if __name__ == "__main__":
    test_identity_and_dense_forward_equivalence()
    test_cayley_and_embedded_transform_are_orthogonal()
    print("LOFT layer tests passed.")
