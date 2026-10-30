"""Fixed-support low-rank orthogonal adaptation for a frozen linear layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LOFTLinear(nn.Module):
    """Apply an orthogonal transform inside a fixed row-orthonormal support.

    For row-vector PyTorch inputs, this implements the transpose of

        S = I + P^T (T - I) P,

    before the frozen linear projection. ``P`` is fixed and ``T`` is obtained
    from a trainable skew-symmetric generator through an exact Cayley map.
    """

    def __init__(
        self,
        base: nn.Linear,
        support: torch.Tensor,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        if support.ndim != 2 or support.shape[1] != base.in_features:
            raise ValueError(
                "support must have shape [rank, base.in_features], got "
                f"{tuple(support.shape)} for in_features={base.in_features}."
            )
        if support.shape[0] <= 0 or support.shape[0] > base.in_features:
            raise ValueError("support rank must be in [1, base.in_features].")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        support = support.detach().to(dtype=torch.float32, device="cpu").contiguous()
        gram = support @ support.T
        identity = torch.eye(support.shape[0], dtype=support.dtype)
        if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
            raise ValueError("support rows must be orthonormal.")

        self.base = base
        self.rank = support.shape[0]
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.register_buffer("support", support)

        rows, columns = torch.triu_indices(self.rank, self.rank, offset=1)
        self.register_buffer("skew_rows", rows, persistent=False)
        self.register_buffer("skew_columns", columns, persistent=False)
        self.skew_values = nn.Parameter(
            torch.zeros(self.rank * (self.rank - 1) // 2)
        )

    @property
    def adapter_parameter_count(self) -> int:
        return self.skew_values.numel()

    def skew_matrix(self) -> torch.Tensor:
        matrix = self.skew_values.new_zeros(self.rank, self.rank)
        matrix[self.skew_rows, self.skew_columns] = self.skew_values
        return matrix - matrix.T

    def orthogonal_matrix(self) -> torch.Tensor:
        """Return T=(I-Q)(I+Q)^-1 from the exact Cayley transform."""
        skew = self.skew_matrix()
        original_dtype = skew.dtype
        if skew.dtype in (torch.float16, torch.bfloat16):
            skew = skew.float()
        identity = torch.eye(self.rank, device=skew.device, dtype=skew.dtype)
        transform = torch.linalg.solve(identity + skew, identity - skew)
        return transform.to(original_dtype)

    def adapt_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply S^T to row-vector inputs without materializing the dense S."""
        support = self.support.to(device=inputs.device, dtype=inputs.dtype)
        projected = F.linear(inputs, support)
        projected = self.dropout(projected)
        transformed = F.linear(projected, self.orthogonal_matrix().to(projected.dtype))
        correction = F.linear(transformed - projected, support.T)
        return inputs + correction

    def dense_orthogonal_matrix(self) -> torch.Tensor:
        support = self.support.to(
            device=self.skew_values.device,
            dtype=self.skew_values.dtype,
        )
        transform = self.orthogonal_matrix()
        identity = torch.eye(
            support.shape[1],
            device=support.device,
            dtype=support.dtype,
        )
        return identity + support.T @ (transform - torch.eye(
            self.rank,
            device=transform.device,
            dtype=transform.dtype,
        )) @ support

    def effective_weight(self) -> torch.Tensor:
        return self.base.weight @ self.dense_orthogonal_matrix().to(self.base.weight.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(self.adapt_features(inputs))
