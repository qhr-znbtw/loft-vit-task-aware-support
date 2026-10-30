"""Construct fixed principal or task-aware LOFT support subspaces."""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from torch import nn

from experiments.loft.injection import iter_target_linears
from vit_core.vit_backbone import ViTForImageClassification


def _right_singular_support(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    if rank > min(matrix.shape):
        raise ValueError(f"rank={rank} exceeds matrix shape {tuple(matrix.shape)}.")
    _, _, right_t = torch.linalg.svd(matrix.float(), full_matrices=False)
    support = right_t[:rank].contiguous()
    # CUDA SVD can leave ~1e-4 orthogonality error. A thin QR preserves the
    # selected row space while enforcing the LOFT support constraint.
    orthonormal_columns, _ = torch.linalg.qr(support.T, mode="reduced")
    return orthonormal_columns.T.contiguous()


def _check_support(support: torch.Tensor, rank: int, input_size: int) -> None:
    if support.shape != (rank, input_size):
        raise RuntimeError(
            f"Expected support {(rank, input_size)}, got {tuple(support.shape)}."
        )
    gram = support @ support.T
    identity = torch.eye(rank, device=support.device, dtype=support.dtype)
    if not torch.allclose(gram, identity, atol=3e-4, rtol=3e-4):
        raise RuntimeError("Constructed LOFT support is not row-orthonormal.")


def build_principal_supports(
    model: ViTForImageClassification,
    *,
    rank: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    """Use the top right singular subspace of each pretrained weight."""
    start = time.perf_counter()
    supports: dict[str, torch.Tensor] = {}
    for path, _, _, layer in iter_target_linears(model):
        weight = layer.weight.detach().to(device=device, dtype=torch.float32)
        support = _right_singular_support(weight, rank)
        _check_support(support, rank, layer.in_features)
        supports[path] = support.cpu()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return supports, time.perf_counter() - start


def build_gradient_supports(
    model: ViTForImageClassification,
    calibration_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    support_type: str,
    rank: int,
    calibration_batches: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    """Estimate one initial task gradient and derive GradSVD or SkewGrad support."""
    if support_type not in {"grad_svd", "skew_grad"}:
        raise ValueError("Gradient support_type must be 'grad_svd' or 'skew_grad'.")
    if calibration_batches <= 0:
        raise ValueError("calibration_batches must be positive.")

    targets = list(iter_target_linears(model))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, _, _, layer in targets:
        layer.weight.requires_grad = True

    model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    start = time.perf_counter()

    used_batches = 0
    for images, labels in calibration_loader:
        if used_batches >= calibration_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = criterion(model(images), labels) / calibration_batches
        loss.backward()
        used_batches += 1

    if used_batches != calibration_batches:
        raise ValueError(
            f"Requested {calibration_batches} calibration batches, got {used_batches}."
        )

    supports: dict[str, torch.Tensor] = {}
    for path, _, _, layer in targets:
        gradient = layer.weight.grad
        if gradient is None:
            raise RuntimeError(f"No calibration gradient was collected for {path}.")
        gradient = gradient.detach().float()

        if support_type == "grad_svd":
            support = _right_singular_support(gradient, rank)
        else:
            weight = layer.weight.detach().float()
            cross = weight.T @ gradient
            skew_signal = 0.5 * (cross - cross.T)
            support = _right_singular_support(skew_signal, rank)

        _check_support(support, rank, layer.in_features)
        supports[path] = support.cpu()
        layer.weight.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    model.zero_grad(set_to_none=True)
    return supports, elapsed
