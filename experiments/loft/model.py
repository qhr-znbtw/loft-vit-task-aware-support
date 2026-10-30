"""Build a pretrained ViT-B/16 and inject fixed-support LOFT adapters."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from experiments.loft.config import LOFT_DROPOUT, LOFT_RANK
from experiments.loft.injection import inject_loft
from experiments.loft.loft_layer import LOFTLinear
from experiments.loft.supports import build_gradient_supports, build_principal_supports
from vit_core.pretrained import load_pretrained_vit
from vit_core.vit_backbone import VIT_BASE_PATCH16_224, ViTForImageClassification


def set_trainable_parameters(model: ViTForImageClassification) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, LOFTLinear):
            module.skew_values.requires_grad = True
    for parameter in model.head.classifier.parameters():
        parameter.requires_grad = True


def build_loft_model(
    num_classes: int,
    support_type: str,
    *,
    device: torch.device,
    criterion: nn.Module,
    calibration_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None,
    calibration_batches: int,
) -> tuple[ViTForImageClassification, dict[str, object]]:
    model = ViTForImageClassification(
        num_classes=num_classes,
        config=VIT_BASE_PATCH16_224,
    )
    loaded_tensors = load_pretrained_vit(model)

    if support_type == "principal":
        supports, support_seconds = build_principal_supports(
            model,
            rank=LOFT_RANK,
            device=device,
        )
        used_calibration_batches = 0
    else:
        if calibration_loader is None:
            raise ValueError(f"{support_type} requires a calibration loader.")
        supports, support_seconds = build_gradient_supports(
            model,
            calibration_loader,
            criterion,
            support_type=support_type,
            rank=LOFT_RANK,
            calibration_batches=calibration_batches,
            device=device,
        )
        used_calibration_batches = calibration_batches

    injected = inject_loft(model, supports, dropout=LOFT_DROPOUT)
    set_trainable_parameters(model)
    metadata: dict[str, object] = {
        "loaded_tensors": loaded_tensors,
        "injected_paths": injected,
        "support_construction_seconds": support_seconds,
        "calibration_batches": used_calibration_batches,
    }
    return model, metadata
