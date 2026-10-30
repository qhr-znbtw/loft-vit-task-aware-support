"""Locate and replace all six target linears in every ViT encoder block."""

from __future__ import annotations

from collections.abc import Iterator

from torch import nn

from experiments.loft.loft_layer import LOFTLinear
from vit_core.vit_backbone import ViTForImageClassification


def iter_target_linears(
    model: ViTForImageClassification,
) -> Iterator[tuple[str, nn.Module, str, nn.Linear]]:
    for index, block in enumerate(model.encoder.blocks):
        targets = (
            (f"encoder.blocks.{index}.attention.query", block.attention, "query"),
            (f"encoder.blocks.{index}.attention.key", block.attention, "key"),
            (f"encoder.blocks.{index}.attention.value", block.attention, "value"),
            (f"encoder.blocks.{index}.attention.output", block.attention, "output"),
            (f"encoder.blocks.{index}.mlp.fc1", block.mlp, "fc1"),
            (f"encoder.blocks.{index}.mlp.fc2", block.mlp, "fc2"),
        )
        for path, parent, attribute in targets:
            layer = getattr(parent, attribute)
            if not isinstance(layer, nn.Linear):
                raise TypeError(
                    f"Expected nn.Linear at {path}, found {type(layer).__name__}."
                )
            yield path, parent, attribute, layer


def inject_loft(
    model: ViTForImageClassification,
    supports: dict[str, object],
    *,
    dropout: float,
) -> list[str]:
    targets = list(iter_target_linears(model))
    expected = {path for path, _, _, _ in targets}
    if set(supports) != expected:
        missing = sorted(expected - set(supports))
        unexpected = sorted(set(supports) - expected)
        raise ValueError(f"Support keys differ from targets. Missing={missing}, extra={unexpected}")

    injected: list[str] = []
    for path, parent, attribute, layer in targets:
        setattr(parent, attribute, LOFTLinear(layer, supports[path], dropout=dropout))
        injected.append(path)
    return injected
