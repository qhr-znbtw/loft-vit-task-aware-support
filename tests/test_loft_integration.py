"""Support construction and ViT injection tests for LOFT."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.loft.injection import inject_loft, iter_target_linears
from experiments.loft.loft_layer import LOFTLinear
from experiments.loft.model import set_trainable_parameters
from experiments.loft.run_loft import build_optimizer, build_scheduler, count_parameters
from experiments.loft.supports import build_gradient_supports, build_principal_supports
from vit_core.vit_backbone import ViTConfig, ViTForImageClassification


def build_tiny_model() -> ViTForImageClassification:
    config = ViTConfig(
        image_size=32,
        patch_size=16,
        hidden_size=8,
        num_heads=2,
        intermediate_size=16,
        num_layers=2,
    )
    return ViTForImageClassification(num_classes=3, config=config)


def assert_supports(supports: dict[str, torch.Tensor], rank: int) -> None:
    assert len(supports) == 12
    for support in supports.values():
        assert support.shape[0] == rank
        assert torch.allclose(
            support @ support.T,
            torch.eye(rank),
            atol=3e-4,
            rtol=3e-4,
        )


def test_all_support_types_and_identity_injection() -> None:
    torch.manual_seed(7)
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    for support_type in ("principal", "grad_svd", "skew_grad"):
        model = build_tiny_model()
        model.eval()
        before = model(images)
        if support_type == "principal":
            supports, _ = build_principal_supports(
                model,
                rank=4,
                device=torch.device("cpu"),
            )
        else:
            supports, _ = build_gradient_supports(
                model,
                loader,
                criterion,
                support_type=support_type,
                rank=4,
                calibration_batches=1,
                device=torch.device("cpu"),
            )
        assert_supports(supports, rank=4)
        paths = inject_loft(model, supports, dropout=0.0)
        set_trainable_parameters(model)
        after = model(images)
        assert len(paths) == 12
        assert torch.equal(before, after)


def test_parameter_groups_and_counts() -> None:
    model = build_tiny_model()
    supports = {
        path: torch.eye(layer.in_features)[:4]
        for path, _, _, layer in iter_target_linears(model)
    }
    inject_loft(model, supports, dropout=0.0)
    set_trainable_parameters(model)
    counts = count_parameters(model)
    expected_adapter = 12 * (4 * 3 // 2)
    expected_classifier = 8 * 3 + 3
    assert counts["adapter"] == expected_adapter
    assert counts["classifier"] == expected_classifier
    assert counts["trainable"] == expected_adapter + expected_classifier
    assert sum(isinstance(module, LOFTLinear) for module in model.modules()) == 12

    optimizer = build_optimizer(model, 2e-4)
    assert [group["lr"] for group in optimizer.param_groups] == [2e-4, 5e-3]
    assert build_scheduler(optimizer, num_epochs=50, warmup_epochs=5) is not None


if __name__ == "__main__":
    test_all_support_types_and_identity_injection()
    test_parameter_groups_and_counts()
    print("LOFT integration tests passed.")
