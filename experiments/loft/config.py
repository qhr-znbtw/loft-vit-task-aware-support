"""Paper-aligned LOFT configuration for ViT-B/16 on Mini-VTAB-3."""

from __future__ import annotations


LOFT_RANK = 46
LOFT_DROPOUT = 0.1
NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
CLASSIFIER_LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-3
CALIBRATION_BATCHES = 1

SUPPORT_TYPES = ("principal", "grad_svd", "skew_grad")

# Task-specific learning rates reported for the rank-46 orthogonal VTAB runs.
LOFT_LEARNING_RATES = {
    "principal": {
        "dtd": 2e-4,
        "eurosat": 2e-3,
        "dsprites_orientation": 2e-3,
    },
    "grad_svd": {
        "dtd": 3.5e-4,
        "eurosat": 1.78e-3,
        "dsprites_orientation": 1.3e-3,
    },
    "skew_grad": {
        "dtd": 2e-4,
        "eurosat": 8e-4,
        "dsprites_orientation": 9e-4,
    },
}


def get_learning_rate(support_type: str, task: str) -> float:
    try:
        return LOFT_LEARNING_RATES[support_type][task]
    except KeyError as error:
        raise ValueError(
            f"No LOFT learning rate for support={support_type!r}, task={task!r}."
        ) from error
