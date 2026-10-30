"""Run Principal, GradSVD, or SkewGrad LOFT on Mini-VTAB-3."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from experiments.loft.config import (
    CALIBRATION_BATCHES,
    CLASSIFIER_LEARNING_RATE,
    LOFT_DROPOUT,
    LOFT_RANK,
    NUM_EPOCHS,
    SUPPORT_TYPES,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    get_learning_rate,
)
from experiments.loft.loft_layer import LOFTLinear
from experiments.loft.model import build_loft_model
from vit_core.evaluation import test
from vit_core.mini_vtab import (
    EVAL_BATCH_SIZE,
    RANDOM_SEED,
    TASK_SPECS,
    TRAIN_BATCH_SIZE,
    build_dataloaders,
    get_task_spec,
)
from vit_core.training_engine import train


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", required=True, choices=SUPPORT_TYPES)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--calibration-batches", type=int, default=CALIBRATION_BATCHES)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, loft_learning_rate: float) -> AdamW:
    adapter_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".skew_values"):
            adapter_parameters.append(parameter)
        elif name.startswith("head.classifier."):
            classifier_parameters.append(parameter)
        else:
            raise RuntimeError(f"Unclassified trainable parameter: {name}")
    if not adapter_parameters or not classifier_parameters:
        raise RuntimeError("LOFT and classifier parameter groups must be non-empty.")
    return AdamW(
        [
            {"params": adapter_parameters, "lr": loft_learning_rate},
            {"params": classifier_parameters, "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def build_scheduler(optimizer: AdamW, *, num_epochs: int, warmup_epochs: int) -> LambdaLR:
    if not 0 <= warmup_epochs < num_epochs:
        raise ValueError("warmup_epochs must be in [0, num_epochs).")

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        progress = (epoch_index - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=multiplier)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    adapter = sum(
        module.skew_values.numel()
        for module in model.modules()
        if isinstance(module, LOFTLinear)
    )
    classifier = sum(parameter.numel() for parameter in model.head.classifier.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != adapter + classifier:
        raise RuntimeError("Trainable count does not equal LOFT plus classifier parameters.")
    return {
        "total": total,
        "adapter": adapter,
        "classifier": classifier,
        "trainable": trainable,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.support != "principal" and args.calibration_batches <= 0:
        raise ValueError("--calibration-batches must be positive.")

    task = get_task_spec(args.task)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = f"loft_{args.support}"
    loft_learning_rate = get_learning_rate(args.support, task.name)
    checkpoint_path = PROJECT_ROOT / "checkpoints" / method / "development_800_200" / task.name / f"seed_{args.seed}.pt"
    result_path = PROJECT_ROOT / "results" / method / "development_800_200" / task.name / f"seed_{args.seed}.json"
    criterion = nn.CrossEntropyLoss()

    calibration_loader = None
    if args.support != "principal":
        calibration_loader, _, _ = build_dataloaders(
            task.name,
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            num_workers=0,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, metadata = build_loft_model(
        task.num_classes,
        args.support,
        device=device,
        criterion=criterion,
        calibration_loader=calibration_loader,
        calibration_batches=args.calibration_batches,
    )
    synchronize(device)
    support_peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    del calibration_loader

    # Rebuild the loader so calibration never advances the formal shuffle order.
    train_loader, val_loader, test_loader = build_dataloaders(
        task.name,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
    )
    model.to(device)
    optimizer = build_optimizer(model, loft_learning_rate)
    scheduler = build_scheduler(
        optimizer,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    counts = count_parameters(model)

    print(f"Task: {task.name} ({task.group}, {task.num_classes} classes)")
    print(f"Support: {args.support}; device: {device}")
    print(f"Dataset sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"LOFT: all {len(metadata['injected_paths'])} Transformer linears, rank={LOFT_RANK}, exact Cayley")
    print(f"Calibration batches: {metadata['calibration_batches']}; support construction: {metadata['support_construction_seconds']:.2f} seconds")
    print(f"Learning rates: LOFT={loft_learning_rate:g}, classifier={CLASSIFIER_LEARNING_RATE:g}")
    print(f"Parameters: {counts['adapter']:,} adapter; {counts['trainable']:,} trainable / {counts['total']:,} total")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_path,
    )
    synchronize(device)
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    test_metrics = test(model, test_loader, criterion, device)
    synchronize(device)
    test_seconds = time.perf_counter() - start
    training_peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    best_epoch = max(history, key=lambda item: float(item["val_accuracy"]))

    result = {
        "protocol": "development_800_200",
        "method": method,
        "support_type": args.support,
        "transform": "orthogonal_exact_cayley",
        "task": task.name,
        "task_group": task.group,
        "num_classes": task.num_classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "scheduler": "linear_warmup_cosine_decay",
        "targets": ["query", "key", "value", "attention_output", "mlp_fc1", "mlp_fc2"],
        "injected_layers": len(metadata["injected_paths"]),
        "rank": LOFT_RANK,
        "factor_depth": 1,
        "dropout": LOFT_DROPOUT,
        "calibration_batches": metadata["calibration_batches"],
        "support_construction_seconds": metadata["support_construction_seconds"],
        "loft_learning_rate": loft_learning_rate,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "dataset_sizes": {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "loaded_backbone_tensors": metadata["loaded_tensors"],
        "total_parameters": counts["total"],
        "adapter_parameters": counts["adapter"],
        "classifier_parameters": counts["classifier"],
        "trainable_parameters": counts["trainable"],
        "trainable_fraction": counts["trainable"] / counts["total"],
        "support_peak_vram_bytes": support_peak_vram,
        "training_seconds": training_seconds,
        "test_seconds": test_seconds,
        "training_peak_vram_bytes": training_peak_vram,
        "best_epoch": int(best_epoch["epoch"]),
        "best_val_accuracy": float(best_epoch["val_accuracy"]),
        "test": test_metrics,
        "history": history,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\nFinal test result")
    print(f"Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Training peak VRAM: {training_peak_vram / 1024**3:.2f} GiB")
    print(f"Result file: {result_path}")


if __name__ == "__main__":
    main()
