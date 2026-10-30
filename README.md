# LOFT Task-Aware Support Study

A transparent ViT-B/16 reproduction of **LOFT: Low-Rank Orthogonal Fine-Tuning via Task-Aware Support Selection**. The project isolates the adaptation support while keeping the rank, orthogonal transform, trainable parameter count, backbone, and data protocol fixed.

Three support choices are implemented:

- **Principal** — top right-singular subspace of the frozen pretrained weight.
- **GradSVD** — top right-singular subspace of the initial downstream gradient.
- **SkewGrad** — top invariant subspace of the task signal `skew(W^T G)`.

Linear Probe, LoRA, and Block OFT are retained as reference baselines.

## Results

All LOFT runs use rank 46, one orthogonal factor, 72 adapted Transformer linears, seed 42, and the best checkpoint on a 200-example validation set.

| Method | DTD | EuroSAT | dSprites Orientation | Mean | Adapter parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13% | 89.24% | 24.48% | 60.28% | 0 |
| LoRA, rank 8, Q+V | 69.68% | 94.24% | 49.58% | 71.17% | 294,912 |
| Block OFT, block 24, Q+V | 70.27% | 93.76% | 46.07% | 70.03% | 211,968 |
| LOFT-Principal | 69.79% | **96.33%** | 45.31% | 70.48% | **74,520** |
| LOFT-GradSVD | 70.05% | 95.85% | 49.72% | 71.87% | **74,520** |
| LOFT-SkewGrad | **70.59%** | 94.19% | **52.24%** | **72.34%** | **74,520** |

SkewGrad improves over Principal by 6.93 points on dSprites Orientation and by 1.86 mean points across the three tasks. It also exceeds the LoRA mean by 1.17 points while using 74.7% fewer adapter parameters. Principal remains strongest on EuroSAT, showing that task-aware support is useful but not universally superior.

These are single-seed Mini-VTAB-3 results, not an exact reproduction of the paper's five-seed VTAB-19 aggregate. See the [full reproduction report](reports/loft_reproduction.md) for protocol details, timing, memory, limitations, and interpretation.

## Method

For a frozen weight `W`, row-orthonormal support `P`, and trainable orthogonal transform `T`, LOFT uses

```text
W+ = W [I + P^T (T - I) P].
```

`P` determines where adaptation acts. `T` is generated from 1,035 independent skew coordinates through an exact Cayley transform. At initialization `T=I`, so the adapted model exactly equals the pretrained model.

For gradient-informed variants, one training mini-batch is used once before training to estimate each target weight gradient. The resulting supports are then fixed for the complete run.

## Configuration

| Setting | Value |
| --- | --- |
| Backbone | `google/vit-base-patch16-224-in21k` |
| Tasks | DTD, EuroSAT, dSprites Orientation |
| Train / validation samples | 800 / 200 per task |
| Target modules | Q, K, V, attention output, MLP fc1, MLP fc2 |
| Injected layers | 72 |
| Rank / factor depth | 46 / 1 |
| Transform | Exact Cayley orthogonal transform |
| Calibration batches | 0 for Principal, 1 for GradSVD/SkewGrad |
| Epochs / warmup | 50 / 5 |
| Classifier learning rate | `5e-3` |
| Batch size | 64 training, 128 evaluation |
| Weight decay / dropout | `1e-3` / `0.1` |
| Optimizer / schedule | AdamW / cosine |
| Seed | 42 |

The LOFT learning rates are the paper's rank-46 task-specific values:

| Support | DTD | EuroSAT | dSprites Orientation |
| --- | ---: | ---: | ---: |
| Principal | `2e-4` | `2e-3` | `2e-3` |
| GradSVD | `3.5e-4` | `1.78e-3` | `1.3e-3` |
| SkewGrad | `2e-4` | `8e-4` | `9e-4` |

## Repository Structure

```text
.
|-- experiments/
|   |-- linear_probe/          # Frozen-backbone baseline
|   |-- lora/                  # Rank-8 Q/V LoRA baseline
|   |-- oft/                   # Block-OFT baseline
|   `-- loft/                  # LOFT layer, support construction, and runner
|-- vit_core/                  # Shared ViT, data, training, and evaluation
|-- scripts/                   # Dataset and pretrained-weight preparation
|-- tests/                     # Algebra, support, injection, and smoke tests
|-- results/                   # Machine-readable formal results
|-- reports/                   # Baseline and LOFT reports
|-- data/                      # Local only; ignored by Git
|-- pretrained/                # Local only; ignored by Git
`-- checkpoints/               # Local only; ignored by Git
```

## Reproduction

After preparing the existing Mini-VTAB-3 data and ViT weights:

```bash
python -m experiments.loft.run_loft --support principal --task dtd
python -m experiments.loft.run_loft --support grad_svd --task dtd
python -m experiments.loft.run_loft --support skew_grad --task dtd
```

Replace `dtd` with `eurosat` or `dsprites_orientation` to reproduce the other tasks.

## Verification

```bash
python -m tests.test_pretrained_loading
python -m tests.test_linear_probe
python -m tests.test_lora
python -m tests.test_oft
python -m tests.test_loft_layer
python -m tests.test_loft_integration
```

## References

- [LOFT: Low-Rank Orthogonal Fine-Tuning via Task-Aware Support Selection](https://arxiv.org/abs/2605.11872)
- [Visual Task Adaptation Benchmark](https://arxiv.org/abs/1910.04867)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Orthogonal Fine-Tuning](https://arxiv.org/abs/2306.07280)
