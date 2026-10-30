# LOFT Support Selection on Mini-VTAB-3

## 1. Research Question

Orthogonal fine-tuning is usually discussed in terms of how an orthogonal matrix is parameterized. LOFT separates this into two choices:

1. the low-dimensional support in which adaptation is allowed to act;
2. the transformation learned inside that support.

This experiment asks:

> When rank, transform, parameter count, backbone, and training protocol are controlled, does a support derived from the downstream task outperform the principal singular subspace of the pretrained weight?

We compare Principal, GradSVD, and SkewGrad support on three visual domains using the same rank-46 exact-Cayley transform.

## 2. Mathematical Formulation

Let

\[
W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
\]

be a frozen pretrained linear weight. LOFT selects a row-orthonormal support

\[
P_r\in\mathbb R^{r\times d_{\mathrm{in}}},
\qquad P_rP_r^T=I_r,
\]

and learns an in-support transform $T_r\in O(r)$:

\[
W^+
=W_0\left[I+P_r^T(T_r-I_r)P_r\right].
\]

The complement of the support is unchanged. Because $T_r$ is orthogonal, the embedded full-space transform is also orthogonal.

The trainable transform is generated from a skew-symmetric matrix $Q$:

\[
T_r=(I-Q)(I+Q)^{-1},
\qquad Q^T=-Q.
\]

At $Q=0$, $T_r=I$, so every variant exactly reproduces the pretrained mapping at initialization.

### Support choices

**Principal** uses the leading right-singular subspace of the pretrained weight:

\[
P_r=V_r^T,
\qquad W_0=U\Sigma V^T.
\]

**GradSVD** uses the leading right-singular subspace of the initial downstream gradient:

\[
P_r=V_{G,r}^T,
\qquad G=\nabla_W L(W_0).
\]

**SkewGrad** uses the dominant invariant subspace of

\[
F=\operatorname{skew}(W_0^TG)
=\frac{W_0^TG-G^TW_0}{2}.
\]

In the implementation, the paired leading singular subspace of the real skew-symmetric (F) is used. Rank 46 is even, so complete singular pairs are retained. A final thin QR removes CUDA SVD numerical drift without changing the selected subspace.

## 3. Controlled Implementation

The same adapter is injected into all six linear maps in every ViT encoder block:

- Query, Key, Value;
- attention output;
- MLP fc1 and fc2.

This gives (12\times6=72) adapted layers. Each rank-46 transform stores

\[
\frac{46(46-1)}2=1{,}035
\]

independent skew coordinates. Therefore every support variant has exactly

\[
72\times1{,}035=74{,}520
\]

adapter parameters. The support matrices are fixed buffers, not trainable parameters.

For GradSVD and SkewGrad, a single training-only mini-batch of 64 examples is used to estimate (G). The calibration loader is separate from the formal training loader, so calibration does not advance or change the formal shuffle order. No validation or test examples are used for support construction.

## 4. Experimental Protocol

| Task | Domain | Classes | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| DTD | Natural textures | 47 | 800 | 200 | 1,880 |
| EuroSAT | Satellite imagery | 10 | 800 | 200 | 5,400 |
| dSprites Orientation | Structured synthetic | 16 | 800 | 200 | 73,728 |

Common configuration:

| Setting | Value |
| --- | --- |
| Backbone | ViT-B/16 pretrained on ImageNet-21k |
| Rank / depth | 46 / 1 |
| Transform | Exact Cayley, strictly orthogonal |
| Epochs / warmup | 50 / 5 |
| Optimizer | AdamW |
| Schedule | Epoch-wise linear warmup then cosine decay |
| Classifier LR | `5e-3` |
| Weight decay / adapter dropout | `1e-3` / `0.1` |
| Train / evaluation batch | 64 / 128 |
| Calibration batches | 1 for task-aware support |
| Seed | 42 |
| Selection | Highest validation accuracy |

Task-specific adapter learning rates follow the rank-46 VTAB table in the paper:

| Support | DTD | EuroSAT | dSprites Orientation |
| --- | ---: | ---: | ---: |
| Principal | `2e-4` | `2e-3` | `2e-3` |
| GradSVD | `3.5e-4` | `1.78e-3` | `1.3e-3` |
| SkewGrad | `2e-4` | `8e-4` | `9e-4` |

## 5. Accuracy Results

| Method | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13% | 89.24% | 24.48% | 60.28% |
| LoRA | 69.68% | 94.24% | 49.58% | 71.17% |
| Block OFT | 70.27% | 93.76% | 46.07% | 70.03% |
| LOFT-Principal | 69.79% | **96.33%** | 45.31% | 70.48% |
| LOFT-GradSVD | 70.05% | 95.85% | 49.72% | 71.87% |
| LOFT-SkewGrad | **70.59%** | 94.19% | **52.24%** | **72.34%** |

### Support-only comparison

| Difference from Principal | DTD | EuroSAT | dSprites | Mean |
| --- | ---: | ---: | ---: | ---: |
| GradSVD − Principal | +0.26 | -0.48 | +4.41 | +1.39 |
| SkewGrad − Principal | +0.80 | -2.14 | +6.93 | +1.86 |

Task-aware support is most valuable on dSprites Orientation, where the task depends on geometric structure rather than appearance. SkewGrad also improves DTD, but Principal is substantially stronger on EuroSAT. The experiment therefore supports task-aware support selection as a meaningful design axis without implying universal dominance.

## 6. Validation and Efficiency

| Variant | Task | Best epoch | Best val. | Support time | Train time | Test time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Principal | DTD | 11 | 70.50% | 2.91 s | 247.97 s | 11.27 s |
| Principal | EuroSAT | 20 | 97.00% | 2.92 s | 223.83 s | 17.82 s |
| Principal | dSprites | 8 | 53.00% | 2.91 s | 221.86 s | 135.43 s |
| GradSVD | DTD | 10 | 67.00% | 3.83 s | 228.32 s | 10.94 s |
| GradSVD | EuroSAT | 13 | 96.00% | 3.46 s | 202.79 s | 16.55 s |
| GradSVD | dSprites | 30 | 52.00% | 3.54 s | 202.43 s | 130.75 s |
| SkewGrad | DTD | 37 | 69.00% | 35.72 s | 219.96 s | 10.89 s |
| SkewGrad | EuroSAT | 7 | 95.50% | 34.08 s | 202.70 s | 16.54 s |
| SkewGrad | dSprites | 29 | 55.50% | 33.40 s | 201.30 s | 130.73 s |

Training peak memory is 5.34–5.35 GiB for every LOFT variant. Gradient calibration peaks at 6.97 GiB. Principal support construction peaks at only 0.05 GiB of additional CUDA allocation because each pretrained matrix is decomposed sequentially while the model remains on CPU.

SkewGrad support construction is roughly ten times slower than GradSVD because each MLP fc2 layer requires a decomposition in a 3,072-dimensional input space. It is still a small one-time cost relative to the complete run.

## 7. Parameter Efficiency

| Method | Adapter parameters | Relative to LoRA |
| --- | ---: | ---: |
| LoRA | 294,912 | 100% |
| Block OFT | 211,968 | 71.9% |
| Any LOFT variant | 74,520 | 25.3% |

SkewGrad exceeds the LoRA mean by 1.17 percentage points while using 74.7% fewer adapter parameters. The support buffers consume memory but are fixed, task-specific state rather than optimized parameters.

## 8. Interpretation and Limitations

The strongest evidence is the controlled comparison among the three LOFT variants: they use the same adapter, rank, number of layers, parameter count, dataset split, and seed. Their only structural difference is the support.

Important limitations remain:

1. The experiment covers three tasks and one seed; the paper reports five-seed VTAB-19 averages.
2. Each support uses its paper-reported task-specific learning rate, so this is a reproduction-oriented comparison rather than a common-learning-rate ablation.
3. A 200-example validation set is noisy: one percentage point corresponds to two examples.
4. Only one calibration mini-batch is used. EuroSAT suggests that a noisy initial gradient can produce a worse support than the principal subspace.
5. The baseline runs use 100 epochs, while the paper-aligned LOFT runs use 50 epochs.

A high-value extension is a calibration-size ablation for SkewGrad with $K\in\{1,2,4\}$, followed by seeds 42, 43, and 44 for the final configuration.

## 9. Verification

The final test suite verifies:

- exact equality with the base layer at zero initialization;
- equality of low-dimensional and dense effective-weight forwards;
- orthogonality of both the rank-46 and embedded full-space transforms;
- gradient flow only to skew coordinates and the task head;
- row orthonormality of Principal, GradSVD, and SkewGrad support;
- output preservation after ViT injection;
- exact adapter parameter counts and optimizer groups;
- compatibility with all baseline and pretrained-loading tests.

All six test modules pass.

## 10. Reproduction Commands

```bash
python -m experiments.loft.run_loft --support principal --task dtd
python -m experiments.loft.run_loft --support grad_svd --task dtd
python -m experiments.loft.run_loft --support skew_grad --task dtd
```

Repeat with `eurosat` and `dsprites_orientation`. Machine-readable outputs are stored in `results/loft_<support>/development_800_200/<task>/seed_42.json`.

## 11. Reference

Zhao et al. *LOFT: Low-Rank Orthogonal Fine-Tuning via Task-Aware Support Selection*. arXiv:2605.11872, 2026. <https://arxiv.org/abs/2605.11872>
