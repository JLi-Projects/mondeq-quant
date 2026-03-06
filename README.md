# Quantisation of Monotone Operator Equilibrium Networks

Code companion for the paper:

> J. Li, P. H. W. Leong, and T. Chaffey, "Quantisation of Monotone Operator Equilibrium Networks," *IEEE Control Systems Letters (L-CSS)*, 2026. [arXiv:TBD]

## Overview

Monotone operator deep equilibrium networks (MonDEQs) parameterise the weight matrix $W$ so that the operator $F(z) = (I - W)z - (Ux + b)$ is strongly monotone with margin $m > 0$, guaranteeing a unique equilibrium $z^\star$ found by operator splitting. This repository provides the code to reproduce the paper's three main results:

1. **Convergence certificate** (Theorem 2 + Corollary 2): weight quantisation preserves monotonicity (and hence convergence) if $\|\Delta W\|_2 < m$, where $\Delta W = W - \widetilde{W}$ is the quantisation perturbation.
2. **Equilibrium displacement bound** (Theorem 3): $\|z^\star - \widetilde{z}^\star\| \leq \frac{\|\Delta W\|_2}{m}\|\widetilde{z}^\star\|$.
3. **Condition number** (Theorem 4): the relative condition number of the equilibrium map is $\kappa_{\mathrm{rel}} = \|W\|_2 / m$.

## Repository structure

```
mondeq-quant/
├── README.md
├── LICENSE                          # MIT
├── pyproject.toml                   # Package metadata
├── src/
│   ├── mondeq/                      # Core MonDEQ package
│   │   ├── operators.py             # Winston-Kolter parameterisation, spectral bounds
│   │   ├── splitting.py             # FB, PR, DR solvers with W override
│   │   ├── prox.py                  # ReLU proximal operator
│   │   ├── quant.py                 # Fake quantisation, iterate quantisers
│   │   └── layers/fc.py             # MonDEQLayerFC with implicit differentiation
│   ├── models/                      # MNIST classifier wrappers
│   │   ├── mnist_mondeq.py          # MNISTMonDEQ (training model)
│   │   └── quant_wrapper.py         # QuantisedMonDEQ for PTQ evaluation
│   ├── training/                    # Training infrastructure
│   │   ├── train_mnist.py           # Training loop with alpha tuning
│   │   ├── train_qat.py             # Quantisation-aware training
│   │   └── evaluate.py              # Evaluation utilities
│   └── utils/
│       └── splitting_stats.py       # Solver statistics
├── experiments/
│   ├── utils.py                     # Shared experiment utilities (model loading, plotting)
│   ├── train_and_quantise.py        # Full training + PTQ pipeline
│   ├── margin_stability_certificate.py  # Experiment 1 / Figure 1
│   ├── splitting_comparison.py          # Experiment 2 / Figure 2
│   ├── qat_vs_ptq.py                   # Experiment 3 / Figure 3
│   ├── iterate_quantisation.py          # Corollary 5 validation (iterate quantisation)
│   ├── displacement_validation.py       # Theorem 3 validation / Figure 4
│   └── regenerate_figures.py            # Regenerate all paper figures
├── tests/                           # Unit tests
├── checkpoints/
│   ├── mnist_mondeq_float.pt        # Pretrained float model (98.22% accuracy)
│   └── experiment_results.json      # Key numerical results
└── figures/                         # Paper figure PDFs
```

## Installation

```bash
git clone https://github.com/JLi-Projects/mondeq-quant.git
cd mondeq-quant
pip install -e ".[dev]"
```

Requirements: Python >= 3.9, PyTorch >= 2.0, torchvision >= 0.15.

## Quick start

Run the pretrained model through the PTQ sweep (no training required):

```bash
python experiments/train_and_quantise.py --skip-training
```

Run individual paper experiments:

```bash
# Experiment 1: Margin stability certificate (Fig 1)
python experiments/margin_stability_certificate.py

# Experiment 2: FB vs PR vs DR splitting comparison (Fig 2)
python experiments/splitting_comparison.py

# Experiment 3: QAT vs PTQ (Fig 3)
python experiments/qat_vs_ptq.py

# Theorem 3 validation: Displacement bound (Fig 4)
python experiments/displacement_validation.py

# Corollary 5: Fixed vs adaptive iterate quantisation
python experiments/iterate_quantisation.py

# Regenerate all paper figures
python experiments/regenerate_figures.py
```

## Mathematical background

A **MonDEQ** defines a fixed-point equation through a monotone inclusion $0 \in F(z) + G(z)$, where $F(z) = (I - W)z - (Ux + b)$ is strongly monotone and $G$ is the ReLU subdifferential (maximal monotone). The equilibrium $z^\star$ is found by forward-backward splitting:

$$z^{k+1} = J_{\alpha G}\bigl((I - \alpha F)z^k\bigr)$$

**Winston-Kolter parameterisation.** The weight matrix is parameterised as $W = (1-m)I - A^\top A + S - S^\top$ where $m > 0$ is the learnable monotonicity margin. This guarantees $\lambda_{\min}(\operatorname{sym}(I - W)) \geq m$.

**Quantisation perturbs the margin.** Under weight quantisation $\widetilde{W} = W + \Delta W$, the perturbed margin satisfies $\widetilde{m} \geq m - \|\Delta W\|_2$ (Theorem 2). The quantised system converges if and only if $\widetilde{m} > 0$; the sufficient condition $\|\Delta W\|_2 < m$ is checked without running the solver.

## Math-to-code mapping

| Mathematical object | File | Function / Class |
|---|---|---|
| $W = (1-m)I - A^\top A + S - S^\top$ | `operators.py` | `WKLinearFC` |
| $m = \lambda_{\min}(\operatorname{sym}(I-W))$, $L = \|I-W\|_2$ | `operators.py` | `spectral_bounds()` |
| $J_{\alpha G} = \operatorname{prox}_{\alpha g}$ (ReLU) | `prox.py` | `relu_prox()` |
| Forward-backward splitting | `splitting.py` | `fb_solve()` |
| Peaceman-Rachford splitting | `splitting.py` | `pr_solve()` |
| Douglas-Rachford splitting | `splitting.py` | `dr_solve()` |
| $Q_b(w) = \Delta \cdot \operatorname{round}(w/\Delta)$ | `quant.py` | `fake_quant_sym()` |
| Fixed iterate quantisation | `quant.py` | `fixed_iterate_quant()` |
| Adaptive iterate quantisation | `quant.py` | `adaptive_iterate_quant()` |
| Implicit differentiation | `layers/fc.py` | `MonDEQLayerFC` |

### Winston-Kolter parameterisation

`WKLinearFC` stores learnable parameters `A`, `S`, `m_raw`. The margin $m = \operatorname{softplus}(m_{\mathrm{raw}})$ is always positive. The weight matrix $W$ is assembled on the fly via `_build_W(A, S, m_raw)` in `splitting.py`.

### Splitting solvers

All three solvers (`fb_solve`, `pr_solve`, `dr_solve`) accept an optional `W=` argument to override the weight matrix. This is how quantised evaluation works: build $W$ from the parameterisation, quantise it with `fake_quant_sym(W, bits)`, and pass `W=W_q` to the solver. The solver sees the quantised operator without modifying the underlying parameters.

### Quantisation

`fake_quant_sym(x, bits)` implements a symmetric uniform mid-tread quantiser: $Q(x) = \Delta \cdot \operatorname{round}(x / \Delta)$, $\Delta = 2 x_{\max} / (2^b - 1)$. Gradients pass through via the straight-through estimator (STE), enabling quantisation-aware training.

## Training

The pretrained model (`checkpoints/mnist_mondeq_float.pt`) uses the following architecture and training setup:

| Parameter | Value |
|---|---|
| Architecture | Single MonDEQ layer, $n = 100$ hidden units |
| Input | MNIST, flattened to 784 dimensions |
| Optimizer | Adam, lr = $10^{-3}$ |
| LR schedule | StepLR, $\gamma = 0.1$ at epoch 10 |
| Epochs | 15 |
| Margin | Learnable via softplus($m_{\mathrm{raw}}$) |
| Solver | Forward-backward, 500 max iterations, tol = $10^{-5}$ |
| Seed | 42 |

**Pretrained model statistics:**

| Metric | Value |
|---|---|
| Test accuracy | 98.22% |
| Monotonicity margin $m$ | 0.2269 |
| Lipschitz constant $L$ | 1.845 |
| Condition number $\kappa = L/m$ | 8.13 |

To retrain from scratch:

```bash
python experiments/train_and_quantise.py --epochs 15 --seed 42
```

## Experiments

### Experiment 1: Margin stability certificate (Figure 1)

**Paper reference:** Theorem 2 (margin perturbation) and Corollary 2 (convergence condition).

**What it tests:** Quantises $W$ at bit depths 3-32 and checks whether the solver converges. The normalised perturbation $\|\Delta W\|_2 / m$ predicts a phase transition at 1: below it, the solver converges; above it, monotonicity is lost.

**Key finding:** Sharp phase transition at $\|\Delta W\|_2 / m = 1$. At 5-bit ($\|\Delta W\|_2/m = 1.25$), the Weyl bound predicts failure, but the actual margin $\widetilde{m} = 0.045 > 0$ and the solver converges — the sufficient condition is conservative.

```bash
python experiments/margin_stability_certificate.py
```

### Experiment 2: Splitting comparison (Figure 2)

**Paper reference:** Corollaries 3-4 (FB and PR contraction rates under quantisation).

**What it tests:** Compares FB, PR, and DR splitting under FP32, 8-bit, and 6-bit weight quantisation. PR and DR converge much faster than FB (condition-number dependence: $O(\kappa)$ vs $O(\kappa^2)$).

**Key finding:** PR converges in ~50 iterations vs FB's ~300+ ($\kappa = 8.13$). Quantisation barely affects PR/DR iteration counts. All three solvers converge to the same fixed point (relative diff $< 3 \times 10^{-7}$).

```bash
python experiments/splitting_comparison.py
```

### Experiment 3: QAT vs PTQ (Figure 3)

**Paper reference:** Section V-C.

**What it tests:** Compares post-training quantisation and quantisation-aware training at 4, 6, 8 bits. QAT trains from scratch with STE gradients through the quantiser.

**Key finding:** QAT rescues 4-bit (96.78% accuracy, $\widetilde{m} = 0.006 > 0$) where PTQ fails ($\widetilde{m} = -0.14$). However, vanilla QAT learns a smaller margin than PTQ inherits from float training (0.12-0.18 vs 0.23).

```bash
python experiments/qat_vs_ptq.py --epochs 15 --seed 42
```

### Experiment 4: Iterate quantisation (Corollary 5)

**Paper reference:** Corollary 5 (inexact convergence with activation quantisation).

**What it tests:** Fixed iterate quantisation produces an error floor proportional to cell width $\delta$. Adaptive (geometrically shrinking) iterate quantisation converges to the exact fixed point.

**Key finding:** Error floor at $\varepsilon/(1-r)$ for fixed $\delta$. Adaptive quantisation with $\gamma = 0.5$ or $\gamma = 0.9$ converges to machine precision.

```bash
python experiments/iterate_quantisation.py
```

### Displacement validation (Figure 4)

**Paper reference:** Theorem 3 (equilibrium displacement bound).

**What it tests:** For 6, 8, 12, 16, 32-bit quantisation, computes the empirical displacement $\|z^\star - \widetilde{z}^\star\| / \|z^\star\|$ on 1000 test samples and compares against the theoretical bound $\|\Delta W\|_2 / m \cdot \|\widetilde{z}^\star\| / \|z^\star\|$.

**Key finding:** Bound satisfied in 91-99% of test samples across bit depths. Points below the $y = x$ line validate Theorem 3.

```bash
python experiments/displacement_validation.py
```

## Results summary

| Bits | Converges | Float Acc | Quant Acc | Drop | $\|\Delta W\|_2$ |
|------|-----------|-----------|-----------|------|-------------------|
| 4 | No | -- | -- | -- | 0.604 > $m$ |
| 6 | Yes | 98.22% | 98.18% | 0.04% | 0.137 |
| 8 | Yes | 98.22% | 98.24% | -0.02% | 0.035 |
| 12 | Yes | 98.22% | 98.22% | 0.00% | 0.002 |
| 16 | Yes | 98.22% | 98.22% | 0.00% | 0.0001 |
| 32 | Yes | 98.22% | 98.22% | 0.00% | ~0 |

## Tests

```bash
python -m pytest tests/ -v
```

The tests verify:
- Winston-Kolter parameterisation produces a strongly monotone operator ($m > 0$)
- Forward-backward solver converges and respects the theoretical contraction rate
- Backward pass (implicit differentiation) produces finite gradients
- Backward gradients match finite differences (relative error $< 5\%$)

## Citation

```bibtex
@article{li2026quantisation,
  author  = {Li, James and Leong, Philip H. W. and Chaffey, Thomas},
  title   = {Quantisation of Monotone Operator Equilibrium Networks},
  journal = {IEEE Control Systems Letters},
  year    = {2026},
  note    = {In preparation}
}
```

## License

MIT. See [LICENSE](LICENSE).
