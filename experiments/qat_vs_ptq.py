#!/usr/bin/env python3
"""
Experiment 4: QAT vs PTQ Comparison

Compares Quantisation-Aware Training (QAT) and Post-Training Quantisation (PTQ)
at various bit depths (4, 6, 8).

For each bit depth:
  - PTQ: quantise the pretrained FP32 model (W quantised directly)
  - QAT: train from scratch with quantisation in the loop (15 epochs)

Compares: accuracy, learned margin m, ||DW||_2, convergence success.

Note: QAT trains from scratch (not fine-tuning the PTQ baseline). This is a
harder optimisation problem and the margin comparison is between independently
trained models. The conclusion is that vanilla QAT (without margin-aware
regularisation) does not preserve the margin that PTQ inherits from float training.

Usage:
    python experiments/qat_vs_ptq.py [--epochs 15] [--seed 42]
"""
from __future__ import annotations
import argparse
from pathlib import Path

_project_root = Path(__file__).parent.parent

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.utils import (
    setup_matplotlib, load_pretrained_model, get_model_params,
    get_mnist_loaders, evaluate_accuracy, DEVICE,
)
from mondeq.splitting import fb_solve, pr_solve, optimal_pr_alpha, _build_W
from mondeq.quant import fake_quant_sym
from mondeq import spectral_bounds
from training.train_qat import train_qat

setup_matplotlib()


def evaluate_ptq(model, bits, x_batch, test_loader):
    """Evaluate PTQ at a given bit depth.

    Uses direct W quantisation (matching Exp 1 methodology) rather than
    quantising A, S parameters individually.
    """
    params = get_model_params(model)
    W = params["W"]
    m = params["m"]
    L = params["L"]

    W_q = fake_quant_sym(W, bits)
    delta_W = float(torch.linalg.norm(W - W_q, 2))

    I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    m_q, L_q = spectral_bounds(I_n - W_q)
    m_q, L_q = float(m_q), float(L_q)

    # Use actual computed margin for convergence check (not parametric lower bound)
    converges = m_q > 0

    # Evaluate accuracy using PR solver with directly-quantised W
    # (consistent with the convergence analysis based on ||Q(W) - W||_2;
    # PR used for speed — converges in ~50 iters vs ~500 for FB)
    acc = None
    if converges:
        U_q = fake_quant_sym(params["U"], bits)
        b_q = fake_quant_sym(params["b"], bits)
        alpha_q = optimal_pr_alpha(m_q, L_q)
        output_layer = model.output
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                x_flat = data.view(data.size(0), -1)
                z_star, _ = pr_solve(
                    params["A_param"], params["S_param"], params["m_raw"],
                    U_q, b_q, x_flat,
                    alpha=alpha_q, max_iters=500, tol=1e-5, W=W_q,
                )
                logits = output_layer(z_star.T)
                correct += logits.argmax(1).eq(target).sum().item()
                total += len(target)
        acc = 100.0 * correct / total

    return {
        "bits": bits,
        "method": "PTQ",
        "accuracy": acc,
        "m": m,
        "m_tilde": m_q,
        "delta_W_norm": delta_W,
        "ratio": delta_W / m,
        "converges": converges,
    }


def evaluate_qat(model_qat, bits, test_loader):
    """Evaluate a QAT-trained model.

    Uses high-precision solver settings for evaluation (same as PTQ)
    to ensure a fair accuracy comparison. Training uses loose
    tol=1e-2/max_iters=100 for speed; evaluation uses tol=1e-5/500.
    """
    core = model_qat.mon.core
    m_val = float(core.m.detach())

    with torch.no_grad():
        W = _build_W(core.A, core.S, core.m_raw)
        W_q = fake_quant_sym(W, bits)
        delta_W = float(torch.linalg.norm(W - W_q, 2))

    I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    m_q, L_q = spectral_bounds(I_n - W_q)
    m_q, L_q = float(m_q), float(L_q)

    # Use actual computed margin for convergence check (not parametric lower bound)
    converges = m_q > 0

    acc = None
    if converges:
        # Use high-precision solver for evaluation (matching PTQ evaluation)
        U_q = fake_quant_sym(core.U.data, bits)
        b_q = fake_quant_sym(core.b.data, bits)
        alpha_q = optimal_pr_alpha(m_q, L_q)
        output_layer = model_qat.output
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                x_flat = data.view(data.size(0), -1)
                z_star, _ = pr_solve(
                    core.A.data, core.S.data, core.m_raw.data,
                    U_q, b_q, x_flat,
                    alpha=alpha_q, max_iters=500, tol=1e-5, W=W_q,
                )
                logits = output_layer(z_star.T)
                correct += logits.argmax(1).eq(target).sum().item()
                total += len(target)
        acc = 100.0 * correct / total

    return {
        "bits": bits,
        "method": "QAT",
        "accuracy": acc,
        "m": m_val,
        "m_tilde": m_q,
        "delta_W_norm": delta_W,
        "ratio": delta_W / m_val if m_val > 0 else float("inf"),
        "converges": converges,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 4: QAT vs PTQ Comparison")
    print("=" * 60)

    # Load pretrained model for PTQ
    model = load_pretrained_model()
    train_loader, test_loader = get_mnist_loaders(train_batch_size=128, test_batch_size=256)

    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.view(x_batch.size(0), -1).to(DEVICE)

    bit_depths = [4, 6, 8]
    all_results = []

    # --- PTQ evaluation ---
    print("\n--- PTQ Results ---")
    for bits in bit_depths:
        result = evaluate_ptq(model, bits, x_batch, test_loader)
        all_results.append(result)
        acc_str = f"{result['accuracy']:.2f}%" if result["accuracy"] is not None else "N/A"
        print(f"  {bits}-bit PTQ: acc={acc_str}, m={result['m']:.4f}, "
              f"||DW||_2={result['delta_W_norm']:.4f}, "
              f"ratio={result['ratio']:.3f}, converges={result['converges']}")

    # --- QAT training and evaluation ---
    print("\n--- QAT Training ---")
    qat_histories = {}
    for bits in bit_depths:
        print(f"\nTraining QAT at {bits}-bit...")
        model_qat, history = train_qat(
            train_loader, test_loader,
            bits=bits,
            epochs=args.epochs,
            seed=args.seed,
            verbose=True,
        )

        qat_histories[bits] = history

        result = evaluate_qat(model_qat, bits, test_loader)
        all_results.append(result)
        acc_str = f"{result['accuracy']:.2f}%" if result["accuracy"] is not None else "N/A"
        print(f"  {bits}-bit QAT: acc={acc_str}, m={result['m']:.4f}, "
              f"||DW||_2={result['delta_W_norm']:.4f}, "
              f"ratio={result['ratio']:.3f}, converges={result['converges']}")

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"{'Method':>8} {'Bits':>5} {'Acc':>8} {'m':>8} {'m_tilde':>8} "
          f"{'||DW||_2':>10} {'Ratio':>8} {'Conv':>6}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: (x["bits"], x["method"])):
        acc_str = f"{r['accuracy']:.2f}" if r["accuracy"] is not None else "N/A"
        print(f"{r['method']:>8} {r['bits']:>5} {acc_str:>8} {r['m']:>8.4f} "
              f"{r['m_tilde']:>8.4f} {r['delta_W_norm']:>10.4f} "
              f"{r['ratio']:>8.3f} {'Yes' if r['converges'] else 'No':>6}")

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    ptq_results = [r for r in all_results if r["method"] == "PTQ"]
    qat_results = [r for r in all_results if r["method"] == "QAT"]

    # Panel (a): Accuracy comparison
    ax = axes[0]
    x_pos = np.arange(len(bit_depths))
    width = 0.35

    ptq_accs = [r["accuracy"] if r["accuracy"] is not None else 0 for r in ptq_results]
    qat_accs = [r["accuracy"] if r["accuracy"] is not None else 0 for r in qat_results]

    bars_ptq = ax.bar(x_pos - width/2, ptq_accs, width, label="PTQ", color="#1f77b4")
    bars_qat = ax.bar(x_pos + width/2, qat_accs, width, label="QAT", color="#ff7f0e")

    # Mark non-converging cases
    for i, r in enumerate(ptq_results):
        if not r["converges"]:
            ax.text(x_pos[i] - width/2, 1, "X", ha="center", va="bottom",
                    fontsize=8, color="red", fontweight="bold")
    for i, r in enumerate(qat_results):
        if not r["converges"]:
            ax.text(x_pos[i] + width/2, 1, "X", ha="center", va="bottom",
                    fontsize=8, color="red", fontweight="bold")

    ax.set_xlabel("Bit depth")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("(a) Accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in bit_depths])
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

    # Panel (b): Learned margin
    ax = axes[1]
    ptq_margins = [r["m"] for r in ptq_results]
    qat_margins = [r["m"] for r in qat_results]

    ax.bar(x_pos - width/2, ptq_margins, width, label="PTQ", color="#1f77b4")
    ax.bar(x_pos + width/2, qat_margins, width, label="QAT", color="#ff7f0e")
    ax.set_xlabel("Bit depth")
    ax.set_ylabel("Margin $m$")
    ax.set_title("(b) Monotonicity margin")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in bit_depths])
    ax.legend(fontsize=7)

    # Panel (c): ||DW||_2 / m ratio
    ax = axes[2]
    ptq_ratios = [r["ratio"] for r in ptq_results]
    qat_ratios = [r["ratio"] for r in qat_results]

    ax.bar(x_pos - width/2, ptq_ratios, width, label="PTQ", color="#1f77b4")
    ax.bar(x_pos + width/2, qat_ratios, width, label="QAT", color="#ff7f0e")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
               label="Threshold")
    ax.set_xlabel("Bit depth")
    ax.set_ylabel(r"$\|\Delta W\|_2 / m$")
    ax.set_title("(c) Convergence ratio")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in bit_depths])
    ax.legend(fontsize=7)

    plt.tight_layout()

    # Save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / "qat_vs_ptq.pdf"
    fig.savefig(out_path, dpi=300)
    fig.savefig(figures_dir / "qat_vs_ptq.png", dpi=150)
    print(f"\nFigure saved to {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
