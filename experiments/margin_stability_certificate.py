#!/usr/bin/env python3
"""
Experiment 1: Margin Stability Certificate

Visualises the monotonicity margin as a stability certificate for quantised
MonDEQ convergence. Shows a sharp phase transition at ||DW||_2 / m = 1.

Two-panel figure:
  (left)  Iterations to convergence vs normalised perturbation ||DW||_2/m
  (right) Final residual (log) vs normalised perturbation ||DW||_2/m

Usage:
    python experiments/margin_stability_certificate.py
"""
from __future__ import annotations
from pathlib import Path

_project_root = Path(__file__).parent.parent

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.utils import (
    setup_matplotlib, load_pretrained_model, get_model_params,
    get_mnist_loaders, DEVICE, BIT_COLORS,
)
from mondeq.splitting import fb_solve
from mondeq.quant import fake_quant_sym
from mondeq import spectral_bounds

setup_matplotlib()


def main():
    print("=" * 60)
    print("Experiment 1: Margin Stability Certificate")
    print("=" * 60)

    # Load model
    model = load_pretrained_model()
    params = get_model_params(model)
    m = params["m"]
    L = params["L"]
    W = params["W"]
    U, b = params["U"], params["b"]
    A_param, S_param, m_raw = params["A_param"], params["S_param"], params["m_raw"]

    print(f"m = {m:.4f}, L = {L:.4f}, kappa = {L/m:.2f}")

    # Get test batch
    _, test_loader = get_mnist_loaders(test_batch_size=64)
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.view(x_batch.size(0), -1).to(DEVICE)

    # Bit depths to test (finer grid around the transition)
    bit_depths = [3, 4, 5, 6, 7, 8, 10, 12, 16, 32]

    max_iters = 2000
    tol = 1e-5

    results = []
    for bits in bit_depths:
        W_q = fake_quant_sym(W, bits)
        delta_W_norm = float(torch.linalg.norm(W - W_q, 2))
        ratio = delta_W_norm / m

        # Compute quantised spectral bounds
        I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        m_q, L_q = spectral_bounds(I_n - W_q)
        m_q, L_q = float(m_q), float(L_q)

        # Use quantised-optimal alpha if convergent, else use a safe default
        if m_q > 0:
            alpha_q = m_q / (L_q ** 2)
        else:
            # System won't converge — use a small alpha and let it run
            alpha_q = 0.01

        z_q, hist_q = fb_solve(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_q, max_iters=max_iters, tol=tol,
            W=W_q,
        )

        iters = hist_q["iters"]
        final_res = hist_q["residual"][-1] if hist_q["residual"] else float("inf")
        converged = iters < max_iters

        results.append({
            "bits": bits,
            "delta_W_norm": delta_W_norm,
            "ratio": ratio,
            "iters": iters,
            "final_res": final_res,
            "converged": converged,
            "m_tilde": m_q,
            "L_tilde": L_q,
        })

        status = "CONVERGED" if converged else "DIVERGED"
        print(f"  {bits:2d}-bit: ||DW||_2/m = {ratio:.3f}, iters = {iters:4d}, "
              f"res = {final_res:.2e}, m_tilde = {m_q:.4f}  [{status}]")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ratios = [r["ratio"] for r in results]
    iters_list = [r["iters"] for r in results]
    residuals = [r["final_res"] for r in results]
    converged = [r["converged"] for r in results]
    bits_list = [r["bits"] for r in results]

    # Left panel: iterations vs ratio
    for i, r in enumerate(results):
        color = BIT_COLORS.get(r["bits"], "#333333")
        marker = "o" if r["converged"] else "x"
        ax1.plot(r["ratio"], r["iters"], marker, color=color, markersize=6,
                 label=f'{r["bits"]}b')

    ax1.axvline(x=1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
                label=r"$\|\Delta W\|_2 / m = 1$")
    ax1.set_xlabel(r"$\|\Delta W\|_2 / m$")
    ax1.set_ylabel("Iterations")
    ax1.set_title("(a) Iterations to convergence")
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.set_xlim(left=0)
    ax1.grid(True, alpha=0.3)

    # Right panel: final residual vs ratio
    for i, r in enumerate(results):
        color = BIT_COLORS.get(r["bits"], "#333333")
        marker = "o" if r["converged"] else "x"
        ax2.semilogy(r["ratio"], r["final_res"], marker, color=color, markersize=6,
                     label=f'{r["bits"]}b')

    ax2.axvline(x=1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.set_xlabel(r"$\|\Delta W\|_2 / m$")
    ax2.set_ylabel("Final residual")
    ax2.set_title("(b) Final residual")
    ax2.set_xlim(left=0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / "margin_stability.pdf"
    fig.savefig(out_path, dpi=300)
    print(f"\nFigure saved to {out_path}")

    # Also save PNG for quick preview
    fig.savefig(figures_dir / "margin_stability.png", dpi=150)

    plt.close(fig)

    # Print summary table
    print("\nSummary:")
    print(f"{'Bits':>5} {'||DW||_2/m':>12} {'Iters':>6} {'Residual':>12} {'Status':>10}")
    print("-" * 50)
    for r in results:
        status = "OK" if r["converged"] else "FAIL"
        print(f'{r["bits"]:>5} {r["ratio"]:>12.3f} {r["iters"]:>6} '
              f'{r["final_res"]:>12.2e} {status:>10}')


if __name__ == "__main__":
    main()
