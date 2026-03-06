#!/usr/bin/env python3
"""
Experiment 2: FB vs PR vs DR Splitting Comparison

Compares Forward-Backward, Peaceman-Rachford, and Douglas-Rachford splitting
methods under weight quantisation (FP32, 8-bit, 6-bit).

Produces a Jonkman-style figure: log-scale residual vs iteration for
3 solvers x 3 weight configs = 9 curves.

Usage:
    python experiments/splitting_comparison.py
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
    get_mnist_loaders, DEVICE, SOLVER_STYLES, BIT_COLORS,
)
from mondeq.splitting import fb_solve, pr_solve, dr_solve, optimal_pr_alpha
from mondeq.quant import fake_quant_sym
from mondeq import spectral_bounds

setup_matplotlib()


def main():
    print("=" * 60)
    print("Experiment 2: FB vs PR vs DR Splitting Comparison")
    print("=" * 60)

    # Load model
    model = load_pretrained_model()
    params = get_model_params(model)
    m = params["m"]
    L = params["L"]
    W = params["W"]
    U, b = params["U"], params["b"]
    A_param, S_param, m_raw = params["A_param"], params["S_param"], params["m_raw"]

    print(f"Float model: m = {m:.4f}, L = {L:.4f}, kappa = {L/m:.2f}")

    # Get test batch
    _, test_loader = get_mnist_loaders(test_batch_size=64)
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.view(x_batch.size(0), -1).to(DEVICE)

    # Configurations: (label, bits, W_override)
    configs = [
        ("FP32", None, W),  # Use original W
        ("8-bit", 8, fake_quant_sym(W, 8)),
        ("6-bit", 6, fake_quant_sym(W, 6)),
    ]

    max_iters = 300
    tol = 1e-10  # Very tight to see full convergence curve

    # Collect all results
    all_results = {}

    for config_label, bits, W_cfg in configs:
        # Compute spectral bounds for this config
        I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        m_cfg, L_cfg = spectral_bounds(I_n - W_cfg)
        m_cfg, L_cfg = float(m_cfg), float(L_cfg)

        if bits is not None:
            delta_W = float(torch.linalg.norm(W - W_cfg, 2))
            print(f"\n{config_label}: m_tilde={m_cfg:.4f}, L_tilde={L_cfg:.4f}, "
                  f"||DW||_2={delta_W:.4f}, ||DW||_2/m={delta_W/m:.3f}")
        else:
            print(f"\n{config_label}: m={m_cfg:.4f}, L={L_cfg:.4f}")

        if m_cfg <= 0:
            print(f"  Skipping {config_label}: margin is non-positive")
            continue

        # Optimal step sizes
        alpha_fb = m_cfg / (L_cfg ** 2)
        alpha_pr = optimal_pr_alpha(m_cfg, L_cfg)

        # FB solve
        z_fb, hist_fb = fb_solve(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_fb, max_iters=max_iters, tol=tol, W=W_cfg,
        )
        all_results[(config_label, "FB")] = hist_fb["residual"]
        print(f"  FB:  {len(hist_fb['residual']):3d} iters, "
              f"alpha={alpha_fb:.4f}, final_res={hist_fb['residual'][-1]:.2e}")

        # PR solve
        z_pr, hist_pr = pr_solve(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_pr, max_iters=max_iters, tol=tol, W=W_cfg,
        )
        all_results[(config_label, "PR")] = hist_pr["residual"]
        print(f"  PR:  {len(hist_pr['residual']):3d} iters, "
              f"alpha={alpha_pr:.4f}, final_res={hist_pr['residual'][-1]:.2e}")

        # DR solve
        z_dr, hist_dr = dr_solve(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_pr, max_iters=max_iters, tol=tol, W=W_cfg,
        )
        all_results[(config_label, "DR")] = hist_dr["residual"]
        print(f"  DR:  {len(hist_dr['residual']):3d} iters, "
              f"alpha={alpha_pr:.4f}, final_res={hist_dr['residual'][-1]:.2e}")

        # Verify all three solvers converge to the same fixed point.
        # PR and DR should agree tightly. FB converges slower (O(1/kappa^2)
        # vs O(1/kappa) for PR), so compare it at its own residual level.
        rel_fb_pr = float((z_fb - z_pr).norm() / (z_fb.norm() + 1e-12))
        rel_fb_dr = float((z_fb - z_dr).norm() / (z_fb.norm() + 1e-12))
        rel_pr_dr = float((z_pr - z_dr).norm() / (z_pr.norm() + 1e-12))
        print(f"  Solver agreement: ||z_FB - z_PR||/||z_FB|| = {rel_fb_pr:.2e}, "
              f"||z_FB - z_DR||/||z_FB|| = {rel_fb_dr:.2e}, "
              f"||z_PR - z_DR||/||z_PR|| = {rel_pr_dr:.2e}")
        assert rel_pr_dr < 1e-4, f"PR and DR disagree: {rel_pr_dr:.2e}"
        # FB-PR gap is bounded by FB's residual — it's converging to the same point
        fb_res = hist_fb["residual"][-1]
        if fb_res < 1e-4:
            assert rel_fb_pr < 1e-2, f"FB and PR disagree: {rel_fb_pr:.2e}"

    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    config_colors = {"FP32": "#000000", "8-bit": "#1f77b4", "6-bit": "#ff7f0e"}

    for (config_label, solver_name), residuals in all_results.items():
        style = SOLVER_STYLES[solver_name]
        color = config_colors[config_label]
        label = f"{solver_name} ({config_label})"

        ax.semilogy(
            range(1, len(residuals) + 1), residuals,
            linestyle=style["linestyle"],
            color=color,
            linewidth=1.0,
            label=label,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative residual")
    ax.set_title("Convergence: FB vs PR vs DR")
    ax.legend(fontsize=6.5, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()

    # Save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / "splitting_comparison.pdf"
    fig.savefig(out_path, dpi=300)
    fig.savefig(figures_dir / "splitting_comparison.png", dpi=150)
    print(f"\nFigure saved to {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
