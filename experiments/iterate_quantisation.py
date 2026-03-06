#!/usr/bin/env python3
"""
Experiment 3: Fixed vs Adaptive Iterate Quantisation

Compares iterate quantisation strategies (Jonkman-inspired):
  - No iterate quantisation (baseline)
  - Fixed iterate quantisation at various delta (shows error floor)
  - Adaptive iterate quantisation with shrinking delta (converges to zero)

Uses 8-bit weight quantisation and PR splitting (fast convergence).
Tracks distance to the true fixed point ||z^k - z*|| / ||z*||.

Usage:
    python experiments/iterate_quantisation.py
"""
from __future__ import annotations
from pathlib import Path

_project_root = Path(__file__).parent.parent

import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.utils import (
    setup_matplotlib, load_pretrained_model, get_model_params,
    get_mnist_loaders, DEVICE,
)
from mondeq.splitting import fb_solve, pr_solve, optimal_pr_alpha, compute_pr_rate
from mondeq.quant import fake_quant_sym, fixed_iterate_quant, adaptive_iterate_quant
from mondeq import spectral_bounds

setup_matplotlib()


@torch.no_grad()
def pr_solve_tracked(
    A, S, m_raw, U, b, x, *,
    alpha, max_iters, W, z_true,
    quant_mode="none", quant_delta=0.01, quant_gamma=0.5,
):
    """
    PR solve with tracking of distance to true fixed point at each iteration.
    Returns list of ||z^k - z*|| / ||z*|| values.

    Note: iterate quantisation is applied to the PR *shadow* sequence (the
    variable that the PR iteration actually evolves), not the primal
    variable z_primal = J_G(R_F(z_shadow)). The theoretical error floor
    delta/(1-rho) from Cor. 5 (inexact iterations) strictly applies to
    whichever sequence is perturbed. Since the primal recovery map
    J_G * R_F is nonexpansive, primal error is bounded by shadow error,
    so the qualitative behaviour (fixed quant -> floor proportional to
    delta; adaptive -> convergence to exact FP) is preserved, though
    the exact floor magnitudes are empirical.
    """
    from mondeq.prox import relu_prox

    device, dtype = A.device, A.dtype
    n = U.size(0)
    B = x.size(0)
    z = torch.zeros((n, B), device=device, dtype=dtype)

    I_n = torch.eye(n, device=device, dtype=dtype)
    resolvent_matrix = I_n + alpha * (I_n - W)
    resolvent_inv = torch.linalg.inv(resolvent_matrix)
    affine = alpha * ((U @ x.T) + b[:, None])

    z_true_norm = z_true.norm() + 1e-12
    distances = []

    for i in range(max_iters):
        # PR step
        J_F_z = resolvent_inv @ (z + affine)
        R_F_z = 2 * J_F_z - z
        prox_G_R_F = relu_prox(R_F_z, alpha)
        z_next = 2 * prox_G_R_F - R_F_z

        # Apply iterate quantisation
        if quant_mode == "fixed":
            z_next = fixed_iterate_quant(z_next, quant_delta)
        elif quant_mode == "adaptive":
            z_diff = z_next - z
            z_diff_q = adaptive_iterate_quant(z_diff, quant_delta, quant_gamma, i)
            z_next = z + z_diff_q

        z = z_next

        # Recover primal and compute distance to true fixed point
        J_F_z_cur = resolvent_inv @ (z + affine)
        R_F_z_cur = 2 * J_F_z_cur - z
        z_primal = relu_prox(R_F_z_cur, alpha)
        dist = (z_primal - z_true).norm() / z_true_norm
        distances.append(float(dist))

    return distances


def main():
    print("=" * 60)
    print("Experiment 3: Fixed vs Adaptive Iterate Quantisation")
    print("=" * 60)

    # Load model
    model = load_pretrained_model()
    params = get_model_params(model)
    m = params["m"]
    L = params["L"]
    W = params["W"]
    U, b = params["U"], params["b"]
    A_param, S_param, m_raw = params["A_param"], params["S_param"], params["m_raw"]
    n = W.size(0)

    # Use 8-bit quantised weights (ensures convergence)
    W_q = fake_quant_sym(W, 8)
    delta_W = float(torch.linalg.norm(W - W_q, 2))
    I_n = torch.eye(n, device=W.device, dtype=W.dtype)
    m_q, L_q = spectral_bounds(I_n - W_q)
    m_q, L_q = float(m_q), float(L_q)

    alpha_pr = optimal_pr_alpha(m_q, L_q)
    rho_pr = compute_pr_rate(m_q, L_q, alpha_pr)

    print(f"8-bit: m_tilde={m_q:.4f}, L_tilde={L_q:.4f}")
    print(f"PR: alpha={alpha_pr:.4f}, rho={rho_pr:.4f}")

    # Get test batch
    _, test_loader = get_mnist_loaders(test_batch_size=64)
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.view(x_batch.size(0), -1).to(DEVICE)

    # Compute true fixed point (high-precision PR)
    z_true, _ = pr_solve(
        A_param, S_param, m_raw, U, b, x_batch,
        alpha=alpha_pr, max_iters=2000, tol=1e-12, W=W_q,
    )
    print(f"True fixed point computed (||z*|| = {z_true.norm():.2f})")

    max_iters = 200

    # --- Baseline ---
    dist_base = pr_solve_tracked(
        A_param, S_param, m_raw, U, b, x_batch,
        alpha=alpha_pr, max_iters=max_iters, W=W_q, z_true=z_true,
    )
    print(f"Baseline: final dist = {dist_base[-1]:.2e}")

    # --- Fixed iterate quantisation ---
    fixed_deltas = [0.001, 0.01, 0.1]
    fixed_results = {}
    for delta in fixed_deltas:
        dists = pr_solve_tracked(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_pr, max_iters=max_iters, W=W_q, z_true=z_true,
            quant_mode="fixed", quant_delta=delta,
        )
        fixed_results[delta] = dists
        print(f"Fixed delta={delta}: final dist = {dists[-1]:.2e}")

    # --- Adaptive iterate quantisation ---
    adaptive_configs = [
        (0.1, 0.5),   # Fast decay
        (0.1, 0.9),   # Slow decay
    ]
    adaptive_results = {}
    for delta_0, gamma in adaptive_configs:
        dists = pr_solve_tracked(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_pr, max_iters=max_iters, W=W_q, z_true=z_true,
            quant_mode="adaptive", quant_delta=delta_0, quant_gamma=gamma,
        )
        adaptive_results[(delta_0, gamma)] = dists
        print(f"Adaptive delta_0={delta_0}, gamma={gamma}: final dist = {dists[-1]:.2e}")

    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    iters = range(1, max_iters + 1)

    # Baseline
    ax.semilogy(iters, dist_base, "k-", linewidth=1.2, label="No iterate quant")

    # Fixed
    fixed_colors = {0.001: "#2ca02c", 0.01: "#1f77b4", 0.1: "#d62728"}
    for delta in fixed_deltas:
        dists = fixed_results[delta]
        ax.semilogy(iters, dists, "--", color=fixed_colors[delta], linewidth=1.0,
                    label=f"Fixed $\\delta$={delta}")

    # Adaptive
    adaptive_styles = {(0.1, 0.5): ("#9467bd", "-."), (0.1, 0.9): ("#ff7f0e", "-.")}
    for (delta_0, gamma), dists in adaptive_results.items():
        color, ls = adaptive_styles[(delta_0, gamma)]
        ax.semilogy(iters, dists, ls, color=color, linewidth=1.0,
                    label=f"Adaptive $\\gamma$={gamma}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|z^k - z^\star\| / \|z^\star\|$")
    ax.set_title("Iterate quantisation (PR, 8-bit weights)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0, right=max_iters)

    plt.tight_layout()

    # Save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / "iterate_quantisation.pdf"
    fig.savefig(out_path, dpi=300)
    fig.savefig(figures_dir / "iterate_quantisation.png", dpi=150)
    print(f"\nFigure saved to {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
