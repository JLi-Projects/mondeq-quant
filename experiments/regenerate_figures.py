#!/usr/bin/env python3
"""
Regenerate all 4 LCSS figures with correct sizing and layout.

Outputs 8 PDF files:
  - figures/margin_stability_a.pdf  (3.40in x 2.2in)
  - figures/margin_stability_b.pdf  (3.40in x 2.2in)
  - figures/splitting_comparison.pdf (3.40in x 2.5in)
  - figures/qat_vs_ptq_a.pdf        (2.25in x 2.2in)
  - figures/qat_vs_ptq_b.pdf        (2.25in x 2.2in)
  - figures/qat_vs_ptq_c.pdf        (2.25in x 2.2in)
  - figures/displacement_vs_bound.pdf (3.40in x 2.5in)

All figures use 8pt font, bbox_inches='tight', pad_inches=0.02.
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
    load_pretrained_model, get_model_params,
    get_mnist_loaders, DEVICE, BIT_COLORS, SOLVER_STYLES,
)
from mondeq.splitting import fb_solve, pr_solve, dr_solve, optimal_pr_alpha
from mondeq.quant import fake_quant_sym
from mondeq import spectral_bounds

# ============================================================================
# Global settings
# ============================================================================
FONT_SIZE = 8
DPI = 300
FIGURES_DIR = _project_root / "figures"

def setup_matplotlib():
    """Configure matplotlib for 8pt publication-quality figures."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 2,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "text.usetex": False,
    })


def save_fig(fig, name: str):
    """Save figure to figures/ directory."""
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    print(f"  Saved {name}")


# ============================================================================
# Figure 1: Margin Stability Certificate (two separate panels)
# ============================================================================
def figure1_margin_stability(model, params):
    print("\n--- Figure 1: Margin Stability Certificate ---")

    m = params["m"]
    W = params["W"]
    U, b = params["U"], params["b"]
    A_param, S_param, m_raw = params["A_param"], params["S_param"], params["m_raw"]

    _, test_loader = get_mnist_loaders(test_batch_size=64)
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.view(x_batch.size(0), -1).to(DEVICE)

    bit_depths = [3, 4, 5, 6, 7, 8, 10, 12, 16, 32]
    max_iters = 2000
    tol = 1e-5

    results = []
    for bits in bit_depths:
        W_q = fake_quant_sym(W, bits)
        delta_W_norm = float(torch.linalg.norm(W - W_q, 2))
        ratio = delta_W_norm / m

        I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        m_q, L_q = spectral_bounds(I_n - W_q)
        m_q, L_q = float(m_q), float(L_q)

        if m_q > 0:
            alpha_q = m_q / (L_q ** 2)
        else:
            alpha_q = 0.01

        z_q, hist_q = fb_solve(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_q, max_iters=max_iters, tol=tol, W=W_q,
        )

        iters = hist_q["iters"]
        final_res = hist_q["residual"][-1] if hist_q["residual"] else float("inf")
        converged = iters < max_iters

        results.append({
            "bits": bits, "ratio": ratio, "iters": iters,
            "final_res": final_res, "converged": converged, "m_tilde": m_q,
        })

        status = "CONVERGED" if converged else "DIVERGED"
        print(f"  {bits:2d}-bit: ||DW||_2/m = {ratio:.3f}, iters = {iters:4d}, "
              f"res = {final_res:.2e}, m_tilde = {m_q:.4f}  [{status}]")

    # --- Panel (a): Iterations to convergence ---
    fig_a, ax1 = plt.subplots(figsize=(3.40, 2.2))
    for r in results:
        color = BIT_COLORS.get(r["bits"], "#333333")
        marker = "o" if r["converged"] else "x"
        ax1.plot(r["ratio"], r["iters"], marker, color=color, markersize=6,
                 label=f'{r["bits"]}b')

    ax1.axvline(x=1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
                label=r"$\|\Delta W\|_2 / m = 1$")
    ax1.set_xlabel(r"$\|\Delta W\|_2 / m$")
    ax1.set_ylabel("Iterations")
    ax1.legend(fontsize=FONT_SIZE - 2, ncol=2, loc="center right")
    ax1.set_xlim(left=0)
    ax1.grid(True, alpha=0.3)
    save_fig(fig_a, "margin_stability_a.pdf")
    plt.close(fig_a)

    # --- Panel (b): Final residual ---
    fig_b, ax2 = plt.subplots(figsize=(3.40, 2.2))
    for r in results:
        color = BIT_COLORS.get(r["bits"], "#333333")
        marker = "o" if r["converged"] else "x"
        ax2.semilogy(r["ratio"], r["final_res"], marker, color=color, markersize=6,
                     label=f'{r["bits"]}b')

    ax2.axvline(x=1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.set_xlabel(r"$\|\Delta W\|_2 / m$")
    ax2.set_ylabel("Final residual")
    ax2.set_xlim(left=0)
    ax2.grid(True, alpha=0.3)
    save_fig(fig_b, "margin_stability_b.pdf")
    plt.close(fig_b)


# ============================================================================
# Figure 2: Splitting Comparison
# ============================================================================
def figure2_splitting_comparison(model, params):
    print("\n--- Figure 2: Splitting Comparison ---")

    m = params["m"]
    W = params["W"]
    U, b = params["U"], params["b"]
    A_param, S_param, m_raw = params["A_param"], params["S_param"], params["m_raw"]

    _, test_loader = get_mnist_loaders(test_batch_size=64)
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.view(x_batch.size(0), -1).to(DEVICE)

    configs = [
        ("FP32", None, W),
        ("8-bit", 8, fake_quant_sym(W, 8)),
        ("6-bit", 6, fake_quant_sym(W, 6)),
    ]

    max_iters = 300
    tol = 1e-10

    all_results = {}
    for config_label, bits, W_cfg in configs:
        I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        m_cfg, L_cfg = spectral_bounds(I_n - W_cfg)
        m_cfg, L_cfg = float(m_cfg), float(L_cfg)

        if m_cfg <= 0:
            print(f"  Skipping {config_label}: margin non-positive")
            continue

        alpha_fb = m_cfg / (L_cfg ** 2)
        alpha_pr = optimal_pr_alpha(m_cfg, L_cfg)

        z_fb, hist_fb = fb_solve(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_fb, max_iters=max_iters, tol=tol, W=W_cfg,
        )
        all_results[(config_label, "FB")] = hist_fb["residual"]
        print(f"  FB ({config_label}): {len(hist_fb['residual'])} iters")

        z_pr, hist_pr = pr_solve(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_pr, max_iters=max_iters, tol=tol, W=W_cfg,
        )
        all_results[(config_label, "PR")] = hist_pr["residual"]
        print(f"  PR ({config_label}): {len(hist_pr['residual'])} iters")

        z_dr, hist_dr = dr_solve(
            A_param, S_param, m_raw, U, b, x_batch,
            alpha=alpha_pr, max_iters=max_iters, tol=tol, W=W_cfg,
        )
        all_results[(config_label, "DR")] = hist_dr["residual"]
        print(f"  DR ({config_label}): {len(hist_dr['residual'])} iters")

    # Plot
    fig, ax = plt.subplots(figsize=(3.40, 2.5))
    config_colors = {"FP32": "#000000", "8-bit": "#1f77b4", "6-bit": "#ff7f0e"}

    for (config_label, solver_name), residuals in all_results.items():
        style = SOLVER_STYLES[solver_name]
        color = config_colors[config_label]
        label = f"{solver_name} ({config_label})"
        ax.semilogy(
            range(1, len(residuals) + 1), residuals,
            linestyle=style["linestyle"], color=color, linewidth=1.0, label=label,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative residual")
    ax.legend(fontsize=FONT_SIZE - 2, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    save_fig(fig, "splitting_comparison.pdf")
    plt.close(fig)


# ============================================================================
# Figure 3: QAT vs PTQ (three separate panels, from cached data)
# ============================================================================
def figure3_qat_vs_ptq():
    print("\n--- Figure 3: QAT vs PTQ (from cached/hardcoded data) ---")

    bit_depths = [4, 6, 8]

    # Hardcoded data from the paper / experiment results
    ptq_data = [
        {"bits": 4, "accuracy": None, "m": 0.2269, "m_tilde": -0.142, "ratio": 2.661, "converges": False},
        {"bits": 6, "accuracy": 98.25, "m": 0.2269, "m_tilde": 0.138, "ratio": 0.606, "converges": True},
        {"bits": 8, "accuracy": 98.29, "m": 0.2269, "m_tilde": 0.208, "ratio": 0.153, "converges": True},
    ]
    qat_data = [
        {"bits": 4, "accuracy": 96.78, "m": 0.1844, "m_tilde": 0.006, "ratio": 1.0, "converges": True},
        {"bits": 6, "accuracy": 97.93, "m": 0.1266, "m_tilde": 0.022, "ratio": 0.8, "converges": True},
        {"bits": 8, "accuracy": 98.16, "m": 0.1229, "m_tilde": 0.098, "ratio": 0.2, "converges": True},
    ]

    x_pos = np.arange(len(bit_depths))
    width = 0.35

    # --- Panel (a): Accuracy ---
    fig_a, ax = plt.subplots(figsize=(2.25, 2.2))
    ptq_accs = [r["accuracy"] if r["accuracy"] is not None else 0 for r in ptq_data]
    qat_accs = [r["accuracy"] if r["accuracy"] is not None else 0 for r in qat_data]

    ax.bar(x_pos - width/2, ptq_accs, width, label="PTQ", color="#1f77b4")
    ax.bar(x_pos + width/2, qat_accs, width, label="QAT", color="#ff7f0e")

    # Mark non-converging (PTQ 4-bit)
    for i, r in enumerate(ptq_data):
        if not r["converges"]:
            ax.text(x_pos[i] - width/2, 1, "X", ha="center", va="bottom",
                    fontsize=FONT_SIZE, color="red", fontweight="bold")

    ax.set_xlabel("Bit depth")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in bit_depths])
    ax.legend(fontsize=FONT_SIZE - 2)
    ax.set_ylim(bottom=0)

    save_fig(fig_a, "qat_vs_ptq_a.pdf")
    plt.close(fig_a)

    # --- Panel (b): Monotonicity margin ---
    fig_b, ax = plt.subplots(figsize=(2.25, 2.2))
    ptq_margins = [r["m"] for r in ptq_data]
    qat_margins = [r["m"] for r in qat_data]

    ax.bar(x_pos - width/2, ptq_margins, width, label="PTQ", color="#1f77b4")
    ax.bar(x_pos + width/2, qat_margins, width, label="QAT", color="#ff7f0e")
    ax.set_xlabel("Bit depth")
    ax.set_ylabel("Margin $m$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in bit_depths])
    ax.legend(fontsize=FONT_SIZE - 2)

    save_fig(fig_b, "qat_vs_ptq_b.pdf")
    plt.close(fig_b)

    # --- Panel (c): Convergence ratio ---
    fig_c, ax = plt.subplots(figsize=(2.25, 2.2))
    ptq_ratios = [r["ratio"] for r in ptq_data]
    qat_ratios = [r["ratio"] for r in qat_data]

    ax.bar(x_pos - width/2, ptq_ratios, width, label="PTQ", color="#1f77b4")
    ax.bar(x_pos + width/2, qat_ratios, width, label="QAT", color="#ff7f0e")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
               label="Threshold")
    ax.set_xlabel("Bit depth")
    ax.set_ylabel(r"$\|\Delta W\|_2 / m$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in bit_depths])
    ax.legend(fontsize=FONT_SIZE - 2)

    save_fig(fig_c, "qat_vs_ptq_c.pdf")
    plt.close(fig_c)


# ============================================================================
# Figure 4: Displacement Validation
# ============================================================================
def figure4_displacement_validation(model, params):
    print("\n--- Figure 4: Displacement Validation ---")

    m = params["m"]
    W = params["W"]
    U, b = params["U"], params["b"]
    A_param, S_param, m_raw = params["A_param"], params["S_param"], params["m_raw"]

    # Load test data (2560 samples = 10 batches of 256)
    _, test_loader = get_mnist_loaders(test_batch_size=256)

    bit_depths = [6, 8, 12, 16]
    n_samples = 2560

    # Collect test data
    all_x = []
    count = 0
    for images, _ in test_loader:
        all_x.append(images.view(images.size(0), -1))
        count += images.size(0)
        if count >= n_samples:
            break
    x_all = torch.cat(all_x, dim=0)[:n_samples].to(DEVICE)

    print(f"  Using {x_all.size(0)} test samples")

    # Compute float equilibrium once
    # Use PR solver for speed (converges in ~50 iters)
    alpha_float = optimal_pr_alpha(float(params["m_computed"]), params["L"])
    z_float, _ = pr_solve(
        A_param, S_param, m_raw, U, b, x_all,
        alpha=alpha_float, max_iters=500, tol=1e-6, W=W,
    )
    z_float_norm = z_float.norm(dim=0)  # (B,)

    # For each bit depth, quantise W directly and compute displacement
    all_empirical = []
    all_bound = []
    all_bits = []

    for bits in bit_depths:
        W_q = fake_quant_sym(W, bits)
        delta_W_norm = float(torch.linalg.norm(W - W_q, 2))

        # Check convergence
        I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        m_q, L_q = spectral_bounds(I_n - W_q)
        m_q, L_q = float(m_q), float(L_q)

        if m_q <= 0:
            print(f"  {bits}-bit: SKIP (m_tilde = {m_q:.4f} <= 0)")
            continue

        alpha_q = optimal_pr_alpha(m_q, L_q)

        # Also quantise U and b for consistency
        U_q = fake_quant_sym(U, bits)
        b_q = fake_quant_sym(b, bits)

        z_quant, _ = pr_solve(
            A_param, S_param, m_raw, U_q, b_q, x_all,
            alpha=alpha_q, max_iters=500, tol=1e-6, W=W_q,
        )
        z_quant_norm = z_quant.norm(dim=0)  # (B,)

        # Empirical displacement
        diff_norm = (z_float - z_quant).norm(dim=0)
        empirical = diff_norm / (z_float_norm + 1e-12)

        # Theoretical bound: ||DW||_2 / m * ||z_quant|| / ||z_float||
        # Add small numerical tolerance for FP precision
        bound = (delta_W_norm / m) * (z_quant_norm / (z_float_norm + 1e-12)) + 1e-7

        all_empirical.append(empirical.cpu().numpy())
        all_bound.append(bound.cpu().numpy())
        all_bits.append(bits)

        frac_satisfied = float((empirical <= bound).float().mean())
        print(f"  {bits:2d}-bit: ||DW||_2/m = {delta_W_norm/m:.4f}, "
              f"max_emp = {float(empirical.max()):.4f}, "
              f"max_bound = {float(bound.max()):.4f}, "
              f"satisfied = {frac_satisfied*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(3.40, 2.5))

    bit_colors_disp = {
        6: "#ff7f0e",
        8: "#1f77b4",
        12: "#2ca02c",
        16: "#9467bd",
    }

    max_val = 0
    for i, bits in enumerate(all_bits):
        empirical = all_empirical[i]
        bound = all_bound[i]

        # Subsample for clarity
        n_plot = min(200, len(empirical))
        idx = np.linspace(0, len(empirical)-1, n_plot, dtype=int)

        color = bit_colors_disp.get(bits, "#333333")
        ax.scatter(
            bound[idx], empirical[idx],
            c=color, s=12, alpha=0.5,
            label=f"{bits}-bit", marker="o", edgecolors="none",
        )

        max_val = max(max_val, float(empirical.max()), float(bound.max()))

    # y=x line
    line_range = np.linspace(0, max_val * 1.1, 100)
    ax.plot(line_range, line_range, "k--", linewidth=1.0, label="$y = x$ (bound)")

    # Shade regions
    ax.fill_between(line_range, 0, line_range, alpha=0.08, color="green")
    ax.fill_between(line_range, line_range, max_val * 1.5, alpha=0.08, color="red")

    ax.set_xlabel("Theoretical bound")
    ax.set_ylabel("Empirical displacement")
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)
    ax.legend(loc="upper left", fontsize=FONT_SIZE - 2, ncol=2, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    save_fig(fig, "displacement_vs_bound.pdf")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    setup_matplotlib()

    print("=" * 60)
    print("Regenerating all LCSS figures")
    print("=" * 60)
    print(f"Font size: {FONT_SIZE}pt")
    print(f"Device: {DEVICE}")

    # Load model
    model = load_pretrained_model()
    params = get_model_params(model)
    print(f"Model: m = {params['m']:.4f}, L = {params['L']:.4f}, "
          f"kappa = {params['L']/params['m']:.2f}")

    # Figure 1: Margin Stability (fast, ~1 min)
    figure1_margin_stability(model, params)

    # Figure 2: Splitting Comparison (fast, ~1 min)
    figure2_splitting_comparison(model, params)

    # Figure 3: QAT vs PTQ (from cached data, instant)
    figure3_qat_vs_ptq()

    # Figure 4: Displacement Validation (~2 min)
    figure4_displacement_validation(model, params)

    print("\n" + "=" * 60)
    print("All figures regenerated.")
    print(f"Saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
