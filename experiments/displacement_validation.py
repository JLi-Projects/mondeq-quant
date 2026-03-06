#!/usr/bin/env python3
"""
Experiment 2: Equilibrium Displacement Validation

Validate the theoretical bound on equilibrium displacement under quantisation.

Theoretical bound (from paper):
    ||z_float - z_quant|| / ||z_float|| <= ||Delta W||_2 / m * ||z_quant|| / ||z_float||

where:
- z_float is the equilibrium of the float model
- z_quant is the equilibrium of the quantised model
- Delta W = W_float - W_quant is the weight perturbation
- m is the strong monotonicity margin

The bound is validated empirically on 1000 test samples.
"""
from __future__ import annotations
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from experiments.utils import (
    setup_matplotlib, FIGURE_WIDTH, FIGURE_HEIGHT, DPI,
    BIT_DEPTHS, SEEDS, HIDDEN_DIM, DEVICE,
    get_mnist_loaders, MonDEQClassifier,
    load_pretrained_model, get_model_params,
    train_model, set_seed, save_results, load_results
)
from mondeq.splitting import _build_W, pr_solve, optimal_pr_alpha
from mondeq.quant import fake_quant_sym
from mondeq import spectral_bounds

setup_matplotlib()


def run_experiment(
    n_samples: int = 1000,
    checkpoint_path: str = "checkpoints/mnist_mondeq_float.pt",
):
    """
    Run the displacement validation experiment using direct W quantisation.

    Quantises W = (1-m)I - A^T A + S - S^T directly, matching the paper's
    perturbation model (Delta W = W - Q(W)).

    Returns dict with:
    - For each bit depth: empirical displacements, bounds, delta_W, m
    """
    bit_depths = [6, 8, 12, 16, 32]

    results = {
        'bit_depths': bit_depths,
        'n_samples': n_samples,
    }

    # Load pretrained model and extract parameters
    model = load_pretrained_model(checkpoint_path)
    params = get_model_params(model)

    W = params["W"]
    U, b = params["U"], params["b"]
    A_param, S_param, m_raw = params["A_param"], params["S_param"], params["m_raw"]
    m = params["m"]

    # Get test data
    _, test_loader = get_mnist_loaders(test_batch_size=n_samples)
    images, _ = next(iter(test_loader))
    x_all = images.view(images.size(0), -1).to(DEVICE)

    # Compute float equilibrium once (PR solver for speed)
    alpha_float = optimal_pr_alpha(float(params["m_computed"]), params["L"])
    with torch.no_grad():
        z_float, _ = pr_solve(
            A_param, S_param, m_raw, U, b, x_all,
            alpha=alpha_float, max_iters=500, tol=1e-6, W=W,
        )
    z_float_norm = z_float.norm(dim=0)  # (B,)

    # Process each bit depth
    for bits in tqdm(bit_depths, desc="Bit depths"):
        print(f"\nProcessing {bits}-bit quantisation...")

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

        with torch.no_grad():
            z_quant, _ = pr_solve(
                A_param, S_param, m_raw, U, b, x_all,
                alpha=alpha_q, max_iters=500, tol=1e-6, W=W_q,
            )
        z_quant_norm = z_quant.norm(dim=0)  # (B,)

        # Empirical displacement
        diff_norm = (z_float - z_quant).norm(dim=0)
        empirical = diff_norm / (z_float_norm + 1e-12)

        # Theoretical bound: ||DW||_2 / m * ||z_quant|| / ||z_float||
        # Add small numerical tolerance for FP precision
        bound = (delta_W_norm / m) * (z_quant_norm / (z_float_norm + 1e-12)) + 1e-7

        # Store results
        frac_satisfied = float((empirical <= bound).float().mean())
        results[f'{bits}bit'] = {
            'empirical': empirical.cpu().numpy(),
            'bound': bound.cpu().numpy(),
            'delta_W': delta_W_norm,
            'm': m,
            'bound_satisfied': bool((empirical <= bound).all()),
            'violation_fraction': 1.0 - frac_satisfied,
        }

        print(f"  ||DW||_2: {delta_W_norm:.6f}")
        print(f"  m (margin): {m:.6f}")
        print(f"  ||DW||_2/m: {delta_W_norm/m:.4f}")
        print(f"  Max empirical: {float(empirical.max()):.6f}")
        print(f"  Max bound: {float(bound.max()):.6f}")
        print(f"  Bound satisfied: {frac_satisfied*100:.1f}%")

    return results


def generate_figure(results: dict, output_path: str = 'figures/displacement_vs_bound.pdf'):
    """Generate scatter plot of empirical displacement vs theoretical bound."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Colors for different bit depths
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(BIT_DEPTHS)))

    # Collect all data for axis limits
    all_empirical = []
    all_bound = []

    for i, bits in enumerate(BIT_DEPTHS):
        data = results[f'{bits}bit']
        empirical = data['empirical']
        bound = data['bound']

        # Subsample for clarity (plot every nth point)
        n_plot = min(100, len(empirical))
        idx = np.linspace(0, len(empirical)-1, n_plot, dtype=int)

        ax.scatter(
            bound[idx], empirical[idx],
            c=[colors[i]], s=15, alpha=0.6,
            label=f'{bits}-bit', marker='o', edgecolors='none'
        )

        all_empirical.extend(empirical)
        all_bound.extend(bound)

    # Plot y=x line (theoretical bound)
    max_val = max(max(all_empirical), max(all_bound))
    min_val = min(min(all_empirical), min(all_bound))
    line_range = np.linspace(0, max_val * 1.1, 100)
    ax.plot(line_range, line_range, 'k--', linewidth=1.0, label='$y = x$ (bound)')

    # Shade region where bound is satisfied
    ax.fill_between(line_range, 0, line_range, alpha=0.1, color='green')
    ax.fill_between(line_range, line_range, max_val * 1.5, alpha=0.1, color='red')

    ax.set_xlabel('Theoretical Bound')
    ax.set_ylabel('Empirical Displacement')
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)

    # Legend outside or inside depending on space
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.9)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=DPI, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Equilibrium Displacement Validation')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mnist_mondeq_float.pt',
                        help='Path to pretrained float checkpoint')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help='Directory to save figures')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Number of samples: {args.n_samples}")

    # Run experiment
    results = run_experiment(
        n_samples=args.n_samples,
        checkpoint_path=args.checkpoint,
    )

    # Save results
    save_results(results, 'displacement_validation.npy', args.results_dir)

    # Generate figure
    output_path = os.path.join(args.figures_dir, 'displacement_vs_bound.pdf')
    generate_figure(results, output_path)

    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    for bits in results['bit_depths']:
        key = f'{bits}bit'
        if key not in results:
            continue
        data = results[key]
        status = "PASS" if data['bound_satisfied'] else "FAIL"
        print(f"{bits:2d}-bit: {status} (violations: {data['violation_fraction']*100:.2f}%)")

    print("="*60)


if __name__ == '__main__':
    main()
