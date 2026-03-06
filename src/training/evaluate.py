# training/evaluate.py
"""
Evaluation utilities for MonDEQ models including quantised inference.

TODO (5.1): Add support for multiple tolerance levels in convergence analysis
TODO (5.2): Add statistical significance tests for accuracy comparisons
TODO (5.3): Add memory profiling for quantised vs FP32 inference
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Optional
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cuda(tensor):
    """Move tensor to GPU if available."""
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def evaluate(model: nn.Module, test_loader: DataLoader, verbose: bool = True) -> float:
    """
    Evaluate model accuracy on test set.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    test_loader : DataLoader
        Test data loader.
    verbose : bool
        Whether to print results.

    Returns
    -------
    float
        Test accuracy as percentage.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = cuda(data), cuda(target)
            logits = model(data)
            preds = logits.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += len(target)
            total_loss += nn.CrossEntropyLoss(reduction="sum")(logits, target).item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total

    if verbose:
        print(f"Test set: Avg loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")

    return accuracy


def evaluate_quantised(
    base_model: nn.Module,
    quant_model: nn.Module,
    test_loader: DataLoader,
    verbose: bool = True,
) -> dict:
    """
    Evaluate quantised model and compare to float baseline.

    Parameters
    ----------
    base_model : nn.Module
        Float precision model.
    quant_model : QuantisedMonDEQ
        Quantised model.
    test_loader : DataLoader
        Test data loader.
    verbose : bool
        Whether to print results.

    Returns
    -------
    dict
        Evaluation results including:
        - float_accuracy: float model accuracy
        - quant_accuracy: quantised model accuracy
        - accuracy_drop: difference in accuracy
        - quant_stats: quantisation statistics
    """
    # Evaluate float model
    float_acc = evaluate(base_model, test_loader, verbose=False)

    # Evaluate quantised model
    quant_acc = evaluate(quant_model, test_loader, verbose=False)

    results = {
        "float_accuracy": float_acc,
        "quant_accuracy": quant_acc,
        "accuracy_drop": float_acc - quant_acc,
        "quant_stats": quant_model.quant_stats,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Quantisation Evaluation ({quant_model.bits}-bit)")
        print(f"{'='*60}")
        print(f"Float model accuracy:     {float_acc:.2f}%")
        print(f"Quantised model accuracy: {quant_acc:.2f}%")
        print(f"Accuracy drop:            {float_acc - quant_acc:.2f}%")
        print(f"{'='*60}")

        stats = quant_model.quant_stats
        print(f"\nQuantisation Statistics:")
        print(f"  Bits:                   {stats['bits']}")
        print(f"  Original margin m:      {stats['m_original']:.6f}")
        print(f"  Quantised margin m~:    {stats['m_tilde']:.6f}")
        print(f"  ||Delta_W||_2:          {stats['delta_W_norm']:.6f}")
        print(f"  Relative error:         {stats['relative_error']:.4%}")
        print(f"  Converges:              {stats['converges']}")
        print(f"  Convergence margin:     {stats['convergence_margin']:.6f}")
        print()

    return results


def verify_displacement_bound(
    base_model: nn.Module,
    quant_model: nn.Module,
    test_loader: DataLoader,
    num_batches: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Verify the theoretical displacement bound empirically.

    Theoretical bound: ||z* - z_tilde*|| <= (||Delta_W||_2 / m) * ||z_tilde*||

    Parameters
    ----------
    base_model : nn.Module
        Float precision model.
    quant_model : QuantisedMonDEQ
        Quantised model.
    test_loader : DataLoader
        Test data loader.
    num_batches : int
        Number of batches to evaluate.
    verbose : bool
        Whether to print results.

    Returns
    -------
    dict
        Displacement bound verification results.
    """
    base_model.eval()
    quant_model.eval()

    empirical_displacements = []
    theoretical_bounds = []

    stats = quant_model.quant_stats
    m = stats["m_original"]
    delta_W_norm = stats["delta_W_norm"]

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_batches:
                break

            data = cuda(data)
            x = data.view(data.size(0), -1)

            # Get float equilibrium
            z_star_float = base_model.mon(x)  # (n, B)

            # Get quantised equilibrium
            # Access the quantised forward pass internals
            from mondeq.splitting import fb_solve
            mon = base_model.mon

            z_star_quant, _ = fb_solve(
                mon.core.A, mon.core.S, mon.core.m_raw,
                quant_model._U_quant, quant_model._b_quant, x,
                alpha=mon.alpha,
                max_iters=mon.max_iters,
                tol=mon.tol,
                W=quant_model._W_quant,
            )

            # Compute displacement for each sample
            delta_z = z_star_float - z_star_quant
            for j in range(delta_z.size(1)):
                emp_disp = delta_z[:, j].norm().item()
                z_tilde_norm = z_star_quant[:, j].norm().item()
                theo_bound = (delta_W_norm / m) * z_tilde_norm

                empirical_displacements.append(emp_disp)
                theoretical_bounds.append(theo_bound)

    # Compute statistics
    import numpy as np
    emp_arr = np.array(empirical_displacements)
    theo_arr = np.array(theoretical_bounds)

    results = {
        "empirical_mean": float(emp_arr.mean()),
        "empirical_max": float(emp_arr.max()),
        "theoretical_mean": float(theo_arr.mean()),
        "theoretical_max": float(theo_arr.max()),
        "ratio_mean": float((emp_arr / (theo_arr + 1e-12)).mean()),
        "ratio_max": float((emp_arr / (theo_arr + 1e-12)).max()),
        "all_satisfied": bool((emp_arr <= theo_arr * 1.01).all()),  # 1% tolerance
        "fraction_satisfied": float((emp_arr <= theo_arr * 1.01).mean()),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Displacement Bound Verification ({quant_model.bits}-bit)")
        print(f"{'='*60}")
        print(f"Empirical displacement:")
        print(f"  Mean: {results['empirical_mean']:.6f}")
        print(f"  Max:  {results['empirical_max']:.6f}")
        print(f"\nTheoretical bound:")
        print(f"  Mean: {results['theoretical_mean']:.6f}")
        print(f"  Max:  {results['theoretical_max']:.6f}")
        print(f"\nRatio (empirical/theoretical):")
        print(f"  Mean: {results['ratio_mean']:.4f}")
        print(f"  Max:  {results['ratio_max']:.4f}")
        print(f"\nBound satisfied: {results['all_satisfied']}")
        print(f"Fraction satisfied: {results['fraction_satisfied']:.2%}")
        print()

    return results


def verify_gradient_convergence(
    base_model: nn.Module,
    quant_model: nn.Module,
    test_loader: DataLoader,
    num_batches: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Verify backward pass convergence for quantised model.

    Key insight (Lemma 4.1-4.2): backward pass uses same (m_tilde, L_tilde) as forward,
    so if forward converges (m_tilde > 0), backward also converges.

    Parameters
    ----------
    base_model : nn.Module
        Float precision model.
    quant_model : QuantisedMonDEQ
        Quantised model.
    test_loader : DataLoader
        Test data loader.
    num_batches : int
        Number of batches to test.
    verbose : bool
        Whether to print results.

    Returns
    -------
    dict
        Gradient convergence verification results.
    """
    base_model.train()  # Enable gradient computation

    gradient_norms_float = []
    gradient_norms_quant = []
    gradient_errors = []
    backward_success = True

    stats = quant_model.quant_stats
    delta_W_norm = stats["delta_W_norm"]

    for i, (data, target) in enumerate(test_loader):
        if i >= num_batches:
            break

        # Use fresh copies for each test
        data, target = cuda(data.clone()), cuda(target.clone())

        # Float model gradients - use fresh forward pass
        base_model.zero_grad()
        logits_float = base_model(data.clone())
        loss_float = nn.CrossEntropyLoss()(logits_float, target)
        loss_float.backward()

        # Collect float gradients
        float_grads = {}
        for name, param in base_model.named_parameters():
            if param.grad is not None:
                float_grads[name] = param.grad.clone()
                gradient_norms_float.append(param.grad.norm().item())

        # Quantised model gradients (using forward_with_grad)
        # Need fresh data to avoid graph issues
        base_model.zero_grad()
        try:
            logits_quant = quant_model.forward_with_grad(data.clone())
            loss_quant = nn.CrossEntropyLoss()(logits_quant, target)
            loss_quant.backward()

            # Collect quantised gradients and compare
            for name, param in base_model.named_parameters():
                if param.grad is not None and name in float_grads:
                    gradient_norms_quant.append(param.grad.norm().item())
                    grad_error = (param.grad - float_grads[name]).norm().item()
                    gradient_errors.append(grad_error)

        except RuntimeError as e:
            backward_success = False
            if verbose:
                print(f"Backward pass failed: {e}")
            break

    import numpy as np
    results = {
        "backward_converges": bool(backward_success),
        "grad_norm_float_mean": float(np.mean(gradient_norms_float)) if gradient_norms_float else 0.0,
        "grad_norm_quant_mean": float(np.mean(gradient_norms_quant)) if gradient_norms_quant else 0.0,
        "grad_error_mean": float(np.mean(gradient_errors)) if gradient_errors else 0.0,
        "grad_error_max": float(np.max(gradient_errors)) if gradient_errors else 0.0,
    }

    # Verify gradient error scales as O(||Delta_W||_2)
    if gradient_errors and delta_W_norm > 0:
        results["grad_error_ratio"] = float(results["grad_error_mean"] / delta_W_norm)
    else:
        results["grad_error_ratio"] = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Gradient Convergence Verification ({quant_model.bits}-bit)")
        print(f"{'='*60}")
        print(f"Backward pass converges: {results['backward_converges']}")
        print(f"\nGradient norms:")
        print(f"  Float mean:  {results['grad_norm_float_mean']:.6f}")
        print(f"  Quant mean:  {results['grad_norm_quant_mean']:.6f}")
        print(f"\nGradient error:")
        print(f"  Mean: {results['grad_error_mean']:.6f}")
        print(f"  Max:  {results['grad_error_max']:.6f}")
        print(f"  Ratio to ||Delta_W||_2: {results['grad_error_ratio']:.4f}")
        print()

    return results


def verify_pr_convergence(
    base_model: nn.Module,
    test_loader: DataLoader,
    num_batches: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Verify Peaceman-Rachford splitting convergence and compare with FB.

    Key insight: PR has contraction rate ρ_PR = max{|1-αm|/(1+αm), |αL-1|/(αL+1)},
    which can be faster than FB for well-conditioned problems.

    Parameters
    ----------
    base_model : nn.Module
        Float precision model with MonDEQ layer.
    test_loader : DataLoader
        Test data loader.
    num_batches : int
        Number of batches to test.
    verbose : bool
        Whether to print results.

    Returns
    -------
    dict
        PR convergence verification results including comparison with FB.
    """
    from mondeq.splitting import fb_solve, pr_solve, compute_pr_rate

    base_model.eval()

    fb_iters_list = []
    pr_iters_list = []
    fb_residuals = []
    pr_residuals = []
    equilibrium_diffs = []

    mon = base_model.mon

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_batches:
                break

            data = cuda(data)
            x = data.view(data.size(0), -1)

            # FB solve
            z_fb, hist_fb = fb_solve(
                mon.core.A, mon.core.S, mon.core.m_raw,
                mon.core.U, mon.core.b, x,
                alpha=mon.alpha,
                max_iters=mon.max_iters,
                tol=mon.tol,
            )
            fb_iters_list.append(hist_fb["iters"])
            if hist_fb["residual"]:
                fb_residuals.append(hist_fb["residual"][-1])

            # PR solve
            z_pr, hist_pr = pr_solve(
                mon.core.A, mon.core.S, mon.core.m_raw,
                mon.core.U, mon.core.b, x,
                alpha=mon.alpha,
                max_iters=mon.max_iters,
                tol=mon.tol,
            )
            pr_iters_list.append(hist_pr["iters"])
            if hist_pr["residual"]:
                pr_residuals.append(hist_pr["residual"][-1])

            # Compare equilibria
            diff = (z_fb - z_pr).norm() / (z_fb.norm() + 1e-12)
            equilibrium_diffs.append(float(diff))

    import numpy as np

    # Compute spectral properties for rate comparison
    m = float(torch.nn.functional.softplus(mon.core.m_raw) + 1e-4)
    W = mon.core.build_W()
    I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    L = float(torch.linalg.norm(I_n - W, ord=2))

    alpha = mon.alpha
    r_fb = float(np.sqrt(1 - 2 * alpha * m + alpha**2 * L**2))
    rho_pr = compute_pr_rate(m, L, alpha)

    results = {
        "fb_iters_mean": float(np.mean(fb_iters_list)),
        "pr_iters_mean": float(np.mean(pr_iters_list)),
        "fb_final_residual_mean": float(np.mean(fb_residuals)) if fb_residuals else 0.0,
        "pr_final_residual_mean": float(np.mean(pr_residuals)) if pr_residuals else 0.0,
        "equilibrium_diff_mean": float(np.mean(equilibrium_diffs)),
        "equilibrium_diff_max": float(np.max(equilibrium_diffs)),
        "r_fb_theoretical": r_fb,
        "rho_pr_theoretical": rho_pr,
        "margin_m": m,
        "lipschitz_L": L,
        "condition_number": L / m,
        "pr_converges": all(i < mon.max_iters for i in pr_iters_list),
    }

    if verbose:
        print(f"\n{'='*60}")
        print("Peaceman-Rachford vs Forward-Backward Comparison")
        print(f"{'='*60}")
        print(f"Spectral properties:")
        print(f"  Margin m:           {m:.6f}")
        print(f"  Lipschitz L:        {L:.6f}")
        print(f"  Condition κ = L/m:  {L/m:.2f}")
        print(f"\nTheoretical rates (α = {alpha}):")
        print(f"  FB rate r_FB:       {r_fb:.6f}")
        print(f"  PR rate ρ_PR:       {rho_pr:.6f}")
        print(f"\nEmpirical iteration counts:")
        print(f"  FB mean iters:      {results['fb_iters_mean']:.1f}")
        print(f"  PR mean iters:      {results['pr_iters_mean']:.1f}")
        print(f"\nEquilibrium agreement:")
        print(f"  Mean diff:          {results['equilibrium_diff_mean']:.2e}")
        print(f"  Max diff:           {results['equilibrium_diff_max']:.2e}")
        print(f"\nPR converges:         {results['pr_converges']}")
        print()

    return results
