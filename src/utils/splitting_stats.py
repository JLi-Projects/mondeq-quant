# utils/splitting_stats.py
"""
Convergence tracking utilities for FB splitting.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from torch import Tensor


@dataclass
class AverageMeter:
    """Computes and stores the average and current value."""
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


@dataclass
class SplittingStats:
    """
    Track forward-backward splitting convergence statistics.

    Attributes
    ----------
    fwd_iters : AverageMeter
        Forward pass iteration counts.
    bwd_iters : AverageMeter
        Backward pass iteration counts (if tracked).
    fwd_residuals : List[List[float]]
        Residual history for each forward pass.
    convergence_failures : int
        Number of times forward pass hit max_iters without converging.
    """
    fwd_iters: AverageMeter = field(default_factory=AverageMeter)
    bwd_iters: AverageMeter = field(default_factory=AverageMeter)
    fwd_residuals: List[List[float]] = field(default_factory=list)
    convergence_failures: int = 0
    max_iters: int = 500

    def reset(self):
        """Reset all statistics."""
        self.fwd_iters.reset()
        self.bwd_iters.reset()
        self.fwd_residuals = []
        self.convergence_failures = 0

    def update_forward(self, iters: int, residuals: Optional[List[float]] = None):
        """
        Update forward pass statistics.

        Parameters
        ----------
        iters : int
            Number of iterations used.
        residuals : List[float], optional
            Residual history from this solve.
        """
        self.fwd_iters.update(iters)
        if residuals is not None:
            self.fwd_residuals.append(residuals)
        if iters >= self.max_iters:
            self.convergence_failures += 1

    def update_backward(self, iters: int):
        """Update backward pass statistics."""
        self.bwd_iters.update(iters)

    def report(self) -> str:
        """Generate a summary report."""
        lines = [
            "Splitting Statistics:",
            f"  Forward iterations:  avg={self.fwd_iters.avg:.1f}, last={self.fwd_iters.val}",
            f"  Convergence failures: {self.convergence_failures}",
        ]
        if self.bwd_iters.count > 0:
            lines.append(f"  Backward iterations: avg={self.bwd_iters.avg:.1f}")
        return "\n".join(lines)

    def get_summary(self) -> dict:
        """Get summary as dictionary."""
        return {
            "fwd_iters_avg": self.fwd_iters.avg,
            "fwd_iters_last": self.fwd_iters.val,
            "bwd_iters_avg": self.bwd_iters.avg,
            "convergence_failures": self.convergence_failures,
            "total_forward_calls": self.fwd_iters.count,
        }


def compute_contraction_rate(residuals: List[float], skip_first: int = 5) -> float:
    """
    Estimate contraction rate from residual history.

    Uses linear regression on log(residuals) to estimate r in ||z_{k+1} - z_k|| ~ r^k.

    Parameters
    ----------
    residuals : List[float]
        Residual values from FB iteration.
    skip_first : int
        Number of initial iterations to skip (transient).

    Returns
    -------
    float
        Estimated contraction rate.
    """
    import numpy as np

    if len(residuals) <= skip_first + 2:
        return float("nan")

    res = np.array(residuals[skip_first:])
    res = res[res > 1e-12]  # Filter zeros

    if len(res) < 2:
        return float("nan")

    log_res = np.log(res)
    k = np.arange(len(log_res))

    # Linear regression: log(res) = log(c) + k * log(r)
    A = np.vstack([k, np.ones(len(k))]).T
    slope, _ = np.linalg.lstsq(A, log_res, rcond=None)[0]

    return float(np.exp(slope))


def theoretical_contraction_rate(m: float, L: float, alpha: float) -> float:
    """
    Compute theoretical FB contraction rate.

    r = sqrt(1 - 2*alpha*m + alpha^2*L^2)

    Parameters
    ----------
    m : float
        Monotonicity margin.
    L : float
        Lipschitz constant.
    alpha : float
        FB step size.

    Returns
    -------
    float
        Theoretical contraction rate.
    """
    import math
    r_sq = 1 - 2 * alpha * m + alpha ** 2 * L ** 2
    return math.sqrt(max(0, r_sq))


def optimal_alpha(m: float, L: float) -> float:
    """
    Compute optimal FB step size.

    alpha* = m / L^2

    Parameters
    ----------
    m : float
        Monotonicity margin.
    L : float
        Lipschitz constant.

    Returns
    -------
    float
        Optimal step size.
    """
    return m / (L ** 2) if L > 0 else 1.0
