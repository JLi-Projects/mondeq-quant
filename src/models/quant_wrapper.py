# models/quant_wrapper.py
"""
Post-Training Quantisation (PTQ) wrapper for MonDEQ models.

Implements quantised forward pass while preserving convergence guarantees.
Key insight: if ||Delta_W||_2 < m, both forward and backward pass converge.
"""
from __future__ import annotations
import warnings
import torch
from torch import nn, Tensor
from typing import Optional

from mondeq import fake_quant_sym, spectral_bounds
from mondeq.splitting import fb_solve, _build_W
from mondeq.prox import relu_prox, relu_jacobian_mask


def compute_quantisation_error(W: Tensor, W_quant: Tensor) -> dict:
    """
    Compute quantisation error metrics.

    Returns
    -------
    dict with keys:
        delta_W_norm : ||W - W_quant||_2
        W_norm : ||W||_2
        relative_error : ||W - W_quant||_2 / ||W||_2
    """
    delta_W = W - W_quant
    delta_W_norm = torch.linalg.norm(delta_W, 2).item()
    W_norm = torch.linalg.norm(W, 2).item()
    return {
        "delta_W_norm": delta_W_norm,
        "W_norm": W_norm,
        "relative_error": delta_W_norm / (W_norm + 1e-12),
    }


class QuantisedMonDEQ(nn.Module):
    """
    Wrapper that applies post-training quantisation to a MonDEQ model.

    The quantised forward pass uses fake-quantised W, U, b while
    maintaining the original model parameters for gradient computation.

    Parameters
    ----------
    base_model : MNISTMonDEQ
        The trained float model to quantise.
    bits : int
        Bit depth for quantisation.
    """

    def __init__(self, base_model: nn.Module, bits: int):
        super().__init__()
        self.base_model = base_model
        self.bits = bits

        # Cache quantised parameters
        self._W_quant: Optional[Tensor] = None
        self._U_quant: Optional[Tensor] = None
        self._b_quant: Optional[Tensor] = None
        self._quant_stats: Optional[dict] = None

        # Quantise parameters
        self._quantise()

    def _quantise(self):
        """Compute and cache quantised parameters."""
        W, U, b = self.base_model.get_params()

        # Quantise each parameter
        self._W_quant = fake_quant_sym(W, self.bits)
        self._U_quant = fake_quant_sym(U, self.bits)
        self._b_quant = fake_quant_sym(b, self.bits)

        # Compute quantisation statistics
        m = self.base_model.get_m().item()
        error_stats = compute_quantisation_error(W, self._W_quant)

        # Check convergence condition: ||Delta_W||_2 < m
        converges = error_stats["delta_W_norm"] < m

        # Warn if convergence is not guaranteed (Issue 6 from code review)
        if not converges:
            warnings.warn(
                f"Convergence NOT guaranteed: ||Delta_W||_2 = {error_stats['delta_W_norm']:.6f} >= m = {m:.6f}. "
                f"Consider using more bits or training with larger margin.",
                RuntimeWarning,
            )

        # Compute spectral bounds for quantised system
        W_quant = self._W_quant
        I = torch.eye(W_quant.size(0), device=W_quant.device, dtype=W_quant.dtype)
        A_quant = I - W_quant
        m_tilde, L_tilde = spectral_bounds(A_quant)

        self._quant_stats = {
            "bits": self.bits,
            "m_original": m,
            "m_tilde": m_tilde.item(),
            "L_tilde": L_tilde.item(),
            "delta_W_norm": error_stats["delta_W_norm"],
            "relative_error": error_stats["relative_error"],
            "converges": converges,
            "convergence_margin": m - error_stats["delta_W_norm"],
        }

    @property
    def quant_stats(self) -> dict:
        """Get quantisation statistics."""
        return self._quant_stats

    def forward(self, x: Tensor) -> Tensor:
        """
        Quantised forward pass.

        Uses fake-quantised W, U, b for the equilibrium solve.
        """
        # Flatten input
        x = x.view(x.size(0), -1)  # (B, 784)

        # Get MonDEQ layer parameters (needed for fb_solve interface)
        mon = self.base_model.mon
        A = mon.core.A
        S = mon.core.S
        m_raw = mon.core.m_raw

        # Run forward-backward with quantised parameters
        with torch.no_grad():
            z_star, hist = fb_solve(
                A, S, m_raw,
                self._U_quant, self._b_quant, x,
                alpha=mon.alpha,
                max_iters=mon.max_iters,
                tol=mon.tol,
                W=self._W_quant,  # Use quantised W
            )

        # Transpose for classifier: (B, hidden_dim)
        z_star = z_star.T

        # Classification (not quantised for now)
        logits = self.base_model.classifier(z_star)
        return logits

    def forward_with_grad(self, x: Tensor) -> Tensor:
        """
        Quantised forward pass with gradient support.

        Uses implicit differentiation through the quantised equilibrium.
        Key insight (Lemma 4.1-4.2): backward pass uses same (m_tilde, L_tilde) as forward,
        so if forward converges (m_tilde > 0), backward also converges.
        """
        # Flatten input
        x = x.view(x.size(0), -1)  # (B, 784)

        # Forward solve with quantised parameters
        mon = self.base_model.mon
        z_star = _QuantisedFBFunction.apply(
            self._W_quant, self._U_quant, self._b_quant, x,
            mon.alpha, mon.max_iters, mon.tol
        )

        # Transpose for classifier: (B, hidden_dim)
        z_star = z_star.T

        # Classification
        logits = self.base_model.classifier(z_star)
        return logits

    def verify_convergence(self) -> bool:
        """Check if quantised system satisfies convergence condition."""
        return self._quant_stats["converges"]

    def get_displacement_bound(self, z_star_float: Tensor, z_star_quant: Tensor) -> dict:
        """
        Compute and validate displacement bound.

        Theoretical bound: ||z* - z_tilde*|| <= (||Delta_W||_2 / m) * ||z_tilde*||
        """
        delta_z = z_star_float - z_star_quant
        delta_z_norm = delta_z.norm(dim=0).mean().item()  # Average over batch
        z_tilde_norm = z_star_quant.norm(dim=0).mean().item()

        m = self._quant_stats["m_original"]
        delta_W_norm = self._quant_stats["delta_W_norm"]

        theoretical_bound = (delta_W_norm / m) * z_tilde_norm
        empirical_displacement = delta_z_norm

        return {
            "empirical": empirical_displacement,
            "theoretical_bound": theoretical_bound,
            "ratio": empirical_displacement / (theoretical_bound + 1e-12),
            "bound_satisfied": empirical_displacement <= theoretical_bound * 1.01,  # 1% tolerance
        }


class _QuantisedFBFunction(torch.autograd.Function):
    """
    Custom autograd for quantised FB fixed-point.

    Forward: solve z* with quantised W, U, b
    Backward: implicit differentiation using quantised parameters

    Key insight: The backward (adjoint) inclusion has the same linear part
    (I - W_tilde) as the forward, so it inherits (m_tilde, L_tilde).
    If forward converges (m_tilde > 0), backward also converges.
    """

    @staticmethod
    def forward(
        ctx,
        W_quant: Tensor,
        U_quant: Tensor,
        b_quant: Tensor,
        x: Tensor,
        alpha: float,
        max_iters: int,
        tol: float,
    ) -> Tensor:
        device, dtype = W_quant.device, W_quant.dtype
        n = W_quant.size(0)
        B = x.size(0)
        z = torch.zeros((n, B), device=device, dtype=dtype)

        # Forward-backward iteration with quantised parameters
        for _ in range(max_iters):
            u = (1.0 - alpha) * z + alpha * (W_quant @ z + (U_quant @ x.T) + b_quant[:, None])
            z_next = relu_prox(u, alpha)
            res = (z_next - z).norm() / (z.norm() + 1e-12)
            z = z_next
            if res < tol:
                break

        # Compute mask for backward pass
        u_star = (1.0 - alpha) * z + alpha * (W_quant @ z + (U_quant @ x.T) + b_quant[:, None])
        mask = relu_jacobian_mask(u_star)

        ctx.save_for_backward(W_quant, U_quant, b_quant, x, z, mask)
        ctx.alpha = alpha
        return z

    @staticmethod
    def backward(ctx, grad_z: Tensor):
        """
        Implicit differentiation for quantised equilibrium.

        Solves: (I - W_tilde^T D*) lambda* = grad_z
        Then computes: grad_x = alpha * U_tilde^T * D* * lambda*
        """
        W_quant, U_quant, b_quant, x, z_star, mask = ctx.saved_tensors
        alpha = ctx.alpha
        device, dtype = W_quant.device, W_quant.dtype
        n, B = z_star.shape

        # S = (1-alpha)I + alpha*W
        I = torch.eye(n, device=device, dtype=dtype)
        S_T = ((1.0 - alpha) * I + alpha * W_quant).T

        # Solve (I - S^T J) lambda = grad_z per-sample
        lam = torch.empty_like(z_star)
        for i in range(B):
            J = torch.diag(mask[:, i])
            M_T = I - S_T @ J
            rhs = grad_z[:, i]
            lam[:, i] = torch.linalg.solve(M_T, rhs)

        # xi = J lambda
        xi = mask * lam

        # Gradient for x
        grad_x = (alpha * xi.T) @ U_quant  # (B, d)

        # No gradients for quantised parameters (they are fixed)
        return None, None, None, grad_x, None, None, None
