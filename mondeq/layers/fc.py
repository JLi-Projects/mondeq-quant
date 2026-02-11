# mondeq/layers/fc.py
from __future__ import annotations
import torch
from torch import nn, Tensor
from ..prox import relu_prox, relu_jacobian_mask
from ..splitting import fb_solve, _build_W


class _FBEquilibriumFunction(torch.autograd.Function):
    """
    Custom autograd for the FB fixed-point with ReLU prox.

    Inputs (tensors tracked for grads):
        A (r,n), S (n,n), m_raw (), U (n,d), b (n,), x (B,d)
    Aux scalars (no grads): alpha, max_iters, tol
    Output:
        z_star (n,B)
    """

    @staticmethod
    def forward(ctx, A: Tensor, S: Tensor, m_raw: Tensor, U: Tensor, b: Tensor, x: Tensor,
                alpha: float, max_iters: int, tol: float) -> Tensor:
        # Solve equilibrium without tracking a giant graph
        with torch.no_grad():
            z_star, _ = fb_solve(A, S, m_raw, U, b, x, alpha=alpha, max_iters=max_iters, tol=tol)

            # Prox input at equilibrium for Jacobian selection
            W = _build_W(A, S, m_raw)
            u_star = (1.0 - alpha) * z_star + alpha * (W @ z_star + (U @ x.T) + b[:, None])
            mask = relu_jacobian_mask(u_star)  # (n,B)

        # Save for backward
        ctx.save_for_backward(A, S, m_raw, U, b, x, z_star, mask)
        ctx.alpha = float(alpha)
        return z_star

    @staticmethod
    def backward(ctx, grad_z: Tensor):
        """
        Implicit differentiation for FB with prox=ReLU.

        Solve for each sample i:

            (I - S^T J_i) λ_i = grad_z_i,

        where S = (1-α) I + α W and J_i = diag(mask[:, i]).
        Then set ξ_i = J_i λ_i and accumulate parameter/input grads via ∂y/∂θ.
        """
        A, S_param, m_raw, U, b, x, z_star, mask = ctx.saved_tensors
        alpha = ctx.alpha
        device, dtype = A.device, A.dtype
        n, B = z_star.shape

        # Build W and S = (1-α)I + αW
        W = _build_W(A, S_param, m_raw)
        I = torch.eye(n, device=device, dtype=dtype)
        S_T = ((1.0 - alpha) * I + alpha * W).T

        # Solve (I - S^T J) λ = grad_z  per-sample
        lam = torch.empty_like(z_star)  # (n,B)
        for i in range(B):
            J = torch.diag(mask[:, i])
            M_T = I - S_T @ J  # (n,n)
            rhs = grad_z[:, i]
            lam[:, i] = torch.linalg.solve(M_T, rhs)

        # ξ = J λ
        xi = mask * lam  # (n,B)

        # Gradients for parameters via ∂y/∂θ; y = (1-α)z + α(W z + U x + b)
        # Effective "G" on W is: G_W = α * (xi @ z_star.T)
        G_W = alpha * (xi @ z_star.T)  # (n,n)

        # dL/dA from W = (1-m)I - A^T A + (S - S^T)  => ∂/∂A tr(G^T W) = - A (G + G^T)
        grad_A = -A @ (G_W + G_W.T)

        # dL/dS from W depends on (S - S^T): ⟨G, dS - dS^T⟩ = ⟨(G - G^T), dS⟩
        grad_S = G_W - G_W.T

        # dL/dm from W = (1 - m) I ... => ∂/∂m tr(G^T W) = - tr(G^T I) = - sum(diag(G))
        grad_m = -torch.einsum("ii->", G_W).unsqueeze(0)  # shape ()

        # chain through m_raw via softplus derivative (sigmoid)
        grad_m_raw = torch.sigmoid(m_raw) * grad_m

        # U, b, x
        grad_U = alpha * (xi @ x)             # (n,d)
        grad_b = alpha * xi.sum(dim=1)        # (n,)
        grad_x = (alpha * xi.T) @ U           # (B,d)

        # Return grads for tensor inputs in order: A, S, m_raw, U, b, x
        return grad_A, grad_S, grad_m_raw, grad_U, grad_b, grad_x, None, None, None


class MonDEQLayerFC(nn.Module):
    """
    Fully connected MON layer with FB equilibrium and implicit backward (ReLU prox).
    """

    def __init__(self, n: int, d: int, alpha: float = 1.0, max_iters: int = 500, tol: float = 1e-6):
        super().__init__()
        from ..operators import WKLinearFC
        self.core = WKLinearFC(n, d)
        self.alpha = float(alpha)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, d) -> z*: (n, B)
        z_star = _FBEquilibriumFunction.apply(
            self.core.A, self.core.S, self.core.m_raw, self.core.U, self.core.b, x,
            self.alpha, self.max_iters, self.tol
        )
        return z_star
