# mondeq/layers/fc.py
from __future__ import annotations
import torch
import time
from torch import nn, Tensor
from ..prox import relu_prox, relu_jacobian_mask
from ..splitting import fb_solve, _build_W


# Thread-local storage for stats from the most recent forward/backward pass
# This allows the layer to expose iteration counts without modifying the autograd API
_last_fwd_iters = 0
_last_fwd_time = 0.0
_last_bwd_iters = 0
_last_bwd_time = 0.0


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
        global _last_fwd_iters, _last_fwd_time
        start = time.time()

        # Solve equilibrium without tracking a giant graph
        with torch.no_grad():
            z_star, hist = fb_solve(A, S, m_raw, U, b, x, alpha=alpha, max_iters=max_iters, tol=tol)

            # Prox input at equilibrium for Jacobian selection
            W = _build_W(A, S, m_raw)
            u_star = (1.0 - alpha) * z_star + alpha * (W @ z_star + (U @ x.T) + b[:, None])
            mask = relu_jacobian_mask(u_star)  # (n,B)

        # Update stats
        _last_fwd_iters = hist["iters"]
        _last_fwd_time = time.time() - start

        # Save for backward
        ctx.save_for_backward(A, S, m_raw, U, b, x, z_star, mask)
        ctx.alpha = float(alpha)
        ctx.max_iters = int(max_iters)
        ctx.tol = float(tol)
        return z_star

    @staticmethod
    def backward(ctx, grad_z: Tensor):
        """
        Implicit differentiation for FB with prox=ReLU.

        Uses iterative backward solve (same FB iteration as forward) instead of
        per-sample direct linear solve. This is O(iters * n^2 * B) instead of O(B * n^3).

        The backward pass solves:
            u = J^T (W^T u + g)
        where J = diag(mask) is the ReLU Jacobian.
        """
        global _last_bwd_iters, _last_bwd_time
        start = time.time()

        A, S_param, m_raw, U, b, x, z_star, mask = ctx.saved_tensors
        alpha = ctx.alpha
        max_iters = ctx.max_iters
        tol = ctx.tol
        device, dtype = A.device, A.dtype
        n, B = z_star.shape

        # Build W
        W = _build_W(A, S_param, m_raw)

        # Iterative backward solve (following locuslab's approach)
        # We need to solve: (I - S^T J) lambda = grad_z
        # Rewritten as fixed-point: lambda = S^T (J lambda) + grad_z
        # With S = (1-alpha)I + alpha*W
        # Using the prox on the ReLU adjoint

        # Compute J derivative values: j = 1 where mask > 0, else 0
        j = mask.float()  # (n, B)
        I_mask = (j == 0)  # where ReLU was inactive
        d = torch.zeros_like(j)
        d[~I_mask] = (1 - j[~I_mask]) / j[~I_mask]  # = 0 where j=1

        v = j * grad_z  # (n, B)
        u = torch.zeros((n, B), device=device, dtype=dtype)

        it = 0
        for it in range(max_iters):
            # FB iteration on adjoint: un = (1-alpha)*u + alpha * W^T @ u
            un = (1 - alpha) * u + alpha * (W.T @ u)
            # Apply adjoint prox
            un = (un + alpha * (1 + d) * v) / (1 + alpha * d)
            un[I_mask] = v[I_mask]  # fix inactive components

            err = (un - u).norm() / (u.norm() + 1e-12)
            u = un
            if err < tol:
                break

        _last_bwd_iters = it + 1
        _last_bwd_time = time.time() - start

        # xi = J @ u (element-wise since J is diagonal)
        xi = j * u  # (n, B)

        # Gradients for parameters via dy/dtheta; y = (1-alpha)z + alpha(W z + U x + b)
        # Effective "G" on W is: G_W = alpha * (xi @ z_star.T)
        G_W = alpha * (xi @ z_star.T)  # (n,n)

        # dL/dA from W = (1-m)I - A^T A + (S - S^T)  => d/dA tr(G^T W) = - A (G + G^T)
        grad_A = -A @ (G_W + G_W.T)

        # dL/dS from W depends on (S - S^T): <G, dS - dS^T> = <(G - G^T), dS>
        grad_S = G_W - G_W.T

        # dL/dm from W = (1 - m) I ... => d/dm tr(G^T W) = - tr(G^T I) = - sum(diag(G))
        grad_m = -torch.einsum("ii->", G_W).unsqueeze(0)  # shape ()

        # chain through m_raw via softplus derivative (sigmoid)
        grad_m_raw = torch.sigmoid(m_raw) * grad_m

        # U, b, x
        grad_U = alpha * (xi @ x)             # (n,d)
        grad_b = alpha * xi.sum(dim=1)        # (n,)
        grad_x = (alpha * xi.T) @ U           # (B,d)

        # Return grads for tensor inputs in order: A, S, m_raw, U, b, x
        return grad_A, grad_S, grad_m_raw, grad_U, grad_b, grad_x, None, None, None


class SplittingMethodStats:
    """
    Track forward-backward splitting convergence statistics.
    Compatible with locuslab's stats interface for alpha tuning.
    """

    def __init__(self):
        self.fwd_iters = _Meter()
        self.bkwd_iters = _Meter()
        self.fwd_time = _Meter()
        self.bkwd_time = _Meter()

    def reset(self):
        self.fwd_iters.reset()
        self.bkwd_iters.reset()
        self.fwd_time.reset()
        self.bkwd_time.reset()

    def report(self):
        print(f"Fwd iters: {self.fwd_iters.avg:.2f}\tFwd Time: {self.fwd_time.avg:.4f}\t"
              f"Bkwd Iters: {self.bkwd_iters.avg:.2f}\tBkwd Time: {self.bkwd_time.avg:.4f}")


class _Meter:
    """Simple meter for tracking values (matches locuslab's Meter interface)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.max = float("-inf")
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)


class MonDEQLayerFC(nn.Module):
    """
    Fully connected MON layer with FB equilibrium and implicit backward (ReLU prox).
    """

    def __init__(self, n: int, d: int, alpha: float = 1.0, max_iters: int = 500, tol: float = 1e-5):
        super().__init__()
        from ..operators import WKLinearFC
        self.core = WKLinearFC(n, d)
        self.alpha = float(alpha)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        # Stats tracking (matches locuslab interface)
        self.stats = SplittingMethodStats()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, d) -> z*: (n, B)
        z_star = _FBEquilibriumFunction.apply(
            self.core.A, self.core.S, self.core.m_raw, self.core.U, self.core.b, x,
            self.alpha, self.max_iters, self.tol
        )
        # Update stats from global variables set by autograd function
        self.stats.fwd_iters.update(_last_fwd_iters)
        self.stats.fwd_time.update(_last_fwd_time)
        # Note: backward stats are tracked via globals (_last_bwd_iters, _last_bwd_time)
        # but not automatically collected since backward runs in autograd context
        return z_star
