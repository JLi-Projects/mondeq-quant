# mondeq/splitting.py
from __future__ import annotations
import torch
from torch import Tensor
from .prox import relu_prox


@torch.no_grad()
def _build_W(A: Tensor, S: Tensor, m_raw: Tensor) -> Tensor:
    m = torch.nn.functional.softplus(m_raw) + 1e-4
    n = A.size(1)
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    AAt = A.transpose(0, 1) @ A
    Bskew = S - S.transpose(0, 1)
    return (1.0 - m) * I - AAt + Bskew


@torch.no_grad()
def fb_solve(
    A: Tensor,
    S: Tensor,
    m_raw: Tensor,
    U: Tensor,
    b: Tensor,
    x: Tensor,
    *,
    alpha: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
    z0: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """
    Forward–Backward fixed-point iteration for fully connected MON:

        z_{k+1} = prox_{alpha f}((1 - alpha) z_k + alpha (W z_k + U x + b)),

    with prox = ReLU-projection (indicator of R_+^n).

    Shapes:
        A: (r, n), S: (n, n), m_raw: (), U: (n, d), b: (n,), x: (B, d)
        Returns z*: (n, B)
    """
    device, dtype = A.device, A.dtype
    n = U.size(0)
    B = x.size(0)
    z = z0.clone() if z0 is not None else torch.zeros((n, B), device=device, dtype=dtype)

    W = _build_W(A, S, m_raw)
    hist = {"residual": []}

    for _ in range(max_iters):
        u = (1.0 - alpha) * z + alpha * (W @ z + (U @ x.T) + b[:, None])
        z_next = relu_prox(u, alpha)
        res = (z_next - z).norm() / (z.norm() + 1e-12)
        hist["residual"].append(float(res))
        z = z_next
        if res < tol:
            break

    return z, hist
