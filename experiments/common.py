# experiments/common.py
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch import Tensor

from mondeq.operators import WKLinearFC, spectral_bounds


torch.set_default_dtype(torch.float64)


def set_seed(s: int):
    torch.manual_seed(s)
    np.random.seed(s)


# --------------------------
# Parameter quantisation (static, symmetric, per-tensor)
# --------------------------
def fake_quant_sym(x: Tensor, bits: int, eps: float = 1e-12) -> Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().amax()
    if scale < eps:
        return torch.zeros_like(x)
    s = scale / (qmax + 0.0)
    y = torch.round(x / s).clamp_(min=-(qmax + 1), max=qmax) * s
    return y


def quantize_params(
    W: Tensor, U: Tensor, b: Tensor,
    bits_w: Optional[int], bits_u: Optional[int], bits_b: Optional[int]
) -> Tuple[Tensor, Tensor, Tensor]:
    Wq = fake_quant_sym(W, bits_w) if bits_w is not None else W
    Uq = fake_quant_sym(U, bits_u) if bits_u is not None else U
    bq = fake_quant_sym(b, bits_b) if bits_b is not None else b
    return Wq, Uq, bq


# --------------------------
# Iterate quantisation on the UPDATE (fixed or decaying step)
# --------------------------
@torch.no_grad()
def fb_solve_with_iterate_quant(
    W: Tensor,
    U: Tensor,
    b: Tensor,
    x: Tensor,
    *,
    alpha: float,
    max_iters: int,
    tol: float,
    bits_iter: int,
    mode: str = "decay",           # "fixed" or "decay"
    rho: float = 0.8,              # geometric decay if mode == "decay"
    z0: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Forward–Backward with quantisation on the UPDATE:
        u_k = (1-α) z_k + α (W z_k + U x + b)
        v_k = ReLU(u_k)
        Δ_k = v_k - z_k
        Δ̂_k = Q_k(Δ_k)           # per-sample symmetric uniform; step may decay
        z_{k+1} = z_k + Δ̂_k

    Per-sample step: s_k[j] = amax(Δ_k[:, j]) / qmax, optionally * rho**k.
    Quantising the update avoids early zeroing by ReLU and yields vanishing
    per-iteration error when mode="decay" with rho in (0,1).
    """
    assert mode in ("fixed", "decay")
    device, dtype = W.device, W.dtype
    n = W.size(0)
    B = x.size(0)
    qmax = (1 << (bits_iter - 1)) - 1

    z = torch.zeros((n, B), device=device, dtype=dtype) if z0 is None else z0.clone()
    residuals = []

    for k in range(max_iters):
        u = (1.0 - alpha) * z + alpha * (W @ z + (U @ x.T) + b[:, None])
        v = torch.relu(u)
        delta = v - z  # n x B

        # per-sample step (1 x B), robust to small amax
        amax = delta.abs().amax(dim=0, keepdim=True)  # 1 x B
        step = torch.where(amax > 0, amax / qmax, torch.zeros_like(amax))
        if mode == "decay":
            step = step * (rho ** k)

        # quantise update and apply
        delta_q = torch.where(
            amax > 0,
            torch.round(delta / step).clamp_(min=-(qmax + 1), max=qmax) * step,
            torch.zeros_like(delta),
        )
        z_next = z + delta_q

        res = (z_next - z).norm() / (z.norm() + 1e-12)
        residuals.append(float(res))
        z = z_next
        if res < tol:
            break

    return z, {"residuals": residuals}


# --------------------------
# Exact float FB solve (no iterate quant)
# --------------------------
def fb_solve_with_W(
    W: Tensor,
    U: Tensor,
    b: Tensor,
    x: Tensor,
    alpha: float,
    max_iters: int,
    tol: float,
    z0: Optional[Tensor] = None
) -> Tuple[Tensor, Dict[str, Any]]:
    n = W.size(0)
    B = x.size(0)
    device, dtype = W.device, W.dtype
    z = torch.zeros((n, B), device=device, dtype=dtype) if z0 is None else z0.clone()
    residuals = []
    with torch.no_grad():
        for _ in range(max_iters):
            u = (1.0 - alpha) * z + alpha * (W @ z + (U @ x.T) + b[:, None])
            z_next = torch.relu(u)
            res = (z_next - z).norm() / (z.norm() + 1e-12)
            residuals.append(float(res))
            z = z_next
            if res < tol:
                break
    return z, {"residuals": residuals}


# --------------------------
# Utilities
# --------------------------
def tail_ratio(residuals: list, tail: int = 50) -> float:
    if len(residuals) < 3:
        return float("inf")
    r = residuals[-tail:] if len(residuals) >= tail else residuals
    ratios = []
    for i in range(len(r) - 1):
        denom = r[i] + 1e-18
        ratios.append(r[i + 1] / denom)
    if not ratios:
        return float("inf")
    ratios = np.array(ratios, dtype=np.float64)
    return float(np.median(ratios))


def diverged(residuals: list, eps: float = 1e-3, patience: int = 10) -> bool:
    if len(residuals) < patience + 1:
        return False
    rr = []
    for i in range(-patience - 1, -1):
        denom = residuals[i] + 1e-18
        rr.append(residuals[i + 1] / denom)
    return np.median(rr) > 1.0 + eps or any(not np.isfinite(v) for v in residuals[-patience:])


@dataclass
class Instance:
    n: int
    d: int
    seed: int
    core: WKLinearFC
    W_f: Tensor
    U_f: Tensor
    b_f: Tensor
    x: Tensor


def build_W_from_core(core: WKLinearFC) -> Tensor:
    return core.W()


def sample_instance(n: int, d: int, B: int, seed: int) -> Instance:
    set_seed(seed)
    core = WKLinearFC(n, d)
    W_f = build_W_from_core(core).detach()
    U_f = core.U.detach().clone()
    b_f = core.b.detach().clone()
    x = torch.randn(B, d, dtype=W_f.dtype, device=W_f.device)
    return Instance(n=n, d=d, seed=seed, core=core, W_f=W_f, U_f=U_f, b_f=b_f, x=x)


def spectral_from_W(W: Tensor) -> Tuple[Tensor, Tensor]:
    n = W.size(0)
    I = torch.eye(n, device=W.device, dtype=W.dtype)
    A = I - W
    return spectral_bounds(A)


def alpha_crit_from_W(W: Tensor) -> float:
    m_tilde, L_tilde = spectral_from_W(W)
    m = float(m_tilde)
    L = float(L_tilde)
    if m <= 0 or not np.isfinite(m) or not np.isfinite(L) or L <= 0:
        return 0.0
    return 2.0 * m / (L * L)


def predicted_rate_from_W(W: Tensor, alpha: float) -> float:
    m_tilde, L_tilde = spectral_from_W(W)
    m, L = float(m_tilde), float(L_tilde)
    val = 1.0 - 2.0 * alpha * m + (alpha * L) ** 2
    return math.sqrt(max(0.0, val))


def deltaW_norm2(W: Tensor, Wq: Tensor) -> float:
    D = (Wq - W)
    try:
        v = torch.linalg.norm(D, 2)
        return float(v)
    except RuntimeError:
        # fallback power-iter
        x = torch.randn(D.size(1), dtype=D.dtype, device=D.device)
        x = x / x.norm()
        for _ in range(50):
            x = (D.T @ (D @ x))
            x = x / (x.norm() + 1e-18)
        val = float(torch.sqrt((x @ (D.T @ (D @ x)))))
        return val
