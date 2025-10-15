from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import torch

Tensor = torch.Tensor

@dataclass
class PRLinear:
    """
    Peaceman–Rachford splitting for linear monotone equation A z = r.

    We solve 0 ∈ F(z) + G(z) - r with the PR iteration:
        x_k = (I + αF)^{-1}(u_k + α r)
        y_k = (I + αG)^{-1}(2 x_k - u_k + α r)
        u_{k+1} = u_k + 2 (y_k - x_k)
    A fixed point satisfies x_* = y_* = z_*.

    We supply F = m I  (strongly monotone, trivial resolvent)
             G = A - m I (monotone PSD if A ⪰ m I)

    That makes:
        (I + αF)^{-1}(v) = (1/(1+α m)) v
        (I + αG)^{-1}(v) = solve( (I + α(A - mI)) y = v )
    """
    alpha: float = 1e-1
    max_iters: int = 2_000
    min_iters: int = 50
    tol: float = 1e-8
    m_split: float = 0.1
    callback: Optional[Callable[[int, Tensor, float, Dict[str, Any]], None]] = None
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64

    def solve(self, A: Tensor, r: Optional[Tensor] = None, u0: Optional[Tensor] = None) -> Tensor:
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = A.shape[-1]
        A = A.to(device=self.device, dtype=self.dtype)
        if r is None:
            r = torch.zeros(n, device=self.device, dtype=self.dtype)
        else:
            r = r.to(device=self.device, dtype=self.dtype)
        if u0 is None:
            u = torch.randn(n, device=self.device, dtype=self.dtype)
        else:
            u = u0.to(device=self.device, dtype=self.dtype).clone()

        I = torch.eye(n, device=self.device, dtype=self.dtype)
        m = float(self.m_split)

        # resolvents
        def JF(v: Tensor) -> Tensor:
            return v / (1.0 + self.alpha * m)

        def JG(v: Tensor) -> Tensor:
            # (I + α(A - mI)) y = v  =>  ((1 - α m) I + α A) y = v
            M = (1.0 - self.alpha * m) * I + self.alpha * A
            return torch.linalg.solve(M, v)

        x = torch.zeros_like(u)
        y = torch.zeros_like(u)

        prev = None
        for k in range(self.max_iters):
            x = JF(u + self.alpha * r)
            y = JG(2 * x - u + self.alpha * r)
            u_next = u + 2 * (y - x)

            # residual in equation A z = r using z ≈ y (or x)
            res = float(torch.linalg.norm(A @ y - r).item())

            # a simple contraction proxy on u
            contr = float((torch.linalg.norm(u_next - u) / (torch.linalg.norm(u - prev) + 1e-16)).item()) if prev is not None else float("nan")

            if self.callback is not None:
                self.callback(k, y.detach(), res, {"measured_contraction_u": contr})

            if k + 1 >= self.min_iters and res < self.tol:
                u = u_next
                break

            prev = u
            u = u_next

        # best current z estimate is y (or x)
        return y
