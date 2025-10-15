from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
import math
import torch

Tensor = torch.Tensor
IterCallback = Callable[[int, Tensor, float, Dict[str, Any]], None]

@dataclass
class FBSolver:
    """
    Forward–Backward (gradient-descent-like on linear A) with per-iteration callbacks.
    Solves z_{k+1} = z_k - α (A z_k - r). For linear tests, this is enough to measure rates.

    Improvements over the earlier version:
      - min_iters: do not exit before collecting a burn-in worth of iterations
      - optional z_star: if provided, we report error-based contraction ||z_{k+1}-z*|| / ||z_k - z*||
    """
    max_iters: int = 2_000
    min_iters: int = 50
    tol: float = 1e-8
    alpha: float = 1e-1
    damping: float = 1.0  # 1 = pure FB; <1 = under-relaxed
    callback: Optional[IterCallback] = None
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64

    def solve_linear(
        self,
        A: Tensor,
        r: Optional[Tensor] = None,
        z0: Optional[Tensor] = None,
        z_star: Optional[Tensor] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Fixed-point / linear solve: A z* = r. Iterate z_{k+1} = z_k - α (A z_k - r).
        If z_star is supplied, we will report error-based contraction factors.
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = A.shape[-1]
        if r is None:
            r = torch.zeros(n, device=self.device, dtype=self.dtype)
        if z0 is None:
            z = torch.randn(n, device=self.device, dtype=self.dtype)  # non-zero start
        else:
            z = z0.to(device=self.device, dtype=self.dtype).clone()

        A = A.to(device=self.device, dtype=self.dtype)
        r = r.to(device=self.device, dtype=self.dtype)
        if z_star is not None:
            z_star = z_star.to(device=self.device, dtype=self.dtype)

        prev = None
        for k in range(self.max_iters):
            grad = A @ z - r
            zk1 = z - self.alpha * grad
            if self.damping != 1.0:
                zk1 = (1 - self.damping) * z + self.damping * zk1

            res = float(torch.linalg.norm(grad).item())

            # Step-ratio contraction (falls back if z_star is missing)
            meas_contr = math.nan
            if prev is not None:
                num = torch.linalg.norm(zk1 - z)
                den = torch.linalg.norm(z - prev) + 1e-16
                meas_contr = float((num / den).item())

            # Error-ratio contraction if z_star is available
            err_contr = math.nan
            if z_star is not None:
                ez  = torch.linalg.norm(z - z_star)
                ez1 = torch.linalg.norm(zk1 - z_star)
                if ez.item() > 0:
                    err_contr = float((ez1 / ez).item())

            if self.callback is not None:
                info = {"residual": res, "measured_contraction": meas_contr, "error_contraction": err_contr}
                if extra:
                    info.update(extra)
                self.callback(k, zk1.detach(), res, info)

            # Only allow early exit after min_iters to ensure we measured something
            if k + 1 >= self.min_iters and res < self.tol:
                z = zk1
                break

            prev = z
            z = zk1

        return z