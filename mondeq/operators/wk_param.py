from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class WKParamMonotone:
    """
    Winston–Kolter parameterisation:
        W = (1 - m) I - A^T A + (B - B^T)
        => I - W = m I + A^T A - (B - B^T)

    Sym(I - W) = m I + A^T A ⪰ m I, so strong monotonicity (m) is guaranteed.
    You can tune the spectrum (Lipschitz) by scaling A and B.

    Args:
        dim: state dimension n
        m: strong monotonicity parameter (> 0)
        a_scale: Frobenius-scale of A (controls PSD part)
        b_scale: Frobenius-scale of B (controls skew part)
        dtype, device: tensor settings
        seed: optional RNG seed for reproducibility
    """
    dim: int
    m: float = 0.2
    a_scale: float = 0.2
    b_scale: float = 0.05
    dtype: torch.dtype = torch.float64
    device: Optional[torch.device] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.m > 0.0, "m must be > 0"
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.seed is not None:
            torch.manual_seed(int(self.seed))

        n = self.dim
        A = torch.randn(n, n, device=self.device, dtype=self.dtype)
        B = torch.randn(n, n, device=self.device, dtype=self.dtype)

        # scale A, B to requested Frobenius norms (a_scale, b_scale)
        def _scale(M: torch.Tensor, target: float) -> torch.Tensor:
            fn = torch.linalg.norm(M, ord='fro') + 1e-12
            return M * (target / fn)

        A = _scale(A, self.a_scale)
        B = _scale(B, self.b_scale)

        # Build W and I-W
        I = torch.eye(n, device=self.device, dtype=self.dtype)
        W = (1.0 - self.m) * I - A.T @ A + (B - B.T)
        self._W = W
        self._I = I
        # cache A=I-W
        self._A_lin = I - W

    def W(self) -> torch.Tensor:
        return self._W

    def A(self) -> torch.Tensor:
        """Return I - W (the monotone linear operator used by FB)."""
        return self._A_lin

    @torch.no_grad()
    def spectral_diagnostics(self) -> Tuple[float, float]:
        """
        Return (m_tilde, L_tilde) where:
          m_tilde = λ_min(Sym(I-W)),  L_tilde = ||I-W||_2
        """
        A = self._A_lin
        S = 0.5 * (A + A.T)
        m_t = float(torch.linalg.eigvalsh(S).min().item())
        L_t = float(torch.linalg.norm(A, ord=2).item())
        return m_t, L_t

