# mondeq/operators.py
from __future__ import annotations
import math
import torch
from torch import nn, Tensor


def _softplus(x: Tensor) -> Tensor:
    return torch.nn.functional.softplus(x)


class WKLinearFC(nn.Module):
    """
    Winston–Kolter parameterisation for fully connected MON.

    Parameters
    ----------
    n : int
        Hidden dimension (state size).
    d : int
        Input dimension.
    rank : int | None
        Rank of A in W = (1 - m)I - A^T A + (S - S^T). If None, defaults to max(4, n // 4).
    m_init : float
        Initial strong monotonicity margin (>0) before softplus.
    """

    def __init__(self, n: int, d: int, rank: int | None = None, m_init: float = 0.2):
        super().__init__()
        self.n, self.d = n, d
        r = rank if rank is not None else max(4, n // 4)

        # Parameters of W
        self.A = nn.Parameter(0.01 * torch.randn(r, n))       # PSD part via A^T A
        self.S = nn.Parameter(0.01 * torch.randn(n, n))       # skew part via S - S^T
        self.m_raw = nn.Parameter(torch.tensor(math.log(math.exp(m_init) - 1.0)))  # softplus -> m > 0

        # Affine injection U, b
        self.U = nn.Parameter(0.01 * torch.randn(n, d))
        self.b = nn.Parameter(torch.zeros(n))

    @property
    def m(self) -> Tensor:
        # small epsilon keeps m bounded away from 0
        return _softplus(self.m_raw) + 1e-4

    def W(self) -> Tensor:
        n = self.n
        I = torch.eye(n, device=self.A.device, dtype=self.A.dtype)
        AAt = self.A.transpose(0, 1) @ self.A              # n x n PSD
        Bskew = self.S - self.S.transpose(0, 1)            # skew-symmetric
        return (1.0 - self.m) * I - AAt + Bskew

    def A_matrix(self) -> Tensor:
        # A := I - W, the monotone linear core
        n = self.n
        I = torch.eye(n, device=self.A.device, dtype=self.A.dtype)
        return I - self.W()

    def forward(self, z: Tensor, x: Tensor | None = None) -> Tensor:
        """
        Compute affine map: W z + U x + b.
        Shapes:
            z : (n, B)
            x : (B, d) or None
            returns (n, B)
        """
        out = self.W() @ z + self.b[:, None]
        if x is not None:
            out = out + (self.U @ x.T)
        return out


def spectral_bounds(A: Tensor) -> tuple[Tensor, Tensor]:
    """
    (m_tilde, L_tilde) for A = I - W.
    m_tilde : min eigenvalue of Sym(A)
    L_tilde : spectral norm of A
    """
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be square"
    S = 0.5 * (A + A.T)
    evals = torch.linalg.eigvalsh(S)
    m_tilde = evals.min()
    L_tilde = torch.linalg.norm(A, 2)
    return m_tilde.real, L_tilde.real
