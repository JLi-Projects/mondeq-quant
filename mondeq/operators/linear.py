from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
from . import sym, spectral_bounds

def _project_psd_min_eig(S: torch.Tensor, m: float) -> torch.Tensor:
    """
    Given a symmetric matrix S, enforce S ≽ m I by shifting along identity if needed.
    Returns S_tilde with eigenvalues >= m (numerically).
    """
    # Compute smallest eigenvalue
    lam_min = torch.linalg.eigvalsh(S).min().real
    shift = torch.clamp(m - lam_min, min=0.0)
    if shift > 0:
        n = S.shape[-1]
        S = S + shift * torch.eye(n, device=S.device, dtype=S.dtype)
    return S

@dataclass
class LinearMonotone:
    """
    Linear monotone operator parameterised so that A := I - W ≽ m I (m-strongly monotone).
    Internally we keep an unconstrained parameter M and construct W via a symmetrised map.

    Construction:
        Let R be a free matrix. Set S = Sym(R R^T) to ensure S ≽ 0.
        Then A = S + m I  (so A ≽ m I), and W = I - A.

    This gives you:
      - forward: y = A x  (the linear operator I - W)
      - accessors for W and A
      - spectral diagnostics (m_tilde, L_tilde) needed by your certificates
    """
    dim: int
    m: float = 0.1
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Free parameter R initialised small so A ≈ m I initially
        self.R = torch.nn.Parameter(1e-2 * torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype))
        # Bias (optional) for affine maps; keep zero by default
        self.b = torch.nn.Parameter(torch.zeros(self.dim, device=self.device, dtype=self.dtype))
        # Register as a tiny nn.Module without subclassing (lightweight container)
        # If you prefer a full nn.Module, convert this dataclass into nn.Module subclass.
        self._params = torch.nn.ParameterList([self.R, self.b])

    @property
    def parameters(self):
        return self._params

    def A(self) -> torch.Tensor:
        """A = I - W, enforced to be ≽ m I."""
        RR = self.R @ self.R.transpose(-1, -2)  # PSD
        S = sym(RR)
        S = _project_psd_min_eig(S, self.m)  # numerical safety
        I = torch.eye(self.dim, device=S.device, dtype=S.dtype)
        A = S + self.m * I
        return A

    def W(self) -> torch.Tensor:
        """W = I - A."""
        I = torch.eye(self.dim, device=self.device, dtype=self.dtype)
        return I - self.A()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A x + b (linear monotone operator acting on x)."""
        return x @ self.A().transpose(0, 1) + self.b

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @torch.no_grad()
    def spectral_diagnostics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (m_tilde, L_tilde) where
          m_tilde = λ_min(Sym(A)),
          L_tilde = ||A||_2.
        """
        A = self.A()
        return spectral_bounds(A)

    @torch.no_grad()
    def alpha_window(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For FB on monotone–Lipschitz A (linear), a classic stepsize window ensuring contraction
        of I - α A in 2-norm is α in (0, 2 m / L^2) when m>0, L>=m.
        """
        m_t, L_t = self.spectral_diagnostics()
        if m_t <= 0 or L_t <= 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        return torch.tensor(0.0, device=self.device), (2.0 * m_t) / (L_t * L_t)

    @torch.no_grad()
    def contraction_factor(self, alpha: float) -> torch.Tensor:
        """
        Predict r(α) = ||I - α A||_2 upper-bounded by sqrt(1 - 2α m + α^2 L^2) for linear A.
        Useful to compare to measured per-iteration contraction.
        """
        m_t, L_t = self.spectral_diagnostics()
        return torch.sqrt(torch.clamp(1 - 2 * alpha * m_t + (alpha * L_t) ** 2, min=0.0))

