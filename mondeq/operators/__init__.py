from __future__ import annotations
from typing import Protocol, Tuple
import torch

class Operator(Protocol):
    """Abstract operator: y = F(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

def sym(A: torch.Tensor) -> torch.Tensor:
    """Symmetric part: (A + A^T)/2."""
    return 0.5 * (A + A.transpose(-1, -2))

@torch.no_grad()
def spectral_bounds(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (lambda_min(Sym(A)), ||A||_2) for square A.
    Uses torch.linalg.eigvalsh on Sym(A) and svdvals for ||A||_2.
    """
    S = sym(A)
    lam_min = torch.linalg.eigvalsh(S).min()
    sigma_max = torch.linalg.svdvals(A).max()
    return lam_min.real, sigma_max.real

