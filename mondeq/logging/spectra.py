from __future__ import annotations
from typing import Tuple
import torch

def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

@torch.no_grad()
def spectral_bounds(A: torch.Tensor) -> Tuple[float, float]:
    S = sym(A)
    lam_min = torch.linalg.eigvalsh(S).min().real.item()
    sigma_max = torch.linalg.svdvals(A).max().real.item()
    return lam_min, sigma_max

@torch.no_grad()
def predicted_fb_rate(alpha: float, m_tilde: float, L_tilde: float) -> float:
    val = 1.0 - 2.0 * alpha * m_tilde + (alpha * L_tilde) ** 2
    return float((val if val > 0 else 0.0) ** 0.5)

