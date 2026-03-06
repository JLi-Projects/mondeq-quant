# mondeq/prox.py
from __future__ import annotations
import torch
from torch import Tensor


@torch.no_grad()
def relu_prox(u: Tensor, alpha: float | Tensor) -> Tensor:
    """
    Prox of the indicator of R_+^n is projection onto R_+^n, i.e., ReLU.
    Alpha is unused (kept for API uniformity).
    u : (n, B)
    returns (n, B)
    """
    return torch.relu(u)


@torch.no_grad()
def relu_jacobian_mask(u: Tensor) -> Tensor:
    """
    Clarke Jacobian selection for ReLU prox at input u:
    J = diag(1_{u > 0}). We return the mask (n, B).
    """
    return (u > 0).to(u.dtype)
