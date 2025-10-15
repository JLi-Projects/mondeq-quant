from __future__ import annotations
from typing import Callable, Tuple
import torch

def power_iteration(op: Callable[[torch.Tensor], torch.Tensor], dim: int, iters: int = 200, dtype=torch.float64, device=None) -> float:
    """Estimate ||op||_2 via power iteration."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(dim, device=device, dtype=dtype)
    x = x / (x.norm() + 1e-12)
    for _ in range(iters):
        y = op(x)
        n = y.norm() + 1e-12
        x = y / n
    return float((op(x).norm() + 1e-12).item())

def power_iteration_tensor(op: Callable[[torch.Tensor], torch.Tensor], shape: Tuple[int, ...], iters: int = 200, dtype=torch.float32, device=None) -> float:
    """Power iteration on tensor-shaped operator (for conv A). Returns spectral norm estimate."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(*shape, device=device, dtype=dtype)
    x = x / (x.norm() + 1e-12)
    for _ in range(iters):
        y = op(x)
        n = y.norm() + 1e-12
        x = y / n
    return float((op(x).norm() + 1e-12).item())
