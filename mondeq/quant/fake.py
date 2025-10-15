from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable
import torch

Tensor = torch.Tensor

def fake_quantize_symmetric(x: Tensor, num_bits: int = 8, scale_boost: float = 1.0) -> Tensor:
    qmax = (2 ** (num_bits - 1)) - 1
    # robust scale: prevent degenerate tiny scales; allow optional noise boost
    scale = torch.maximum(x.abs().max() / qmax, torch.tensor(1e-12, device=x.device, dtype=x.dtype))
    scale = scale * float(scale_boost)
    x_q = torch.round(x / scale).clamp(-qmax-1, qmax) * scale
    return x_q

@dataclass
class QuantWrapperLinearA:
    """
    Fake-quant wrapper for a *linear* operator A = I - W, provided by a callable a_fn().
    """
    a_fn: Callable[[], Tensor]
    num_bits: int = 8
    per_tensor: bool = True
    dtype: torch.dtype = torch.float64
    scale_boost: float = 1.0  # >1 inflates quantization noise (for stress/failure demos)

    def A_q(self) -> Tensor:
        A = self.a_fn().to(dtype=self.dtype)
        return fake_quantize_symmetric(A, self.num_bits, self.scale_boost)

    @torch.no_grad()
    def deltas(self) -> Tuple[float, float]:
        A = self.a_fn().to(dtype=self.dtype)
        Aq = self.A_q()
        dA = torch.linalg.norm(Aq - A, ord=2).item()
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        W = I - A
        Wq = I - Aq
        dW = torch.linalg.norm(Wq - W, ord=2).item()
        return dA, dW
