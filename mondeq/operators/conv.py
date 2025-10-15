from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
from mondeq.utils.power import power_iteration_tensor

@dataclass
class ConvMonotone:
    """
    Convolutional monotone map on 4D tensors (N,C,H,W), channel-wise operator:
      A x = m x + K^T (K x)
    with K = Conv2d(C→C, kernel_size=3, padding=1, groups=C), so K^T K is PSD.
    Thus A ⪰ m I; we use m>0. L (spectral norm) is estimated by power iteration.
    """
    channels: int
    m: float = 0.1
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels, bias=False).to(self.device, self.dtype)
        nn.init.normal_(self.K.weight, mean=0.0, std=1e-2)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # A x = m x + K^T (K x)
        y = self.K(x)
        # ConvTranspose with tied weights = adjoint in the circular-padding ideal; we approximate by conv with flipped kernels.
        # Simpler: use conv again with weight transposed per-group (depthwise), equivalent in this grouped setting.
        w = self.K.weight
        # 180-degree rotate spatial kernel for adjoint-like effect
        wT = torch.flip(w, dims=[-1, -2])
        y = torch.nn.functional.conv2d(y, wT, bias=None, stride=1, padding=1, groups=self.channels)
        return self.m * x + y

    @torch.no_grad()
    def spectral_norm_estimate(self, H: int, W: int, iters: int = 100) -> float:
        def op(u: torch.Tensor) -> torch.Tensor:
            return self.apply(u)
        return power_iteration_tensor(op, (1, self.channels, H, W), iters=iters, dtype=self.dtype, device=self.device)
