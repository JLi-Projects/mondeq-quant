# mondeq/quant.py
from __future__ import annotations
import torch
from torch import Tensor

def fake_quant_sym(x: Tensor, bits: int, eps: float = 1e-12) -> Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().amax() / (qmax + eps)
    if scale < eps:
        return torch.zeros_like(x)
    y = torch.clamp(torch.round(x / scale), -qmax-1, qmax) * scale
    return y

def quantize_params(W: Tensor, U: Tensor, b: Tensor, bits_w: int, bits_u: int | None = None, bits_b: int | None = None):
    U_bits = bits_u if bits_u is not None else bits_w
    b_bits = bits_b if bits_b is not None else bits_w
    Wq = fake_quant_sym(W, bits_w)
    Uq = fake_quant_sym(U, U_bits)
    bq = fake_quant_sym(b, b_bits)
    return Wq, Uq, bq
