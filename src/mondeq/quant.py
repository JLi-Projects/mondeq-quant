# mondeq/quant.py
from __future__ import annotations
import torch
from torch import Tensor


def fake_quant_sym(x: Tensor, bits: int, eps: float = 1e-12) -> Tensor:
    """
    Symmetric fake quantisation to `bits` bits.

    Maps x to the nearest point in {-qmax*scale, ..., 0, ..., qmax*scale}
    where qmax = 2^(bits-1) - 1 and scale = max|x| / qmax.
    """
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().amax() / (qmax + eps)
    if scale < eps:
        return torch.zeros_like(x)
    y = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    return y


def fixed_iterate_quant(z: Tensor, delta: float) -> Tensor:
    """
    Fixed-resolution iterate quantisation (uniform grid rounding).

    Maps each element of z to the nearest multiple of delta:
        Q(z) = delta * round(z / delta)

    This introduces a constant per-element error bounded by delta/2.

    Parameters
    ----------
    z : Tensor
        Iterate tensor, shape (n, B).
    delta : float
        Quantisation cell width (resolution).

    Returns
    -------
    Tensor
        Quantised iterate, same shape as z.
    """
    if delta <= 0:
        return z
    return delta * torch.round(z / delta)


def adaptive_iterate_quant(z_diff: Tensor, delta_0: float, gamma: float, k: int) -> Tensor:
    """
    Adaptive (Jonkman-style) iterate quantisation with shrinking cell width.

    Quantises the iterate *difference* z^{k+1} - z^k with resolution
    delta_k = delta_0 * gamma^k.  Because gamma < 1, the errors
    {delta_k} are summable (geometric series), which guarantees
    convergence to the exact fixed point (not just an error ball).

    Parameters
    ----------
    z_diff : Tensor
        Iterate difference z^{k+1} - z^k, shape (n, B).
    delta_0 : float
        Initial quantisation cell width.
    gamma : float
        Decay rate, must be in (0, 1) for summability.
    k : int
        Current iteration index (0-based).

    Returns
    -------
    Tensor
        Quantised iterate difference, same shape as z_diff.
    """
    delta_k = delta_0 * (gamma ** k)
    if delta_k < 1e-15:
        return z_diff
    return delta_k * torch.round(z_diff / delta_k)


