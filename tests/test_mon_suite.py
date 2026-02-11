# tests/test_mon_suite.py
import math
import torch
import pytest

from mondeq.layers.fc import MonDEQLayerFC
from mondeq.operators import spectral_bounds

torch.set_default_dtype(torch.float64)

def _random_layer(n=16, d=8, alpha=0.9, iters=400, tol=1e-7):
    torch.manual_seed(0)
    layer = MonDEQLayerFC(n, d, alpha=alpha, max_iters=iters, tol=tol).double()
    # Bias away from the ReLU kink so J = I a.e. for the finite-difference test
    with torch.no_grad():
        layer.core.b.add_(1.0)
    return layer

def test_wk_strong_monotonicity():
    layer = _random_layer()
    A = layer.core.A_matrix().detach()
    m_tilde, L_tilde = spectral_bounds(A)
    assert float(m_tilde) > 0.0
    assert float(L_tilde) >= float(m_tilde)

def test_forward_converges_and_respects_bound():
    n, d, B = 32, 12, 4
    layer = _random_layer(n=n, d=d, alpha=0.9, iters=600, tol=1e-8)
    x = torch.randn(B, d, dtype=torch.float64)
    z = layer(x)  # (n,B)
    assert z.shape == (n, B)

    # Check residual tail contraction ≤ sqrt(1 - 2α m + (α L)^2) + ε
    A = layer.core.A_matrix().detach()
    m_tilde, L_tilde = spectral_bounds(A)
    alpha = layer.alpha
    r_bound = math.sqrt(max(0.0, 1.0 - 2.0 * alpha * float(m_tilde) + (alpha * float(L_tilde)) ** 2))
    # Re-run once to record residuals
    from mondeq.splitting import fb_solve
    with torch.no_grad():
        _, hist = fb_solve(layer.core.A, layer.core.S, layer.core.m_raw,
                           layer.core.U, layer.core.b, x,
                           alpha=alpha, max_iters=400, tol=1e-12)
    # Use median tail ratio to avoid early transients
    res = hist["residual"]
    tail = res[-50:] if len(res) >= 50 else res
    tail_ratio = sum(tail[i+1]/(tail[i]+1e-18) for i in range(len(tail)-1)) / max(1, len(tail)-1)
    assert tail_ratio <= r_bound + 5e-2  # allow slack due to nonlinearity/prox

def test_backward_parameter_grads_exist_and_are_finite():
    n, d, B = 20, 10, 3
    layer = _random_layer(n=n, d=d, alpha=0.9, iters=500, tol=1e-7)
    x = torch.randn(B, d, dtype=torch.float64, requires_grad=True)
    z = layer(x)
    loss = (z**2).mean()
    loss.backward()
    assert layer.core.A.grad is not None and torch.isfinite(layer.core.A.grad).all()
    assert layer.core.S.grad is not None and torch.isfinite(layer.core.S.grad).all()
    assert layer.core.m_raw.grad is not None and torch.isfinite(layer.core.m_raw.grad).all()
    assert layer.core.U.grad is not None and torch.isfinite(layer.core.U.grad).all()
    assert layer.core.b.grad is not None and torch.isfinite(layer.core.b.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_backward_wrt_x_matches_finite_difference():
    n, d, B = 12, 6, 2
    layer = _random_layer(n=n, d=d, alpha=0.8, iters=600, tol=1e-8)
    x = torch.randn(B, d, dtype=torch.float64, requires_grad=True)
    z = layer(x)
    loss = (z**2).sum()  # scalar
    loss.backward()
    autograd_grad = x.grad.detach().clone()

    # Finite differences around x (central diff), staying away from ReLU kinks via positive bias in b
    eps = 1e-5
    num_grad = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(B):
            for j in range(d):
                e = torch.zeros_like(x)
                e[i, j] = eps
                lp = (layer(x + e)**2).sum()
                lm = (layer(x - e)**2).sum()
                num_grad[i, j] = (lp - lm) / (2 * eps)

    # Relative error tolerance; larger near kinks, but b shift should keep us away
    rel_err = (autograd_grad - num_grad).norm() / (num_grad.norm() + 1e-12)
    assert rel_err < 5e-2
