# tests/test_fc_mon_backward.py
import torch
from mondeq.layers.fc import MonDEQLayerFC


def test_backward_gradients_exist():
    torch.manual_seed(0)
    n, d, B = 12, 6, 5
    layer = MonDEQLayerFC(n, d, alpha=0.9, max_iters=300, tol=1e-6)
    x = torch.randn(B, d, requires_grad=True)
    z = layer(x)
    loss = z.pow(2).mean()
    loss.backward()

    # Parameter grads
    assert layer.core.A.grad is not None
    assert layer.core.S.grad is not None
    assert layer.core.m_raw.grad is not None
    assert layer.core.U.grad is not None
    assert layer.core.b.grad is not None

    # Input grad
    assert x.grad is not None
