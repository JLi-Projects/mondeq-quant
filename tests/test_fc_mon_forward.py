# tests/test_fc_mon_forward.py
import torch
from mondeq.layers.fc import MonDEQLayerFC
from mondeq.operators import spectral_bounds


def test_forward_runs_and_converges_small():
    torch.manual_seed(0)
    n, d, B = 16, 8, 4
    layer = MonDEQLayerFC(n, d, alpha=0.9, max_iters=400, tol=1e-7)
    x = torch.randn(B, d)
    z = layer(x)
    assert z.shape == (n, B)

    # Check strong monotonicity of A = I - W
    A = layer.core.A_matrix().detach()
    m_tilde, L_tilde = spectral_bounds(A)
    assert float(m_tilde) > 0.0
    assert float(L_tilde) >= float(m_tilde)
