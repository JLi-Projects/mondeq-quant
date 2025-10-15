import torch
from mondeq.operators.linear import LinearMonotone

def test_linear_monotone_spectra_and_window():
    torch.manual_seed(0)
    op = LinearMonotone(dim=8, m=0.2, dtype=torch.float64, device=torch.device("cpu"))
    m_t, L_t = op.spectral_diagnostics()
    assert m_t > 0.0
    assert L_t >= m_t
    a_lo, a_hi = op.alpha_window()
    assert a_hi > 0.0
    r = op.contraction_factor(float(a_hi) * 0.5)
    assert 0.0 <= float(r) <= 1.0

