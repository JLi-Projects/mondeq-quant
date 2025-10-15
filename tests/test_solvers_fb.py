import torch
from mondeq.operators.linear import LinearMonotone
from mondeq.solvers.fb import FBSolver

def test_fb_converges_linear_window():
    torch.manual_seed(0)
    op = LinearMonotone(dim=16, m=0.3, dtype=torch.float64, device=torch.device("cpu"))
    m_t, L_t = op.spectral_diagnostics()
    a_hi = float((2.0 * m_t) / (L_t * L_t))
    alpha = 0.5 * a_hi
    solver = FBSolver(max_iters=2000, tol=1e-10, alpha=alpha, damping=1.0, dtype=torch.float64, device=torch.device("cpu"))
    z_star = solver.solve_linear(op.A())
    assert torch.isfinite(z_star).all()
    # residual should be tiny
    res = torch.linalg.norm(op.A() @ z_star).item()
    assert res < 1e-6

