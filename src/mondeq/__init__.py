# mondeq/__init__.py
from .operators import WKLinearFC, spectral_bounds
from .prox import relu_prox, relu_jacobian_mask
from .splitting import fb_solve, pr_solve, dr_solve, _build_W, compute_pr_rate, compute_dr_rate, optimal_pr_alpha
from .layers.fc import MonDEQLayerFC
from .quant import fake_quant_sym, fixed_iterate_quant, adaptive_iterate_quant

__all__ = [
    "WKLinearFC",
    "spectral_bounds",
    "relu_prox",
    "relu_jacobian_mask",
    "fb_solve",
    "pr_solve",
    "dr_solve",
    "_build_W",
    "compute_pr_rate",
    "compute_dr_rate",
    "optimal_pr_alpha",
    "MonDEQLayerFC",
    "fake_quant_sym",
    "fixed_iterate_quant",
    "adaptive_iterate_quant",
]
