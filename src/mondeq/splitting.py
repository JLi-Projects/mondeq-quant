# mondeq/splitting.py
"""
Operator splitting solvers for monotone deep equilibrium networks.

Implements Forward-Backward (FB), Peaceman-Rachford (PR), and
Douglas-Rachford (DR) splitting schemes.
"""
from __future__ import annotations
import torch
from torch import Tensor
from .prox import relu_prox


@torch.no_grad()
def _build_W(A: Tensor, S: Tensor, m_raw: Tensor) -> Tensor:
    m = torch.nn.functional.softplus(m_raw) + 1e-4
    n = A.size(1)
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    AAt = A.transpose(0, 1) @ A
    Bskew = S - S.transpose(0, 1)
    return (1.0 - m) * I - AAt + Bskew


@torch.no_grad()
def fb_solve(
    A: Tensor,
    S: Tensor,
    m_raw: Tensor,
    U: Tensor,
    b: Tensor,
    x: Tensor,
    *,
    alpha: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-5,
    z0: Tensor | None = None,
    W: Tensor | None = None,
    quant_mode: str = "none",
    quant_delta: float = 0.01,
    quant_gamma: float = 0.5,
) -> tuple[Tensor, dict]:
    """
    Forward-Backward fixed-point iteration for fully connected MON:

        z_{k+1} = J_{alpha G}(z_k - alpha F(z_k))
                = ReLU((1 - alpha) z_k + alpha (W z_k + U x + b))

    where F(z) = (I-W)z - (Ux+b) is the affine monotone operator and
    G = partial iota_{R+} with resolvent J_{alpha G} = proj_{R+} = ReLU
    (independent of alpha).

    Shapes:
        A: (r, n), S: (n, n), m_raw: (), U: (n, d), b: (n,), x: (B, d)
        Returns z*: (n, B)

    If W is provided, it overrides the W computed from A, S, m_raw (useful for quantised forward pass).

    Parameters for iterate quantisation:
        quant_mode : "none" (default), "fixed", or "adaptive"
        quant_delta : cell width for fixed mode, or initial cell width for adaptive
        quant_gamma : decay rate for adaptive mode (must be in (0, 1))
    """
    from .quant import fixed_iterate_quant, adaptive_iterate_quant

    device, dtype = A.device, A.dtype
    n = U.size(0)
    B = x.size(0)
    z = z0.clone() if z0 is not None else torch.zeros((n, B), device=device, dtype=dtype)

    if W is None:
        W = _build_W(A, S, m_raw)
    hist = {"residual": [], "iters": 0, "quant_errors": []}

    for i in range(max_iters):
        u = (1.0 - alpha) * z + alpha * (W @ z + (U @ x.T) + b[:, None])
        z_next = relu_prox(u, alpha)

        # Apply iterate quantisation
        if quant_mode == "fixed":
            z_next = fixed_iterate_quant(z_next, quant_delta)
            hist["quant_errors"].append(quant_delta / 2.0)
        elif quant_mode == "adaptive":
            z_diff = z_next - z
            z_diff_q = adaptive_iterate_quant(z_diff, quant_delta, quant_gamma, i)
            quant_err = (z_diff_q - z_diff).norm().item()
            z_next = z + z_diff_q
            hist["quant_errors"].append(quant_err)

        res = (z_next - z).norm() / (z.norm() + 1e-12)
        hist["residual"].append(float(res))
        z = z_next
        if res < tol:
            hist["iters"] = i + 1
            break
    else:
        hist["iters"] = max_iters

    return z, hist


# TODO: Add unit tests for PR splitting. Test cases should include:
# - Convergence to same fixed point as FB splitting
# - Verification of theoretical contraction rate bounds
# - Edge cases: m close to 0, high condition number kappa
# - Comparison with optimal alpha = 1/sqrt(m*L)

@torch.no_grad()
def pr_solve(
    A: Tensor,
    S: Tensor,
    m_raw: Tensor,
    U: Tensor,
    b: Tensor,
    x: Tensor,
    *,
    alpha: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-5,
    z0: Tensor | None = None,
    W: Tensor | None = None,
    quant_mode: str = "none",
    quant_delta: float = 0.01,
    quant_gamma: float = 0.5,
) -> tuple[Tensor, dict]:
    """
    Peaceman-Rachford fixed-point iteration for fully connected MON.

    The PR iteration alternates reflected resolvent operators:
        z_{k+1} = R_G R_F z_k
    where R_T = 2 J_T - I is the reflected resolvent.

    For the MonDEQ problem 0 ∈ F(z) + G(z) with F(z) = (I - W)z - (Ux + b) affine
    and G = ∂ι_{R_+} (indicator of nonnegative orthant), we have:

    DERIVATION OF R_F:
    -----------------
    The resolvent J_{αF}(z) solves: find y such that z ∈ y + αF(y).
    For affine F(y) = (I-W)y - c where c = Ux + b:
        z = y + α((I-W)y - c)
        z + αc = (I + α(I-W))y
        y = (I + α(I-W))^{-1}(z + αc)

    So J_F(z) = (I + α(I-W))^{-1}(z + α(Ux + b)).

    The reflected resolvent is R_F(z) = 2 J_F(z) - z.

    DERIVATION OF R_G:
    -----------------
    For G = ∂ι_{R_+}, the resolvent J_G is the projection onto R_+^n:
        J_G(z) = max(0, z)  (componentwise)

    So R_G(z) = 2 max(0, z) - z = |z| (the reflection about R_+^n boundary).

    The PR contraction rate (Lions-Mercier/Giselsson, valid for non-symmetric I-W):
        ρ_PR = sqrt(1 - 4αm/(1 + αL)^2)

    where m is the monotonicity margin and L = ||I - W||_2 is the Lipschitz constant.
    Optimal stepsize: α* = 1/L, giving rate sqrt(1 - 1/κ) where κ = L/m.

    Shapes:
        A: (r, n), S: (n, n), m_raw: (), U: (n, d), b: (n,), x: (B, d)
        Returns z*: (n, B)

    If W is provided, it overrides the W computed from A, S, m_raw.
    """
    device, dtype = A.device, A.dtype
    n = U.size(0)
    B = x.size(0)
    z = z0.clone() if z0 is not None else torch.zeros((n, B), device=device, dtype=dtype)

    if W is None:
        W = _build_W(A, S, m_raw)

    # Precompute (I + α(I - W))^{-1} for the linear resolvent
    I_n = torch.eye(n, device=device, dtype=dtype)
    resolvent_matrix = I_n + alpha * (I_n - W)
    resolvent_inv = torch.linalg.inv(resolvent_matrix)

    # Precompute affine term: α(Ux + b)
    affine = alpha * ((U @ x.T) + b[:, None])  # (n, B)

    from .quant import fixed_iterate_quant, adaptive_iterate_quant

    hist = {"residual": [], "iters": 0, "quant_errors": []}

    for i in range(max_iters):
        # Reflected resolvent of F: R_F(z) = 2 J_F(z) - z
        # J_F(z) = (I + α(I-W))^{-1}(z + affine)
        J_F_z = resolvent_inv @ (z + affine)
        R_F_z = 2 * J_F_z - z

        # Reflected resolvent of G: R_G(z) = 2 prox_G(z) - z = 2 max(0, z) - z
        prox_G_R_F = relu_prox(R_F_z, alpha)
        z_next = 2 * prox_G_R_F - R_F_z

        # Apply iterate quantisation
        if quant_mode == "fixed":
            z_next = fixed_iterate_quant(z_next, quant_delta)
            hist["quant_errors"].append(quant_delta / 2.0)
        elif quant_mode == "adaptive":
            z_diff = z_next - z
            z_diff_q = adaptive_iterate_quant(z_diff, quant_delta, quant_gamma, i)
            quant_err = (z_diff_q - z_diff).norm().item()
            z_next = z + z_diff_q
            hist["quant_errors"].append(quant_err)

        res = (z_next - z).norm() / (z.norm() + 1e-12)
        hist["residual"].append(float(res))
        z = z_next

        if res < tol:
            hist["iters"] = i + 1
            break
    else:
        hist["iters"] = max_iters

    # Recover primal solution: the fixed point of PR shadow sequence
    # The actual equilibrium is J_G(R_F(z)) = prox_G(R_F(z))
    J_F_z = resolvent_inv @ (z + affine)
    R_F_z = 2 * J_F_z - z
    z_star = relu_prox(R_F_z, alpha)

    return z_star, hist


@torch.no_grad()
def dr_solve(
    A: Tensor,
    S: Tensor,
    m_raw: Tensor,
    U: Tensor,
    b: Tensor,
    x: Tensor,
    *,
    alpha: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-5,
    z0: Tensor | None = None,
    W: Tensor | None = None,
    quant_mode: str = "none",
    quant_delta: float = 0.01,
    quant_gamma: float = 0.5,
) -> tuple[Tensor, dict]:
    """
    Douglas-Rachford splitting for fully connected MON.

    DR is the Krasnosel'skii-Mann (KM) relaxation of Peaceman-Rachford
    with relaxation parameter 1/2:

        z_{k+1} = (1/2) z_k + (1/2) R_G(R_F(z_k))

    where R_T = 2 J_T - I is the reflected resolvent. Like PR, DR
    converges for any α > 0 when F is strongly monotone and G is maximal
    monotone (the same conditions). The KM averaging ensures firm
    nonexpansiveness, giving weak convergence even for merely maximal
    monotone operators (where unrelaxed PR may diverge). In the MonDEQ
    setting both PR and DR converge; DR is strictly slower due to the
    1/2 averaging (rate ≈ (1 + rho_PR)/2).

    The equilibrium is recovered as z* = J_G(R_F(z_fix)) where z_fix
    is the fixed point of the DR iteration.

    Shapes:
        A: (r, n), S: (n, n), m_raw: (), U: (n, d), b: (n,), x: (B, d)
        Returns z*: (n, B)

    If W is provided, it overrides the W computed from A, S, m_raw.
    """
    device, dtype = A.device, A.dtype
    n = U.size(0)
    B = x.size(0)
    z = z0.clone() if z0 is not None else torch.zeros((n, B), device=device, dtype=dtype)

    if W is None:
        W = _build_W(A, S, m_raw)

    # Precompute (I + α(I - W))^{-1} for the linear resolvent
    I_n = torch.eye(n, device=device, dtype=dtype)
    resolvent_matrix = I_n + alpha * (I_n - W)
    resolvent_inv = torch.linalg.inv(resolvent_matrix)

    # Precompute affine term: α(Ux + b)
    affine = alpha * ((U @ x.T) + b[:, None])  # (n, B)

    from .quant import fixed_iterate_quant, adaptive_iterate_quant

    hist = {"residual": [], "iters": 0, "quant_errors": []}

    for i in range(max_iters):
        # Reflected resolvent of F: R_F(z) = 2 J_F(z) - z
        J_F_z = resolvent_inv @ (z + affine)
        R_F_z = 2 * J_F_z - z

        # Reflected resolvent of G: R_G(z) = 2 J_G(z) - z = 2 max(0, z) - z
        prox_G_R_F = relu_prox(R_F_z, alpha)
        R_G_R_F_z = 2 * prox_G_R_F - R_F_z

        # KM averaging: z_{k+1} = (1/2) z_k + (1/2) R_G(R_F(z_k))
        z_next = 0.5 * z + 0.5 * R_G_R_F_z

        # Apply iterate quantisation
        if quant_mode == "fixed":
            z_next = fixed_iterate_quant(z_next, quant_delta)
            hist["quant_errors"].append(quant_delta / 2.0)
        elif quant_mode == "adaptive":
            z_diff = z_next - z
            z_diff_q = adaptive_iterate_quant(z_diff, quant_delta, quant_gamma, i)
            quant_err = (z_diff_q - z_diff).norm().item()
            z_next = z + z_diff_q
            hist["quant_errors"].append(quant_err)

        res = (z_next - z).norm() / (z.norm() + 1e-12)
        hist["residual"].append(float(res))
        z = z_next

        if res < tol:
            hist["iters"] = i + 1
            break
    else:
        hist["iters"] = max_iters

    # Recover primal solution: z* = J_G(R_F(z_fix)) = ReLU(R_F(z_fix))
    J_F_z = resolvent_inv @ (z + affine)
    R_F_z = 2 * J_F_z - z
    z_star = relu_prox(R_F_z, alpha)

    return z_star, hist


def compute_pr_rate(m: float, L: float, alpha: float) -> float:
    """
    Compute the Peaceman-Rachford contraction rate (Lions-Mercier/Giselsson).

    ρ_PR = sqrt(1 - 4αm/(1 + αL)^2)

    This formula is valid for non-symmetric (I-W) (i.e., when the operator
    is strongly monotone but NOT necessarily cocoercive). The older
    max-of-fractions formula requires cocoercivity which MonDEQ's I-W
    does not satisfy due to the skew part B - B^T.

    For optimal α = 1/L, the rate is sqrt(1 - 1/κ) where κ = L/m.

    Parameters
    ----------
    m : float
        Monotonicity margin (must be > 0).
    L : float
        Lipschitz constant (must be > 0).
    alpha : float
        Step size.

    Returns
    -------
    float
        Contraction rate ρ_PR ∈ [0, 1).
    """
    if m <= 0 or L <= 0 or alpha <= 0:
        return float('inf')

    import math
    val = 1.0 - 4.0 * alpha * m / (1.0 + alpha * L) ** 2
    if val < 0:
        return 0.0  # Perfect contraction (should not happen in practice)
    return math.sqrt(val)


def optimal_pr_alpha(m: float, L: float) -> float:
    """
    Compute optimal step size for Peaceman-Rachford splitting.

    For the Lions-Mercier/Giselsson rate sqrt(1 - 4αm/(1+αL)^2),
    the optimal α = 1/L, giving rate sqrt(1 - 1/κ) where κ = L/m.

    Parameters
    ----------
    m : float
        Monotonicity margin.
    L : float
        Lipschitz constant.

    Returns
    -------
    float
        Optimal step size.
    """
    return 1.0 / L


def compute_dr_rate(m: float, L: float, alpha: float) -> float:
    """
    Compute contraction rate for Douglas-Rachford splitting.

    DR is KM-relaxed PR with parameter 1/2, so its contraction rate is
    the average of the identity and the PR operator:
        rho_DR = (1 + rho_PR) / 2
    This is strictly slower than PR but converges for any alpha > 0.

    Parameters
    ----------
    m : float
        Monotonicity margin.
    L : float
        Lipschitz constant.
    alpha : float
        Step size.

    Returns
    -------
    float
        DR contraction rate.
    """
    rho_pr = compute_pr_rate(m, L, alpha)
    return (1.0 + rho_pr) / 2.0
