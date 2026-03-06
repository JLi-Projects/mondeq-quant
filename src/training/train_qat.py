# training/train_qat.py
"""
Quantisation-Aware Training (QAT) for MonDEQ.

Trains a MonDEQ from scratch with fake quantisation in the forward pass,
using Straight-Through Estimator (STE) for gradient flow through the
quantiser. The key insight: MonDEQ's backward pass solves a monotone
inclusion with the same (m_tilde, L_tilde) as the forward pass, so
the STE approximation only enters through the r_tilde term (not through
the solver itself).
"""
from __future__ import annotations
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np

from mondeq.quant import fake_quant_sym
from mondeq.splitting import fb_solve, _build_W
from mondeq.prox import relu_jacobian_mask


class _QATForwardFunction(torch.autograd.Function):
    """
    Custom autograd for QAT forward pass with STE.

    Forward: build W from (A, S, m_raw), apply STE fake quantisation,
             run fb_solve with quantised W.
    Backward: implicit differentiation through the quantised equilibrium.
    """

    @staticmethod
    def forward(ctx, A, S, m_raw, U, b, x, bits, alpha, max_iters, tol):
        # Build W from learnable parameters
        with torch.no_grad():
            W = _build_W(A, S, m_raw)

            # STE quantisation: W_q = W + (fake_quant(W) - W).detach()
            # In no_grad mode, we just use fake_quant directly for the solve
            W_q = fake_quant_sym(W, bits)
            U_q = fake_quant_sym(U, bits)
            b_q = fake_quant_sym(b, bits)

            # Run forward-backward solver with quantised parameters
            z_star, hist = fb_solve(
                A, S, m_raw, U_q, b_q, x,
                alpha=alpha, max_iters=max_iters, tol=tol,
                W=W_q,
            )

            # Compute mask for backward pass
            u_star = (1.0 - alpha) * z_star + alpha * (W_q @ z_star + (U_q @ x.T) + b_q[:, None])
            mask = relu_jacobian_mask(u_star)

        ctx.save_for_backward(A, S, m_raw, U, b, x, z_star, mask, W_q)
        ctx.alpha = float(alpha)
        ctx.max_iters = int(max_iters)
        ctx.tol = float(tol)
        ctx.bits = int(bits)
        ctx.hist = hist
        return z_star

    @staticmethod
    def backward(ctx, grad_z):
        """
        Implicit differentiation with STE.

        The backward solve uses the quantised W for the adjoint iteration
        (same m_tilde, L_tilde as forward). Gradients flow through the
        STE to the original (A, S, m_raw) parameters.
        """
        A, S_param, m_raw, U, b, x, z_star, mask, W_q = ctx.saved_tensors
        alpha = ctx.alpha
        max_iters = ctx.max_iters
        tol = ctx.tol
        device, dtype = A.device, A.dtype
        n, B = z_star.shape

        # Iterative backward solve using quantised W
        j = mask.float()
        I_mask = (j == 0)
        d = torch.zeros_like(j)
        d[~I_mask] = (1 - j[~I_mask]) / j[~I_mask]

        v = j * grad_z
        u = torch.zeros((n, B), device=device, dtype=dtype)

        for it in range(max_iters):
            un = (1 - alpha) * u + alpha * (W_q.T @ u)
            un = (un + alpha * (1 + d) * v) / (1 + alpha * d)
            un[I_mask] = v[I_mask]

            err = (un - u).norm() / (u.norm() + 1e-12)
            u = un
            if err < tol:
                break

        xi = j * u

        # Gradients for W via STE
        # G_W = alpha * (xi @ z_star.T) is the gradient w.r.t. W
        G_W = alpha * (xi @ z_star.T)

        # STE: gradients pass through quantiser as if it were identity
        # dL/dA from W = (1-m)I - A^T A + (S - S^T)
        grad_A = -A @ (G_W + G_W.T)
        grad_S = G_W - G_W.T
        grad_m = -torch.einsum("ii->", G_W).unsqueeze(0)
        grad_m_raw = torch.sigmoid(m_raw) * grad_m
        grad_U = alpha * (xi @ x)
        grad_b = alpha * xi.sum(dim=1)
        grad_x = (alpha * xi.T) @ U

        return grad_A, grad_S, grad_m_raw, grad_U, grad_b, grad_x, None, None, None, None


class QATMonDEQClassifier(nn.Module):
    """
    MonDEQ classifier with quantisation-aware training support.

    Architecture is identical to MonDEQClassifier but the forward pass
    applies fake quantisation with STE to all MonDEQ parameters.
    """

    def __init__(
        self,
        in_dim: int = 784,
        hidden_dim: int = 100,
        out_dim: int = 10,
        bits: int = 8,
        alpha: float = 1.0,
        max_iters: int = 500,
        tol: float = 1e-5,
    ):
        super().__init__()
        from mondeq import MonDEQLayerFC
        self.mon = MonDEQLayerFC(
            n=hidden_dim, d=in_dim,
            alpha=alpha, max_iters=max_iters, tol=tol,
        )
        self.output = nn.Linear(hidden_dim, out_dim)
        self.bits = bits

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)

        z_star = _QATForwardFunction.apply(
            self.mon.core.A, self.mon.core.S, self.mon.core.m_raw,
            self.mon.core.U, self.mon.core.b, x,
            self.bits, self.mon.alpha, self.mon.max_iters, self.mon.tol,
        )

        z_star = z_star.T
        logits = self.output(z_star)
        return logits


def train_qat(
    train_loader: DataLoader,
    test_loader: DataLoader,
    bits: int = 8,
    hidden_dim: int = 100,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Train a MonDEQ with QAT from scratch.

    Parameters
    ----------
    train_loader, test_loader : DataLoader
    bits : int
        Bit depth for fake quantisation.
    hidden_dim : int
        Hidden dimension.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    seed : int
        Random seed.
    device : torch.device, optional
    verbose : bool

    Returns
    -------
    tuple[nn.Module, dict]
        Trained model and training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Use a conservative alpha and loose tolerance for training speed.
    # W&K use tol=1e-2 in their code; high-precision solve is only needed
    # at evaluation time, not during training.
    model = QATMonDEQClassifier(
        bits=bits, hidden_dim=hidden_dim,
        alpha=0.05, max_iters=100, tol=1e-2,
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    history = {"train_losses": [], "test_accs": [], "margins": []}

    def _tune_alpha_qat(model, x_sample):
        """Tune alpha for QAT model based on current margin/Lipschitz."""
        with torch.no_grad():
            W = _build_W(model.mon.core.A, model.mon.core.S, model.mon.core.m_raw)
            W_q = fake_quant_sym(W, model.bits)
            I_n = torch.eye(W.size(0), device=W.device)
            from mondeq import spectral_bounds
            m_q, L_q = spectral_bounds(I_n - W_q)
            m_q, L_q = float(m_q), float(L_q)
            if m_q > 0 and L_q > 0:
                # Use optimal FB alpha with safety margin
                model.mon.alpha = 0.8 * m_q / (L_q ** 2)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Tune alpha periodically (every 100 batches)
            if batch_idx % 100 == 0:
                _tune_alpha_qat(model, data)

            optimizer.zero_grad()

            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Record margin
        m_val = float(model.mon.core.m.detach())
        history["margins"].append(m_val)

        avg_loss = float(np.mean(epoch_losses))
        history["train_losses"].append(avg_loss)

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                correct += logits.argmax(1).eq(target).sum().item()
                total += len(target)
        acc = 100.0 * correct / total
        history["test_accs"].append(acc)

        if verbose:
            dt = time.time() - t0
            print(f"[QAT {bits}b] Epoch {epoch:2d}: loss={avg_loss:.4f}, "
                  f"acc={acc:.2f}%, m={m_val:.4f}, time={dt:.1f}s")

    history["final_acc"] = history["test_accs"][-1]
    history["final_margin"] = history["margins"][-1]
    return model, history
