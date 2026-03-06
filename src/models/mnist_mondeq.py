# models/mnist_mondeq.py
"""
MNIST classifier using MonDEQ equilibrium layer.

Architecture:
    Input: Flatten 28x28 -> 784
    MonDEQ equilibrium layer: 784 -> hidden_dim (default 100)
    Output: Linear hidden_dim -> 10
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from mondeq import MonDEQLayerFC


class MNISTMonDEQ(nn.Module):
    """
    MonDEQ MNIST classifier with a single equilibrium layer.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the MonDEQ layer.
    alpha : float
        Forward-Backward step size.
    max_iters : int
        Maximum FB iterations.
    tol : float
        Convergence tolerance for FB solver.
    """

    def __init__(
        self,
        hidden_dim: int = 100,
        alpha: float = 1.0,
        max_iters: int = 500,
        tol: float = 1e-5,
    ):
        super().__init__()
        self.input_dim = 784  # 28 x 28 flattened
        self.hidden_dim = hidden_dim
        self.num_classes = 10

        # MonDEQ equilibrium layer
        self.mon = MonDEQLayerFC(
            n=hidden_dim,
            d=self.input_dim,
            alpha=alpha,
            max_iters=max_iters,
            tol=tol,
        )

        # Output classification head
        self.classifier = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, 1, 28, 28) or (B, 784).

        Returns
        -------
        Tensor
            Logits of shape (B, 10).
        """
        # Flatten input
        x = x.view(x.size(0), -1)  # (B, 784)

        # MonDEQ equilibrium: z* = (n, B)
        z_star = self.mon(x)  # (hidden_dim, B)

        # Transpose for classifier: (B, hidden_dim)
        z_star = z_star.T

        # Classification
        logits = self.classifier(z_star)  # (B, 10)
        return logits

    @property
    def alpha(self) -> float:
        """Current FB step size."""
        return self.mon.alpha

    @alpha.setter
    def alpha(self, value: float):
        """Set FB step size."""
        self.mon.alpha = float(value)

    def get_W(self) -> Tensor:
        """Get the current W matrix from the MonDEQ layer."""
        return self.mon.core.W()

    def get_m(self) -> Tensor:
        """Get the current monotonicity margin m."""
        return self.mon.core.m

    def get_params(self) -> tuple[Tensor, Tensor, Tensor]:
        """Get (W, U, b) parameters."""
        W = self.mon.core.W()
        U = self.mon.core.U.data
        b = self.mon.core.b.data
        return W, U, b

    @property
    def max_iters(self) -> int:
        """Get max iterations for FB solver."""
        return self.mon.max_iters

    @max_iters.setter
    def max_iters(self, value: int):
        """Set max iterations for FB solver."""
        self.mon.max_iters = int(value)
