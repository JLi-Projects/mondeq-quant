"""
Shared utilities for numerical experiments.
Provides consistent plotting settings, data loaders, and helper functions.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple, Optional
import torchvision.datasets as dset
import torchvision.transforms as transforms


# =============================================================================
# Publication-Quality Figure Settings (IEEE single column)
# =============================================================================
FIGURE_WIDTH = 3.5  # inches
FIGURE_HEIGHT = 2.5  # inches
FONT_SIZE = 10
DPI = 300

def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 2,
        'figure.figsize': (FIGURE_WIDTH, FIGURE_HEIGHT),
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'text.usetex': False,  # Set True if LaTeX available
    })

setup_matplotlib()


# =============================================================================
# Experiment Configuration
# =============================================================================
BIT_DEPTHS = [4, 6, 8, 12, 16, 32]  # Exclude FP32 (will be baseline)
SEEDS = [0, 1, 2, 3, 4]
HIDDEN_DIM = 100  # Matching locuslab pattern
INPUT_DIM = 784  # MNIST flattened
OUTPUT_DIM = 10  # MNIST classes

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Data Loading
# =============================================================================
def get_mnist_loaders(train_batch_size: int = 128, test_batch_size: int = 256):
    """Get MNIST data loaders with standard normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = dset.MNIST(
        root='data', train=True, download=True, transform=transform
    )
    test_dataset = dset.MNIST(
        root='data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    return train_loader, test_loader


# =============================================================================
# Model Definition
# =============================================================================
class MonDEQClassifier(nn.Module):
    """
    MonDEQ-based MNIST classifier.
    Architecture: Flatten -> MonDEQ Layer -> Linear Output
    """
    def __init__(
        self,
        in_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        out_dim: int = OUTPUT_DIM,
        alpha: float = 1.0,
        max_iters: int = 500,
        tol: float = 1e-6
    ):
        super().__init__()
        from mondeq import MonDEQLayerFC
        self.mon = MonDEQLayerFC(
            n=hidden_dim, d=in_dim,
            alpha=alpha, max_iters=max_iters, tol=tol
        )
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, 28, 28) -> flatten to (B, 784)
        x = x.view(x.size(0), -1)
        # MonDEQ layer returns (n, B), need to transpose
        z = self.mon(x)  # (n, B)
        z = z.T  # (B, n)
        return self.output(z)  # (B, 10)


# =============================================================================
# Quantisation Helpers
# =============================================================================


# =============================================================================
# Evaluation Helpers
# =============================================================================
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, test_loader, device: torch.device = DEVICE) -> float:
    """Evaluate model accuracy on test set."""
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


@torch.no_grad()
def compute_equilibrium_with_residuals(
    model: nn.Module,
    x: Tensor,
    max_iters: int = 500,
    tol: float = 1e-6
) -> Tuple[Tensor, list]:
    """
    Run forward-backward iteration and return equilibrium point with residual history.
    """
    from mondeq.splitting import _build_W
    from mondeq.prox import relu_prox

    model.eval()
    core = model.mon.core
    alpha = model.mon.alpha

    # Extract parameters
    A, S, m_raw = core.A.data, core.S.data, core.m_raw.data
    U, b = core.U.data, core.b.data

    device, dtype = A.device, A.dtype
    n = U.size(0)
    B = x.size(0)

    # Initialize z
    z = torch.zeros((n, B), device=device, dtype=dtype)

    # Build W matrix
    W = _build_W(A, S, m_raw)

    residuals = []
    for _ in range(max_iters):
        u = (1.0 - alpha) * z + alpha * (W @ z + (U @ x.T) + b[:, None])
        z_next = relu_prox(u, alpha)
        res = (z_next - z).norm() / (z.norm() + 1e-12)
        residuals.append(float(res))
        z = z_next
        if res < tol:
            break

    return z, residuals


# =============================================================================
# Spectral Analysis
# =============================================================================
def compute_spectral_properties(model: nn.Module) -> dict:
    """
    Compute spectral properties of the model.
    Returns dict with m_tilde (strong monotonicity), L_tilde (Lipschitz),
    and theoretical convergence rate.
    """
    from mondeq import spectral_bounds

    core = model.mon.core
    A_matrix = core.A_matrix()  # I - W

    m_tilde, L_tilde = spectral_bounds(A_matrix)

    alpha = model.mon.alpha
    # Theoretical convergence rate: r = sqrt(1 - 2*alpha*m_tilde + alpha^2*L_tilde^2)
    r_squared = 1 - 2 * alpha * m_tilde.detach() + alpha**2 * L_tilde.detach()**2
    r = torch.sqrt(torch.clamp(r_squared, min=0.0))

    return {
        'm_tilde': float(m_tilde.detach()),
        'L_tilde': float(L_tilde.detach()),
        'm': float(core.m.detach()),
        'alpha': alpha,
        'r_theory': float(r.detach()),
        'convergent': float(r.detach()) < 1.0
    }


# =============================================================================
# Training Helpers
# =============================================================================
def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    verbose: bool = True
) -> nn.Module:
    """Train the MonDEQ classifier."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        acc = evaluate_accuracy(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc

        if verbose:
            print(f"Epoch {epoch:2d}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Acc={acc:.2f}%, Best={best_acc:.2f}%")

    return model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Results Storage
# =============================================================================
def save_results(data: dict, filename: str, results_dir: str = 'results'):
    """Save results to a numpy file."""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    np.save(filepath, data)
    print(f"Saved results to {filepath}")


def load_results(filename: str, results_dir: str = 'results') -> dict:
    """Load results from a numpy file."""
    filepath = os.path.join(results_dir, filename)
    return np.load(filepath, allow_pickle=True).item()


# =============================================================================
# Solver Colour/Linestyle Constants (for consistent figures)
# =============================================================================
SOLVER_STYLES = {
    "FB":  {"color": "#1f77b4", "linestyle": "-",  "label": "FB"},
    "PR":  {"color": "#ff7f0e", "linestyle": "--", "label": "PR"},
    "DR":  {"color": "#2ca02c", "linestyle": ":",  "label": "DR"},
}

BIT_COLORS = {
    "FP32": "#000000",
    32:      "#333333",
    16:      "#555555",
    12:      "#777777",
    10:      "#888888",
    8:       "#1f77b4",
    7:       "#2ca02c",
    6:       "#ff7f0e",
    5:       "#d62728",
    4:       "#9467bd",
    3:       "#8c564b",
}


# =============================================================================
# Model Loading
# =============================================================================
def load_pretrained_model(
    checkpoint_path: str = "checkpoints/mnist_mondeq_float.pt",
    hidden_dim: int = HIDDEN_DIM,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Load a pretrained MonDEQ MNIST model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint file.
    hidden_dim : int
        Hidden dimension (must match checkpoint).
    device : torch.device, optional
        Device to load to. Defaults to DEVICE.

    Returns
    -------
    nn.Module
        The loaded model in eval mode.
    """
    if device is None:
        device = DEVICE

    model = MonDEQClassifier(hidden_dim=hidden_dim)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle key mismatch: MNISTMonDEQ uses "classifier", MonDEQClassifier uses "output"
    remapped = {}
    for k, v in state_dict.items():
        new_key = k.replace("classifier.", "output.")
        remapped[new_key] = v
    model.load_state_dict(remapped)

    model = model.to(device)
    model.eval()
    return model


def get_model_params(model: nn.Module) -> dict:
    """
    Extract W, U, b, m, L, alpha from a MonDEQ model.

    Returns
    -------
    dict with keys: W, U, b, m, L, alpha, kappa, A_param, S_param, m_raw
    """
    from mondeq import spectral_bounds
    from mondeq.splitting import _build_W

    core = model.mon.core
    A_param = core.A.data
    S_param = core.S.data
    m_raw = core.m_raw.data
    U = core.U.data
    b = core.b.data
    W = _build_W(A_param, S_param, m_raw)
    m = float(core.m.detach())
    I_n = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    m_computed, L = spectral_bounds(I_n - W)
    m_computed = float(m_computed)
    L = float(L)
    alpha = model.mon.alpha

    return {
        "W": W, "U": U, "b": b,
        "m": m, "m_computed": m_computed, "L": L,
        "alpha": alpha, "kappa": L / m,
        "A_param": A_param, "S_param": S_param, "m_raw": m_raw,
    }
