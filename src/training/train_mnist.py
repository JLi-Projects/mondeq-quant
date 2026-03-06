# training/train_mnist.py
"""
Training loop for MonDEQ MNIST classifier.

Adapted from monotone_op_net/train.py with modifications for MonDEQ architecture.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Optional, Callable


def cuda(tensor):
    """Move tensor to GPU if available."""
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def mnist_loaders(
    train_batch_size: int,
    test_batch_size: Optional[int] = None,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """
    Create MNIST data loaders with standard normalisation.

    Parameters
    ----------
    train_batch_size : int
        Batch size for training.
    test_batch_size : int, optional
        Batch size for testing. Defaults to train_batch_size.
    data_dir : str
        Directory to store/load MNIST data.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, test_loader)
    """
    if test_batch_size is None:
        test_batch_size = train_batch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        dset.MNIST(data_dir, train=True, download=True, transform=transform),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dset.MNIST(data_dir, train=False, transform=transform),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def tune_alpha(model, x: torch.Tensor, max_alpha: float = 1.0, verbose: bool = True):
    """
    Tune alpha to find the value with fewest iterations.

    Following locuslab's approach (monotone_op_net/train.py:105-131):
    1. Start with max_alpha, record iteration count
    2. Halve alpha repeatedly while iterations decrease
    3. Stop when iterations start increasing (found optimal)

    Parameters
    ----------
    model : MNISTMonDEQ
        The model to tune. Must have model.mon.stats.fwd_iters.val
    x : torch.Tensor
        Sample input batch for testing convergence.
    max_alpha : float
        Maximum alpha to try.
    verbose : bool
        Whether to print tuning progress.
    """
    if verbose:
        print("---- Tuning alpha ----")
        print(f"Current alpha: {model.alpha:.6f}")

    orig_alpha = model.alpha

    # Reset stats and test max_alpha
    model.mon.stats.reset()
    model.alpha = max_alpha
    with torch.no_grad():
        _ = model(x)

    iters = model.mon.stats.fwd_iters.val
    model.mon.stats.reset()
    iters_n = iters
    max_iter_limit = model.max_iters

    if verbose:
        print(f"  alpha={model.alpha:.6f}\titers={iters_n}")

    # Halve alpha while iterations decrease
    while model.alpha > 1e-4 and iters_n <= iters:
        model.alpha = model.alpha / 2
        with torch.no_grad():
            _ = model(x)
        iters = iters_n
        iters_n = model.mon.stats.fwd_iters.val
        if verbose:
            print(f"  alpha={model.alpha:.6f}\titers={iters_n}")
        model.mon.stats.reset()

    # Check if anything converged
    if iters == max_iter_limit:
        if verbose:
            print("None converged, resetting to original alpha")
        model.alpha = orig_alpha
    else:
        # Go back to the previous alpha (before iterations started increasing)
        model.alpha = model.alpha * 2
        if verbose:
            print(f"Setting alpha to: {model.alpha:.6f}")

    if verbose:
        print("--------------------\n")


# Default random seed for reproducibility
DEFAULT_SEED = 42


def train(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: nn.Module,
    epochs: int = 15,
    max_lr: float = 1e-3,
    print_freq: int = 100,
    lr_mode: str = "step",
    step: int = 10,
    tune_alpha_flag: bool = False,
    max_alpha: float = 1.0,
    model_path: Optional[str] = None,
    seed: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> dict:
    """
    Train a MonDEQ model on MNIST.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    test_loader : DataLoader
        Test data loader.
    model : nn.Module
        MonDEQ model to train.
    epochs : int
        Number of training epochs.
    max_lr : float
        Maximum learning rate.
    print_freq : int
        How often to print training progress (in batches).
    lr_mode : str
        Learning rate schedule: "step", "1cycle", or "constant".
    step : int
        Step size for StepLR scheduler.
    tune_alpha_flag : bool
        Whether to tune alpha during training.
    max_alpha : float
        Maximum alpha for tuning.
    model_path : str, optional
        Path to save checkpoints.
    seed : int, optional
        Random seed for reproducibility. Defaults to DEFAULT_SEED (42).
    callback : Callable, optional
        Function called after each epoch with (epoch, train_loss, test_acc).

    Returns
    -------
    dict
        Training history with keys: train_losses, test_accs, final_test_acc
    """
    # Use default seed if not provided (ensures reproducibility)
    if seed is None:
        seed = DEFAULT_SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Move model to GPU if available
    model = cuda(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=max_lr)

    # Learning rate scheduler
    if lr_mode == "1cycle":
        lr_schedule = lambda t: np.interp(
            [t],
            [0, (epochs - 5) // 2, epochs - 5, epochs],
            [1e-3, max_lr, 1e-3, 1e-3]
        )[0]
    elif lr_mode == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step, gamma=0.1)
    elif lr_mode != "constant":
        raise ValueError(f"lr_mode must be one of 'constant', 'step', '1cycle', got {lr_mode}")

    # History tracking
    history = {
        "train_losses": [],
        "test_accs": [],
        "epochs": [],
    }

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        n_processed = 0
        n_train = len(train_loader.dataset)
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            # Alpha tuning at specific points
            if tune_alpha_flag and (batch_idx == 30 or batch_idx == len(train_loader) // 2):
                tune_alpha(model, cuda(data), max_alpha)

            # LR scheduling for 1cycle
            if lr_mode == "1cycle":
                lr = lr_schedule(epoch - 1 + batch_idx / len(train_loader))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            data, target = cuda(data), cuda(target)
            optimizer.zero_grad()

            # Forward pass
            logits = model(data)

            # Loss
            loss = nn.CrossEntropyLoss()(logits, target)
            loss.backward()

            epoch_losses.append(loss.item())
            n_processed += len(data)

            # Print progress
            if batch_idx % print_freq == 0 and batch_idx > 0:
                preds = logits.argmax(dim=1)
                incorrect = preds.ne(target).sum().item()
                err = 100.0 * incorrect / len(data)
                partial_epoch = epoch + batch_idx / len(train_loader) - 1
                print(
                    f"Epoch: {partial_epoch:.2f} [{n_processed}/{n_train} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]  "
                    f"Loss: {loss.item():.4f}  Error: {err:.2f}%"
                )
                # Report convergence stats (like locuslab)
                if hasattr(model, 'mon') and hasattr(model.mon, 'stats'):
                    model.mon.stats.report()
                    model.mon.stats.reset()

            optimizer.step()

        # Step LR scheduler
        if lr_mode == "step":
            lr_scheduler.step()

        # Save checkpoint
        if model_path is not None:
            torch.save(model.state_dict(), model_path)

        train_time = time.time() - start_time
        avg_loss = np.mean(epoch_losses)
        history["train_losses"].append(avg_loss)

        print(f"\nEpoch {epoch} training time: {train_time:.1f}s, avg loss: {avg_loss:.4f}")

        # Evaluation
        test_acc = evaluate(model, test_loader)
        history["test_accs"].append(test_acc)
        history["epochs"].append(epoch)

        print(f"Test accuracy: {test_acc:.2f}%\n")

        # Callback
        if callback is not None:
            callback(epoch, avg_loss, test_acc)

    history["final_test_acc"] = history["test_accs"][-1]
    return history


def evaluate(model: nn.Module, test_loader: DataLoader) -> float:
    """
    Evaluate model accuracy on test set.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    test_loader : DataLoader
        Test data loader.

    Returns
    -------
    float
        Test accuracy as percentage.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = cuda(data), cuda(target)
            logits = model(data)
            preds = logits.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += len(target)

    return 100.0 * correct / total
