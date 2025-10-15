from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import torch
import matplotlib.pyplot as plt

from mondeq.operators.linear import LinearMonotone
from mondeq.quant.fake import QuantWrapperLinearA
from mondeq.logging.spectra import spectral_bounds

def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.t())

@torch.no_grad()
def shifted_norm(x: torch.Tensor, G: torch.Tensor, alpha: float) -> float:
    n = x.numel()
    I = torch.eye(n, dtype=x.dtype, device=x.device)
    M = I + alpha * G
    # keep everything as tensors; clamp the tensor, then take .item()
    val = (x @ (M @ x))
    val = torch.clamp(val, min=0.0)
    return float(torch.sqrt(val).item())

def phi_b(p: torch.Tensor, A: torch.Tensor, G: torch.Tensor, r: torch.Tensor, alpha: float) -> torch.Tensor:
    # Φ_b(p) = (I + α G)^{-1} ((I - α A) p + α r)
    n = p.numel()
    I = torch.eye(n, dtype=p.dtype, device=p.device)
    M = I + alpha * G
    rhs = (I - alpha * A) @ p + alpha * r
    return torch.linalg.solve(M, rhs)

def predicted_rate(alpha: float, Ab: torch.Tensor) -> float:
    m_t, L_t = spectral_bounds(Ab)
    val = 1.0 - 2.0 * alpha * m_t + (alpha * L_t) ** 2
    return float((val if val > 0 else 0.0) ** 0.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--outdir", type=str, default="figures/backward_shifted_norm")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device, dtype = torch.device("cpu"), torch.float64
    op = LinearMonotone(dim=args.dim, m=args.m, dtype=dtype, device=device)
    A_fp = op.A().detach()

    # PSD G_b
    G = torch.diag(torch.abs(torch.randn(args.dim, device=device, dtype=dtype))) * 0.1
    r = torch.randn(args.dim, device=device, dtype=dtype)

    bits_list = [None, 8, 6, 4]
    labels, preds, meas = [], [], []

    for bits in bits_list:
        if bits is None:
            A_use = A_fp
        else:
            qw = QuantWrapperLinearA(op.A, num_bits=bits, dtype=dtype)
            A_use = qw.A_q().detach()

        Ab = A_use + G
        preds.append(predicted_rate(args.alpha, Ab))

        p1 = torch.randn(args.dim, device=device, dtype=dtype)
        p2 = torch.randn(args.dim, device=device, dtype=dtype)
        tail: List[float] = []
        for _ in range(args.steps):
            q1 = phi_b(p1, A_use, G, r, args.alpha)
            q2 = phi_b(p2, A_use, G, r, args.alpha)
            num = shifted_norm(q1 - q2, G, args.alpha)
            den = shifted_norm(p1 - p2, G, args.alpha) + 1e-16
            tail.append(num / den)
            p1, p2 = q1, q2
        last = tail[-50:] if len(tail) >= 50 else tail
        meas.append(sum(last) / len(last) if last else float("nan"))
        labels.append(32 if bits is None else bits)

    plt.figure()
    plt.plot(labels, preds, marker="o", label="predicted")
    plt.plot(labels, meas, marker="x", label="measured")
    plt.gca().invert_xaxis()
    plt.xlabel("bitwidth")
    plt.ylabel("shifted-norm contraction")
    plt.title("Backward map contraction (shifted norm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    f = Path(args.outdir) / "backward_shifted_norm.png"
    plt.tight_layout(); plt.savefig(f)
    print(f"[saved] {f}")

if __name__ == "__main__":
    main()
