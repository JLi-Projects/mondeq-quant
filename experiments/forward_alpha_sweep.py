from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import matplotlib.pyplot as plt

from mondeq.operators.linear import LinearMonotone
from mondeq.solvers.fb import FBSolver
from mondeq.quant.fake import QuantWrapperLinearA
from mondeq.logging.spectra import spectral_bounds

def exact_rate(alpha: float, A: torch.Tensor) -> float:
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    # exact ||I - α A||_2 via top singular value
    return float(torch.linalg.svdvals(I - alpha * A).max().item())

def measure(alpha: float, A: torch.Tensor, r: torch.Tensor, steps: int) -> float:
    """Return tail-avg error contraction for FB at stepsize alpha."""
    device, dtype = A.device, A.dtype
    z0 = torch.randn(A.shape[-1], device=device, dtype=dtype)
    z_star = torch.linalg.solve(A, r)

    tail: List[float] = []
    def cb(k, zk1, res, info):
        ez  = torch.linalg.norm(z_solver[0] - z_star)
        ez1 = torch.linalg.norm(zk1 - z_star)
        if ez.item() > 0:
            tail.append(float((ez1 / ez).item()))
        z_solver[0] = zk1
    z_solver = [z0.clone()]

    solver = FBSolver(max_iters=steps, min_iters=50, tol=1e-12, alpha=alpha, callback=cb, device=device, dtype=dtype)
    solver.solve_linear(A, r=r, z0=z0, z_star=z_star)
    last = tail[-50:] if len(tail) >= 50 else tail
    return float(sum(last)/len(last)) if last else float("nan")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--m", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--alphas", type=int, default=30)
    p.add_argument("--outdir", type=str, default="figures/forward_alpha_sweep")
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device, dtype = torch.device("cpu"), torch.float64
    op = LinearMonotone(dim=args.dim, m=args.m, dtype=dtype, device=device)
    r = torch.randn(args.dim, device=device, dtype=dtype)

    configs = [(None,1.0), (8,1.0), (6,1.0), (4,1.0)]
    colors = {None:"C0", 8:"C1", 6:"C2", 4:"C3"}

    # α grid relative to FP32 window (use certified edge 2m/L^2 as a guide)
    m_fp, L_fp = spectral_bounds(op.A())
    a_edge = (2*m_fp)/(L_fp*L_fp)
    alphas = torch.linspace(0.05*a_edge, 1.2*a_edge, args.alphas).tolist()

    plt.figure()
    for bits,boost in configs:
        if bits is None:
            A_use = op.A().detach()
        else:
            qw = QuantWrapperLinearA(op.A, num_bits=bits, dtype=dtype, scale_boost=boost)
            A_use = qw.A_q().detach()

        preds = [ exact_rate(a, A_use) for a in alphas ]
        meas  = [ measure(a, A_use, r, args.steps) for a in alphas ]
        lab = "fp32" if bits is None else f"{bits}-bit"
        plt.plot(alphas, preds, linestyle="-",  label=f"exact {lab}", color=colors[bits])
        plt.plot(alphas, meas,  linestyle="--", label=f"meas {lab}",  color=colors[bits])

    plt.xlabel("alpha")
    plt.ylabel("contraction factor  ||I - αA||₂")
    plt.title("FB: exact rate vs measured across α")
    plt.grid(True, alpha=0.3)
    plt.legend()
    f1 = Path(args.outdir) / "alpha_sweep_rates.png"
    plt.tight_layout(); plt.savefig(f1)

    # Fixed-point displacement vs bitwidth
    dWs, disps, labels = [], [], []
    z_star_fp = torch.linalg.solve(op.A().detach(), r)
    for bits,boost in configs[1:]:
        qw = QuantWrapperLinearA(op.A, num_bits=bits, dtype=dtype, scale_boost=boost)
        A_q = qw.A_q().detach()
        z_star_q = torch.linalg.solve(A_q, r)
        _, dW = qw.deltas()
        dWs.append(dW)
        disps.append(float(torch.linalg.norm(z_star_q - z_star_fp).item()))
        labels.append(bits)

    plt.figure()
    plt.plot(labels, disps, marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("bitwidth")
    plt.ylabel(r"$\|z^\star_q - z^\star\|$")
    plt.title("Fixed-point displacement vs bitwidth")
    plt.grid(True, alpha=0.3)
    f2 = Path(args.outdir) / "fp_displacement_vs_bits.png"
    plt.tight_layout(); plt.savefig(f2)

    print(f"[saved] {f1}")
    print(f"[saved] {f2}")

if __name__ == "__main__":
    main()
