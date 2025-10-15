from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import torch
import matplotlib.pyplot as plt

from mondeq.operators.linear import LinearMonotone
from mondeq.solvers.fb import FBSolver
from mondeq.solvers.pr import PRLinear
from mondeq.quant.fake import QuantWrapperLinearA
from mondeq.logging.spectra import spectral_bounds

def tail_mean(xs: List[float], k: int = 50) -> float:
    xs = xs[-k:] if len(xs) >= k else xs
    return float(sum(xs) / len(xs)) if xs else float("nan")

def measure_fb(alpha: float, A: torch.Tensor, r: torch.Tensor, steps: int) -> float:
    device, dtype = A.device, A.dtype
    z0 = torch.randn(A.shape[-1], device=device, dtype=dtype)
    z_star = torch.linalg.solve(A, r)
    tail: List[float] = []
    def cb(k, zk1, res, info):
        ez  = torch.linalg.norm(meas[0] - z_star)
        ez1 = torch.linalg.norm(zk1 - z_star)
        if ez.item() > 0:
            tail.append(float((ez1/ez).item()))
        meas[0] = zk1
    meas = [z0.clone()]
    FBSolver(max_iters=steps, min_iters=50, tol=1e-12, alpha=alpha, callback=cb, device=device, dtype=dtype)\
        .solve_linear(A, r=r, z0=z0, z_star=z_star)
    return tail_mean(tail, 50)

def measure_pr(alpha: float, A: torch.Tensor, r: torch.Tensor, m_split: float, steps: int) -> float:
    device, dtype = A.device, A.dtype
    z_star = torch.linalg.solve(A, r)
    tail: List[float] = []
    def cb(k, yk1, res, info):
        ez  = torch.linalg.norm(pr_state[0] - z_star)
        ez1 = torch.linalg.norm(yk1 - z_star)
        if ez.item() > 0:
            tail.append(float((ez1/ez).item()))
        pr_state[0] = yk1
    pr_state = [torch.randn(A.shape[-1], device=device, dtype=dtype)]
    PRLinear(alpha=alpha, max_iters=steps, min_iters=50, tol=1e-12, m_split=m_split, callback=cb, device=device, dtype=dtype)\
        .solve(A, r=r, u0=pr_state[0])
    return tail_mean(tail, 50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--m", type=float, default=0.08)      # small m to be near edge
    ap.add_argument("--alpha", type=float, default=0.9)   # aggressive
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--outdir", type=str, default="figures/fb_vs_pr")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device, dtype = torch.device("cpu"), torch.float64
    op = LinearMonotone(dim=args.dim, m=args.m, dtype=dtype, device=device)
    r = torch.randn(args.dim, device=device, dtype=dtype)

    settings = [
        ("8-bit",         8, 1.0,  False),
        ("4-bit-stress",  4, 6.0,  True),   # inflate quant noise to reduce strong monotonicity
    ]
    labels, fb_rates, pr_rates, stable_flags = [], [], [], []

    for name, bits, boost, stressed in settings:
        qw = QuantWrapperLinearA(op.A, num_bits=bits, dtype=dtype, scale_boost=boost)
        A_use = qw.A_q().detach()
        m_t, L_t = spectral_bounds(A_use)
        # certified FB window
        fb_stable = (args.alpha < (2*m_t)/(L_t*L_t)) if (L_t > 0) else False
        stable_flags.append(fb_stable)

        # choose m_split not larger than the measured m_t to keep G monotone; clamp at small positive
        m_split = max(min(float(m_t), args.m), 1e-4)

        fb_rates.append(measure_fb(args.alpha, A_use, r, args.steps))
        pr_rates.append(measure_pr(args.alpha, A_use, r, m_split, args.steps))
        labels.append(name)

    plt.figure()
    xs = range(len(labels))
    plt.plot(xs, fb_rates, "o-", label="FB measured rate")
    plt.plot(xs, pr_rates, "x--", label="PR measured rate")
    for i, ok in enumerate(stable_flags):
        plt.text(i, max(fb_rates[i], pr_rates[i]) + 0.02, "FB stable" if ok else "FB near/unstable", ha="center")
    plt.xticks(xs, labels)
    plt.ylabel("contraction factor (error ratio)")
    plt.title(f"FB vs PR (true splitting) @ α={args.alpha}")
    plt.grid(True, alpha=0.3)
    f = Path(args.outdir) / "fb_vs_pr.png"
    plt.tight_layout(); plt.savefig(f)
    print(f"[saved] {f}")

if __name__ == "__main__":
    main()
