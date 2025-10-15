from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import matplotlib.pyplot as plt

from mondeq.operators.linear import LinearMonotone
from mondeq.solvers.fb import FBSolver
from mondeq.quant.fake import QuantWrapperLinearA
from mondeq.logging.spectra import spectral_bounds, predicted_fb_rate

def run_once(dim: int, m: float, alpha: float, num_bits: int | None, steps: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    dtype = torch.float64

    op = LinearMonotone(dim=dim, m=m, dtype=dtype, device=device)
    A_fp = op.A().detach()

    # Use a non-zero RHS and a random z0 to avoid trivial early exit
    r = torch.randn(dim, device=device, dtype=dtype)
    z0 = torch.randn(dim, device=device, dtype=dtype)

    if num_bits is not None:
        qw = QuantWrapperLinearA(a_fn=op.A, num_bits=num_bits, dtype=dtype)
        A_use = qw.A_q().detach()
        dA, dW = qw.deltas()
    else:
        A_use = A_fp
        dA, dW = 0.0, 0.0

    # Ground-truth fixed point z* = A^{-1} r for linear system
    z_star = torch.linalg.solve(A_use, r)
    m_tilde, L_tilde = spectral_bounds(A_use)
    r_pred = predicted_fb_rate(alpha, m_tilde, L_tilde)

    step_ratios: List[float] = []
    err_ratios: List[float] = []

    def cb(k, zk1, res, info):
        mc = info.get("measured_contraction", float("nan"))
        ec = info.get("error_contraction", float("nan"))
        if not torch.isnan(torch.tensor(mc)):
            step_ratios.append(mc)
        if not torch.isnan(torch.tensor(ec)):
            err_ratios.append(ec)

    solver = FBSolver(
        max_iters=steps,
        min_iters=50,
        tol=1e-12,
        alpha=alpha,
        damping=1.0,
        callback=cb,
        dtype=dtype,
        device=device,
    )
    _ = solver.solve_linear(A_use, r=r, z0=z0, z_star=z_star)

    # Average last few values for stability
    def tail_mean(xs: List[float], tail: int = 50) -> float:
        if not xs:
            return float("nan")
        t = xs[-tail:] if len(xs) >= tail else xs
        return float(sum(t) / len(t))

    meas = tail_mean(step_ratios, 50)
    meas_err = tail_mean(err_ratios, 50)

    return {
        "alpha": alpha,
        "bits": num_bits if num_bits is not None else 32,
        "m_tilde": m_tilde,
        "L_tilde": L_tilde,
        "pred_rate": r_pred,
        "meas_rate": meas,
        "meas_err_rate": meas_err,
        "dA": dA,
        "dW": dW,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--m", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--outdir", type=str, default="figures/forward_certificates")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for bits in [None, 8, 6, 4]:
        for s in range(args.seeds):
            r = run_once(dim=args.dim, m=args.m, alpha=args.alpha, num_bits=(None if bits is None else bits),
                         steps=args.steps, seed=1234 + s)
            runs.append(r)

    # Aggregate by bitwidth
    by_bits: Dict[int, Dict[str, float]] = {}
    for r in runs:
        b = int(r["bits"])
        d = by_bits.setdefault(b, {"pred": 0.0, "meas": 0.0, "meas_err": 0.0, "cnt": 0, "dW": 0.0})
        d["pred"] += r["pred_rate"]
        if not torch.isnan(torch.tensor(r["meas_rate"])):
            d["meas"] += r["meas_rate"]
        if not torch.isnan(torch.tensor(r["meas_err_rate"])):
            d["meas_err"] += r["meas_err_rate"]
        d["dW"] += r["dW"]
        d["cnt"] += 1

    bits_sorted = sorted(by_bits.keys(), reverse=True)
    preds = [by_bits[b]["pred"] / max(by_bits[b]["cnt"], 1) for b in bits_sorted]
    meas  = [by_bits[b]["meas"] / max(by_bits[b]["cnt"], 1) for b in bits_sorted]
    measE = [by_bits[b]["meas_err"] / max(by_bits[b]["cnt"], 1) for b in bits_sorted]
    dWs   = [by_bits[b]["dW"] / max(by_bits[b]["cnt"], 1) for b in bits_sorted]

    # Contraction rates vs bitwidth (predicted vs step-ratio vs error-ratio)
    plt.figure()
    plt.plot(bits_sorted, preds, marker="o", label="predicted r(α)")
    plt.plot(bits_sorted, meas, marker="x", label="measured (step ratio)")
    plt.plot(bits_sorted, measE, marker="^", label="measured (error ratio)")
    plt.gca().invert_xaxis()
    plt.xlabel("bitwidth")
    plt.ylabel("contraction factor")
    plt.title(f"FB contraction (α={args.alpha})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1 = Path(args.outdir) / "contraction_vs_bits.png"
    plt.savefig(fig1)

    # ||ΔW||_2 vs bitwidth
    plt.figure()
    plt.plot(bits_sorted, dWs, marker="s")
    plt.gca().invert_xaxis()
    plt.xlabel("bitwidth")
    plt.ylabel(r"$\|\Delta W\|_2$")
    plt.title("Quantisation perturbation vs bitwidth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2 = Path(args.outdir) / "deltaW_vs_bits.png"
    plt.savefig(fig2)

    print(f"[saved] {fig1}")
    print(f"[saved] {fig2}")

if __name__ == "__main__":
    main()
