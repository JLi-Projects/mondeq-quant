# experiments/alpha_sweep.py
import argparse
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import torch

from experiments.common import (
    sample_instance,
    quantize_params,
    fb_solve_with_W,
    tail_ratio,
    predicted_rate_from_W,
    alpha_crit_from_W,
)


def run(bits: Optional[int], n: int, d: int, B: int, seeds: List[int],
        alpha_min: float, alpha_max: float, num_alpha: int,
        max_iters: int, tol: float, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    alphas = np.linspace(alpha_min, alpha_max, num_alpha, dtype=np.float64)

    measured = []
    predicted = []

    for seed in seeds:
        inst = sample_instance(n=n, d=d, B=B, seed=seed)
        Wf, Uf, bf = inst.W_f, inst.U_f, inst.b_f
        if bits is None:
            Wq, Uq, bq = Wf, Uf, bf
            tag = "fp"
        else:
            Wq, Uq, bq = quantize_params(Wf, Uf, bf, bits, bits, bits)
            tag = f"{bits}bit"

        alpha_c = alpha_crit_from_W(Wq)
        if alpha_c <= 0:
            continue

        seed_meas = []
        seed_pred = []

        for a in alphas:
            z, hist = fb_solve_with_W(Wq, Uq, bq, inst.x, alpha=a, max_iters=max_iters, tol=tol)
            tr = tail_ratio(hist["residuals"], tail=50)
            seed_meas.append(tr)
            seed_pred.append(predicted_rate_from_W(Wq, a))

        measured.append(seed_meas)
        predicted.append(seed_pred)

    if not measured:
        return

    measured = np.array(measured)  # S x A
    predicted = np.array(predicted)

    m_mu = measured.mean(axis=0)
    m_lo = np.percentile(measured, 25, axis=0)
    m_hi = np.percentile(measured, 75, axis=0)
    p_mu = predicted.mean(axis=0)

    plt.figure()
    plt.plot(alphas, p_mu, label="predicted rate")
    plt.fill_between(alphas, m_lo, m_hi, alpha=0.3)
    plt.plot(alphas, m_mu, linestyle="--", label="measured tail ratio")
    plt.axhline(1.0, linestyle=":")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("contraction / tail ratio")
    plt.title(f"Measured vs predicted FB rate ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"alpha_sweep_{tag}.png", dpi=180)

    meta = {
        "bits": bits if bits is not None else "fp",
        "n": n, "d": d, "B": B,
        "seeds": seeds,
        "alpha_grid": alphas.tolist(),
    }
    (outdir / f"alpha_sweep_{tag}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=-1, help="-1 for float; else bit-depth")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--alpha-min", type=float, default=0.0)
    p.add_argument("--alpha-max", type=float, default=2.0)
    p.add_argument("--num-alpha", type=int, default=41)
    p.add_argument("--max-iters", type=int, default=800)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--outdir", type=Path, default=Path("figures/alpha_sweep"))
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    bits_opt = None if args.bits < 0 else args.bits
    run(bits_opt, args.n, args.d, args.B, args.seeds,
        args.alpha_min, args.alpha_max, args.num_alpha,
        args.max_iters, args.tol, args.outdir)
