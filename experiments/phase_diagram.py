# experiments/phase_diagram.py
import argparse
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
    alpha_crit_from_W,
)


def run(bits_list: List[int], n: int, d: int, B: int, seeds: List[int],
        alpha_factor_max: float, num_alpha: int,
        max_iters: int, tol: float, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    factors = np.linspace(0.2, alpha_factor_max, num_alpha, dtype=np.float64)

    heat = np.zeros((len(bits_list), num_alpha), dtype=np.float64)

    for bi, bits in enumerate(bits_list):
        for seed in seeds:
            inst = sample_instance(n=n, d=d, B=B, seed=seed)
            Wf, Uf, bf = inst.W_f, inst.U_f, inst.b_f
            Wq, Uq, bq = quantize_params(Wf, Uf, bf, bits, bits, bits)
            alpha_c = alpha_crit_from_W(Wq)
            if alpha_c <= 0:
                continue
            for ai, f in enumerate(factors):
                alpha = f * alpha_c
                _, hist = fb_solve_with_W(Wq, Uq, bq, inst.x, alpha=alpha, max_iters=max_iters, tol=tol)
                tr = tail_ratio(hist["residuals"], tail=50)
                success = float(min(hist["residuals"]) < tol and tr < 1.0 + 1e-3)
                heat[bi, ai] += success

    heat /= len(seeds)

    plt.figure()
    plt.imshow(heat, aspect="auto", origin="lower",
               extent=[factors[0], factors[-1], bits_list[0], bits_list[-1]])
    plt.colorbar(label="success rate")
    plt.axvline(1.0, linestyle=":")
    plt.xlabel(r"$\alpha / \alpha_c$")
    plt.ylabel("bit-depth")
    plt.title("Convergence phase diagram")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, nargs="+", default=[4,6,8,12,16,24,32])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5])
    p.add_argument("--alpha-factor-max", type=float, default=1.6)
    p.add_argument("--num-alpha", type=int, default=36)
    p.add_argument("--max-iters", type=int, default=800)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--out", type=Path, default=Path("figures/phase/phase_diagram.png"))
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    run(args.bits, args.n, args.d, args.B, args.seeds,
        args.alpha_factor_max, args.num_alpha, args.max_iters, args.tol, args.out)
