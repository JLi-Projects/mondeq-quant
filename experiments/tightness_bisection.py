# experiments/tightness_bisection.py
import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch

from experiments.common import (
    sample_instance, quantize_params, fb_solve_with_W,
    tail_ratio, alpha_crit_from_W
)


def converges(W, U, b, x, alpha, max_iters, tol) -> bool:
    _, hist = fb_solve_with_W(W, U, b, x, alpha=alpha, max_iters=max_iters, tol=tol)
    tr = tail_ratio(hist["residuals"], tail=50)
    return min(hist["residuals"]) < tol and tr < 1.0 + 1e-3


def alpha_empirical_ratio(W, U, b, x, max_iters, tol) -> float:
    alpha_c = alpha_crit_from_W(W)
    if alpha_c <= 0:
        return 0.0
    lo, hi = 0.1 * alpha_c, 1.6 * alpha_c
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if converges(W, U, b, x, mid, max_iters, tol):
            lo = mid
        else:
            hi = mid
    return lo / alpha_c


def run(bits_list: List[int], n: int, d: int, B: int, seeds: List[int],
        max_iters: int, tol: float, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    ratios = {bits: [] for bits in bits_list}
    for bits in bits_list:
        for seed in seeds:
            inst = sample_instance(n=n, d=d, B=B, seed=seed)
            Wf, Uf, bf = inst.W_f, inst.U_f, inst.b_f
            Wq, Uq, bq = quantize_params(Wf, Uf, bf, bits, bits, bits)
            r = alpha_empirical_ratio(Wq, Uq, bq, inst.x, max_iters, tol)
            if r > 0:
                ratios[bits].append(r)

    labels = []
    data = []
    for bits in bits_list:
        labels.append(str(bits))
        data.append(ratios[bits])

    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.axhline(1.0, linestyle=":")
    plt.ylabel(r"$\alpha_{\mathrm{emp}} / \alpha_c$")
    plt.xlabel("bit-depth")
    plt.title("Tightness of the FB stepsize certificate")
    plt.tight_layout()
    plt.savefig(out, dpi=180)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, nargs="+", default=[4,6,8,12,16,24,32])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5])
    p.add_argument("--max-iters", type=int, default=800)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--out", type=Path, default=Path("figures/tightness/alpha_empirical_ratio.png"))
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    run(args.bits, args.n, args.d, args.B, args.seeds, args.max_iters, args.tol, args.out)
