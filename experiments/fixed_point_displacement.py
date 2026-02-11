# experiments/fixed_point_displacement.py
import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch

from experiments.common import (
    sample_instance, quantize_params, fb_solve_with_W,
    alpha_crit_from_W, spectral_from_W, deltaW_norm2
)


def run(bits_list: List[int], n: int, d: int, B: int, seeds: List[int],
        rel_alpha: float, max_iters: int, tol: float, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for bits in bits_list:
        for seed in seeds:
            inst = sample_instance(n=n, d=d, B=B, seed=seed)
            Wf, Uf, bf = inst.W_f, inst.U_f, inst.b_f
            alpha_c_f = alpha_crit_from_W(Wf)
            alpha_f = rel_alpha * alpha_c_f if alpha_c_f > 0 else 0.0
            zf, _ = fb_solve_with_W(Wf, Uf, bf, inst.x, alpha=alpha_f, max_iters=max_iters, tol=tol)

            Wq, Uq, bq = quantize_params(Wf, Uf, bf, bits, bits, bits)
            alpha_c_q = alpha_crit_from_W(Wq)
            alpha_q = rel_alpha * alpha_c_q if alpha_c_q > 0 else 0.0
            zq, _ = fb_solve_with_W(Wq, Uq, bq, inst.x, alpha=alpha_q, max_iters=max_iters, tol=tol)

            disp = (zq - zf).norm().item()
            nz = zf.norm().item() + 1e-12
            rel_disp = disp / nz
            m_t, _ = spectral_from_W(Wq)
            dwn = deltaW_norm2(Wf, Wq)
            bound_proxy = dwn / (float(m_t) + 1e-12)
            rows.append((bits, rel_disp, bound_proxy))

    bits = sorted(set(b for (b, _, _) in rows))
    plt.figure()
    for b in bits:
        vals = [(rel, bd) for (bb, rel, bd) in rows if bb == b]
        rels = [v[0] for v in vals]
        bds = [v[1] for v in vals]
        jitter = (np.random.rand(len(rels)) - 0.5) * 0.2
        x = np.full(len(rels), b, dtype=np.float64) + jitter
        plt.scatter(x, rels)
    plt.xlabel("bit-depth")
    plt.ylabel("relative fixed-point displacement")
    plt.title(f"Quantised vs float equilibrium displacement (alpha={rel_alpha}·alpha_c)")
    plt.tight_layout()
    plt.savefig(out.with_name("fp_displacement.png"), dpi=180)

    plt.figure()
    xs = [bd for (_, _, bd) in rows]
    ys = [rel for (_, rel, _) in rows]
    plt.scatter(xs, ys)
    plt.xlabel(r"$\| \Delta W \|_2 / \tilde m$")
    plt.ylabel("relative displacement")
    plt.title("Displacement vs ΔW/m proxy")
    plt.tight_layout()
    plt.savefig(out.with_name("fp_displacement_vs_bound_proxy.png"), dpi=180)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, nargs="+", default=[4,6,8,12,16,24,32])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--rel-alpha", type=float, default=0.9)
    p.add_argument("--max-iters", type=int, default=800)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--out", type=Path, default=Path("figures/displacement/placeholder.png"))
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    run(args.bits, args.n, args.d, args.B, args.seeds,
        args.rel_alpha, args.max_iters, args.tol, args.out)
