# experiments/compare_fp_displacement_decaying.py
import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

from experiments.common import (
    sample_instance,
    quantize_params,
    fb_solve_with_W,
    fb_solve_with_iterate_quant,
    alpha_crit_from_W,
    deltaW_norm2,
)


def run(
    n: int,
    d: int,
    B: int,
    seeds: List[int],
    bits_param: Optional[int],
    bits_iter: int,
    rho: float,
    rel_alpha: float,
    max_iters: int,
    tol: float,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []  # (seed, disp_param, disp_iter_decay, dW_over_m)

    for seed in seeds:
        inst = sample_instance(n=n, d=d, B=B, seed=seed)
        Wf, Uf, bf, x = inst.W_f, inst.U_f, inst.b_f, inst.x

        # Baseline: exact float equilibrium
        alpha_c_f = alpha_crit_from_W(Wf)
        if alpha_c_f <= 0:
            continue
        alpha_f = rel_alpha * alpha_c_f
        z_float, _ = fb_solve_with_W(Wf, Uf, bf, x, alpha=alpha_f, max_iters=max_iters, tol=tol)

        # Original non-decaying quantiser: static parameter quantisation
        if bits_param is not None and bits_param > 0:
            Wq, Uq, bq = quantize_params(Wf, Uf, bf, bits_param, bits_param, bits_param)
            alpha_c_q = alpha_crit_from_W(Wq)
            if alpha_c_q > 0:
                alpha_q = rel_alpha * alpha_c_q
                z_param, _ = fb_solve_with_W(Wq, Uq, bq, x, alpha=alpha_q, max_iters=max_iters, tol=tol)
                disp_param = (z_param - z_float).norm().item() / (z_float.norm().item() + 1e-12)
                # proxy driver for displacement: ||ΔW||_2 / m_tilde (Sym(I-Wq))
                I = torch.eye(Wq.size(0), device=Wq.device, dtype=Wq.dtype)
                Ssym = 0.5 * ((I - Wq) + (I - Wq).T)
                m_t = torch.linalg.eigvalsh(Ssym).min().item()
                dW = deltaW_norm2(Wf, Wq)
                dW_over_m = dW / (m_t + 1e-12)
            else:
                disp_param, dW_over_m = np.nan, np.nan
        else:
            disp_param, dW_over_m = 0.0, 0.0

        # Decaying iterate quantiser: parameters float; quantise u_k with s_k ~ rho^k
        z_decay, _ = fb_solve_with_iterate_quant(
            Wf, Uf, bf, x,
            alpha=alpha_f, max_iters=max_iters, tol=tol,
            bits_iter=bits_iter, mode="decay", rho=rho
        )
        disp_decay = (z_decay - z_float).norm().item() / (z_float.norm().item() + 1e-12)

        rows.append((seed, disp_param, disp_decay, dW_over_m))

    if not rows:
        return

    seeds_out = [r[0] for r in rows]
    disp_param = np.array([r[1] for r in rows], dtype=np.float64)
    disp_decay = np.array([r[2] for r in rows], dtype=np.float64)
    dW_over_m = np.array([r[3] for r in rows], dtype=np.float64)

    # Boxplot: relative displacement for the two quantisers
    plt.figure()
    data = []
    labels = []
    if bits_param is not None and bits_param > 0:
        data.append(disp_param[~np.isnan(disp_param)])
        labels.append(f"param-quant ({bits_param}b)")
    data.append(disp_decay)
    labels.append(f"iter-quant decay (bits={bits_iter}, rho={rho})")
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("relative fixed-point displacement vs float")
    plt.title(f"Displacement comparison (n={n}, d={d}, B={B}, rel α={rel_alpha})")
    plt.tight_layout()
    plt.savefig(outdir / "fp_displacement_compare.png", dpi=180)

    # Optional scatter: parameter-quant displacement vs ΔW/m proxy
    if bits_param is not None and bits_param > 0:
        mask = ~np.isnan(dW_over_m)
        plt.figure()
        plt.scatter(dW_over_m[mask], disp_param[mask])
        plt.xlabel(r"$\|\Delta W\|_2 / \tilde m$ (proxy)")
        plt.ylabel("relative displacement (param-quant)")
        plt.title("Param-quant displacement vs ΔW/m proxy")
        plt.tight_layout()
        plt.savefig(outdir / "param_quant_disp_vs_proxy.png", dpi=180)

    # CSV dump
    with open(outdir / "fp_displacement_compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "disp_param", "disp_iter_decay", "dW_over_m"])
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9])

    # param quant (original non-decaying); set to -1 to disable
    p.add_argument("--bits-param", type=int, default=8)

    # iterate quant (new decaying)
    p.add_argument("--bits-iter", type=int, default=8)
    p.add_argument("--rho", type=float, default=0.8, help="geometric decay factor for iterate quant step")

    p.add_argument("--rel-alpha", type=float, default=0.9)
    p.add_argument("--max-iters", type=int, default=800)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--outdir", type=Path, default=Path("figures/compare_quant"))
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    bits_param_opt = None if args.bits_param < 0 else args.bits_param
    run(
        n=args.n, d=args.d, B=args.B, seeds=args.seeds,
        bits_param=bits_param_opt, bits_iter=args.bits_iter, rho=args.rho,
        rel_alpha=args.rel_alpha, max_iters=args.max_iters, tol=args.tol,
        outdir=args.outdir
    )
