# experiments/safe_window_vs_bits.py
import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch

from experiments.common import sample_instance, quantize_params, spectral_from_W, alpha_crit_from_W


def run(bits_list: List[int], n: int, d: int, B: int, seeds: List[int], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    alpha_c_stats = []
    m_stats = []
    L_stats = []
    x_bits = []

    for bits in bits_list:
        acs = []
        ms = []
        Ls = []
        for seed in seeds:
            inst = sample_instance(n=n, d=d, B=B, seed=seed)
            Wf, Uf, bf = inst.W_f, inst.U_f, inst.b_f
            Wq, Uq, bq = quantize_params(Wf, Uf, bf, bits, bits, bits)
            m_t, L_t = spectral_from_W(Wq)
            ac = alpha_crit_from_W(Wq)
            acs.append(ac)
            ms.append(float(m_t))
            Ls.append(float(L_t))
        alpha_c_stats.append((np.mean(acs), np.std(acs)))
        m_stats.append((np.mean(ms), np.std(ms)))
        L_stats.append((np.mean(Ls), np.std(Ls)))
        x_bits.append(bits)

    bits_arr = np.array(x_bits, dtype=np.float64)
    ac_mu, ac_sd = np.array([s[0] for s in alpha_c_stats]), np.array([s[1] for s in alpha_c_stats])
    m_mu, m_sd = np.array([s[0] for s in m_stats]), np.array([s[1] for s in m_stats])
    L_mu, L_sd = np.array([s[0] for s in L_stats]), np.array([s[1] for s in L_stats])

    plt.figure()
    plt.errorbar(bits_arr, ac_mu, yerr=ac_sd, fmt="-o")
    plt.xlabel("bit-depth")
    plt.ylabel(r"$\alpha_c = 2\tilde m / \tilde L^2$")
    plt.title("Certified safe stepsize vs bit-depth")
    plt.tight_layout()
    plt.savefig(out.with_name("alpha_c_vs_bits.png"), dpi=180)

    plt.figure()
    plt.errorbar(bits_arr, m_mu, yerr=m_sd, fmt="-o")
    plt.xlabel("bit-depth")
    plt.ylabel(r"$\tilde m$")
    plt.title("Strong monotonicity margin vs bit-depth")
    plt.tight_layout()
    plt.savefig(out.with_name("m_tilde_vs_bits.png"), dpi=180)

    plt.figure()
    plt.errorbar(bits_arr, L_mu, yerr=L_sd, fmt="-o")
    plt.xlabel("bit-depth")
    plt.ylabel(r"$\tilde L$")
    plt.title("Operator norm vs bit-depth")
    plt.tight_layout()
    plt.savefig(out.with_name("L_tilde_vs_bits.png"), dpi=180)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, nargs="+", default=[4,6,8,12,16,24,32])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7])
    p.add_argument("--out", type=Path, default=Path("figures/window/window.png"))
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    run(args.bits, args.n, args.d, args.B, args.seeds, args.out)
