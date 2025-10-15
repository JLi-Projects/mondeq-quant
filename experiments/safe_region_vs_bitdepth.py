from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import math

import torch
import matplotlib.pyplot as plt

from mondeq.operators.linear import LinearMonotone                  # legacy param (≈ mI + PSD)
from mondeq.operators.wk_param import WKParamMonotone              # Winston–Kolter A,B param
from mondeq.quant.fake import QuantWrapperLinearA
from mondeq.logging.spectra import spectral_bounds

def exact_rate(alpha: float, A: torch.Tensor) -> float:
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return float(torch.linalg.svdvals(I - alpha * A).max().item())

@torch.no_grad()
def eval_m_L(A: torch.Tensor) -> Tuple[float, float]:
    m_t, L_t = spectral_bounds(A)
    return float(max(m_t, 0.0)), float(max(L_t, 0.0))

def collect_stats(
    make_A, bits_list: List[int|None], seeds: int, dtype=torch.float64, device=torch.device("cpu"),
    scale_boost: float = 1.0
) -> Dict[int, Dict[str, List[float]]]:
    stats: Dict[int, Dict[str, List[float]]] = {}
    for b in bits_list:
        key = 32 if b is None else int(b)
        stats[key] = {"m": [], "L": [], "A": []}
    for s in range(seeds):
        A_fp = make_A(seed=1234 + s).to(device=device, dtype=dtype)
        m_t, L_t = eval_m_L(A_fp)
        stats[32]["m"].append(m_t); stats[32]["L"].append(L_t); stats[32]["A"].append(A_fp)
        for b in bits_list:
            if b is None:
                continue
            qw = QuantWrapperLinearA(lambda: A_fp, num_bits=int(b), dtype=dtype, scale_boost=scale_boost)
            A_q = qw.A_q().detach()
            m_t, L_t = eval_m_L(A_q)
            stats[int(b)]["m"].append(m_t); stats[int(b)]["L"].append(L_t); stats[int(b)]["A"].append(A_q)
    return stats

def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs: return (float("nan"), float("nan"))
    mu = sum(xs) / len(xs)
    if len(xs) == 1: return (mu, 0.0)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return (mu, math.sqrt(max(var, 0.0)))

def main():
    p = argparse.ArgumentParser()
    # model options
    p.add_argument("--model", choices=["wk", "linear"], default="wk",
                   help="wk: Winston–Kolter A,B parameterisation; linear: older mI+PSD generator")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--m", type=float, default=0.2)
    # WK-specific knobs
    p.add_argument("--a-scale", type=float, default=0.25, help="Frobenius norm for A (controls PSD part)")
    p.add_argument("--b-scale", type=float, default=0.08, help="Frobenius norm for B (controls skew part)")
    # alpha choice
    p.add_argument("--alpha", type=float, default=None, help="if not set, alpha = frac * (2 m_fp32 / L_fp32^2)")
    p.add_argument("--alpha-frac", type=float, default=0.9)
    # quantisation + sampling
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--bits", type=int, nargs="*", default=[8,6,4])
    p.add_argument("--scale-boost", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="figures/safe_region")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    device, dtype = torch.device("cpu"), torch.float64

    # factory to build A = I - W
    if args.model == "wk":
        def makeA(seed: int):
            op = WKParamMonotone(dim=args.dim, m=args.m, a_scale=args.a_scale, b_scale=args.b_scale,
                                 dtype=dtype, device=device, seed=seed)
            return op.A().detach()
    else:
        def makeA(seed: int):
            # legacy "linear" class (kept for ablation)
            torch.manual_seed(seed)
            op = LinearMonotone(dim=args.dim, m=args.m, dtype=dtype, device=device)
            return op.A().detach()

    # establish alpha
    A_ref = makeA(seed=777)
    m_fp, L_fp = eval_m_L(A_ref)
    margin_fp = (2.0 * m_fp) / (L_fp * L_fp) if L_fp > 0 else 0.0
    alpha = args.alpha if args.alpha is not None else args.alpha_frac * margin_fp

    bits_list: List[int|None] = [None] + [int(b) for b in args.bits]
    stats = collect_stats(makeA, bits_list=bits_list, seeds=args.seeds, dtype=dtype, device=device, scale_boost=args.scale_boost)

    rows = []
    for b in sorted(stats.keys(), reverse=True):
        m_mu, m_sd = mean_std(stats[b]["m"])
        L_mu, L_sd = mean_std(stats[b]["L"])
        margin = (2.0 * m_mu) / (L_mu * L_mu) if L_mu > 0 else 0.0
        r_exact_vals = [exact_rate(alpha, A) for A in stats[b]["A"]]
        r_mu, r_sd = mean_std(r_exact_vals)
        rows.append((b, m_mu, m_sd, L_mu, L_sd, margin, r_mu, r_sd))

    # ---- Fig 1: Safe region ----
    Ls_max = max(max(r[3] + r[4] for r in rows), 1.0)
    Ls = torch.linspace(0.0, Ls_max, 200).tolist()
    Ms = [ (alpha/2.0) * (L*L) for L in Ls ]

    plt.figure()
    plt.plot(Ls, Ms, "k--", linewidth=1.0, label=fr"boundary $m=(\alpha/2)L^2, \ \alpha={alpha:.3f}$")
    L_line = torch.linspace(0.0, Ls_max, 50).tolist()
    plt.plot(L_line, L_line, "k:", linewidth=1.0, label=r"$m\leq L$")
    for (b, m_mu, m_sd, L_mu, L_sd, margin, r_mu, r_sd) in rows:
        lbl = "fp32" if b == 32 else f"{b}-bit"
        plt.errorbar([L_mu], [m_mu], xerr=[L_sd], yerr=[m_sd], fmt="o", label=f"{lbl}")
        plt.text(L_mu, m_mu, f"  {lbl}", va="bottom", fontsize=9)
    plt.xlabel(r"$\tilde L$")
    plt.ylabel(r"$\tilde m$")
    plt.title(fr"Safe region @ $\alpha={alpha:.3f}$  (model={args.model})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    f1 = Path(args.outdir) / "safe_region_m_vs_L.png"
    plt.tight_layout(); plt.savefig(f1, bbox_inches="tight")

    # ---- Fig 2: Certified margin vs bitwidth ----
    bits_sorted = [r[0] for r in rows]
    margins     = [r[5] for r in rows]

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(bits_sorted, margins, "o-", label=r"certified margin $2\tilde m/\tilde L^2$")
    ax1.axhline(alpha, color="k", linestyle="--", linewidth=1.0, label=fr"probe $\alpha={alpha:.3f}$")
    ax1.invert_xaxis()
    ax1.set_xlabel("bitwidth")
    ax1.set_ylabel("margin")
    ax1.set_title("Certified step-size margin vs bitwidth")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    f2 = Path(args.outdir) / "margin_vs_bitwidth.png"
    plt.tight_layout(); plt.savefig(f2, bbox_inches="tight")

    # ---- Fig 3: Exact rate at alpha vs bitwidth ----
    rates_mu    = [r[6] for r in rows]
    rates_sd    = [r[7] for r in rows]
    plt.figure()
    plt.errorbar(bits_sorted, rates_mu, yerr=rates_sd, fmt="o-")
    plt.gca().invert_xaxis()
    plt.xlabel("bitwidth")
    plt.ylabel(r"exact rate $\|I-\alpha A\|_2$")
    plt.title(fr"Exact contraction @ $\alpha={alpha:.3f}$ vs bitwidth")
    plt.grid(True, alpha=0.3)
    f3 = Path(args.outdir) / "exact_rate_vs_bitwidth.png"
    plt.tight_layout(); plt.savefig(f3, bbox_inches="tight")

    print("[summary]")
    for (b, m_mu, m_sd, L_mu, L_sd, margin, r_mu, r_sd) in rows:
        label = "fp32" if b == 32 else f"{b}-bit"
        ok = alpha < margin
        print(f"{label:>8}: m~={m_mu:.4f}±{m_sd:.4f}  L~={L_mu:.4f}±{L_sd:.4f}  "
              f"margin={margin:.4f}  exact_rate@α={r_mu:.4f}±{r_sd:.4f}  "
              f"{'STABLE' if ok else 'UNSAFE'}")

    print(f"[saved] {f1}")
    print(f"[saved] {f2}")
    print(f"[saved] {f3}")

if __name__ == "__main__":
    main()
