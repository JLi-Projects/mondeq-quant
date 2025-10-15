from __future__ import annotations
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from mondeq.operators.conv import ConvMonotone
from mondeq.quant.fake import fake_quantize_symmetric

def predicted_rate(alpha: float, m_tilde: float, L_tilde: float) -> float:
    val = 1.0 - 2.0 * alpha * m_tilde + (alpha * L_tilde) ** 2
    return float((val if val > 0 else 0.0) ** 0.5)

@torch.no_grad()
def measure_rate(op: ConvMonotone, alpha: float, H: int, W: int, steps: int, qbits: int | None) -> float:
    device, dtype = op.device, op.dtype
    x0 = torch.randn(1, op.channels, H, W, device=device, dtype=dtype)
    r  = torch.randn_like(x0)
    if qbits is not None:
        op.K.weight[:] = fake_quantize_symmetric(op.K.weight, qbits)

    zs = []
    z = torch.randn_like(x0)
    def A(u): return op.apply(u)
    for _ in range(steps):
        z1 = z - alpha * (A(z) - r)
        # use error step ratio in norm; detach to avoid grad warnings
        rate = (z1 - z).detach().norm() / (z.detach().norm() + 1e-12)
        zs.append(float(rate.item()))
        z = z1
    tail = zs[-50:] if len(zs) >= 50 else zs
    return float(sum(tail)/len(tail)) if tail else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--H", type=int, default=28)
    ap.add_argument("--W", type=int, default=28)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--outdir", type=str, default="figures/conv_forward")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    op = ConvMonotone(channels=args.channels, m=args.m, dtype=dtype, device=device)

    L_est = op.spectral_norm_estimate(args.H, args.W, iters=100)
    m_tilde = args.m
    pred = predicted_rate(args.alpha, m_tilde, L_est)

    results = []
    for bits in [None, 8, 4]:
        op = ConvMonotone(channels=args.channels, m=args.m, dtype=dtype, device=device)
        rate = measure_rate(op, args.alpha, args.H, args.W, args.steps, qbits=(None if bits is None else bits))
        results.append((32 if bits is None else bits, rate))

    plt.figure()
    xs = [b for b,_ in results]
    ys = [r for _,r in results]
    plt.axhline(pred, color="C0", linestyle="-", label="pred ref (fp32 L estimate)")
    plt.plot(xs, ys, "o--", label="measured (conv)")
    plt.gca().invert_xaxis()
    plt.xlabel("bitwidth")
    plt.ylabel("contraction factor")
    plt.title("Conv MON forward contraction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    f = Path(args.outdir) / "conv_forward_rates.png"
    plt.tight_layout(); plt.savefig(f)
    print(f"[saved] {f}")

if __name__ == "__main__":
    main()
