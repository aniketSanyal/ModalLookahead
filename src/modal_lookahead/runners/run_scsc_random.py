import argparse, numpy as np
from ..problems.scsc import ScScConfig, ScScQuadratic
from ..optim.lookahead import ModalLookahead, FixedLookahead
from ..plotting.plots import plot_distance_curves

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=120)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--mu_x", type=float, default=0.18)
    p.add_argument("--mu_y", type=float, default=0.18)
    p.add_argument("--sigma_min", type=float, default=0.25)
    p.add_argument("--sigma_max", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mode", type=str, default="grad_bilinear", choices=["grad_bilinear","hvp"])
    p.add_argument("--outfile", type=str, default="scsc_modal_vs_fixed")
    args=p.parse_args()

    cfg=ScScConfig(d=args.d, gamma=args.gamma, mu_x=args.mu_x, mu_y=args.mu_y,
                   sigma_min=args.sigma_min, sigma_max=args.sigma_max, seed=args.seed)
    problem=ScScQuadratic(cfg)

    rng=np.random.default_rng(args.seed+7)
    x0=rng.normal(size=(args.d,))*5.0
    y0=rng.normal(size=(args.d,))*5.0

    modal=ModalLookahead(mode=args.mode, alpha=args.alpha)
    fixed=FixedLookahead(k=5, alpha=0.5)

    y_modal, info = modal.run(problem, args.T, x0, y0)
    y_fix = fixed.run(problem, args.T, x0, y0)

    curves=[(f"Modal Lookahead (k={info['k']}, alpha={args.alpha:.2f})", y_modal),
            ("Fixed Lookahead (k=5, alpha=0.5)", y_fix)]
    title=(
        f"SC-SC Random (d={args.d}, gamma={args.gamma}, mu=({args.mu_x},{args.mu_y}), "
        f"sigma in [{args.sigma_min},{args.sigma_max}])\nBackend: {args.mode}"
    )
    plot_distance_curves(curves, title, args.outfile)
    print("Modal selection:", info)

if __name__ == "__main__":
    main()
