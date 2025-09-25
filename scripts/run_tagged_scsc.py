#!/usr/bin/env python3
import argparse
import numpy as np

from modal_lookahead.problems.scsc import ScScConfig, ScScQuadratic
from modal_lookahead.optim.lookahead import (
    AdaptiveModalLookahead, AdaptiveLAConfig, WeightedModalConfig,
    FixedLookahead, choose_modal_params_weighted,
)
from modal_lookahead.optim.first_order import Extragradient, OGDA
from modal_lookahead.plotting.plots import plot_distance_curves


def build_problem(d, mu_x, mu_y, gamma, sigma_min, sigma_max, seed):
    cfg = ScScConfig(
        d=d, gamma=gamma, mu_x=mu_x, mu_y=mu_y,
        sigma_min=sigma_min, sigma_max=sigma_max, seed=seed
    )
    return ScScQuadratic(cfg)


def jacobian_eigs(prob: ScScQuadratic):
    d = prob.d
    mu_x, mu_y = prob.mu_x, prob.mu_y
    J = np.block([
        [mu_x * np.eye(d), prob.A.T],
        [-prob.A,          mu_y * np.eye(d)]
    ])
    return np.linalg.eigvals(J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=120)
    ap.add_argument("--T", type=int, default=300, help="TOTAL base iterations")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--mu_x", type=float, default=0.5)
    ap.add_argument("--mu_y", type=float, default=0.5)
    ap.add_argument("--sigma_min", type=float, default=0.5)
    ap.add_argument("--sigma_max", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed_k", type=int, default=5)
    ap.add_argument("--fixed_alpha", type=float, default=0.5)
    ap.add_argument("--outfile", type=str, default="scsc_iterwise")
    ap.add_argument("--no_eg", action="store_true")
    ap.add_argument("--no_ogda", action="store_true")
    ap.add_argument("--no_modal", action="store_true")
    ap.add_argument("--no_fixed", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prob = build_problem(args.d, args.mu_x, args.mu_y, args.gamma,
                         args.sigma_min, args.sigma_max, args.seed)

    # Initial state
    x0 = rng.normal(size=(prob.d,)) * 5.0
    y0 = rng.normal(size=(prob.d,)) * 5.0

    methods = {}
    if not args.no_fixed:
        methods["Fixed LA (k=5, α=0.5)"] = {
            "obj": FixedLookahead(k=args.fixed_k, alpha=args.fixed_alpha),
            "state": (x0.copy(), y0.copy()),
            "traj": [],
        }
    if not args.no_modal:
        la = AdaptiveModalLookahead(
            AdaptiveLAConfig(
                init_k=args.fixed_k,
                init_alpha=args.fixed_alpha,
                weighted=WeightedModalConfig(
                    gamma=args.gamma, kmin=5, kmax=2000, alpha_grid=None
                ),
            )
        )
        methods["Modal LA (dynamic)"] = {"obj": la, "state": (x0.copy(), y0.copy()), "traj": []}
    if not args.no_eg:
        methods["Extragradient"] = {
            "obj": Extragradient(gamma=args.gamma),
            "state": (x0.copy(), y0.copy()),
            "traj": [],
        }
    if not args.no_ogda:
        methods["OGDA"] = {
            "obj": OGDA(gamma=args.gamma),
            "state": (x0.copy(), y0.copy()),
            "traj": [],
        }

    def gda_step(x, y):
        return prob.step_gda(x, y)

    total_steps = int(args.T)
    for it in range(total_steps):
        # --- Fixed LA ---
        if not args.no_fixed:
            m = methods["Fixed LA (k=5, α=0.5)"]
            fl, (x, y) = m["obj"], m["state"]
            if getattr(fl, "_anchor", None) is None:
                fl._anchor, fl._step_in_cycle = fl._clone_state((x, y)), 0
            x, y = gda_step(x, y)
            fl._step_in_cycle += 1
            m["traj"].append(prob.distance(x, y))
            if fl._step_in_cycle >= fl.k:
                x, y = ((1 - fl.alpha) * fl._anchor[0] + fl.alpha * x,
                        (1 - fl.alpha) * fl._anchor[1] + fl.alpha * y)
                fl._anchor, fl._step_in_cycle = fl._clone_state((x, y)), 0
            m["state"] = (x, y)

        # --- Modal LA ---
        if not args.no_modal:
            m = methods["Modal LA (dynamic)"]
            la, (x, y) = m["obj"], m["state"]
            if getattr(la, "_anchor", None) is None:
                la._anchor, la._step_in_cycle = la._clone_state((x, y)), 0
            x, y = gda_step(x, y)
            la._step_in_cycle += 1
            m["traj"].append(prob.distance(x, y))
            if la._step_in_cycle >= la.k:
                x, y = ((1 - la.alpha) * la._anchor[0] + la.alpha * x,
                        (1 - la.alpha) * la._anchor[1] + la.alpha * y)
                la._anchor, la._step_in_cycle = la._clone_state((x, y)), 0
                eigs = jacobian_eigs(prob)
                cc = choose_modal_params_weighted(
                    eigs, WeightedModalConfig(gamma=args.gamma, kmin=5, kmax=2000)
                )
                la.k, la.alpha = int(cc.k), float(cc.alpha)
            m["state"] = (x, y)

        # --- EG ---
        if not args.no_eg:
            m = methods["Extragradient"]
            eg, (x, y) = m["obj"], m["state"]
            x, y = eg.step(prob, x, y)
            m["traj"].append(prob.distance(x, y))
            m["state"] = (x, y)

        # --- OGDA ---
        if not args.no_ogda:
            m = methods["OGDA"]
            og, (x, y) = m["obj"], m["state"]
            x, y = og.step(prob, x, y)
            m["traj"].append(prob.distance(x, y))
            m["state"] = (x, y)

    # --- Plot ---
    curves = [(name, m["traj"]) for name, m in methods.items()]
    title = (
        f"SC–SC (per-iteration): d={args.d}, μx={args.mu_x}, μy={args.mu_y}, γ={args.gamma}, "
        f"σ∈[{args.sigma_min},{args.sigma_max}]"
    )
    plot_distance_curves(curves, title, args.outfile)


if __name__ == "__main__":
    main()
