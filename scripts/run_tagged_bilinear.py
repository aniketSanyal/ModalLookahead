#!/usr/bin/env python3
import argparse
import numpy as np

from modal_lookahead.problems.bilinear import BilinearGame
from modal_lookahead.optim.lookahead import (
    AdaptiveModalLookahead, AdaptiveLAConfig, WeightedModalConfig,
    FixedLookahead, choose_modal_params_weighted,
)
from modal_lookahead.optim.first_order import Extragradient, OGDA
from modal_lookahead.plotting.plots import plot_distance_curves


def build_problem(d: int, beta: float, gamma: float, seed: int) -> BilinearGame:
    rng = np.random.default_rng(seed)
    A = beta * rng.normal(size=(d, d)) / np.sqrt(d)
    return BilinearGame(A=A, gamma=gamma)


def jacobian_eigs(A: np.ndarray):
    d = A.shape[0]
    J = np.block([[np.zeros((d, d)), A.T], [-A, np.zeros((d, d))]])
    return np.linalg.eigvals(J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=120)
    ap.add_argument("--T", type=int, default=300, help="TOTAL base iterations")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed_k", type=int, default=5)
    ap.add_argument("--fixed_alpha", type=float, default=0.5)
    ap.add_argument("--outfile", type=str, default="bilinear_iterwise")
    ap.add_argument("--no_eg", action="store_true")
    ap.add_argument("--no_ogda", action="store_true")
    ap.add_argument("--no_modal", action="store_true")
    ap.add_argument("--no_fixed", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prob = build_problem(args.d, args.beta, args.gamma, args.seed)

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
                eigs = jacobian_eigs(prob.A)
                cc = choose_modal_params_weighted(
                    eigs, WeightedModalConfig(gamma=args.gamma, kmin=5, kmax=2000)
                )
                la.k, la.alpha = int(cc.k), float(cc.alpha)
            m["state"] = (x, y)

        if not args.no_eg:
            m = methods["Extragradient"]
            eg, (x, y) = m["obj"], m["state"]
            x, y = eg.step(prob, x, y)
            m["traj"].append(prob.distance(x, y))
            m["state"] = (x, y)

        if not args.no_ogda:
            m = methods["OGDA"]
            og, (x, y) = m["obj"], m["state"]
            x, y = og.step(prob, x, y)
            m["traj"].append(prob.distance(x, y))
            m["state"] = (x, y)

    curves = [(name, m["traj"]) for name, m in methods.items()]
    title = f"Bilinear (per-iteration): d={args.d}, β={args.beta}, γ={args.gamma}"
    plot_distance_curves(curves, title, args.outfile)


if __name__ == "__main__":
    main()
