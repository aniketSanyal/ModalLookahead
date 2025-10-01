#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import torch
from torch import optim as torch_optim
import copy

  
# adding src to the system path
sys.path.insert(0, '/Users/baraah/Downloads/ModalLookahead/src/')
  
from modal_lookahead.problems.scsc import ScScConfig, ScScQuadratic
from modal_lookahead.optim.lookahead import (
    AdaptiveModalLookahead, AdaptiveLAConfig, WeightedModalConfig,
    FixedLookahead, choose_modal_params_weighted,
)
from modal_lookahead.optim.first_order import Extragradient, OGDA, GD
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
    ap.add_argument("--no_sgd", action="store_true")
    ap.add_argument("--no_adam", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prob = build_problem(args.d, args.mu_x, args.mu_y, args.gamma,
                         args.sigma_min, args.sigma_max, args.seed)

    # Initial state
    x0 = rng.normal(size=(prob.d,)) * 5.0
    y0 = rng.normal(size=(prob.d,)) * 5.0

    methods = {}
    if not args.no_fixed:
        methods[f"Fixed LA (k={args.fixed_k}, α={args.fixed_alpha})"] = {
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

    if not args.no_sgd:
        methods["GD"] = {
            "obj": GD(gamma=args.gamma),
            "state": (x0.copy(), y0.copy()),
            "traj": [],
        }
    
    if not args.no_adam:
        a = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(y0, dtype=torch.float32, requires_grad=True) 
        opt_x = torch.optim.Adam([a], lr=args.gamma)                    # minimize
        opt_y = torch.optim.Adam([b], lr=args.gamma)     # ascent
        methods["Adam"] = {
            "opt_x": opt_x,
            "opt_y": opt_y,
            "state": (a,b),
            "traj": [],
        }

    def gda_step(x, y):
        return prob.step_gda(x, y)

    total_steps = int(args.T)
    tensor_A = torch.as_tensor(prob.A, dtype=torch.float32)
    for it in range(total_steps):
        # --- Fixed LA ---
        if not args.no_fixed:
            m = methods[f"Fixed LA (k={args.fixed_k}, α={args.fixed_alpha})"]
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

        # --- SGD ---
        if not args.no_sgd:
            m = methods["GD"]
            sgd, (x, y) = m["obj"], m["state"]
            x, y = sgd.step(prob, x, y)
            m["traj"].append(prob.distance(x, y))
            m["state"] = (x, y)

        # --- Adam ---
        if not args.no_adam:
            m = methods["Adam"]
            (x, y) = m["state"]
            opt_x, opt_y = m["opt_x"], m["opt_y"]
            opt_x.zero_grad()
            opt_y.zero_grad()
            val = x @ tensor_A @ y               # f(x, y)
            gx, gy = torch.autograd.grad(val, (x, y))
            x.grad = gx
            y.grad = -gy                    # computes grads for both x and y
            opt_x.step()                  # x ← x - Adam(∇x f)
            opt_y.step()                  # y ← y + Adam(∇y f)  (because maximize=True)
            with torch.no_grad():
                m["traj"].append(prob.distance(x.detach().cpu().numpy(),
                                   y.detach().cpu().numpy()))
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
