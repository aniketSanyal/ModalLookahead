#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import torch
import copy
from time import time, process_time
  
# adding src to the system path
sys.path.insert(0, '/Users/baraah/Downloads/ModalLookahead/src/')
  
from modal_lookahead.problems.bilinear import BilinearGame
from modal_lookahead.optim.lookahead import (
    AdaptiveModalLookahead, AdaptiveLAConfig, ModalConfig,
    FixedLookahead, choose_modal_params_weighted_from_jacobian_eigs_weighted,
    choose_modal_params_dominant_from_jacobian_eigs_dom, choose_modal_params_from_jacobian_eigs_grid,
)
from modal_lookahead.optim.first_order import Extragradient, OGDA, GD
from modal_lookahead.plotting.plots import plot_distance_curves, plot_distances_wc_time


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
    ap.add_argument("--no_modal_weighted", action="store_true")
    ap.add_argument("--no_modal_dom", action="store_true")
    ap.add_argument("--no_modal_grid", action="store_true")
    ap.add_argument("--no_fixed", action="store_true")
    ap.add_argument("--no_sgd", action="store_true")
    ap.add_argument("--no_adam", action="store_true")
    ap.add_argument("--no_adamla", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prob = build_problem(args.d, args.beta, args.gamma, args.seed)

    # Initial state
    x0 = rng.normal(size=(prob.d,)) * 5.0
    y0 = rng.normal(size=(prob.d,)) * 5.0

    methods = {}
    if not args.no_fixed:
        methods[f"Fixed LA-GD (k={args.fixed_k}, α={args.fixed_alpha})"] = {
            "obj": FixedLookahead(k=args.fixed_k, alpha=args.fixed_alpha),
            "state": (x0.copy(), y0.copy()),
            "traj": [],
            "wc_times": [],
        }
    if not args.no_modal_weighted:
        la = AdaptiveModalLookahead(
            AdaptiveLAConfig(
                init_k=args.fixed_k,
                init_alpha=args.fixed_alpha,
                # weighted=ModalConfig(
                #     gamma=args.gamma, kmin=5, kmax=2000, alpha_grid=None
                # ),
            )
        )
        methods["Modal LA (dynamic) weighted"] = {"obj": la, "state": (x0.copy(), y0.copy()), "traj": [], "wc_times": []}

    if not args.no_modal_dom:
        la = AdaptiveModalLookahead(
            AdaptiveLAConfig(
                init_k=args.fixed_k,
                init_alpha=args.fixed_alpha,
                # dominant=ModalConfig(
                #     gamma=args.gamma, kmin=5, kmax=2000, alpha_grid=None
                # ),
            )
        )
        methods["Modal LA (dynamic) dom"] = {"obj": la, "state": (x0.copy(), y0.copy()), "traj": [], "wc_times": []}

    if not args.no_modal_grid:
        la = AdaptiveModalLookahead(
            AdaptiveLAConfig(
                init_k=args.fixed_k,
                init_alpha=args.fixed_alpha,
                modal=ModalConfig(
                    gamma=args.gamma, kmin=5, kmax=2000, alpha_grid=None
                ),
            )
        )
        methods["Modal LA (dynamic) grid"] = {"obj": la, "state": (x0.copy(), y0.copy()), "traj": [], "wc_times": []}

    if not args.no_eg:
        methods["Extragradient"] = {
            "obj": Extragradient(gamma=args.gamma),
            "state": (x0.copy(), y0.copy()),
            "traj": [],
            "wc_times": [],
        }
    if not args.no_ogda:
        methods["OGDA"] = {
            "obj": OGDA(gamma=args.gamma),
            "state": (x0.copy(), y0.copy()),
            "traj": [],
            "wc_times": [],
        }

    if not args.no_sgd:
        methods["GD"] = {
            "obj": GD(gamma=args.gamma),
            "state": (x0.copy(), y0.copy()),
            "traj": [],
            "wc_times": [],
        }

    if not args.no_adam:
        a = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(y0, dtype=torch.float32, requires_grad=True) 
        opt_x = torch.optim.Adam([a], lr=args.gamma)                    # minimize
        opt_y = torch.optim.Adam([b], lr=args.gamma, maximize=True)       # ascent
        methods["adam"] = {
            "opt_x": opt_x,
            "opt_y": opt_y,
            "state": (a,b),
            "traj": [],
            "wc_times": [],
        }

    if not args.no_adamla:
        a = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(y0, dtype=torch.float32, requires_grad=True) 
        opt_x = torch.optim.Adam([a], lr=args.gamma)                    # minimize
        opt_y = torch.optim.Adam([b], lr=args.gamma, maximize=True)    # ascent
        methods[f"Fixed LA-adam (k={args.fixed_k}, α={args.fixed_alpha})"] = {
            "obj": FixedLookahead(k=args.fixed_k, alpha=args.fixed_alpha),
            "opt_x": opt_x,
            "opt_y": opt_y,
            "state": (a,b),
            "traj": [],
            "wc_times": [],
        }

    def gda_step(x, y):
        return prob.step_gda(x, y)

    total_steps = int(args.T)
    tensor_A = torch.as_tensor(prob.A, dtype=torch.float32)
    
    if not args.no_fixed:
        fixed_t1=process_time()
        m = methods[f"Fixed LA-GD (k={args.fixed_k}, α={args.fixed_alpha})" ]
        m["wc_times"].append(fixed_t1)
        
        fl, (x, y) = m["obj"], m["state"]
        m["traj"].append(prob.distance(x, y))
        for i in range(total_steps):
            if getattr(fl, "_anchor", None) is None:
                fl._anchor, fl._step_in_cycle = fl._clone_state((x, y)), 0
            x, y = gda_step(x, y)
            fl._step_in_cycle += 1
            m["traj"].append(prob.distance(x, y))
            if fl._step_in_cycle >= fl.k:
                x, y = ((1 - fl.alpha) * fl._anchor[0] + fl.alpha * x,
                        (1 - fl.alpha) * fl._anchor[1] + fl.alpha * y)
                fl._anchor, fl._step_in_cycle = fl._clone_state((x, y)), 0
            # m["state"] = (x, y)
            fixed_t2=process_time()
            m["wc_times"].append(fixed_t2 - fixed_t1)

    if not args.no_modal_weighted:
        modal_t1=process_time()
        m = methods["Modal LA (dynamic) weighted"]
        m["wc_times"].append(modal_t1)
        eigs = jacobian_eigs(prob.A)
        cc = choose_modal_params_weighted_from_jacobian_eigs_weighted(
            eigs, ModalConfig(gamma=args.gamma, kmin=5, kmax=2000),  stability_all_modes=True, score_all_modes=True
        )
        
        la, (x, y) = m["obj"], m["state"]
        la.k, la.alpha = int(cc.k), float(cc.alpha)
        m["traj"].append(prob.distance(x, y))
        for i in range(total_steps):
            if getattr(la, "_anchor", None) is None:
                la._anchor, la._step_in_cycle = la._clone_state((x, y)), 0
            x, y = gda_step(x, y)
            la._step_in_cycle += 1
            m["traj"].append(prob.distance(x, y))
            if la._step_in_cycle >= la.k:
                x, y = ((1 - la.alpha) * la._anchor[0] + la.alpha * x,
                        (1 - la.alpha) * la._anchor[1] + la.alpha * y)
                la._anchor, la._step_in_cycle = la._clone_state((x, y)), 0
                
            #m["state"] = (x, y)
            modal_t2=process_time()
            m["wc_times"].append(modal_t2-modal_t1)

    if not args.no_modal_dom:
        modal_t1=process_time()
        m = methods["Modal LA (dynamic) dom"]
        m["wc_times"].append(modal_t1)
        eigs = jacobian_eigs(prob.A)
        cc = choose_modal_params_dominant_from_jacobian_eigs_dom(
            eigs, ModalConfig(gamma=args.gamma, kmin=5, kmax=2000), stability_all_modes=True, score_all_modes=True
        )
        
        la, (x, y) = m["obj"], m["state"]
        la.k, la.alpha = int(cc.k), float(cc.alpha)
        m["traj"].append(prob.distance(x, y))
        for i in range(total_steps):
            if getattr(la, "_anchor", None) is None:
                la._anchor, la._step_in_cycle = la._clone_state((x, y)), 0
            x, y = gda_step(x, y)
            la._step_in_cycle += 1
            m["traj"].append(prob.distance(x, y))
            if la._step_in_cycle >= la.k:
                x, y = ((1 - la.alpha) * la._anchor[0] + la.alpha * x,
                        (1 - la.alpha) * la._anchor[1] + la.alpha * y)
                la._anchor, la._step_in_cycle = la._clone_state((x, y)), 0
                
            #m["state"] = (x, y)
            modal_t2=process_time()
            m["wc_times"].append(modal_t2-modal_t1)

    if not args.no_modal_grid:
        modal_t1=process_time()
        m = methods["Modal LA (dynamic) grid"]
        m["wc_times"].append(modal_t1)
        eigs = jacobian_eigs(prob.A)
        cc = choose_modal_params_from_jacobian_eigs_grid(
            eigs, ModalConfig(gamma=args.gamma, kmin=5, kmax=2000)
        )
        
        la, (x, y) = m["obj"], m["state"]
        la.k, la.alpha = int(cc.k), float(cc.alpha)
        m["traj"].append(prob.distance(x, y))
        for i in range(total_steps):
            if getattr(la, "_anchor", None) is None:
                la._anchor, la._step_in_cycle = la._clone_state((x, y)), 0
            x, y = gda_step(x, y)
            la._step_in_cycle += 1
            m["traj"].append(prob.distance(x, y))
            if la._step_in_cycle >= la.k:
                x, y = ((1 - la.alpha) * la._anchor[0] + la.alpha * x,
                        (1 - la.alpha) * la._anchor[1] + la.alpha * y)
                la._anchor, la._step_in_cycle = la._clone_state((x, y)), 0
                
            #m["state"] = (x, y)
            modal_t2=process_time()
            m["wc_times"].append(modal_t2-modal_t1)

    if not args.no_eg:
        eg_t1=process_time()
        m = methods["Extragradient"]
        m["wc_times"].append(eg_t1)
        
        eg, (x, y) = m["obj"], m["state"]
        m["traj"].append(prob.distance(x, y))
        for i in range(total_steps):
            x, y = eg.step(prob, x, y)
            m["traj"].append(prob.distance(x, y))
            # m["state"] = (x, y)
            eg_t2=process_time()
            m["wc_times"].append(eg_t2 - eg_t1)

    if not args.no_ogda:
        og_t1=process_time()
        m = methods["OGDA"]
        m["wc_times"].append(og_t1)
        
        og, (x, y) = m["obj"], m["state"]
        m["traj"].append(prob.distance(x, y))
        for i in range(total_steps):
            x, y = og.step(prob, x, y)
            m["traj"].append(prob.distance(x, y))
            #m["state"] = (x, y)
            og_t2=process_time()
            m["wc_times"].append(og_t2 - og_t1)

    if not args.no_sgd:
        gd_t1=process_time()
        m = methods["GD"]
        m["wc_times"].append(gd_t1)
        sgd, (x, y) = m["obj"], m["state"]
        m["traj"].append(prob.distance(x, y))
        for i in range(total_steps):
            x, y = sgd.step(prob, x, y)
            m["traj"].append(prob.distance(x, y))
            #m["state"] = (x, y)
            gd_t2=process_time()
            m["wc_times"].append(gd_t2 - gd_t1)

    if not args.no_adam:
        ada_t1=process_time()
        m = methods["adam"]
        m["wc_times"].append(ada_t1)
        
        (x, y) = m["state"]
        m["traj"].append(prob.distance(x.detach().cpu().numpy(),
                                    y.detach().cpu().numpy()))
        for i in range(total_steps):
            opt_x, opt_y = m["opt_x"], m["opt_y"]
            opt_x.zero_grad(set_to_none=True)
            opt_y.zero_grad(set_to_none=True)

            val = x @ tensor_A @ y                     # f(x,y) = x^T A y
            val.backward()                      # sets x.grad = ∇_x f, y.grad = ∇_y f

            opt_x.step()                        # x ← x − Adam(∇_x f)
            opt_y.step()                        # y ← y + Adam(∇_y f) because maximize=True

            with torch.no_grad():
                m["traj"].append(prob.distance(x.detach().cpu().numpy(),
                                    y.detach().cpu().numpy()))
            #m["state"] = (x, y)
            ada_t2=process_time()
            m["wc_times"].append(ada_t2 - ada_t1)

        if not args.no_adamla:
            adamla_t1=process_time()
            m = methods[f"Fixed LA-adam (k={args.fixed_k}, α={args.fixed_alpha})" ]
            m["wc_times"].append(adamla_t1)
            
            fl, (x, y) = m["obj"], m["state"]
            m["traj"].append(prob.distance(x.detach().cpu().numpy(),
                                    y.detach().cpu().numpy()))
            for i in range(total_steps):
                if getattr(fl, "_anchor", None) is None:
                    fl._anchor, fl._step_in_cycle = fl._clone_state((x.detach().clone(), y.detach().clone())), 0
                opt_x, opt_y = m["opt_x"], m["opt_y"]
                opt_x.zero_grad(set_to_none=True)
                opt_y.zero_grad(set_to_none=True)

                val = x @ tensor_A @ y                     # f(x,y) = x^T A y
                val.backward()                      # sets x.grad = ∇_x f, y.grad = ∇_y f

                opt_x.step()                        # x ← x − Adam(∇_x f)
                opt_y.step()                        # y ← y + Adam(∇_y f) because maximize=True

                with torch.no_grad():
                    m["traj"].append(prob.distance(x.detach().cpu().numpy(),
                                    y.detach().cpu().numpy()))
                fl._step_in_cycle += 1
                if fl._step_in_cycle >= fl.k:
                    with torch.no_grad():
                     # x = (1 - alpha) * anchor_x + alpha * x
                        x.mul_(fl.alpha).add_(fl._anchor[0], alpha=1.0 - fl.alpha)
                        y.mul_(fl.alpha).add_(fl._anchor[1], alpha=1.0 - fl.alpha)
                    fl._anchor, fl._step_in_cycle = fl._clone_state((x.detach().clone(), y.detach().clone())), 0
                # m["state"] = (x, y)
                adamla_t2=process_time()
                m["wc_times"].append(adamla_t2 - adamla_t1)

    curves = [(name, m["traj"]) for name, m in methods.items()]
    curves_wc = [(name, m["traj"], m["wc_times"]) for name, m in methods.items() if "wc_times" in m]
    title = f"Bilinear (per-iteration): d={args.d}, β={args.beta}, γ={args.gamma}"
    plot_distance_curves(curves, title, args.outfile)
    title_wc = f"Bilinear (wall-clock time): d={args.d}, β={args.beta}, γ={args.gamma}"
    plot_distances_wc_time(curves_wc, title_wc, args.outfile+"_wctime")


if __name__ == "__main__":
    main()
