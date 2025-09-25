#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterSciNotation

from problems.bilinear import BilinearGame
from optim.lookahead import (
    AdaptiveModalLookahead, AdaptiveLAConfig, WeightedModalConfig,
    FixedLookahead, choose_modal_params_weighted,
)
from optim.first_order import Extragradient, OGDA


# --------------------------- Aesthetic ----------------------
def _colors20():
    base = [
        (31,119,180),(174,199,232),(255,127,14),(255,187,120),
        (44,160,44),(152,223,138),(214,39,40),(255,152,150),
        (148,103,189),(197,176,213),(140,86,75),(196,156,148),
        (227,119,194),(247,182,210),(127,127,127),(199,199,199),
        (188,189,34),(219,219,141),(23,190,207),(158,218,229)
    ]
    return [(r/255., g/255., b/255.) for (r,g,b) in base]

MARKERS = ['o','v','^','<','>','8','s','d','h','H','+','x','X','D','d','|','_','p']
LINEWIDTH = 3
ALPHA_LINE = 0.85
MARKER_SIZE = 4
MARKER_EDGE_WIDTH = 0.8
USE_TEX = False

METHOD_COLORS = {
    "Fixed LA (k=5, α=0.5)": (31/255., 119/255., 180/255.),   # blue
    "Modal LA (dynamic)":     (214/255., 39/255., 40/255.),   # red
    "Extragradient":          (255/255., 127/255., 14/255.),  # orange
    "OGDA":                   (148/255., 103/255., 189/255.), # purple
}

def _apply_style():
    rc = plt.rcParams
    rc['figure.dpi'] = 190
    rc['savefig.dpi'] = 800
    rc['text.usetex'] = USE_TEX
    rc['font.family'] = 'lmodern' if USE_TEX else rc.get('font.family', 'sans-serif')
    rc['axes.spines.top'] = True
    rc['axes.spines.right'] = True

def _plot_curves_iterwise(curves, title, outfile, xlabel):
    _apply_style()
    colors = _colors20()

    def _plot(ax, logy=False):
        for i, (label, y) in enumerate(curves):
            x = np.arange(len(y))
            c = METHOD_COLORS.get(label, colors[i % len(colors)])
            m = MARKERS[i % len(MARKERS)]
            if logy:
                ax.semilogy(
                    x, y,
                    lw=LINEWIDTH, color=c, alpha=ALPHA_LINE,
                    marker=m, markersize=MARKER_SIZE,
                    markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor='k',
                    markevery=max(1, len(y)//50),
                    label=label,
                )
            else:
                ax.plot(
                    x, y,
                    lw=LINEWIDTH, color=c, alpha=ALPHA_LINE,
                    marker=m, markersize=MARKER_SIZE,
                    markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor='k',
                    markevery=max(1, len(y)//50),
                    label=label,
                )

        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel("Distance to the equilibrium", fontsize=18)
        ax.set_title(title + (" (log y)" if logy else " (linear y)"), fontsize=16)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=13, width=1.5)
        ax.legend(fontsize=11, borderaxespad=0., loc='best', labelspacing=0.2, framealpha=0.9)

    # --- Plot log scale ---
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _plot(ax, logy=True)
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig(f"{outfile}_log.{ext}", bbox_inches="tight")
    plt.close(fig)

    # --- Plot linear scale ---
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _plot(ax, logy=False)
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig(f"{outfile}_linear.{ext}", bbox_inches="tight")
    plt.close(fig)



# ------------------------------ Problem helpers --------------------------------
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
    ap.add_argument("--T", type=int, default=300)
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

    x0 = rng.normal(size=(prob.d,)) * 5.0
    y0 = rng.normal(size=(prob.d,)) * 5.0

    methods = {}
    if not args.no_fixed:
        methods["Fixed LA (k=5, α=0.5)"] = {"obj": FixedLookahead(k=args.fixed_k, alpha=args.fixed_alpha), "state": (x0.copy(), y0.copy()), "traj": []}
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
        methods["Extragradient"] = {"obj": Extragradient(gamma=args.gamma), "state": (x0.copy(), y0.copy()), "traj": []}
    if not args.no_ogda:
        methods["OGDA"] = {"obj": OGDA(gamma=args.gamma), "state": (x0.copy(), y0.copy()), "traj": []}

    def gda_step(x, y):
        return prob.step_gda(x, y)

    total_steps = int(args.T)
    for _ in range(total_steps):
        for name, m in methods.items():
            obj, (x, y) = m["obj"], m["state"]

            if isinstance(obj, FixedLookahead):
                if getattr(obj, "_anchor", None) is None:
                    obj._anchor, obj._step_in_cycle = obj._clone_state((x, y)), 0
                x, y = gda_step(x, y)
                obj._step_in_cycle += 1
                m["traj"].append(prob.distance(x, y))
                if obj._step_in_cycle >= obj.k:
                    x, y = ((1 - obj.alpha) * obj._anchor[0] + obj.alpha * x,
                            (1 - obj.alpha) * obj._anchor[1] + obj.alpha * y)
                    obj._anchor, obj._step_in_cycle = obj._clone_state((x, y)), 0
            elif isinstance(obj, AdaptiveModalLookahead):
                if getattr(obj, "_anchor", None) is None:
                    obj._anchor, obj._step_in_cycle = obj._clone_state((x, y)), 0
                x, y = gda_step(x, y)
                obj._step_in_cycle += 1
                m["traj"].append(prob.distance(x, y))
                if obj._step_in_cycle >= obj.k:
                    x, y = ((1 - obj.alpha) * obj._anchor[0] + obj.alpha * x,
                            (1 - obj.alpha) * obj._anchor[1] + obj.alpha * y)
                    obj._anchor, obj._step_in_cycle = obj._clone_state((x, y)), 0
                    eigs = jacobian_eigs(prob.A)
                    cc = choose_modal_params_weighted(
                        eigs, WeightedModalConfig(gamma=args.gamma, kmin=5, kmax=2000)
                    )
                    obj.k, obj.alpha = int(cc.k), float(cc.alpha)
            elif isinstance(obj, Extragradient) or isinstance(obj, OGDA):
                x, y = obj.step(prob, x, y)
                m["traj"].append(prob.distance(x, y))

            m["state"] = (x, y)

    curves = [(name, m["traj"]) for name, m in methods.items()]
    title = f"Bilinear (per-iteration): d={args.d}, β={args.beta}, γ={args.gamma}"
    _plot_curves_iterwise(curves, title, args.outfile, xlabel="Iterations (base steps)")


if __name__ == "__main__":
    main()
