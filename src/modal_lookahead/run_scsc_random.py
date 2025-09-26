#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from problems.scsc import ScScConfig, ScScQuadratic  # your local module

# ---------------------------------------------------------------------
# Plot style knobs
# ---------------------------------------------------------------------
LINEWIDTH = 2.4
MARKER_SIZE = 3.2
MARKER_EDGE_WIDTH = 0.6
ALPHA_LINE = 0.95
FILL_ALPHA = 0.18

METHOD_COLORS = {
    "Fixed LA (k=5, α=0.5)": "#1f77b4",  # blue
    "Modal LA (dynamic)":    "#6a0dad",  # purple
    "Extragradient (EG)":    "#ff7f0e",  # orange
    "OGDA":                  "#8c564b",  # brown/purple
}
MARKERS = ["o", "v", "^", "s", "d", "P", "X", ">"]


# ---------------------------------------------------------------------
# Helpers: metric, CI math, plotting with CI
# ---------------------------------------------------------------------
def _measure(x, y, mode="norm"):
    if mode == "norm2":
        return float(x @ x + y @ y)
    return float(np.sqrt(x @ x + y @ y))


def _z_for_ci(ci):
    if abs(ci - 0.68) < 1e-9: return 1.0
    if abs(ci - 0.90) < 1e-9: return 1.645
    if abs(ci - 0.95) < 1e-9: return 1.96
    if abs(ci - 0.99) < 1e-9: return 2.576
    return 1.96


def _plot_with_ci(curves, title, outfile, xlabel, ylabel, logy, ci_level):
    fig, ax = plt.subplots(figsize=(10.8, 5.6))

    for (label, runs, color, marker) in curves:
        Y = np.stack(runs, axis=0)
        T = Y.shape[1]
        x = np.arange(T)
        mean = Y.mean(axis=0)
        std = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
        z = _z_for_ci(ci_level)
        sem = std / max(1, np.sqrt(Y.shape[0]))
        lower = np.maximum(1e-300, mean - z * sem)
        upper = mean + z * sem

        if logy:
            ax.semilogy(
                x, mean, lw=LINEWIDTH, color=color, alpha=ALPHA_LINE,
                marker=marker, markersize=MARKER_SIZE,
                markevery=max(1, T // 50),
                markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor="k",
                label=label,
            )
            ax.fill_between(x, lower, upper, color=color, alpha=FILL_ALPHA, linewidth=0.0)
        else:
            ax.plot(
                x, mean, lw=LINEWIDTH, color=color, alpha=ALPHA_LINE,
                marker=marker, markersize=MARKER_SIZE,
                markevery=max(1, T // 50),
                markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor="k",
                label=label,
            )
            ax.fill_between(x, lower, upper, color=color, alpha=FILL_ALPHA, linewidth=0.0)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title + (" (log y)" if logy else " (linear y)"), fontsize=16)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13, width=1.5)
    if logy:
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.legend(fontsize=11, framealpha=0.9, labelspacing=0.2, borderaxespad=0.2)

    fig.tight_layout()
    base = outfile + ("_log" if logy else "_linear")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# SC–SC base dynamics (GDA) and spectral M
# f(x,y)= 0.5 μx ||x||^2 + x^T A y - 0.5 μy ||y||^2 (min in x, max in y)
# GDA step: x' = x - γ(μx x + A^T y),  y' = y + γ(A x - μy y)
# Linear one-step map on z=[x;y]:
#   M = [[I - γ μx I,   -γ A^T],
#        [ γ A,         I - γ μy I]]
# ---------------------------------------------------------------------
def gda_step_scsc(x, y, A, mu_x, mu_y, gamma):
    return x - gamma * (mu_x * x + A.T @ y), y + gamma * (A @ x - mu_y * y)


def M_scsc(A, mu_x, mu_y, gamma):
    d = A.shape[0]
    I = np.eye(d)
    top_left = (1.0 - gamma * mu_x) * I
    top_right = -gamma * A.T
    bot_left = gamma * A
    bot_right = (1.0 - gamma * mu_y) * I
    return np.block([[top_left, top_right],
                     [bot_left, bot_right]])


def choose_modal_params_from_M(M, kmin=3, kmax=2000, alpha_grid=None, alpha_max=1.0):
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 1.0, 40)
    w = np.linalg.eigvals(M)
    # find complex modes
    idx_c = np.where(np.imag(w) != 0)[0]
    if len(idx_c) == 0:
        # fallback: use largest magnitude real mode angle ~ 0
        theta_bar = 1e-9
    else:
        mags = np.abs(w[idx_c])
        thetas = np.abs(np.angle(w[idx_c]))
        j_dom = idx_c[np.argmax(mags)]
        # "divergent" = mag > 1 if any, else 2nd largest
        divergers = idx_c[mags > 1.0]
        if len(divergers) >= 1:
            j_div = divergers[np.argmax(np.abs(w[divergers]))]
        elif len(idx_c) > 1:
            order = idx_c[np.argsort(-mags)]
            j_div = order[1]
        else:
            j_div = j_dom
        Rdom, Rdiv = abs(w[j_dom]), abs(w[j_div])
        th_dom, th_div = abs(np.angle(w[j_dom])), abs(np.angle(w[j_div]))
        theta_bar = (Rdom * th_dom + Rdiv * th_div) / max(1e-12, (Rdom + Rdiv))

    k_star = int(np.clip(np.round(np.pi / max(theta_bar, 1e-9)), kmin, kmax))
    Mk = np.linalg.matrix_power(M, int(k_star))
    I = np.eye(M.shape[0])
    best_alpha, best_rho = None, np.inf
    for a in alpha_grid:
        if a > alpha_max:
            continue
        L = (1.0 - a) * I + a * Mk
        rho = max(abs(np.linalg.eigvals(L)))
        if rho < best_rho:
            best_rho, best_alpha = rho, a
    if best_alpha is None:
        best_alpha = min(1.0, alpha_max)
    return int(k_star), float(best_alpha)


# Baselines for SC–SC
def eg_step_scsc(x, y, A, mu_x, mu_y, gamma):
    # predictor
    x1 = x - gamma * (mu_x * x + A.T @ y)
    y1 = y + gamma * (A @ x - mu_y * y)
    # corrector
    x2 = x - gamma * (mu_x * x + A.T @ y1)
    y2 = y + gamma * (A @ x1 - mu_y * y)
    return x2, y2


def ogda_step_scsc(x, y, A, mu_x, mu_y, gamma, state):
    # current gradients
    gx = mu_x * x + A.T @ y
    gy = A @ x - mu_y * y
    gx_prev, gy_prev = state
    x_new = x - gamma * (2 * gx - gx_prev)
    y_new = y + gamma * (2 * gy - gy_prev)
    return x_new, y_new, (gx, gy)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=120)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--mu_x", type=float, default=0.5)
    ap.add_argument("--mu_y", type=float, default=0.5)
    ap.add_argument("--sigma_min", type=float, default=0.5)
    ap.add_argument("--sigma_max", type=float, default=0.5)
    ap.add_argument("--outfile", type=str, default="scsc_ci")
    ap.add_argument("--scale", choices=["norm", "norm2"], default="norm")
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--ci", type=float, default=0.95, choices=[0.68, 0.90, 0.95, 0.99])
    ap.add_argument("--fixed_k", type=int, default=5)
    ap.add_argument("--fixed_alpha", type=float, default=0.5)
    ap.add_argument("--alpha_max", type=float, default=1.0)
    args = ap.parse_args()

    runs = {
        "Fixed LA (k=5, α=0.5)": [],
        "Modal LA (dynamic)": [],
        "Extragradient (EG)": [],
        "OGDA": [],
    }

    for r in range(args.n_runs):
        seed = args.base_seed + r
        # Build problem
        cfg = ScScConfig(
            d=args.d,
            gamma=args.gamma,
            mu_x=args.mu_x,
            mu_y=args.mu_y,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            seed=seed,
        )
        prob = ScScQuadratic(cfg)

        rng = np.random.default_rng(seed)
        x = rng.normal(size=(prob.d,)) * 1.0
        y = rng.normal(size=(prob.d,)) * 1.0

        # Fixed LA
        xf, yf = x.copy(), y.copy()
        ax, ay = xf.copy(), yf.copy()
        step = 0
        traj = []
        for t in range(args.T):
            xf, yf = gda_step_scsc(xf, yf, prob.A, prob.mu_x, prob.mu_y, args.gamma)
            step += 1
            traj.append(_measure(xf, yf, args.scale))
            if step >= args.fixed_k:
                xf = (1 - args.fixed_alpha) * ax + args.fixed_alpha * xf
                yf = (1 - args.fixed_alpha) * ay + args.fixed_alpha * yf
                ax, ay = xf.copy(), yf.copy()
                step = 0
        runs["Fixed LA (k=5, α=0.5)"].append(np.asarray(traj, dtype=float))

        # Modal LA
        xm, ym = x.copy(), y.copy()
        axm, aym = xm.copy(), ym.copy()
        k, alpha = args.fixed_k, args.fixed_alpha
        step = 0
        traj = []
        for t in range(args.T):
            xm, ym = gda_step_scsc(xm, ym, prob.A, prob.mu_x, prob.mu_y, args.gamma)
            step += 1
            traj.append(_measure(xm, ym, args.scale))
            if step >= k:
                xm = (1 - alpha) * axm + alpha * xm
                ym = (1 - alpha) * aym + alpha * ym
                axm, aym = xm.copy(), ym.copy()
                step = 0
                # update (k, alpha) from local linear map
                M = M_scsc(prob.A, prob.mu_x, prob.mu_y, args.gamma)
                k, alpha = choose_modal_params_from_M(M, kmin=3, kmax=2000, alpha_max=args.alpha_max)
        runs["Modal LA (dynamic)"].append(np.asarray(traj, dtype=float))

        # EG
        xe, ye = x.copy(), y.copy()
        traj = []
        for t in range(args.T):
            xe, ye = eg_step_scsc(xe, ye, prob.A, prob.mu_x, prob.mu_y, args.gamma)
            traj.append(_measure(xe, ye, args.scale))
        runs["Extragradient (EG)"].append(np.asarray(traj, dtype=float))

        # OGDA
        xo, yo = x.copy(), y.copy()
        state = (np.zeros_like(x), np.zeros_like(y))
        traj = []
        for t in range(args.T):
            xo, yo, state = ogda_step_scsc(xo, yo, prob.A, prob.mu_x, prob.mu_y, args.gamma, state)
            traj.append(_measure(xo, yo, args.scale))
        runs["OGDA"].append(np.asarray(traj, dtype=float))

    # Build curves list
    curves = []
    for i, (name, rr) in enumerate(runs.items()):
        color = METHOD_COLORS.get(name, f"C{i}")
        marker = MARKERS[i % len(MARKERS)]
        curves.append((name, rr, color, marker))

    ylabel = "Distance to the equilibrium" if args.scale == "norm" else r"Distance to equilibrium $\|[x;y]\|_2^2$"
    title = (
        f"SC–SC: d={args.d}, μx={args.mu_x}, μy={args.mu_y}, γ={args.gamma}, "
        f"σ∈[{args.sigma_min},{args.sigma_max}]  (mean ± {int(args.ci*100)}% CI, n={args.n_runs})"
    )

    # Export plots
    _plot_with_ci(curves, title, args.outfile, xlabel="Iterations (base steps)", ylabel=ylabel, logy=True,  ci_level=args.ci)
    _plot_with_ci(curves, title, args.outfile, xlabel="Iterations (base steps)", ylabel=ylabel, logy=False, ci_level=args.ci)


if __name__ == "__main__":
    main()
