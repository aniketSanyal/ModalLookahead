#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------
# Plot style knobs (feel free to tweak)
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
        return float(x * x + y * y)
    return float(np.sqrt(x * x + y * y))


def _z_for_ci(ci):
    if abs(ci - 0.68) < 1e-9: return 1.0
    if abs(ci - 0.90) < 1e-9: return 1.645
    if abs(ci - 0.95) < 1e-9: return 1.96
    if abs(ci - 0.99) < 1e-9: return 2.576
    return 1.96


def _plot_with_ci(curves, title, outfile, xlabel, ylabel, logy, ci_level):
    """
    curves: list of tuples (label, runs, color, marker)
        runs is list[np.ndarray of shape (T,)]
    """
    fig, ax = plt.subplots(figsize=(10.8, 5.6))

    for (label, runs, color, marker) in curves:
        Y = np.stack(runs, axis=0)   # (n_runs, T)
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
# Bilinear game & methods
# f(x,y)=x*y, min in x, max in y
# Base step (GDA): x_{t+1}=x - γ y, y_{t+1}=y + γ x
# ---------------------------------------------------------------------
def gda_step(x, y, gamma):
    return x - gamma * y, y + gamma * x


def M_bilinear(gamma):
    # one-step linear map in z=[x;y]
    return np.array([[1.0, -gamma], [gamma, 1.0]], dtype=float)


def choose_modal_params(M, kmin=3, kmax=2000, alpha_grid=None, alpha_max=1.0):
    """Choose (k, alpha) using phase of complex eigenmode and α-grid minimization."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 1.0, 40)
    w = np.linalg.eigvals(M)
    # dominant complex mode phase (bilinear has conjugate pair)
    j = np.argmax(np.imag(w))
    theta = abs(np.angle(w[j]))
    theta = max(theta, 1e-9)
    k_star = int(np.clip(np.round(np.pi / theta), kmin, kmax))

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


def run_fixed_lookahead(T, x0, y0, gamma, k, alpha, scale):
    traj = []
    x, y = x0, y0
    ax, ay = x, y
    step = 0
    for t in range(T):
        x, y = gda_step(x, y, gamma)
        step += 1
        traj.append(_measure(x, y, scale))
        if step >= k:
            x = (1 - alpha) * ax + alpha * x
            y = (1 - alpha) * ay + alpha * y
            ax, ay = x, y
            step = 0
    return np.asarray(traj, dtype=float)


def run_modal_lookahead(T, x0, y0, gamma, k_init, alpha_init, scale, alpha_max=1.0):
    traj = []
    x, y = x0, y0
    ax, ay = x, y
    k, alpha = k_init, alpha_init
    step = 0
    M = M_bilinear(gamma)
    for t in range(T):
        x, y = gda_step(x, y, gamma)
        step += 1
        traj.append(_measure(x, y, scale))
        if step >= k:
            x = (1 - alpha) * ax + alpha * x
            y = (1 - alpha) * ay + alpha * y
            ax, ay = x, y
            step = 0
            # update (k, alpha)
            k, alpha = choose_modal_params(M, kmin=3, kmax=2000, alpha_max=alpha_max)
    return np.asarray(traj, dtype=float)


# --- First-order baselines (self-contained for bilinear) ---
def eg_step(x, y, gamma):
    # ExtraGradient for bilinear
    x1 = x - gamma * y
    y1 = y + gamma * x
    x2 = x - gamma * y1
    y2 = y + gamma * x1
    return x2, y2


def ogda_step(x, y, gamma, state):
    # OGDA with simple "last gradient" memory in state
    # g_t = (y, -x)
    gx, gy = y, -x
    gx_prev, gy_prev = state
    x_new = x - gamma * (2 * gx - gx_prev)
    y_new = y + gamma * (2 * gy - gy_prev)
    return x_new, y_new, (gx, gy)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--outfile", type=str, default="bilinear_ci")
    ap.add_argument("--scale", choices=["norm", "norm2"], default="norm")
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--ci", type=float, default=0.95, choices=[0.68, 0.90, 0.95, 0.99])
    ap.add_argument("--fixed_k", type=int, default=5)
    ap.add_argument("--fixed_alpha", type=float, default=0.5)
    ap.add_argument("--alpha_max", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=120, help="Dimension of the bilinear game")
    ap.add_argument("--beta", type=float, default=0.6, help="Singular value scaling for the bilinear game")

    args = ap.parse_args()

    # Aggregate runs
    runs = {
        "Fixed LA (k=5, α=0.5)": [],
        "Modal LA (dynamic)": [],
        "Extragradient (EG)": [],
        "OGDA": [],
    }

    for r in range(args.n_runs):
        rng = np.random.default_rng(args.base_seed + r)
        # random initial condition per run
        x0 = float(rng.normal(scale=1.0))
        y0 = float(rng.normal(scale=1.0))

        # Fixed Lookahead
        runs["Fixed LA (k=5, α=0.5)"].append(
            run_fixed_lookahead(args.T, x0, y0, args.gamma, args.fixed_k, args.fixed_alpha, args.scale)
        )

        # Modal Lookahead
        runs["Modal LA (dynamic)"].append(
            run_modal_lookahead(args.T, x0, y0, args.gamma, args.fixed_k, args.fixed_alpha, args.scale, alpha_max=args.alpha_max)
        )

        # EG
        traj = []
        x, y = x0, y0
        for t in range(args.T):
            x, y = eg_step(x, y, args.gamma)
            traj.append(_measure(x, y, args.scale))
        runs["Extragradient (EG)"].append(np.asarray(traj, dtype=float))

        # OGDA
        traj = []
        x, y = x0, y0
        state = (0.0, 0.0)  # (gx_prev, gy_prev)
        for t in range(args.T):
            x, y, state = ogda_step(x, y, args.gamma, state)
            traj.append(_measure(x, y, args.scale))
        runs["OGDA"].append(np.asarray(traj, dtype=float))

    # Build curves list with colors/markers
    curves = []
    for i, (name, rr) in enumerate(runs.items()):
        color = METHOD_COLORS.get(name, f"C{i}")
        marker = MARKERS[i % len(MARKERS)]
        curves.append((name, rr, color, marker))

    ylabel = "Distance to the equilibrium" if args.scale == "norm" else r"Distance to equilibrium $\|[x;y]\|_2^2$"
    title = f"Bilinear: γ={args.gamma}  (mean ± {int(args.ci*100)}% CI, n={args.n_runs})"

    # Log and linear exports
    _plot_with_ci(curves, title, args.outfile, xlabel="Iterations (base steps)", ylabel=ylabel, logy=True,  ci_level=args.ci)
    _plot_with_ci(curves, title, args.outfile, xlabel="Iterations (base steps)", ylabel=ylabel, logy=False, ci_level=args.ci)


if __name__ == "__main__":
    main()
