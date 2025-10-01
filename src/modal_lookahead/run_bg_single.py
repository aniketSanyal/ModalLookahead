#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# If running from elsewhere, uncomment to make local imports work:
# HERE = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, HERE)

from optim.first_order import Extragradient, OGDA
from optim.lookahead import (
    FixedLookahead,
    AdaptiveModalLookahead,
    ModalUpdatedLookahead,
    AdaptiveLAConfig,
    UpdatedLAConfig,
    ModalConfig,
)

# --------------------- Problem setup ---------------------
class BilinearProblem:
    def __init__(self, A: np.ndarray):
        self.A = A

def build_matrix_A(d: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, d)) / np.sqrt(d)
    A *= beta
    return A

def measure(x: np.ndarray, y: np.ndarray, mode: str = "norm") -> float:
    if mode == "norm2":
        return float(np.dot(x, x) + np.dot(y, y))
    return float(np.sqrt(np.dot(x, x) + np.dot(y, y)))

def gda_step(problem: BilinearProblem, x: np.ndarray, y: np.ndarray, gamma: float):
    A = problem.A
    gx = A @ y
    gy = A.T @ x
    return x - gamma * gx, y + gamma * gy

# ----------------- Modal eigs providers (± i σ_i) -----------------
def eigs_all(A: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(A, compute_uv=False)
    return np.concatenate([1j * s, -1j * s]).astype(np.complex128)

def eigs_top(A: np.ndarray) -> np.ndarray:
    s_max = float(np.max(np.linalg.svd(A, compute_uv=False)))
    return np.array([1j * s_max, -1j * s_max], dtype=np.complex128)

def eigs_threshold(A: np.ndarray, gamma: float, tol: float) -> np.ndarray:
    s = np.linalg.svd(A, compute_uv=False)
    lam = np.concatenate([1j * s, -1j * s]).astype(np.complex128)
    T = 1.0 - gamma * lam
    mask = np.abs(T - 1.0) > tol
    if not np.any(mask):
        s_max = float(np.max(s))
        lam = np.array([1j * s_max, -1j * s_max], dtype=np.complex128)
    else:
        lam = lam[mask]
    return lam

def get_modal_eigs(A: np.ndarray, gamma: float, mode: str, tol: float) -> np.ndarray:
    if mode == "top":
        return eigs_top(A)
    if mode == "threshold":
        return eigs_threshold(A, gamma=gamma, tol=tol)
    return eigs_all(A)

# ---------------------- Runners (return xs, ys) -------------------------
def run_fixed_lookahead(T, x0, y0, gamma, problem, k, alpha, scale):
    la = FixedLookahead(k=k, alpha=alpha)
    state = (x0.copy(), y0.copy())

    xs, ys = [0.0], [measure(state[0], state[1], scale)]  # include t=0
    step_ctr = 0.0

    for t in range(T):
        def on_point(st, phase):
            nonlocal step_ctr
            if phase == "base":
                step_ctr += 1.0
                xs.append(step_ctr)
                ys.append(measure(st[0], st[1], scale))
            elif phase == "avg":
                xs.append(step_ctr + 0.5)  # averaging snapshot
                ys.append(measure(st[0], st[1], scale))

        state = la.step_cycle(
            state,
            inner_step=lambda st: gda_step(problem, st[0], st[1], gamma),
            on_point=on_point,
        )
        logger.info(f"[Fixed LA] iter {t+1}/{T} done (k={la.k}, alpha={la.alpha:.4f})")
    return np.asarray(xs), np.asarray(ys)


def run_modal_adaptive(T, x0, y0, gamma, problem, cfg_modal, eigs: np.ndarray, scale):
    la = AdaptiveModalLookahead(
        AdaptiveLAConfig(init_k=cfg_modal.kmin, init_alpha=0.5, modal=cfg_modal)
    )
    state = (x0.copy(), y0.copy())

    xs, ys = [0.0], [measure(state[0], state[1], scale)]
    step_ctr = 0.0

    for t in range(T):
        def on_point(st, phase):
            nonlocal step_ctr
            if phase == "base":
                step_ctr += 1.0
                xs.append(step_ctr)
                ys.append(measure(st[0], st[1], scale))
            elif phase == "avg":
                xs.append(step_ctr + 0.5)
                ys.append(measure(st[0], st[1], scale))

        state = la.step_cycle(
            state,
            inner_step=lambda st: gda_step(problem, st[0], st[1], gamma),
            jacobian_eigs_fn=lambda: eigs,
            on_point=on_point,
        )
        logger.info(f"[Modal Adaptive] iter {t+1}/{T} done (k={la.k}, alpha={la.alpha:.4f})")
    return np.asarray(xs), np.asarray(ys)


def run_modal_updated(T, x0, y0, gamma, problem, cfg_modal, eigs: np.ndarray, scale):
    la = ModalUpdatedLookahead(
        UpdatedLAConfig(modal=cfg_modal, fallback_k=cfg_modal.kmin, fallback_alpha=0.5)
    )
    state = (x0.copy(), y0.copy())

    xs, ys = [0.0], [measure(state[0], state[1], scale)]
    step_ctr = 0.0

    for t in range(T):
        def on_point(st, phase):
            nonlocal step_ctr
            if phase == "base":
                step_ctr += 1.0
                xs.append(step_ctr)
                ys.append(measure(st[0], st[1], scale))
            elif phase == "avg":
                xs.append(step_ctr + 0.5)
                ys.append(measure(st[0], st[1], scale))

        state = la.step_cycle(
            state,
            inner_step=lambda st: gda_step(problem, st[0], st[1], gamma),
            jacobian_eigs_fn=lambda: eigs,
            on_point=on_point,
        )
        logger.info(f"[Modal Updated] iter {t+1}/{T} done (k={la.k}, alpha={la.alpha:.4f})")
    return np.asarray(xs), np.asarray(ys)


def run_extragradient(T, x0, y0, gamma, problem, scale):
    eg = Extragradient(gamma=gamma)
    x, y = x0.copy(), y0.copy()
    xs, ys = [0.0], [measure(x, y, scale)]
    for t in range(T):
        x, y = eg.step(problem, x, y)
        xs.append(xs[-1] + 1.0)
        ys.append(measure(x, y, scale))
        logger.info(f"[EG] iter {t+1}/{T} done")
    return np.asarray(xs), np.asarray(ys)


def run_ogda(T, x0, y0, gamma, problem, scale):
    og = OGDA(gamma=gamma)
    x, y = x0.copy(), y0.copy()
    xs, ys = [0.0], [measure(x, y, scale)]
    for t in range(T):
        x, y = og.step(problem, x, y)
        xs.append(xs[-1] + 1.0)
        ys.append(measure(x, y, scale))
        logger.info(f"[OGDA] iter {t+1}/{T} done")
    return np.asarray(xs), np.asarray(ys)


# ---------------------- Plot helper (uses xs, ys) ---------------------
LINEWIDTH = 2.2
MARKER_SIZE = 2.8
MARKER_EDGE_WIDTH = 0.6
ALPHA_LINE = 0.95
METHOD_COLORS = {
    "Fixed LA (k=5, α=0.5)": "#1f77b4",
    "Modal LA (adaptive)":    "#6a0dad",
    "Modal LA (updated)":     "#2ca02c",
    "Extragradient (EG)":     "#ff7f0e",
    "OGDA":                   "#8c564b",
}
MARKERS = ["o", "v", "^", "s", "d", "P", "X", ">"]

def plot_curves(curves, title, outfile, xlabel, ylabel, logy):
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    for i, (label, xs, ys) in enumerate(curves):
        color = METHOD_COLORS.get(label, f"C{i}")
        marker = MARKERS[i % len(MARKERS)]
        if logy:
            ax.semilogy(
                xs, np.maximum(1e-300, ys),
                lw=LINEWIDTH, color=color, alpha=ALPHA_LINE,
                marker=marker, markersize=MARKER_SIZE,
                markevery=max(1, len(xs)//50 if len(xs) > 0 else 1),
                markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor="k",
                label=label,
            )
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            ax.yaxis.set_minor_locator(
                ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
            )
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.plot(
                xs, ys,
                lw=LINEWIDTH, color=color, alpha=ALPHA_LINE,
                marker=marker, markersize=MARKER_SIZE,
                markevery=max(1, len(xs)//50 if len(xs) > 0 else 1),
                markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor="k",
                label=label,
            )
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title + (" (log y)" if logy else " (linear y)"), fontsize=16)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13, width=1.5)
    ax.legend(fontsize=11, framealpha=0.9, labelspacing=0.2, borderaxespad=0.2)
    fig.tight_layout()
    base = outfile + ("_log" if logy else "_linear")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Single-run d-dimensional bilinear game (no CIs)."
    )
    ap.add_argument("--T", type=int, default=100, help="Outer iterations")
    ap.add_argument("--gamma", type=float, default=0.1, help="Base step size")
    ap.add_argument("--beta", type=float, default=0.6, help="Spectral scaling for A")
    ap.add_argument("--d", type=int, default=120, help="Dimension of x and y")
    ap.add_argument("--outfile", type=str, default="bilinear_single", help="Output file prefix")
    ap.add_argument("--scale", choices=["norm", "norm2"], default="norm")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")

    # Lookahead params
    ap.add_argument("--fixed_k", type=int, default=5)
    ap.add_argument("--fixed_alpha", type=float, default=0.5)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=512)

    # Modal eigs selection mode
    ap.add_argument("--eigs_mode", choices=["all", "top", "threshold"], default="all")
    ap.add_argument("--eigs_tol", type=float, default=1e-3)

    # Method toggles
    ap.add_argument("--skip_fixed", action="store_true")
    ap.add_argument("--skip_modal", action="store_true")
    ap.add_argument("--skip_eg",    action="store_true")
    ap.add_argument("--skip_ogda",  action="store_true")

    # Use one-shot modal lookahead
    ap.add_argument("--use_modal_updated", action="store_true")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    A = build_matrix_A(args.d, args.beta, rng)
    problem = BilinearProblem(A)
    x0 = rng.normal(size=args.d)
    y0 = rng.normal(size=args.d)

    eigs = get_modal_eigs(A, gamma=args.gamma, mode=args.eigs_mode, tol=args.eigs_tol)

    curves = []

    if not args.skip_fixed:
        xs, ys = run_fixed_lookahead(
            args.T, x0, y0, args.gamma, problem,
            k=args.fixed_k, alpha=args.fixed_alpha, scale=args.scale
        )
        curves.append(("Fixed LA (k=5, α=0.5)", xs, ys))

    if not args.skip_modal:
        cfg_modal = ModalConfig(gamma=args.gamma, kmin=args.kmin, kmax=args.kmax)
        if args.use_modal_updated:
            xs, ys = run_modal_updated(args.T, x0, y0, args.gamma, problem, cfg_modal, eigs, args.scale)
            curves.append(("Modal LA (updated)", xs, ys))
        else:
            xs, ys = run_modal_adaptive(args.T, x0, y0, args.gamma, problem, cfg_modal, eigs, args.scale)
            curves.append(("Modal LA (adaptive)", xs, ys))

    if not args.skip_eg:
        xs, ys = run_extragradient(args.T, x0, y0, args.gamma, problem, args.scale)
        curves.append(("Extragradient (EG)", xs, ys))

    if not args.skip_ogda:
        xs, ys = run_ogda(args.T, x0, y0, args.gamma, problem, args.scale)
        curves.append(("OGDA", xs, ys))

    if len(curves) == 0:
        raise SystemExit("All methods were skipped. Enable at least one method.")

    ylabel = "Distance to the equilibrium" if args.scale == "norm" else r"Distance $\|[x;y]\|_2^2$"
    title  = f"Bilinear d={args.d}, β={args.beta}, γ={args.gamma} — single run"

    # Linear & log, both on the same base-step x-axis
    plot_curves(curves, title, args.outfile, xlabel="Base steps (avg at +0.5)", ylabel=ylabel, logy=False)
    plot_curves(curves, title, args.outfile, xlabel="Base steps (avg at +0.5)", ylabel=ylabel, logy=True)

    logger.info(f"Saved: {args.outfile}_linear.(png|pdf|svg) and {args.outfile}_log.(png|pdf|svg)")


if __name__ == "__main__":
    main()
