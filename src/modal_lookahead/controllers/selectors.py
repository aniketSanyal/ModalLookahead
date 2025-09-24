# src/modal_lookahead/controllers/selectors.py
"""
Modal parameter selection helpers for Lookahead.

This module chooses (k, alpha) by *full grid search* each macro step:
1) Sweep alpha in (0,1], discretized.
2) For each alpha, sweep integer k in [k_min, k_max].
3) Enforce stability alpha <= alpha_max(k).
4) Pick the (k, alpha) pair that minimizes modal contraction rho.

It expects a spectral estimator that returns the dominant one-step mode
z = R * exp(-i * theta) of the inner map near the current iterate.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Callable, Dict
import numpy as np

# ---------------------------------------------------------------------
# Core modal math
# ---------------------------------------------------------------------

def alpha_max_of_k(R: float, theta: float, k: int) -> float:
    """Critical averaging for stability at integer k (capped to 1)."""
    s = R ** k
    c = math.cos(k * theta)
    denom = 1.0 + s * s - 2.0 * s * c
    if denom <= 1e-14:
        return 1.0
    amax = 2.0 * (1.0 - s * c) / denom
    return max(0.0, min(1.0, amax))

def rho_modal(R: float, theta: float, k: int, alpha: float) -> float:
    """Modal contraction |(1 - alpha) + alpha * (R^k e^{-ik theta})|."""
    s = R ** k
    re = (1.0 - alpha) + alpha * s * math.cos(k * theta)
    im = -alpha * s * math.sin(k * theta)
    return math.hypot(re, im)

# ---------------------------------------------------------------------
# Full (alpha, k) grid selection
# ---------------------------------------------------------------------

@dataclass
class GridConfig:
    alpha_min: float = 0.02
    alpha_max: float = 0.98
    alpha_points: int = 80
    k_min: int = 1
    k_max: int = 120

def build_alpha_grid(cfg: GridConfig) -> np.ndarray:
    a0 = max(1e-8, cfg.alpha_min)
    a1 = min(1.0, cfg.alpha_max)
    return np.linspace(a0, a1, num=cfg.alpha_points, dtype=np.float64)

@dataclass
class ModalChoice:
    k: Optional[int]
    alpha: Optional[float]
    rho: float

def pick_alpha_k_by_full_grid(
    R: float,
    theta: float,
    cfg: GridConfig,
    alpha_grid: Optional[np.ndarray] = None,
) -> ModalChoice:
    """
    Full sweep over alpha and k with stability check alpha <= alpha_max(k).
    Returns the pair with smallest modal contraction rho.
    """
    if alpha_grid is None:
        alpha_grid = build_alpha_grid(cfg)

    best = ModalChoice(k=None, alpha=None, rho=float("inf"))
    for a in alpha_grid:
        best_k_for_a = None
        best_rho_for_a = float("inf")

        for k in range(cfg.k_min, cfg.k_max + 1):
            amax = alpha_max_of_k(R, theta, k)
            if not (a > 0.0 and a <= amax):
                continue
            r = rho_modal(R, theta, k, a)
            if r < best_rho_for_a:
                best_rho_for_a, best_k_for_a = r, k

        if best_k_for_a is not None and best_rho_for_a < best.rho:
            best = ModalChoice(k=best_k_for_a, alpha=float(a), rho=float(best_rho_for_a))

    return best

# ---------------------------------------------------------------------
# Adapter around user-provided spectral estimator
# ---------------------------------------------------------------------

def choose_modal_params_full_grid(
    spectral_estimator: Callable[..., Tuple[float, float]],
    estimator_kwargs: Dict,
    grid_cfg: GridConfig,
) -> ModalChoice:
    """
    spectral_estimator(...) -> (R, theta)
    estimator_kwargs: whatever your estimator needs (models, data, gamma, HVP/JVP flags, etc).
    """
    R, theta = spectral_estimator(**estimator_kwargs)  # must return floats (R>0, theta in (0,pi))
    return pick_alpha_k_by_full_grid(R, theta, grid_cfg)
