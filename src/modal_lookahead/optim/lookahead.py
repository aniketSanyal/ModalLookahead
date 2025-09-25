"""
Adaptive (dynamic) Modal Lookahead + Fixed Lookahead baseline.

- AdaptiveModalLookahead:
  Updates (k, alpha) at the *end of every lookahead cycle* using your
  weighted-mode selection:
    1) T_i = 1 - gamma * lambda_i  (lambda_i are eigenvalues of J)
    2) choose z_div (worst under snapshot k0=5, a0=0.5) and z_dom (largest |T|)
    3) weighted hybrid z_mix
    4) k_amp from amplitude match, k_phase from phase match -> k
    5) global stability cap for alpha then grid search to minimize |(1-a)+a z_mix^k|

- FixedLookahead:
  Simple baseline with constant k and alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple
import numpy as np


# ----------------------------- Configs ----------------------------------------
@dataclass
class WeightedModalConfig:
    gamma: float = 0.1
    kmin: int = 5
    kmax: int = 2000
    alpha_grid: Optional[Iterable[float]] = None  # if None -> linspace(0.05,1.0,50)


@dataclass
class AdaptiveLAConfig:
    # Used for the very first cycle only (before the first spectral selection).
    init_k: int = 5
    init_alpha: float = 0.5
    weighted: WeightedModalConfig = field(default_factory=WeightedModalConfig)


# ------------------------ Weighted-mode selection -----------------------------
@dataclass
class ModalChoice:
    k: int
    alpha: float


def choose_modal_params_weighted(eigs: np.ndarray, cfg: WeightedModalConfig) -> ModalChoice:
    """
    Choose (k, alpha) using k_amp / k_phase + stability-bounded alpha grid search.

    Parameters
    ----------
    eigs : array-like of complex
        Eigenvalues of the current Jacobian J.
    cfg  : WeightedModalConfig
    """
    eigs = np.asarray(eigs)
    if eigs.size == 0:
        return ModalChoice(k=cfg.kmin, alpha=0.5)

    # One-step multipliers of the base update (I - gamma J)
    T = 1.0 - cfg.gamma * eigs

    # --- Reference snapshot to identify a "divergent" proxy ---
    k0, a0 = 5, 0.5
    vals = (1 - a0) + a0 * (T ** k0)
    z_div = T[int(np.argmax(np.abs(vals)))]     # worst under snapshot
    z_dom = T[int(np.argmax(np.abs(T)))]        # largest |T|

    # --- Weighted hybrid mode (magnitude weights) ---
    denom = (np.abs(z_dom) + np.abs(z_div))
    z_mix = 0.0 if denom == 0 else (np.abs(z_dom) * z_div + np.abs(z_div) * z_dom) / denom
    R = float(np.abs(z_mix))
    theta = float(np.angle(z_mix))

    # --- k from amplitude & phase matching ---
    if R == 1.0:
        k_amp = cfg.kmin
    else:
        try:
            k_amp = int(np.round(np.log((1 - a0) / a0) / np.log(R)))
        except Exception:
            k_amp = cfg.kmin

    k_phase = cfg.kmin if theta == 0.0 else int(np.round(np.pi / abs(theta)))

    k = int(np.clip(int(np.round((k_amp + k_phase) / 2)), cfg.kmin, cfg.kmax))

    # --- Global stability cap for alpha over all modes ---
    alpha_max_all = np.inf
    for z in np.atleast_1d(T):
        w = z ** k
        denom = abs(1 - w) ** 2
        if denom > 1e-12 and (1 - np.real(w)) > 0:
            alpha_max = 2 * (1 - np.real(w)) / denom
            alpha_max_all = min(alpha_max_all, alpha_max)
    if not np.isfinite(alpha_max_all):
        alpha_max_all = 1.0

    # --- Grid search alpha on (0, alpha_max_all] to minimize modal contraction ---
    alpha_grid = (
        np.linspace(0.05, 1.0, 50)
        if cfg.alpha_grid is None
        else np.asarray(list(cfg.alpha_grid), dtype=float)
    )

    best_alpha, best_rho = None, np.inf
    for alpha in alpha_grid:
        if alpha <= 0 or alpha > alpha_max_all:
            continue
        rho = abs((1 - alpha) + alpha * (z_mix ** k))
        if rho < best_rho:
            best_rho, best_alpha = rho, float(alpha)

    if best_alpha is None:
        best_alpha = float(min(0.5, alpha_max_all))

    return ModalChoice(k=int(k), alpha=float(best_alpha))


# --------------------------- Lookahead wrappers --------------------------------
class AdaptiveModalLookahead:
    """
    Lookahead wrapper that *adapts* (k, alpha) every cycle.

    Usage:
        la = AdaptiveModalLookahead(AdaptiveLAConfig(weighted=WeightedModalConfig(gamma=0.1)))

        def inner_step(state):  # ONE base optimizer step (e.g., GDA)
            x, y = state
            return problem.step_gda(x, y)

        def spectral_eigs_fn():  # return eigenvalues of current J (or proxy)
            J = ...
            return np.linalg.eigvals(J)

        state = (x0, y0)
        for t in range(T):
            state = la.step_cycle(state, inner_step, spectral_eigs_fn)
            # la.k and la.alpha are updated for the NEXT cycle

    Notes:
      * `state` is assumed to be a tuple (x, y) of arrays/tensors. Override
        `_clone_state` if your state is a dict or framework object.
      * Computing a full eigendecomposition every cycle can be expensive;
        feel free to return a small proxy spectrum instead.
    """

    def __init__(self, cfg: AdaptiveLAConfig | None = None):
        if cfg is None:
            cfg = AdaptiveLAConfig()
        self.cfg = cfg

        self.k = int(cfg.init_k)
        self.alpha = float(cfg.init_alpha)

        self._anchor = None
        self._step_in_cycle = 0

    # -- utilities --
    def _clone_state(self, state):
        x, y = state
        return (x.copy(), y.copy())

    def _maybe_init_anchor(self, state):
        if self._anchor is None:
            self._anchor = self._clone_state(state)

    def _average(self, state):
        ax, ay = self._anchor
        x, y = state
        return ((1 - self.alpha) * ax + self.alpha * x,
                (1 - self.alpha) * ay + self.alpha * y)

    # -- main API --
    def step_cycle(
        self,
        state,
        inner_step: Callable[[Tuple], Tuple],
        spectral_eigs_fn: Callable[[], np.ndarray],
    ):
        """
        Run ONE lookahead cycle:
          1) k inner steps with base optimizer
          2) average with current alpha
          3) recompute (k, alpha) for the NEXT cycle via weighted-modal selector
        """
        self._maybe_init_anchor(state)

        # k inner steps
        z = self._clone_state(state)
        for _ in range(int(self.k)):
            z = inner_step(z)

        # lookahead averaging
        new_state = self._average(z)

        # prepare next cycle
        self._anchor = self._clone_state(new_state)
        self._step_in_cycle = 0

        # dynamic (k, alpha) update for the NEXT cycle
        eigs = spectral_eigs_fn()
        choice = choose_modal_params_weighted(eigs, self.cfg.weighted)
        self.k = int(np.clip(int(choice.k), self.cfg.weighted.kmin, self.cfg.weighted.kmax))
        self.alpha = float(min(1.0, max(1e-12, float(choice.alpha))))

        return new_state


class FixedLookahead:
    """Baseline lookahead with constant (k, alpha)."""

    def __init__(self, k: int = 5, alpha: float = 0.5):
        self.k = int(k)
        self.alpha = float(alpha)
        self._anchor = None

    def _clone_state(self, state):
        x, y = state
        return (x.copy(), y.copy())

    def _maybe_init_anchor(self, state):
        if self._anchor is None:
            self._anchor = self._clone_state(state)

    def _average(self, state):
        ax, ay = self._anchor
        x, y = state
        return ((1 - self.alpha) * ax + self.alpha * x,
                (1 - self.alpha) * ay + self.alpha * y)

    def step_cycle(self, state, inner_step: Callable[[Tuple], Tuple]):
        self._maybe_init_anchor(state)
        z = self._clone_state(state)
        for _ in range(int(self.k)):
            z = inner_step(z)
        new_state = self._average(z)
        self._anchor = self._clone_state(new_state)
        return new_state
