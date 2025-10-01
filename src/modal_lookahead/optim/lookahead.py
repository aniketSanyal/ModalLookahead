# lookahead.py — Modal Lookahead with k from (k_phase, k_amp) alignment, α by grid search.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ----------------------------- Configs ----------------------------------------
@dataclass
class ModalConfig:
    gamma: float = 0.1                  # inner stepsize used to form T = I - γJ
    kmin: int = 2                       # k >= 2 (k=1 makes lookahead trivial)
    kmax: int = 512                     # hard cap on k
    alpha_grid: Optional[Iterable[float]] = None  # if None -> np.linspace(0.02, 0.98, 97)


@dataclass
class AdaptiveLAConfig:
    # used only for the very first cycle (before the first spectral selection)
    init_k: int = 5
    init_alpha: float = 0.5
    modal: ModalConfig = field(default_factory=ModalConfig)


# ------------------------ helpers (theory-backed) ------------------------------
def _alpha_cap_for_modes(T: np.ndarray, k: int) -> float:
    """
    Global Schur-stability cap on α for the Lookahead map:
        α_max_all(k) = min_i  2(1 - Re(T_i^k)) / |1 - T_i^k|^2 ,  restricted to positive numerators
    (Appendix E.2). If no positive bound exists, fall back to α_max = 1.0.
    """
    amax = np.inf
    for z in np.atleast_1d(T):
        w = z ** k
        denom = abs(1.0 - w) ** 2
        num = 1.0 - np.real(w)
        if denom > 1e-16 and num > 0:
            amax = min(amax, 2.0 * num / denom)
    if not np.isfinite(amax):
        amax = 1.0
    return float(max(1e-12, min(1.0, amax)))


def _k_candidates_for_alpha(T: np.ndarray, alpha: float, kmin: int, kmax: int) -> np.ndarray:
    """
    For a given α, build union of k-candidates implied by per-mode alignment:
      T_i = R_i e^{-iθ_i}, targets: phase φ=π (mod 2π), amplitude s*=(1-α)/α
      k_phase_i(m) = π(2m+1)/θ_i,   k_amp_i = ln((1-α)/α)/ln R_i
    We aggregate per-mode candidates and clip to [kmin, kmax].
    """
    eps_th = 1e-12
    eps_ln = 1e-12

    Ks = set()
    for z in np.atleast_1d(T):
        R = float(abs(z))
        theta = float(np.angle(z))

        # amplitude target
        if abs(np.log(max(R, eps_ln))) < 1e-14:
            k_amp = None
        else:
            k_amp = np.log((1.0 - alpha) / alpha) / np.log(max(R, eps_ln))

        # phase targets
        if abs(theta) < eps_th:
            if k_amp is not None and np.isfinite(k_amp):
                k0 = float(k_amp)
                for ki in (np.floor(k0), np.ceil(k0)):
                    if np.isfinite(ki):
                        Ks.add(int(np.clip(int(ki), kmin, kmax)))
            continue

        if (k_amp is None) or (not np.isfinite(k_amp)):
            k_amp = 0.0
        m_star = round((theta * k_amp / np.pi - 1.0) / 2.0)
        m_star = max(0, int(m_star))

        k_phase = np.pi * (2 * m_star + 1) / theta
        k0 = abs(k_phase)
        for ki in (np.floor(k0), np.ceil(k0)):
            if np.isfinite(ki):
                Ks.add(int(np.clip(int(ki), kmin, kmax)))

        if np.isfinite(k_amp):
            for ki in (np.floor(k_amp), np.ceil(k_amp)):
                if np.isfinite(ki):
                    Ks.add(int(np.clip(int(ki), kmin, kmax)))

    if not Ks:
        Ks = {kmin}
    return np.array(sorted(Ks), dtype=int)


# ---------------------- main selector: k by alignment, α by grid ----------------
@dataclass
class ModalChoice:
    k: int
    alpha: float
    rho: float  # achieved worst-mode spectral radius


def choose_modal_params_from_jacobian_eigs(
    eigs: np.ndarray,
    cfg: ModalConfig,
) -> ModalChoice:
    """
    1) Map Jacobian eigs λ_i to base multipliers T_i = 1 - γ λ_i.
    2) For each α in a grid, build k-candidates from per-mode alignment,
       evaluate ρ_all(k, α) = max_i |(1-α) + α T_i^k| on those candidates.
    3) Enforce stability with α ≤ α_max_all(k).
    4) Return the best (k, α).
    """
    eigs = np.asarray(eigs, dtype=complex)
    if eigs.size == 0:
        return ModalChoice(k=max(2, cfg.kmin), alpha=0.5, rho=1.0)

    T = 1.0 - cfg.gamma * eigs

    alpha_grid = (
        np.linspace(0.02, 0.98, 97)
        if cfg.alpha_grid is None
        else np.asarray(list(cfg.alpha_grid), dtype=float)
    )

    best = ModalChoice(k=int(cfg.kmin), alpha=0.5, rho=np.inf)

    for alpha in alpha_grid:
        if not (0.0 < alpha < 1.0):
            continue

        Kc = _k_candidates_for_alpha(T, alpha, cfg.kmin, cfg.kmax)
        if Kc.size == 0:
            continue

        for k in Kc:
            alpha_cap = _alpha_cap_for_modes(T, k)
            if alpha > alpha_cap + 1e-15:
                continue

            rho = float(np.max(np.abs((1.0 - alpha) + alpha * (T ** k))))
            if rho < best.rho:
                best = ModalChoice(k=int(k), alpha=float(alpha), rho=rho)

    if not np.isfinite(best.rho):
        k_fb = int(cfg.kmin)
        alpha_cap = _alpha_cap_for_modes(T, k_fb)
        alpha_fb = min(0.5, alpha_cap)
        rho_fb = float(np.max(np.abs((1.0 - alpha_fb) + alpha_fb * (T ** k_fb))))
        best = ModalChoice(k=k_fb, alpha=alpha_fb, rho=rho_fb)

    return best


# --------------------------- Lookahead wrappers --------------------------------
class AdaptiveModalLookahead:
    """
    Adapts (k, α) every cycle from lookahead modes; now supports per-point callback.
    """

    def __init__(self, cfg: AdaptiveLAConfig | None = None):
        if cfg is None:
            cfg = AdaptiveLAConfig()
        self.cfg = cfg
        self.k = int(cfg.init_k)
        self.alpha = float(cfg.init_alpha)
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

    def step_cycle(
        self,
        state,
        inner_step: Callable[[Tuple], Tuple],
        jacobian_eigs_fn: Callable[[], np.ndarray],
        on_point: Optional[Callable[[Tuple, str], None]] = None,  # NEW
    ):
        self._maybe_init_anchor(state)

        # k inner steps (emit every base point)
        z = self._clone_state(state)
        for _ in range(int(self.k)):
            z = inner_step(z)
            if on_point is not None:
                on_point(z, "base")

        # average
        new_state = self._average(z)
        if on_point is not None:
            on_point(new_state, "avg")

        # next cycle prep
        self._anchor = self._clone_state(new_state)

        # update params for next cycle
        eigs = jacobian_eigs_fn()
        choice = choose_modal_params_from_jacobian_eigs(eigs, self.cfg.modal)
        self.k = int(np.clip(choice.k, self.cfg.modal.kmin, self.cfg.modal.kmax))
        self.alpha = float(min(1.0, max(1e-12, float(choice.alpha))))
        logger.info(f"Updated lookahead params: k={self.k}, alpha={self.alpha:.4f}, rho={choice.rho:.4f}")
        return new_state


class FixedLookahead:
    """Baseline lookahead with constant (k, α); supports per-point callback."""

    def __init__(self, k: int = 50, alpha: float = 0.5):
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

    def step_cycle(
        self,
        state,
        inner_step: Callable[[Tuple], Tuple],
        on_point: Optional[Callable[[Tuple, str], None]] = None,  # NEW
    ):
        self._maybe_init_anchor(state)
        z = self._clone_state(state)
        for _ in range(int(self.k)):
            z = inner_step(z)
            if on_point is not None:
                on_point(z, "base")
        new_state = self._average(z)
        if on_point is not None:
            on_point(new_state, "avg")
        self._anchor = self._clone_state(new_state)
        return new_state


# --------------------------- One-shot modal lookahead --------------------------
@dataclass
class UpdatedLAConfig:
    """Configuration for the one-shot modal lookahead that freezes (k, α) after first selection."""
    modal: ModalConfig = field(default_factory=ModalConfig)
    fallback_k: int = 5
    fallback_alpha: float = 0.5


class ModalUpdatedLookahead:
    """
    Selects (k, α) exactly ONCE at the start of the first cycle, then keeps them fixed.
    Emits every base step and the averaging point via callback.
    """

    def __init__(self, cfg: UpdatedLAConfig | None = None):
        if cfg is None:
            cfg = UpdatedLAConfig()
        self.cfg = cfg
        self.k = int(cfg.fallback_k)
        self.alpha = float(cfg.fallback_alpha)
        self._anchor = None
        self._chosen_once = False

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

    def step_cycle(
        self,
        state,
        inner_step: Callable[[Tuple], Tuple],
        jacobian_eigs_fn: Callable[[], np.ndarray],
        on_point: Optional[Callable[[Tuple, str], None]] = None,  # NEW
    ):
        self._maybe_init_anchor(state)

        if not self._chosen_once:
            eigs = jacobian_eigs_fn()
            choice = choose_modal_params_from_jacobian_eigs(eigs, self.cfg.modal)
            self.k = int(np.clip(choice.k, self.cfg.modal.kmin, self.cfg.modal.kmax))
            self.alpha = float(min(1.0, max(1e-12, float(choice.alpha))))
            self._chosen_once = True
            logger.info(f"[ModalUpdated] Chosen once: k={self.k}, alpha={self.alpha:.4f}, rho={choice.rho:.4f}")

        z = self._clone_state(state)
        for _ in range(int(self.k)):
            z = inner_step(z)
            if on_point is not None:
                on_point(z, "base")

        new_state = self._average(z)
        if on_point is not None:
            on_point(new_state, "avg")

        self._anchor = self._clone_state(new_state)
        return new_state


# Optional alias
modal_updated_lookahead = ModalUpdatedLookahead
