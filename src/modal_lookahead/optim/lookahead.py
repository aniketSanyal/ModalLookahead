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
    (Appendix E.2, Eq. (37)).  If no positive bound exists, fall back to α_max = 1.0.
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
    For a given α, build the union of k-candidates implied by per-mode alignment:
        T_i = R_i e^{-iθ_i},  targets:  phase ϕ = π (mod 2π), amplitude s* = (1-α)/α
        k_phase_i(m) = π(2m+1)/θ_i,   k_amp_i = ln((1-α)/α) / ln R_i
        choose m_i* ≈ argmin_m |k_phase_i(m) - k_amp_i|
        k_i° = k_phase_i(m_i*),  integer candidates {⌊k_i°⌋, ⌈k_i°⌉}
    (Appendix E.3.) We aggregate candidates across modes, clip to [kmin,kmax].
    """
    # numerical guards
    eps_th = 1e-12
    eps_ln = 1e-12

    Ks = set()
    for z in np.atleast_1d(T):
        R = float(abs(z))
        theta = float(np.angle(z))

        # amplitude target
        if abs(np.log(max(R, eps_ln))) < 1e-14:  # ln R ≈ 0 → ignore amplitude match
            k_amp = None
        else:
            k_amp = np.log((1.0 - alpha) / alpha) / np.log(max(R, eps_ln))

        # phase targets
        if abs(theta) < eps_th:  # no rotation: rely on amplitude only if available
            if k_amp is not None and np.isfinite(k_amp):
                k0 = float(k_amp)
                for ki in (np.floor(k0), np.ceil(k0)):
                    if np.isfinite(ki):
                        Ks.add(int(np.clip(int(ki), kmin, kmax)))
            continue

        # pick m* nearest to k_amp; if k_amp missing, use m* from phase-only (k_amp=0 heuristic)
        if (k_amp is None) or (not np.isfinite(k_amp)):
            k_amp = 0.0
        m_star = round((theta * k_amp / np.pi - 1.0) / 2.0)
        m_star = max(0, int(m_star))

        k_phase = np.pi * (2 * m_star + 1) / theta
        k0 = abs(k_phase)  # sign flip corresponds to adding 2m cycles
        for ki in (np.floor(k0), np.ceil(k0)):
            if np.isfinite(ki):
                Ks.add(int(np.clip(int(ki), kmin, kmax)))

        # also add the amplitude-only candidate when meaningful
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
    Choose (k, α) as follows:
      1) Map Jacobian eigs λ_i to base multipliers T_i = 1 - γ λ_i.
      2) For each α in a grid, build k-candidates from per-mode (k_phase, k_amp) alignment,
         union them across modes, evaluate the worst-mode contraction
             ρ_all(k, α) = max_i |(1-α) + α T_i^k|
         only on those candidates (no k-grid sweep).
      3) Enforce stability: α ≤ α_max_all(k) = min_i 2(1 - Re(T_i^k)) / |1 - T_i^k|^2.
      4) Return the (k(α*), α*) minimizing ρ_all.
    """
    eigs = np.asarray(eigs, dtype=complex)
    if eigs.size == 0:
        return ModalChoice(k=max(2, cfg.kmin), alpha=0.5, rho=1.0)

    # base multipliers (one inner Euler step)
    T = 1.0 - cfg.gamma * eigs

    # α grid
    alpha_grid = (
        np.linspace(0.02, 0.98, 97)
        if cfg.alpha_grid is None
        else np.asarray(list(cfg.alpha_grid), dtype=float)
    )

    best = ModalChoice(k=int(cfg.kmin), alpha=0.5, rho=np.inf)

    for alpha in alpha_grid:
        if not (0.0 < alpha < 1.0):
            continue

        # k candidates induced by this α
        Kc = _k_candidates_for_alpha(T, alpha, cfg.kmin, cfg.kmax)
        if Kc.size == 0:
            continue

        for k in Kc:
            alpha_cap = _alpha_cap_for_modes(T, k)
            if alpha > alpha_cap + 1e-15:
                continue  # violates Schur stability; skip

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
    Lookahead wrapper that adapts (k, α) every cycle from *lookahead modes*,
    using the alignment rule for k and a grid search only over α.

    Usage:
        la = AdaptiveModalLookahead(AdaptiveLAConfig(modal=ModalConfig(gamma=0.1)))

        def inner_step(state):   # EXACTLY ONE base step, e.g., GDA
            x, y = state
            return base_step(x, y)

        def jacobian_eigs_fn():  # current eigs of J (or a proxy)
            return eigs

        state = (x0, y0)
        for t in range(T):
            state = la.step_cycle(state, inner_step, jacobian_eigs_fn)
            # la.k and la.alpha are updated for the NEXT cycle
    """

    def __init__(self, cfg: AdaptiveLAConfig | None = None):
        if cfg is None:
            cfg = AdaptiveLAConfig()
        self.cfg = cfg
        self.k = int(cfg.init_k)
        self.alpha = float(cfg.init_alpha)
        self._anchor = None

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
        jacobian_eigs_fn: Callable[[], np.ndarray],
    ):
        """
        One lookahead cycle:
          (i) run k inner steps with the base optimizer
          (ii) average with α against the saved anchor
          (iii) refresh anchor and recompute (k, α) for the *next* cycle
                using the modal (alignment+α-grid) selector
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

        # update (k, α) from LOOKAHEAD modes computed using current J
        eigs = jacobian_eigs_fn()
        choice = choose_modal_params_from_jacobian_eigs(eigs, self.cfg.modal)
        self.k = int(np.clip(choice.k, self.cfg.modal.kmin, self.cfg.modal.kmax))
        self.alpha = float(min(1.0, max(1e-12, float(choice.alpha))))
        logger.info(f"Updated lookahead params: k={self.k}, alpha={self.alpha:.4f}, rho={choice.rho:.4f}")

        return new_state


class FixedLookahead:
    """Baseline lookahead with constant (k, α)."""

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

    def step_cycle(self, state, inner_step: Callable[[Tuple], Tuple]):
        self._maybe_init_anchor(state)
        z = self._clone_state(state)
        for _ in range(int(self.k)):
            z = inner_step(z)
        new_state = self._average(z)
        self._anchor = self._clone_state(new_state)
        return new_state


# --------------------------- One-shot modal lookahead --------------------------
@dataclass
class UpdatedLAConfig:
    """
    Configuration for the one-shot modal lookahead that *freezes* (k, α)
    after computing them once at the beginning of the first cycle.
    """
    modal: ModalConfig = field(default_factory=ModalConfig)
    fallback_k: int = 5
    fallback_alpha: float = 0.5


class ModalUpdatedLookahead:
    """
    Like AdaptiveModalLookahead, but (k, α) are selected exactly ONCE
    at the beginning of the FIRST cycle using the same selector (alignment + α-grid),
    and then kept FIXED for all subsequent cycles.
    """

    def __init__(self, cfg: UpdatedLAConfig | None = None):
        if cfg is None:
            cfg = UpdatedLAConfig()
        self.cfg = cfg
        self.k = int(cfg.fallback_k)
        self.alpha = float(cfg.fallback_alpha)
        self._anchor = None
        self._chosen_once = False  # toggled after first (k, α) selection

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
        jacobian_eigs_fn: Callable[[], np.ndarray],
    ):
        """
        One lookahead cycle with fixed (k, α) after the first selection.

        At the BEGINNING of the FIRST cycle:
          - compute eigs = jacobian_eigs_fn()
          - choose (k, α) with choose_modal_params_from_jacobian_eigs
          - freeze them for the remainder of the run

        Then every cycle:
          - run k inner steps
          - average with α against the anchor
          - refresh anchor
        """
        self._maybe_init_anchor(state)

        # One-shot selection BEFORE running the very first k inner steps
        if not self._chosen_once:
            eigs = jacobian_eigs_fn()
            choice = choose_modal_params_from_jacobian_eigs(eigs, self.cfg.modal)
            self.k = int(np.clip(choice.k, self.cfg.modal.kmin, self.cfg.modal.kmax))
            self.alpha = float(min(1.0, max(1e-12, float(choice.alpha))))
            self._chosen_once = True
            logger.info(f"[ModalUpdated] Chosen once: k={self.k}, alpha={self.alpha:.4f}, rho={choice.rho:.4f}")

        # Run k inner steps
        z = self._clone_state(state)
        for _ in range(int(self.k)):
            z = inner_step(z)

        # Lookahead averaging
        new_state = self._average(z)

        # Refresh anchor (params remain fixed)
        self._anchor = self._clone_state(new_state)

        return new_state


# Optional lowercase alias if someone prefers importing by this name
modal_updated_lookahead = ModalUpdatedLookahead
