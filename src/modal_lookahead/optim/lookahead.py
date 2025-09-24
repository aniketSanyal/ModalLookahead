# src/modal_lookahead/optim/lookahead.py
"""
Lookahead wrapper with optional *Modal* adaptation.

Modes:
- fixed:     keep (k, alpha) constant.
- modal_grid: at each macro step, estimate (R, theta) and pick (k, alpha)
              via full grid search to minimize the modal contraction.

This module is framework-agnostic for (k, alpha) selection. Your inner
optimizer and model updates can be PyTorch/JAX/etc; pass a spectral
estimator callable that returns (R, theta).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
from .typing import ArrayLike  # if you have a typing helper; else remove
from ..controllers.selectors import (
    GridConfig, choose_modal_params_full_grid, ModalChoice
)

@dataclass
class LookaheadConfig:
    k: int = 5
    alpha: float = 0.5
    mode: str = "fixed"  # "fixed" or "modal_grid"
    # grid config for the modal selector
    grid: GridConfig = GridConfig()

class Lookahead:
    """
    Generic Lookahead wrapper for min-max / games.

    You provide:
      - inner_step(x, y): applies ONE base-optimizer step (e.g., simultaneous GDA) in-place or returns new states.
      - spectral_estimator(**kwargs) -> (R, theta): dominant one-step mode near current iterate.
      - estimator_kwargs_fn(): callable that returns kwargs for spectral_estimator at runtime.
    """

    def __init__(
        self,
        cfg: LookaheadConfig,
        inner_step: Callable[..., Tuple],
        spectral_estimator: Optional[Callable[..., Tuple[float, float]]] = None,
        estimator_kwargs_fn: Optional[Callable[[], Dict]] = None,
    ):
        self.k = int(cfg.k)
        self.alpha = float(cfg.alpha)
        self.mode = cfg.mode
        self.grid_cfg = cfg.grid

        self.inner_step = inner_step
        self.spectral_estimator = spectral_estimator
        self.estimator_kwargs_fn = estimator_kwargs_fn

        # Lookahead anchor (reference copy)
        self._anchor = None
        self._step_in_cycle = 0  # counts inner steps since last averaging

    # -- utility ---------------------------------------------------------

    def _maybe_init_anchor(self, state):
        if self._anchor is None:
            self._anchor = self._clone_state(state)

    def _clone_state(self, state):
        # Assumes (x, y) tuple of arrays/tensors. Override if you have a model dict.
        x, y = state
        return (x.copy(), y.copy())

    def _average(self, state):
        # z <- (1-alpha) * anchor + alpha * state
        ax, ay = self._anchor
        x, y = state
        x_new = (1.0 - self.alpha) * ax + self.alpha * x
        y_new = (1.0 - self.alpha) * ay + self.alpha * y
        return (x_new, y_new)

    # -- main loop -------------------------------------------------------

    def step(self, state):
        """
        Performs one inner step, and if k steps have elapsed,
        performs the lookahead averaging and (optionally) adapts (k, alpha).
        """
        # 1) inner base optimizer step
        new_state = self.inner_step(state)

        # 2) bookkeeping
        self._maybe_init_anchor(state if self._step_in_cycle == 0 else self._anchor)
        self._step_in_cycle += 1

        # 3) averaging + modal selection if cycle finished
        if self._step_in_cycle >= self.k:
            # average
            new_state = self._average(new_state)

            # prepare next cycle
            self._anchor = self._clone_state(new_state)
            self._step_in_cycle = 0

            # optional: adapt (k, alpha)
            if self.mode == "modal_grid":
                assert self.spectral_estimator is not None and self.estimator_kwargs_fn is not None, \
                    "Modal mode requires spectral_estimator and estimator_kwargs_fn."
                choice: ModalChoice = choose_modal_params_full_grid(
                    spectral_estimator=self.spectral_estimator,
                    estimator_kwargs=self.estimator_kwargs_fn(),
                    grid_cfg=self.grid_cfg,
                )
                if choice.k is not None and choice.alpha is not None:
                    self.k = int(max(1, choice.k))
                    self.alpha = float(min(1.0, max(1e-12, choice.alpha)))

        return new_state
