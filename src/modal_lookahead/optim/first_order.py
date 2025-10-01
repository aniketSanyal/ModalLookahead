from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

# ---------------------------------------------------------------------
# Gradient adapters that work for BOTH SC–SC and Bilinear problems.
# They prefer explicit closed forms (fast & unambiguous), and fall
# back to problem.grad_x / problem.grad_y if needed.
# ---------------------------------------------------------------------

def _gx(problem, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return g_x for current (x, y).

    SC–SC: g_x = mu_x * x + A @ y
    Bilinear: g_x = A @ y
    Fallbacks:
      - problem.grad_x(x, y)    (two-arg API)
      - problem.grad_x(y)       (bilinear one-arg API)
    """
    A = getattr(problem, "A", None)
    mu_x = getattr(problem, "mu_x", None)

    if A is not None and mu_x is not None:
        return mu_x * x + A @ y
    if A is not None and mu_x is None:
        return A @ y

    if hasattr(problem, "grad_x"):
        try:
            return problem.grad_x(x, y)  # two-arg API
        except TypeError:
            return problem.grad_x(y)     # one-arg API (bilinear style)

    raise AttributeError("Cannot compute grad_x: missing A/μ_x and grad_x().")


def _gy(problem, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return g_y for current (x, y) using the ascent convention used in your code.

    SC–SC: g_y = A.T @ x - mu_y * y
    Bilinear: g_y = A.T @ x
    Fallbacks:
      - problem.grad_y(x, y)    (two-arg API)
      - problem.grad_y(x)       (bilinear one-arg API)
    """
    A = getattr(problem, "A", None)
    mu_y = getattr(problem, "mu_y", None)

    if A is not None and mu_y is not None:
        return A.T @ x - mu_y * y
    if A is not None and mu_y is None:
        return A.T @ x

    if hasattr(problem, "grad_y"):
        try:
            return problem.grad_y(x, y)  # two-arg API
        except TypeError:
            return problem.grad_y(x)     # one-arg API (bilinear style)

    raise AttributeError("Cannot compute grad_y: missing A/μ_y and grad_y().")


# ---------------------------- Extragradient -----------------------------------

@dataclass
class Extragradient:
    """
    Mirror-Prox / Extragradient with constant step γ.

    Predict:  x̂ = x - γ g_x(x,y),  ŷ = y + γ g_y(x,y)
    Correct:  x⁺ = x - γ g_x(x̂,ŷ), y⁺ = y + γ g_y(x̂,ŷ)
    """
    gamma: float

    def step(self, problem, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx, gy = _gx(problem, x, y), _gy(problem, x, y)
        x_hat = x - self.gamma * gx
        y_hat = y + self.gamma * gy
        gx_hat, gy_hat = _gx(problem, x_hat, y_hat), _gy(problem, x_hat, y_hat)
        x_next = x - self.gamma * gx_hat
        y_next = y + self.gamma * gy_hat
        return x_next, y_next


# --------------------------------- OGDA ---------------------------------------

@dataclass
class OGDA:
    """
    Optimistic Gradient Descent–Ascent with constant step γ.

    x_{t+1} = x_t - 2γ g_x(x_t,y_t) + γ g_x(x_{t-1},y_{t-1})
    y_{t+1} = y_t + 2γ g_y(x_t,y_t) - γ g_y(x_{t-1},y_{t-1})
    """
    gamma: float
    _gx_prev: Optional[np.ndarray] = None
    _gy_prev: Optional[np.ndarray] = None

    def reset(self):
        self._gx_prev, self._gy_prev = None, None

    def step(self, problem, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx, gy = _gx(problem, x, y), _gy(problem, x, y)
        if self._gx_prev is None:
            # zero warmup is standard; swap to one-step warmup if you prefer
            self._gx_prev = np.zeros_like(gx)
            self._gy_prev = np.zeros_like(gy)
        x_next = x - 2.0 * self.gamma * gx + self.gamma * self._gx_prev
        y_next = y + 2.0 * self.gamma * gy - self.gamma * self._gy_prev
        self._gx_prev, self._gy_prev = gx, gy
        return x_next, y_next


@dataclass
class GD:
    """
    Gradient descent with constant step γ.
    """
  
    gamma: float

    def step(self, problem, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx, gy = _gx(problem, x, y), _gy(problem, x, y)
        x_next = x - self.gamma * gx
        y_next = y + self.gamma * gy
        return x_next, y_next