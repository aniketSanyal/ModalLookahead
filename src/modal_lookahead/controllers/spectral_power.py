import numpy as np
from ..utils.linalg import complex_power_iteration_G

def dominant_sigma_from_grads(grad_x, grad_y, d, iters=120):
    rng = np.random.default_rng(123)
    v = rng.normal(size=d); v /= np.linalg.norm(v)+1e-12
    for _ in range(iters):
        Av = grad_x(v)      # A v
        ATAv = grad_y(Av)   # A^T (A v)
        nrm = np.linalg.norm(ATAv)
        if nrm < 1e-20: break
        v = ATAv / (nrm + 1e-12)
    sigma2 = float(v @ grad_y(grad_x(v)))
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    return sigma

def dominant_mode_from_hvp(problem):
    d2 = 2 * problem.d
    lam, R, theta = complex_power_iteration_G(problem.hvp, problem.gamma, d2, iters=40)
    return R, theta, lam
