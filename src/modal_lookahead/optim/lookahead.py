import numpy as np, math
from ..controllers.spectral_power import dominant_sigma_from_grads, dominant_mode_from_hvp

class FixedLookahead:
    def __init__(self, k=5, alpha=0.5):
        self.k=int(k); self.alpha=float(alpha)
    def run(self, problem, T, x0, y0):
        x=x0.copy(); y=y0.copy(); xa=x.copy(); ya=y.copy(); dists=[]; t=0
        for _ in range(T):
            x,y=problem.step_gda(x,y); t+=1
            if t==self.k:
                x=(1.0-self.alpha)*xa + self.alpha*x
                y=(1.0-self.alpha)*ya + self.alpha*y
                xa=x.copy(); ya=y.copy(); t=0
            dists.append(float(np.linalg.norm(np.concatenate([x,y]))))
        return dists

class ModalLookahead:
    """
    Modal Lookahead using the k_phase / k_amp rule.
    Backends:
      - mode='grad_bilinear': uses grad_x, grad_y (first-order) to estimate sigma_max.
      - mode='hvp': uses problem.hvp (second-order) to estimate dominant eigen of G.
    In SC-SC isotropic (mu_x=mu_y=mu), per-mode eigen of G: z = (1 - gamma*mu) +/- i * (gamma*sigma).
    """
    def __init__(self, mode='grad_bilinear', alpha=0.5):
        assert mode in ('grad_bilinear','hvp')
        self.mode = mode
        self.alpha = float(alpha)

    @staticmethod
    def _k_phase_k_amp(alpha, R, theta):
        if abs(R-1.0) < 1e-12:
            k_amp = 1e9
        else:
            k_amp = math.log((1.0 - alpha)/alpha)/math.log(R)
        k_circ = math.pi/max(theta,1e-12)
        k_floor, k_ceil = int(math.floor(k_circ)), int(math.ceil(k_circ))
        def rho(k):
            s = R**k
            return math.sqrt((1.0-alpha)**2 + 2.0*alpha*(1.0-alpha)*s*math.cos(k*theta) + (alpha**2)*(s**2))
        k_best = max(1, min(k_floor, k_ceil, key=rho))
        return k_best

    def run(self, problem, T, x0, y0):
        if self.mode == 'grad_bilinear':
            sigma = dominant_sigma_from_grads(problem.grad_x, problem.grad_y, problem.d, iters=120)
            mu = getattr(problem, "mu_x", 0.0) * 0.5 + getattr(problem, "mu_y", 0.0) * 0.5
            R = ((1.0 - problem.gamma*mu)**2 + (problem.gamma*sigma)**2)**0.5
            theta = math.atan2(problem.gamma*sigma, max(1e-12, 1.0 - problem.gamma*mu))
        else:
            R, theta, _ = dominant_mode_from_hvp(problem)
        k = self._k_phase_k_amp(self.alpha, R, theta)

        x=x0.copy(); y=y0.copy(); xa=x.copy(); ya=y.copy(); dists=[]; t=0
        for _ in range(T):
            x,y=problem.step_gda(x,y); t+=1
            if t==k:
                x=(1.0-self.alpha)*xa + self.alpha*x
                y=(1.0-self.alpha)*ya + self.alpha*y
                xa=x.copy(); ya=y.copy(); t=0
            dists.append(float(np.linalg.norm(np.concatenate([x,y]))))
        return dists, {"k":k, "alpha":self.alpha, "R":R, "theta":theta}
