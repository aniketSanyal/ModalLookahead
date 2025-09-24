import numpy as np

class ScScConfig:
    def __init__(self, d=120, gamma=0.1, mu_x=0.18, mu_y=0.18,
                 sigma_min=0.25, sigma_max=0.45, seed=42):
        self.d=d; self.gamma=gamma; self.mu_x=mu_x; self.mu_y=mu_y
        self.sigma_min=sigma_min; self.sigma_max=sigma_max; self.seed=seed

class ScScQuadratic:
    """
    f(x,y) = 1/2 x^T (mu_x I) x - 1/2 y^T (mu_y I) y + x^T A y
    J = [[mu_x I, A], [-A^T, mu_y I]]
    """
    def __init__(self, cfg: ScScConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d = cfg.d
        U,_ = np.linalg.qr(rng.normal(size=(d,d)))
        V,_ = np.linalg.qr(rng.normal(size=(d,d)))
        sigmas = np.linspace(cfg.sigma_min, cfg.sigma_max, d)
        self.A = U @ np.diag(sigmas) @ V.T
        self.mu_x = cfg.mu_x; self.mu_y = cfg.mu_y
        self.gamma = cfg.gamma
        self.d = d

    def grad_x(self, y, x=None):
        if x is None:
            return self.A @ y
        return self.mu_x * x + self.A @ y

    def grad_y(self, x, y=None):
        if y is None:
            return self.A.T @ x
        return self.A.T @ x - self.mu_y * y

    def step_gda(self, x, y):
        g = self.gamma
        gx = self.mu_x * x + self.A @ y
        gy = self.A.T @ x - self.mu_y * y
        x = x - g * gx
        y = y + g * gy
        return x, y

    def distance(self, x, y):
        return float(np.linalg.norm(np.concatenate([x,y])))

    def hvp(self, v):
        d = self.d
        vx, vy = v[:d], v[d:]
        top = self.mu_x * vx + self.A @ vy
        bot = - self.A.T @ vx + self.mu_y * vy
        return np.concatenate([top, bot])
