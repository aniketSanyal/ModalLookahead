import numpy as np

class BilinearGame:
    """
    f(x,y) = x^T A y
    grad_x = A y, grad_y = A^T x
    J = [[0, A], [-A^T, 0]]
    """
    def __init__(self, A, gamma=1.0):
        self.A = A
        self.d = A.shape[0]
        self.gamma = gamma

    def grad_x(self, y):
        return self.A @ y

    def grad_y(self, x):
        return self.A.T @ x

    def step_gda(self, x, y, gamma=None):
        g = self.gamma if gamma is None else gamma
        gx = self.grad_x(y)
        gy = self.grad_y(x)
        x = x - g * gx
        y = y + g * gy
        return x, y

    def distance(self, x, y):
        return float(np.linalg.norm(np.concatenate([x,y])))

    def hvp(self, v):
        d = self.d
        vx, vy = v[:d], v[d:]
        top = self.A @ vy
        bot = - self.A.T @ vx
        return np.concatenate([top, bot])
