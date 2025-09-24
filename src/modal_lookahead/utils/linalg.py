import numpy as np

def power_iteration_sym(matvec, d, iters=100, tol=1e-10, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    v = rng.normal(size=d)
    v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        w = matvec(v)
        nrm = np.linalg.norm(w)
        if nrm < 1e-20:
            return 0.0, v
        v_next = w / nrm
        if np.linalg.norm(v_next - v) < tol:
            v = v_next
            lam = float(v @ matvec(v))
            return lam, v
        v = v_next
        lam = float(v @ matvec(v))
    return lam, v

def complex_power_iteration_G(hvp_J, gamma, d2, iters=40, tol=1e-10, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    vr = rng.normal(size=d2); vr /= np.linalg.norm(vr)+1e-12
    vi = np.zeros_like(vr)
    for _ in range(iters):
        J_vr = hvp_J(vr); J_vi = hvp_J(vi)
        wr = vr - gamma * J_vr
        wi = vi - gamma * J_vi
        nrm = np.sqrt(np.linalg.norm(wr)**2 + np.linalg.norm(wi)**2) + 1e-12
        vr, vi = wr/nrm, wi/nrm
    J_vr = hvp_J(vr); J_vi = hvp_J(vi)
    Gr = vr - gamma * J_vr
    Gi = vi - gamma * J_vi
    num_real = vr @ Gr + vi @ Gi
    num_imag = vr @ Gi - vi @ Gr
    den = vr @ vr + vi @ vi + 1e-12
    lam = complex(num_real/den, num_imag/den)
    R = abs(lam); theta = float(np.angle(lam))
    return lam, R, theta
