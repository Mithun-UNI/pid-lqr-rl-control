import numpy as np
from scipy.linalg import solve_continuous_are

# wheeled inverted pendulum, 5 kg on a 0.5 m pole
P = dict(
    M=5.0,
    L=0.5,
    mw=1.0,
    r=0.15,
    Ib=0.02,
    g=9.81,
)
TAU_MAX = 6.0
DELAY = 2
DT = 0.01
FALL_ANGLE = 0.8


# state = [x, xd, th, thd], lagrange EOM
def dynamics(s, tau, p):
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    M, L, mw, r, Ib, g = p["M"], p["L"], p["mw"], p["r"], p["Ib"], p["g"]
    Iw = 0.5 * mw * r**2
    a11 = mw + Iw / r**2 + M
    a12 = M * L * np.cos(th)
    a22 = M * L**2 + Ib
    b1 = tau / r + M * L * np.sin(th) * thd**2
    b2 = -tau + M * g * L * np.sin(th)
    det = a11 * a22 - a12**2
    xdd = (b1 * a22 - a12 * b2) / det
    thdd = (a11 * b2 - a12 * b1) / det
    return np.stack([xd, xdd, thd, thdd], axis=-1)


def step(s, tau, p, dt=DT):
    tau = np.clip(tau, -TAU_MAX, TAU_MAX)
    k1 = dynamics(s, tau, p)
    k2 = dynamics(s + 0.5 * dt * k1, tau, p)
    k3 = dynamics(s + 0.5 * dt * k2, tau, p)
    k4 = dynamics(s + dt * k3, tau, p)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def linear_model(p=P):
    eps = 1e-6
    s0 = np.zeros(4)
    A = np.zeros((4, 4))
    for i in range(4):
        d = np.zeros(4); d[i] = eps
        A[:, i] = (dynamics(s0 + d, 0.0, p) - dynamics(s0 - d, 0.0, p)) / (2 * eps)
    B = ((dynamics(s0, eps, p) - dynamics(s0, -eps, p)) / (2 * eps)).reshape(4, 1)
    return A, B


# outer loop: position -> lean angle, inner loop: lean -> torque
class PID:

    def __init__(self, kp=40.0, ki=10.0, kd=4.0, kx=-0.03, kv=-0.08):
        self.kp, self.ki, self.kd, self.kx, self.kv = kp, ki, kd, kx, kv
        self.integ = 0.0

    def reset(self):
        self.integ = 0.0

    def __call__(self, s):
        x, xd, th, thd = s
        th_ref = self.kx * x + self.kv * xd
        e = th - th_ref
        self.integ = np.clip(self.integ + e * DT, -1.0, 1.0)
        return self.kp * e + self.ki * self.integ + self.kd * thd


class LQR:

    def __init__(self, p=P, Q=np.diag([1.0, 1.0, 100.0, 1.0]), R=np.array([[0.2]])):
        A, B = linear_model(p)
        S = solve_continuous_are(A, B, Q, R)
        self.K = (np.linalg.solve(R, B.T @ S)).ravel()

    def reset(self):
        pass

    def __call__(self, s):
        return float(-self.K @ s)


OBS_SCALE = np.array([1.0, 1.0, 0.3, 2.0])
HIDDEN = 16
N_PARAMS = 4 * HIDDEN + HIDDEN + HIDDEN + 1


def unpack(w):
    i = 0
    W1 = w[..., i:i + 4 * HIDDEN].reshape(w.shape[:-1] + (HIDDEN, 4)); i += 4 * HIDDEN
    b1 = w[..., i:i + HIDDEN]; i += HIDDEN
    W2 = w[..., i:i + HIDDEN]; i += HIDDEN
    b2 = w[..., i]
    return W1, b1, W2, b2


def policy_forward(w, s):
    W1, b1, W2, b2 = unpack(w)
    o = s / OBS_SCALE
    h = np.tanh(np.einsum("...ij,...j->...i", W1, o) + b1)
    return TAU_MAX * np.tanh(np.einsum("...i,...i->...", W2, h) + b2)


class RLPolicy:
    def __init__(self, path="rl_policy.npz"):
        self.w = np.load(path)["w"]

    def reset(self):
        pass

    def __call__(self, s):
        return float(policy_forward(self.w, np.asarray(s)))


def simulate(ctrl, s0, p=P, T=8.0, push=None, sensor_noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ctrl.reset()
    n = int(T / DT)
    s = np.array(s0, float)
    S = np.zeros((n + 1, 4)); U = np.zeros(n); S[0] = s
    fell = False
    buf = [0.0] * DELAY
    for k in range(n):
        meas = s + sensor_noise * rng.standard_normal(4) * np.array([0.01, 0.02, 0.005, 0.02])
        u_cmd = np.clip(ctrl(meas), -TAU_MAX, TAU_MAX)
        buf.append(u_cmd)
        u = buf.pop(0)
        U[k] = u
        s = step(s, u, p)
        if push is not None and push[0] <= k * DT < push[0] + push[2]:
            F = push[1]
            s[3] += F * p["L"] * np.cos(s[2]) / (p["M"] * p["L"] ** 2 + p["Ib"]) * DT
        S[k + 1] = s
        if abs(s[2]) > FALL_ANGLE:
            fell = True
            S[k + 2:] = np.nan; U[k + 1:] = np.nan
            break
    t = np.arange(n + 1) * DT
    return t, S, U, fell


def metrics(t, S, U, fell, band=np.deg2rad(1.0)):
    th = S[:, 2]
    if fell:
        return dict(fell=True, settle_s=np.nan, max_tilt_deg=np.nan, effort=np.nan, final_x=np.nan)
    outside = np.where(np.abs(th) > band)[0]
    settle = (t[outside[-1] + 1] if outside[-1] + 1 < len(t) else np.nan) if len(outside) else 0.0
    return dict(
        fell=False,
        settle_s=float(settle),
        max_tilt_deg=float(np.rad2deg(np.nanmax(np.abs(th)))),
        effort=float(np.nansum(U ** 2) * DT),
        final_x=float(S[-1, 0]),
    )
