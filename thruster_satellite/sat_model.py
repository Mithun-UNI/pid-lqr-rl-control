import numpy as np
from scipy.optimize import lsq_linear
from scipy.linalg import solve_continuous_are

# 2D sat, one thruster per side, all mounted off-centre
W, H = 2.0, 1.0
MASS = 100.0
F_MAX = 2.0
ISP = 70.0
G0 = 9.80665
DT = 0.01
CTRL_DT = 0.1
SUB = int(round(CTRL_DT / DT))


def inertia(m=MASS):
    return m * (W**2 + H**2) / 12.0


OFF = 0.8
THR_NAMES = ["T", "B", "L", "R"]
THR_POS = np.array([[+OFF * W / 2, +H / 2],
                    [-OFF * W / 2, -H / 2],
                    [-W / 2, -OFF * H / 2],
                    [+W / 2, +OFF * H / 2]])
THR_DIR = np.array([[0, -1], [0, +1], [+1, 0], [-1, 0]], float)
THR_TORQUE_ARM = THR_POS[:, 0] * THR_DIR[:, 1] - THR_POS[:, 1] * THR_DIR[:, 0]
B_ALLOC = np.vstack([THR_DIR[:, 0], THR_DIR[:, 1], THR_TORQUE_ARM])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# duty = fraction of the 0.1 s cycle each thruster is on (10 ms steps)
def plant_step(s, duty, p):
    on_steps = np.round(np.clip(duty, 0, 1) * SUB)
    thrust = p["F"] * p["health"]
    fuel = np.sum(on_steps * DT * thrust, axis=-1) / (ISP * G0)
    pulses = np.sum(on_steps > 0, axis=-1)
    x, y, vx, vy, th, om = [s[..., i] for i in range(6)]
    for k in range(SUB):
        f = thrust * (on_steps > k)
        fxb = f @ THR_DIR[:, 0]; fyb = f @ THR_DIR[:, 1]; tau = f @ THR_TORQUE_ARM
        c, sn = np.cos(th), np.sin(th)
        ax = (c * fxb - sn * fyb) / p["m"]; ay = (sn * fxb + c * fyb) / p["m"]
        vx = vx + ax * DT; vy = vy + ay * DT; om = om + tau / p["I"] * DT
        x = x + vx * DT; y = y + vy * DT; th = th + om * DT
    return np.stack([x, y, vx, vy, th, om], axis=-1), fuel, pulses


def nominal_params(m=MASS, health=(1, 1, 1, 1)):
    return dict(m=m, I=inertia(m), F=F_MAX, health=np.array(health, float))


# non-negative thrusts that best give [Fx, Fy, tau]
def allocate(wrench_body):
    res = lsq_linear(B_ALLOC, wrench_body, bounds=(0, F_MAX), method="bvls")
    return res.x / F_MAX


class PID:

    def __init__(self, kp=9.0, ki=0.05, kd=30.0, kv=40.0):
        self.kp, self.ki, self.kd, self.kv = kp, ki, kd, kv
        self.reset()

    def reset(self):
        self.integ = 0.0

    def wrench(self, s):
        x, y, vx, vy, th, om = s
        e = wrap(th)
        self.integ = np.clip(self.integ + e * CTRL_DT, -0.5, 0.5)
        tau = -(self.kp * e + self.ki * self.integ + self.kd * om)
        F_in = -self.kv * np.array([vx, vy])
        c, sn = np.cos(th), np.sin(th)
        F_b = np.array([c * F_in[0] + sn * F_in[1], -sn * F_in[0] + c * F_in[1]])
        return np.array([F_b[0], F_b[1], tau])

    def __call__(self, s):
        w = self.wrench(s)
        if abs(wrap(s[4])) < np.deg2rad(0.3) and abs(s[5]) < np.deg2rad(0.05) and np.hypot(s[2], s[3]) < 0.002:
            return np.zeros(4)
        return allocate(w)


class LQR(PID):

    def __init__(self, I=inertia(), m=MASS, q_th=20.0, q_om=400.0, r_tau=2.0, q_v=400.0, r_f=0.1):
        A = np.array([[0, 1], [0, 0]]); Bm = np.array([[0], [1 / I]])
        S = solve_continuous_are(A, Bm, np.diag([q_th, q_om]), np.array([[r_tau]]))
        K = (Bm.T @ S / r_tau).ravel()
        Sv = solve_continuous_are(np.array([[0.0]]), np.array([[1 / m]]), np.array([[q_v]]), np.array([[r_f]]))
        kv = float(Sv[0, 0] / m / r_f)
        super().__init__(kp=K[0], ki=0.0, kd=K[1], kv=kv)
        self.K = K


HID = 24
N_IN = 6
N_PARAMS = N_IN * HID + HID + HID * 4 + 4


def obs(s):
    th, om = s[..., 4], s[..., 5]
    c, sn = np.cos(th), np.sin(th)
    vxb = c * s[..., 2] + sn * s[..., 3]; vyb = -sn * s[..., 2] + c * s[..., 3]
    return np.stack([np.sin(th), np.cos(th) - 1, om / 0.5, om / 0.02, vxb / 0.1, vyb / 0.1], axis=-1)


def policy_forward(w, s):
    i = 0
    W1 = w[..., i:i + N_IN * HID].reshape(w.shape[:-1] + (HID, N_IN)); i += N_IN * HID
    b1 = w[..., i:i + HID]; i += HID
    W2 = w[..., i:i + HID * 4].reshape(w.shape[:-1] + (4, HID)); i += HID * 4
    b2 = w[..., i:i + 4]
    h = np.tanh(np.einsum("...ij,...j->...i", W1, np.clip(obs(s), -5, 5)) + b1)
    return np.clip(np.einsum("...ij,...j->...i", W2, h) + b2, 0, 1)


class RLPolicy:
    def __init__(self, path="rl_policy.npz"):
        self.w = np.load(path)["w"]

    def reset(self):
        pass

    def __call__(self, s):
        return policy_forward(self.w, np.asarray(s))


def simulate(ctrl, s0, p=None, T=90.0, noise=0.0, seed=0):
    p = p or nominal_params()
    rng = np.random.default_rng(seed)
    ctrl.reset()
    n = int(T / CTRL_DT)
    s = np.array(s0, float)
    S = np.zeros((n + 1, 6)); D = np.zeros((n, 4)); S[0] = s
    fuel = 0.0; pulses = 0
    for k in range(n):
        meas = s + noise * rng.standard_normal(6) * np.array([0, 0, 0.002, 0.002, np.deg2rad(0.1), np.deg2rad(0.02)])
        d = ctrl(meas)
        D[k] = d
        s, f, pc = plant_step(s, d, p)
        fuel += f; pulses += pc
        S[k + 1] = s
    t = np.arange(n + 1) * CTRL_DT
    return t, S, D, fuel, pulses


def metrics(t, S, fuel, pulses):
    om_ok = np.abs(np.rad2deg(S[:, 5])) < 0.1
    th_ok = np.abs(np.rad2deg(wrap(S[:, 4]))) < 1.0
    v_ok = np.hypot(S[:, 2], S[:, 3]) < 0.005
    def settle(ok):
        bad = np.where(~ok)[0]
        if len(bad) == 0: return 0.0
        return np.nan if bad[-1] + 1 >= len(t) else t[bad[-1] + 1]
    return dict(
        t_rate=settle(om_ok),
        t_all=settle(om_ok & th_ok & v_ok),
        fuel_g=1000 * fuel,
        pulses=int(pulses),
        final_rate=float(np.rad2deg(S[-1, 5])),
        final_att=float(np.rad2deg(wrap(S[-1, 4]))),
        final_v_mm=float(1000 * np.hypot(S[-1, 2], S[-1, 3])),
    )
