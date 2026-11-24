import numpy as np
from scipy.linalg import solve_continuous_are

# 2D sat, reaction wheels + one thruster on the left side
W, H = 2.0, 1.0
MASS = 100.0
I_BODY = MASS * (W**2 + H**2) / 12.0

N_WHEELS = 3
J_W = 0.02
TAU_W = 0.2
OMEGA_W_MAX = 6000 * 2 * np.pi / 60
H_W_MAX = J_W * OMEGA_W_MAX

F_THR = 2.0
ISP, G0 = 70.0, 9.80665
THR_POS = np.array([-W / 2, 0.03])
THR_DIR = np.array([1.0, 0.0])
DT, CTRL_DT = 0.01, 0.1
SUB = int(round(CTRL_DT / DT))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def params(m=MASS, wheels=(1, 1, 1), offset=0.03):
    return dict(m=m, I=m * (W**2 + H**2) / 12.0, wheels=np.array(wheels, float), offset=offset)


def plant_step(s, cmd, p, sub=SUB):
    tau_req = cmd[..., 0]
    duty = np.clip(cmd[..., 1], 0, 1)
    dt = CTRL_DT / sub
    on = np.round(duty * sub)
    wheels = p["wheels"]
    n_ok = np.maximum(np.sum(wheels, -1), 1e-9)
    s = s.copy()
    fuel = on * dt * F_THR / (ISP * G0)
    arm = -p["offset"]  # thruster sits above CoM -> small torque on every burn
    for k in range(sub):
        x, y, vx, vy, th, om = [s[..., i] for i in range(6)]
        Ow = s[..., 6:9]
        tw = np.clip(-(tau_req / n_ok)[..., None], -TAU_W, TAU_W) * wheels
        sat_hi = (Ow >= OMEGA_W_MAX) & (tw > 0); sat_lo = (Ow <= -OMEGA_W_MAX) & (tw < 0)
        tw = np.where(sat_hi | sat_lo, 0.0, tw)  # wheel at max rpm can't push further
        f = F_THR * (on > k)
        tau_thr = f * arm
        om_dot = (-np.sum(tw, -1) + tau_thr) / p["I"]
        c, sn = np.cos(th), np.sin(th)
        vx = vx + c * f / p["m"] * dt; vy = vy + sn * f / p["m"] * dt
        om = om + om_dot * dt
        Ow = np.clip(Ow + tw / J_W * dt - np.expand_dims(om_dot, -1) * dt, -OMEGA_W_MAX, OMEGA_W_MAX)
        s = np.concatenate([np.stack([x + vx * dt, y + vy * dt, vx, vy, th + om * dt, om], -1), Ow], -1)
    return s, fuel, (on > 0).astype(float)


# detumble -> aim -> burn -> return -> hold
class TurnAndBurn:

    def __init__(self, kp=8.0, ki=0.002, kd=40.0, name="PID"):
        self.kp, self.ki, self.kd, self.name = kp, ki, kd, name
        self.reset()

    def reset(self):
        self.mode, self.integ, self.log = "DETUMBLE", 0.0, []

    def att(self, err, om):
        self.integ = np.clip(self.integ + err * CTRL_DT, -0.3, 0.3)
        return -(self.kp * err + self.ki * self.integ + self.kd * om)

    def __call__(self, s):
        x, y, vx, vy, th, om = s[:6]
        v = np.hypot(vx, vy)
        th_burn = np.arctan2(-vy, -vx)
        duty = 0.0
        if self.mode == "DETUMBLE":
            tau = -self.kd * om
            if abs(om) < np.deg2rad(0.2):
                self.mode = "AIM" if v > 0.004 else "RETURN"; self.integ = 0
        if self.mode == "AIM":
            tau = self.att(wrap(th - th_burn), om)
            if abs(wrap(th - th_burn)) < np.deg2rad(2) and abs(om) < np.deg2rad(0.3):
                self.mode = "BURN"
        if self.mode == "BURN":
            e = wrap(th - th_burn)
            tau = self.att(e, om)
            duty = np.clip(v / 0.02, 0, 1) if abs(e) < np.deg2rad(8) else 0.0
            if v < 0.002:
                self.mode = "RETURN"; self.integ = 0
            elif abs(e) > np.deg2rad(8):
                self.mode = "AIM"
        if self.mode == "RETURN":
            tau = self.att(wrap(th), om)
            if abs(wrap(th)) < np.deg2rad(0.5) and abs(om) < np.deg2rad(0.05):
                self.mode = "HOLD"
        if self.mode == "HOLD":
            tau = self.att(wrap(th), om)
            if v > 0.006:
                self.mode = "AIM"
        self.log.append(self.mode)
        return np.array([tau, duty])


def LQR(q_th=100.0, q_om=1000.0, r=1.0):
    A = np.array([[0, 1], [0, 0.0]]); B = np.array([[0], [1 / I_BODY]])
    S = solve_continuous_are(A, B, np.diag([q_th, q_om]), np.array([[r]]))
    K = (B.T @ S / r).ravel()
    return TurnAndBurn(kp=K[0], ki=0.0, kd=K[1], name="LQR")


HID = 32
N_IN = 8
N_PARAMS = N_IN * HID + HID + HID * 2 + 2


def obs(s):
    th, om = s[..., 4], s[..., 5]
    c, sn = np.cos(th), np.sin(th)
    vxb = c * s[..., 2] + sn * s[..., 3]; vyb = -sn * s[..., 2] + c * s[..., 3]
    hw = np.sum(s[..., 6:9], -1) * J_W / (N_WHEELS * H_W_MAX)
    return np.stack([np.sin(th), np.cos(th) - 1, om / 0.3, om / 0.02, vxb / 0.1, vyb / 0.1,
                     np.tanh(np.hypot(vxb, vyb) / 0.01), hw], -1)


def policy_forward(w, s):
    i = 0
    W1 = w[..., i:i + N_IN * HID].reshape(w.shape[:-1] + (HID, N_IN)); i += N_IN * HID
    b1 = w[..., i:i + HID]; i += HID
    W2 = w[..., i:i + HID * 2].reshape(w.shape[:-1] + (2, HID)); i += HID * 2
    b2 = w[..., i:i + 2]
    h = np.tanh(np.einsum("...ij,...j->...i", W1, np.clip(obs(s), -5, 5)) + b1)
    o = np.einsum("...ij,...j->...i", W2, h) + b2
    return np.stack([N_WHEELS * TAU_W * np.tanh(o[..., 0]), np.clip(o[..., 1], 0, 1)], -1)


class RLPolicy:
    name = "RL"

    def __init__(self, path="rl_policy.npz"):
        self.w = np.load(path)["w"]

    def reset(self):
        self.log = []

    def __call__(self, s):
        return policy_forward(self.w, np.asarray(s))


def simulate(ctrl, s0, p=None, T=200.0, noise=0.0, seed=0):
    p = p or params()
    rng = np.random.default_rng(seed)
    ctrl.reset()
    n = int(T / CTRL_DT)
    s = np.array(list(s0) + [0, 0, 0], float) if len(s0) == 6 else np.array(s0, float)
    S = np.zeros((n + 1, 9)); C = np.zeros((n, 2)); S[0] = s
    fuel = 0.0; burns = 0
    for k in range(n):
        meas = s.copy()
        meas[:6] += noise * rng.standard_normal(6) * np.array([0, 0, 0.002, 0.002, np.deg2rad(0.1), np.deg2rad(0.02)])
        c = ctrl(meas); C[k] = c
        s, f, b = plant_step(s, c, p)
        fuel += f; burns += b; S[k + 1] = s
    return np.arange(n + 1) * CTRL_DT, S, C, fuel, burns


def metrics(t, S, fuel, burns):
    om_ok = np.abs(np.rad2deg(S[:, 5])) < 0.1
    ok = om_ok & (np.abs(np.rad2deg(wrap(S[:, 4]))) < 1.0) & (np.hypot(S[:, 2], S[:, 3]) < 0.005)
    def settle(o):
        bad = np.where(~o)[0]
        if len(bad) == 0: return 0.0
        return np.nan if bad[-1] + 1 >= len(t) else t[bad[-1] + 1]
    rpm = np.max(np.abs(S[:, 6:9])) * 60 / (2 * np.pi)
    return dict(t_rate=settle(om_ok), t_all=settle(ok), fuel_g=1000 * fuel, burns=int(burns),
                peak_rpm=float(rpm), final_rate=float(np.rad2deg(S[-1, 5])),
                final_att=float(np.rad2deg(wrap(S[-1, 4]))), final_v_mm=float(1000 * np.hypot(S[-1, 2], S[-1, 3])))


class Hybrid:
    name = "RL+PID"

    def __init__(self, path="rl_policy.npz"):
        self.rl = RLPolicy(path); self.fine = TurnAndBurn()

    def reset(self):
        self.rl.reset(); self.fine.reset(); self.k = 0; self.handed = False; self.log = []

    def __call__(self, s):
        self.k += 1
        if not self.handed and ((abs(s[5]) < np.deg2rad(0.5) and np.hypot(s[2], s[3]) < 0.02) or self.k * CTRL_DT > 60):
            self.handed = True
        self.log.append("PID" if self.handed else "RL")
        return self.fine(s) if self.handed else self.rl(s)
