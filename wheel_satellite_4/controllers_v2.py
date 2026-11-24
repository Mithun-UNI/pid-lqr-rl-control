import numpy as np
import wheel_model as wm

D2R = np.pi / 180
NAMES = ["DETUMBLE", "AIM", "BURN", "RETURN", "HOLD"]


# same mode logic as TurnAndBurn but vectorised for training
class VecTurnAndBurn:

    def __init__(self, shape=(), kp=8.0, ki=0.002, kd=40.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.mode = np.zeros(shape, int); self.integ = np.zeros(shape)

    def __call__(self, s):
        vx, vy, th, om = s[..., 2], s[..., 3], s[..., 4], s[..., 5]
        v = np.hypot(vx, vy); thb = np.arctan2(-vy, -vx)
        md = self.mode.copy(); duty = np.zeros_like(v); tau = np.zeros_like(v)
        reset_int = np.zeros_like(v, bool)
        m0 = md == 0
        tau = np.where(m0, -self.kd * om, tau)
        go = m0 & (np.abs(om) < 0.2 * D2R)
        md = np.where(go, np.where(v > 0.004, 1, 3), md); reset_int |= go
        e_b = wm.wrap(th - thb)
        m1 = md == 1
        go = m1 & (np.abs(e_b) < 2 * D2R) & (np.abs(om) < 0.3 * D2R)
        md = np.where(go, 2, md)
        m2 = md == 2
        duty = np.where(m2 & (np.abs(e_b) < 8 * D2R), np.clip(v / 0.02, 0, 1), 0.0)
        to_ret = m2 & (v < 0.002); to_aim = m2 & ~to_ret & (np.abs(e_b) > 8 * D2R)
        md = np.where(to_ret, 3, np.where(to_aim, 1, md)); reset_int |= to_ret
        m3 = md == 3
        e0 = wm.wrap(th)
        go = m3 & (np.abs(e0) < 0.5 * D2R) & (np.abs(om) < 0.05 * D2R)
        md = np.where(go, 4, md)
        m4 = md == 4
        md = np.where(m4 & (v > 0.006), 1, md)
        integ = np.where(reset_int, 0.0, self.integ)
        e = np.where((md == 1) | (md == 2), e_b, e0)
        use_pid = md != 0
        integ = np.where(use_pid, np.clip(integ + e * wm.CTRL_DT, -0.3, 0.3), integ)
        tau = np.where(use_pid & ~m0, -(self.kp * e + self.ki * integ + self.kd * om), tau)
        self.mode, self.integ = md, integ
        return np.stack([tau, duty], -1)


HID = 32
N_IN = 12


def obs(s):
    th, om = s[..., 4], s[..., 5]
    c, sn = np.cos(th), np.sin(th)
    vxb = c * s[..., 2] + sn * s[..., 3]; vyb = -sn * s[..., 2] + c * s[..., 3]
    v = np.hypot(vxb, vyb)
    hw = np.sum(s[..., 6:6 + wm.N_WHEELS], -1) * wm.J_W / (wm.N_WHEELS * wm.H_W_MAX)
    e = wm.wrap(th)
    return np.stack([np.sin(th), np.cos(th) - 1, om / 0.3, np.tanh(om / (0.5 * D2R)),
                     vxb / 0.1, vyb / 0.1, np.tanh(vxb / 0.005), np.tanh(vyb / 0.005), np.tanh(v / 0.01),
                     np.tanh(e / (2 * D2R)), np.tanh(e / (20 * D2R)), hw], -1)


def n_params(n_out):
    return N_IN * HID + HID + HID * n_out + n_out


def mlp(w, s, n_out):
    i = 0
    W1 = w[..., i:i + N_IN * HID].reshape(w.shape[:-1] + (HID, N_IN)); i += N_IN * HID
    b1 = w[..., i:i + HID]; i += HID
    W2 = w[..., i:i + HID * n_out].reshape(w.shape[:-1] + (n_out, HID)); i += HID * n_out
    b2 = w[..., i:i + n_out]
    h = np.tanh(np.einsum("...ij,...j->...i", W1, np.clip(obs(s), -5, 5)) + b1)
    return np.einsum("...ij,...j->...i", W2, h) + b2


TAU_TOT = wm.N_WHEELS * wm.TAU_W


def rl_v2_cmd(w, s):
    o = mlp(w, s, 3)
    tau = TAU_TOT * np.tanh(o[..., 0]) + 0.05 * np.tanh(o[..., 1])
    return np.stack([np.clip(tau, -TAU_TOT, TAU_TOT), np.clip(o[..., 2], 0, 1)], -1)


RES_TAU, RES_DUTY = 0.05, 0.2


# RL only nudges PID, and can only touch the thruster while PID is already burning
def residual_cmd(w, s, base_cmd):
    o = mlp(w, s, 2)
    tau = base_cmd[..., 0] + RES_TAU * np.tanh(o[..., 0])
    burning = base_cmd[..., 1] > 0
    duty = np.where(burning, np.clip(base_cmd[..., 1] + RES_DUTY * np.tanh(o[..., 1]), 0, 1), 0.0)
    return np.stack([tau, duty], -1)


class RLv2:
    name = "RL v2"

    def __init__(self, path="rl_v2.npz"):
        self.w = np.load(path)["w"]

    def reset(self):
        self.log = []

    def __call__(self, s):
        self.log.append("")
        return rl_v2_cmd(self.w, np.asarray(s))


class ResidualRL:
    name = "PID+RL residual"

    def __init__(self, path="rl_residual.npz"):
        self.w = np.load(path)["w"]

    def reset(self):
        self.base = VecTurnAndBurn(); self.log = []

    def __call__(self, s):
        s = np.asarray(s)
        b = self.base(s)
        self.log.append(NAMES[int(self.base.mode)] + "+RL")
        return residual_cmd(self.w, s, b)


class Hybrid:
    name = "RL v2→PID"

    def __init__(self, path="rl_v2.npz"):
        self.rl = RLv2(path); self.fine = wm.TurnAndBurn()

    def reset(self):
        self.rl.reset(); self.fine.reset(); self.k = 0; self.handed = False; self.log = []

    def __call__(self, s):
        self.k += 1
        if not self.handed and ((abs(s[5]) < 0.5 * D2R and np.hypot(s[2], s[3]) < 0.02) or self.k * wm.CTRL_DT > 60):
            self.handed = True
        self.log.append("PID" if self.handed else "RL")
        return self.fine(s) if self.handed else self.rl(s)
