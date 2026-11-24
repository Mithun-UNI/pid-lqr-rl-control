import numpy as np
import wheel_model as wm

rng = np.random.default_rng(3)
N_DIR, TOP_B, STEP, NOISE = 32, 16, 0.02, 0.05
EPIS, T_EP, ITERS, SUB_TRAIN = 8, 150.0, 300, 2
DRIFT_W = 5.0


def scenarios(seed, n=None):
    n = n or EPIS
    r = np.random.default_rng(seed)
    om0 = np.deg2rad(r.uniform(5, 35, n)) * r.choice([-1, 1], n)
    vm, va = r.uniform(0, 0.3, n), r.uniform(-np.pi, np.pi, n)
    s0 = np.zeros((n, 9))
    s0[:, 2], s0[:, 3], s0[:, 4], s0[:, 5] = vm * np.cos(va), vm * np.sin(va), r.uniform(-np.pi, np.pi, n), om0
    m = r.uniform(80, 130, n)
    wheels = np.ones((n, 3))
    dead = r.random(n) < 0.25
    wheels[dead, r.integers(0, 3, dead.sum())] = 0
    return s0, dict(m=m, I=m * (wm.W**2 + wm.H**2) / 12.0, wheels=wheels, offset=r.uniform(0, 0.06, n))


def rollout_batch(Wt, seed):
    s0, p = scenarios(seed)
    n = Wt.shape[0]
    s = np.broadcast_to(s0, (n,) + s0.shape).copy()
    Wb = np.repeat(Wt[:, None, :], EPIS, axis=1)
    ret = np.zeros((n, EPIS))
    for k in range(int(T_EP / wm.CTRL_DT)):
        c = wm.policy_forward(Wb, s)
        s, fuel, _ = wm.plant_step(s, c, p, sub=SUB_TRAIN)
        om_d = np.abs(np.rad2deg(s[..., 5])); th_d = np.abs(np.rad2deg(wm.wrap(s[..., 4])))
        v = np.hypot(s[..., 2], s[..., 3])
        hw = np.max(np.abs(s[..., 6:9]), -1) / wm.OMEGA_W_MAX
        ok = (om_d < 0.1) & (th_d < 1) & (v < 0.005)
        ret += -(0.05 * om_d + 0.01 * th_d + DRIFT_W * v) - 0.03 * c[..., 1] - 1.0 * np.maximum(0, hw - 0.9) + 0.2 * ok + 0.05 * (v < 0.005)
    return ret.mean(1)


# ARS (Mania et al. 2018)
def train(resume=False):
    w = np.zeros(wm.N_PARAMS)
    w[: wm.N_IN * wm.HID] = 0.3 * rng.standard_normal(wm.N_IN * wm.HID)
    i2 = wm.N_IN * wm.HID + wm.HID
    w[i2: i2 + wm.HID * 2] = 0.1 * rng.standard_normal(wm.HID * 2)
    hist = []
    if resume:
        d = np.load("rl_policy.npz"); w = d["w"]; hist = list(d["history"])
    for it in range(len(hist), len(hist) + ITERS):
        delta = rng.standard_normal((N_DIR, wm.N_PARAMS))
        R = rollout_batch(np.concatenate([w + NOISE * delta, w - NOISE * delta, w[None]]), seed=it)
        rp, rm, r0 = R[:N_DIR], R[N_DIR:2 * N_DIR], R[-1]
        top = np.argsort(np.maximum(rp, rm))[::-1][:TOP_B]
        sig = np.concatenate([rp[top], rm[top]]).std() + 1e-8
        w += STEP / (TOP_B * sig) * ((rp[top] - rm[top]) @ delta[top])
        hist.append(r0)
        if it % 20 == 0:
            print(f"iter {it:3d}  return {r0:8.1f}", flush=True)
            np.savez("rl_policy.npz", w=w, history=np.array(hist))
    np.savez("rl_policy.npz", w=w, history=np.array(hist))


if __name__ == "__main__":
    import sys
    if "--refine" in sys.argv:
        globals().update(EPIS=12, ITERS=200, NOISE=0.03, STEP=0.015, DRIFT_W=30.0)
        train(resume=True)
    else:
        train()
