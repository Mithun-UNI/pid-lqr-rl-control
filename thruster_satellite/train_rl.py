import numpy as np
import sat_model as sm

rng = np.random.default_rng(1)
N_DIR, TOP_B, STEP, NOISE = 32, 16, 0.02, 0.05
EPIS, T_EP, ITERS = 6, 60.0, 400


def scenarios(seed, n=None):
    n = n or EPIS
    r = np.random.default_rng(seed)
    om0 = np.deg2rad(r.uniform(5, 40, n)) * r.choice([-1, 1], n)
    th0 = r.uniform(-np.pi, np.pi, n)
    vm = r.uniform(0, 0.3, n); va = r.uniform(-np.pi, np.pi, n)
    s0 = np.stack([np.zeros(n), np.zeros(n), vm * np.cos(va), vm * np.sin(va), th0, om0], -1)
    m = r.uniform(80, 130, n)
    p = dict(m=m, I=sm.inertia(m), F=sm.F_MAX, health=r.uniform(0.8, 1.0, (n, 4)))
    return s0, p


def rollout_batch(Wt, seed):
    s0, p = scenarios(seed)
    n = Wt.shape[0]
    s = np.broadcast_to(s0, (n,) + s0.shape).copy()
    Wb = np.repeat(Wt[:, None, :], EPIS, axis=1)
    ret = np.zeros((n, EPIS))
    for k in range(int(T_EP / sm.CTRL_DT)):
        d = sm.policy_forward(Wb, s)
        s, fuel, _ = sm.plant_step(s, d, p)
        om_d = np.abs(np.rad2deg(s[..., 5])); th_d = np.abs(np.rad2deg(sm.wrap(s[..., 4])))
        v = np.hypot(s[..., 2], s[..., 3])
        ok = (om_d < 0.1) & (th_d < 1.0) & (v < 0.005)
        ret += -(0.05 * om_d + 0.01 * th_d + 5 * v) - 0.03 * d.sum(-1) + 0.2 * ok
    return ret.mean(1)


# ARS (Mania et al. 2018)
def train(resume=False):
    w = np.zeros(sm.N_PARAMS)
    w[: sm.N_IN * sm.HID] = 0.3 * rng.standard_normal(sm.N_IN * sm.HID)
    i2 = sm.N_IN * sm.HID + sm.HID
    w[i2: i2 + sm.HID * 4] = 0.1 * rng.standard_normal(sm.HID * 4)
    hist = []
    if resume:
        d = np.load("rl_policy.npz"); w = d["w"]; hist = list(d["history"])
    for it in range(len(hist), len(hist) + ITERS):
        delta = rng.standard_normal((N_DIR, sm.N_PARAMS))
        R = rollout_batch(np.concatenate([w + NOISE * delta, w - NOISE * delta, w[None]]), seed=it)
        rp, rm, r0 = R[:N_DIR], R[N_DIR:2 * N_DIR], R[-1]
        top = np.argsort(np.maximum(rp, rm))[::-1][:TOP_B]
        sig = np.concatenate([rp[top], rm[top]]).std() + 1e-8
        w += STEP / (TOP_B * sig) * ((rp[top] - rm[top]) @ delta[top])
        hist.append(r0)
        if it % 20 == 0 or it == ITERS - 1:
            print(f"iter {it:3d}  return {r0:8.1f}", flush=True)
            np.savez("rl_policy.npz", w=w, history=np.array(hist))
    np.savez("rl_policy.npz", w=w, history=np.array(hist))


if __name__ == "__main__":
    import sys
    if "--refine" in sys.argv:
        global_ = None
        EPIS, T_EP, ITERS, NOISE, STEP = 12, 90.0, 220, 0.03, 0.015
        train(resume=True)
    else:
        train()
