import sys
import numpy as np
import wheel_model as wm
import controllers_v2 as cv

MODE = sys.argv[1] if len(sys.argv) > 1 else "rl"
RESUME = "--resume" in sys.argv
OUT = "rl_v2.npz" if MODE == "rl" else "rl_residual.npz"
N_OUT = 3 if MODE == "rl" else 2
NP = cv.n_params(N_OUT)
rng = np.random.default_rng(5 if MODE == "rl" else 6)
N_DIR, TOP_B = 32, 16
STEP, NOISE = (0.02, 0.05) if MODE == "rl" else (0.01, 0.03)
EPIS, T_EP, SUB = 10, 150.0, 2
ITERS = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 300
TOTAL_PLANNED = 600 if MODE == "rl" else 250
D2R = np.pi / 180
DRIFT_W = 20.0 if ("--drift20" in sys.argv or MODE == "residual") else 3.0


def scenarios(seed, d):
    r = np.random.default_rng(seed)
    n = EPIS
    om0 = D2R * r.uniform(2, 5 + 35 * d, n) * r.choice([-1, 1], n)
    vm, va = r.uniform(0, 0.3 * d, n), r.uniform(-np.pi, np.pi, n)
    s0 = np.zeros((n, 6 + wm.N_WHEELS))
    s0[:, 2], s0[:, 3] = vm * np.cos(va), vm * np.sin(va)
    s0[:, 4], s0[:, 5] = r.uniform(-np.pi, np.pi, n) * d, om0
    m = r.uniform(80, 130, n)
    wheels = np.ones((n, wm.N_WHEELS))
    dead = r.random(n) < 0.25
    wheels[dead, r.integers(0, wm.N_WHEELS, dead.sum())] = 0
    return s0, dict(m=m, I=m * (wm.W**2 + wm.H**2) / 12.0, wheels=wheels, offset=r.uniform(0, 0.06, n))


def reward(s, cmd):
    om = np.abs(s[..., 5]) / D2R; th = np.abs(wm.wrap(s[..., 4])) / D2R; v = np.hypot(s[..., 2], s[..., 3])
    hw = np.max(np.abs(s[..., 6:6 + wm.N_WHEELS]), -1) / wm.OMEGA_W_MAX
    settled = (om < 0.1) & (th < 1) & (v < 0.005)
    tight = (om < 0.02) & (th < 0.2) & (v < 0.002)
    cost = (0.3 * np.log1p(om / 0.05) + 0.3 * np.log1p(th / 0.5) + 0.3 * np.log1p(v / 0.002)
            + 0.02 * om + 0.003 * th + DRIFT_W * v + 0.03 * cmd[..., 1] + 1.0 * np.maximum(0, hw - 0.9))
    return -cost + 1.0 * settled + 0.5 * tight


def rollout_batch(Wt, seed, d):
    s0, p = scenarios(seed, d)
    n = Wt.shape[0]
    s = np.broadcast_to(s0, (n,) + s0.shape).copy()
    Wb = np.repeat(Wt[:, None, :], EPIS, axis=1)
    base = cv.VecTurnAndBurn(shape=(n, EPIS)) if MODE == "residual" else None
    ret = np.zeros((n, EPIS))
    for k in range(int(T_EP / wm.CTRL_DT)):
        c = cv.rl_v2_cmd(Wb, s) if MODE == "rl" else cv.residual_cmd(Wb, s, base(s))
        s, _, _ = wm.plant_step(s, c, p, sub=SUB)
        ret += reward(s, c)
    return ret.mean(1)


# ARS (Mania et al. 2018)
def main():
    if RESUME:
        dat = np.load(OUT); w = dat["w"]; hist = list(dat["history"])
    else:
        w = np.zeros(NP); hist = []
        if MODE == "rl":
            w[: cv.N_IN * cv.HID] = 0.3 * rng.standard_normal(cv.N_IN * cv.HID)
            i2 = cv.N_IN * cv.HID + cv.HID
            w[i2: i2 + cv.HID * N_OUT] = 0.1 * rng.standard_normal(cv.HID * N_OUT)
        else:
            w[: cv.N_IN * cv.HID] = 0.3 * rng.standard_normal(cv.N_IN * cv.HID)
    start = len(hist)
    for it in range(start, start + ITERS):
        d = min(1.0, 0.15 + it / (0.6 * TOTAL_PLANNED)) if MODE == "rl" else 1.0
        delta = rng.standard_normal((N_DIR, NP))
        R = rollout_batch(np.concatenate([w + NOISE * delta, w - NOISE * delta, w[None]]), seed=10_000 + it, d=d)
        rp, rm, r0 = R[:N_DIR], R[N_DIR:2 * N_DIR], R[-1]
        top = np.argsort(np.maximum(rp, rm))[::-1][:TOP_B]
        sig = np.concatenate([rp[top], rm[top]]).std() + 1e-8
        w = w + STEP / (TOP_B * sig) * ((rp[top] - rm[top]) @ delta[top])
        hist.append(r0)
        if it % 20 == 0:
            print(f"[{MODE}] iter {it:3d}  difficulty {d:.2f}  return {r0:8.1f}", flush=True)
            np.savez(OUT, w=w, history=np.array(hist))
    np.savez(OUT, w=w, history=np.array(hist))
    print("done", flush=True)


if __name__ == "__main__":
    main()
