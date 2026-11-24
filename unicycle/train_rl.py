import numpy as np
import unicycle_model as um

rng = np.random.default_rng(42)

N_DIR = 24
TOP_B = 12
STEP = 0.02
NOISE = 0.05
EPISODES = 4
T_EP = 6.0
ITERS = 300


def rollout_batch(W, seeds):
    n = W.shape[0]
    r = np.random.default_rng(seeds)
    M = r.uniform(3.0, 8.0, EPISODES)
    th0 = r.uniform(-0.35, 0.35, EPISODES)
    push_t = r.uniform(1.0, 4.0, EPISODES)
    push_F = r.uniform(-25, 25, EPISODES)

    Wb = np.repeat(W[:, None, :], EPISODES, axis=1)
    s = np.zeros((n, EPISODES, 4)); s[..., 2] = th0
    p = dict(um.P); p["M"] = M[None, :]
    alive = np.ones((n, EPISODES), bool)
    ret = np.zeros((n, EPISODES))
    buf = [np.zeros((n, EPISODES))] * um.DELAY
    for k in range(int(T_EP / um.DT)):
        noise = r.standard_normal(s.shape) * np.array([0.01, 0.02, 0.005, 0.02])
        buf.append(um.policy_forward(Wb, s + noise))
        u = buf.pop(0)
        s = um.step(s, u, p)
        t = k * um.DT
        hit = (t >= push_t) & (t < push_t + 0.1)
        s[..., 3] += np.where(hit, push_F * um.P["L"] / (M * um.P["L"] ** 2 + um.P["Ib"]) * um.DT, 0)
        x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        rew = 1.0 - (th**2 + 0.1 * thd**2 + 0.05 * x**2 + 0.01 * xd**2 + 0.0005 * u**2)
        ret += np.where(alive, rew, 0.0)
        alive &= np.abs(th) < um.FALL_ANGLE
        s = np.where(alive[..., None], s, 0.0)
    return ret.mean(axis=1)


# ARS (Mania et al. 2018)
def train():
    w = np.zeros(um.N_PARAMS)
    w[: 4 * um.HIDDEN] = 0.1 * rng.standard_normal(4 * um.HIDDEN)
    history = []
    for it in range(ITERS):
        delta = rng.standard_normal((N_DIR, um.N_PARAMS))
        W = np.concatenate([w + NOISE * delta, w - NOISE * delta, w[None]])
        R = rollout_batch(W, seeds=it)
        r_plus, r_minus, r_now = R[:N_DIR], R[N_DIR:2 * N_DIR], R[-1]
        top = np.argsort(np.maximum(r_plus, r_minus))[::-1][:TOP_B]
        sigma = np.concatenate([r_plus[top], r_minus[top]]).std() + 1e-8
        w = w + STEP / (TOP_B * sigma) * ((r_plus[top] - r_minus[top]) @ delta[top])
        history.append(r_now)
        if it % 10 == 0 or it == ITERS - 1:
            print(f"iter {it:3d}  return {r_now:7.1f}  (max possible ~{T_EP/um.DT:.0f})", flush=True)
    np.savez("rl_policy.npz", w=w, history=np.array(history))
    return w, history


if __name__ == "__main__":
    train()
