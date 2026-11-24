import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
import sat_model as sm

deg = np.deg2rad
CTRL = {"PID": sm.PID(), "LQR": sm.LQR(), "RL": sm.RLPolicy()}
COL = {"PID": "#1f77b4", "LQR": "#7f7f7f", "RL": "#d62728"}
T = 120.0
out = []


def s0(spin_dps, att_deg, v, v_dir_deg=30):
    a = deg(v_dir_deg)
    return [0, 0, v * np.cos(a), v * np.sin(a), deg(att_deg), deg(spin_dps)]


SCEN = {
    "1. Normal separation: 30°/s spin, 20° off, 0.18 m/s drift": dict(s0=s0(30, 20, 0.18)),
    "2. Violent separation: -40°/s, 150° off, 0.3 m/s drift":    dict(s0=s0(-40, 150, 0.3, -120)),
    "3. Heavier than expected: 130 kg (designed for 100)":       dict(s0=s0(30, 20, 0.18), p=sm.nominal_params(m=130)),
    "4. Weak thruster: top thruster at 60 %":                      dict(s0=s0(30, 20, 0.18), p=sm.nominal_params(health=(0.6, 1, 1, 1))),
    "5. Dead thruster: right thruster failed":                     dict(s0=s0(30, 20, 0.18), p=sm.nominal_params(health=(1, 1, 1, 0))),
    "6. Noisy gyro and star tracker":                              dict(s0=s0(30, 20, 0.18), noise=1.0),
}

hdr = f"{'Scenario':60s} {'Ctrl':4s} {'Spin<0.1°/s at':>14s} {'All normal at':>13s} {'Fuel(g)':>8s} {'Pulses':>7s} {'Final °/s':>9s} {'Final att°':>10s} {'Drift mm/s':>10s}"
out.append(hdr)
fig, ax = plt.subplots(len(SCEN), 3, figsize=(13, 2.5 * len(SCEN)), sharex=True)
runs = {}
for r, (name, sc) in enumerate(SCEN.items()):
    for c, ctrl in CTRL.items():
        t, S, D, fuel, pulses = sm.simulate(ctrl, sc["s0"], p=sc.get("p"), T=T, noise=sc.get("noise", 0.0))
        runs[(r, c)] = (t, S, D)
        m = sm.metrics(t, S, fuel, pulses)
        f = lambda v: "never" if np.isnan(v) else f"{v:.1f} s"
        out.append(f"{name:60s} {c:4s} {f(m['t_rate']):>14s} {f(m['t_all']):>13s} {m['fuel_g']:8.0f} {m['pulses']:7d} "
                   f"{m['final_rate']:9.3f} {m['final_att']:10.2f} {m['final_v_mm']:10.1f}")
        ax[r, 0].plot(t, np.rad2deg(S[:, 5]), color=COL[c], label=f"{c} ({m['fuel_g']:.0f} g)")
        ax[r, 1].plot(t, np.rad2deg(sm.wrap(S[:, 4])), color=COL[c])
        ax[r, 2].plot(t, 1000 * np.hypot(S[:, 2], S[:, 3]), color=COL[c])
    ax[r, 0].set_title(name, loc="left", fontsize=9)
    ax[r, 0].set_ylabel("spin (°/s)"); ax[r, 1].set_ylabel("attitude (°)"); ax[r, 2].set_ylabel("drift (mm/s)")
    ax[r, 0].legend(fontsize=7)
    for a in ax[r]: a.grid(alpha=.3)
for a in ax[-1]: a.set_xlabel("time (s)")
fig.suptitle("Detumble after separation — PID vs LQR vs RL (4 on/off thrusters, one per side)", fontweight="bold")
fig.tight_layout(); fig.savefig("scenarios.png", dpi=120); plt.close(fig)

rng = np.random.default_rng(11)
N = 120
cases = []
for i in range(N):
    cases.append(dict(s0=s0(rng.uniform(5, 45) * rng.choice([-1, 1]), rng.uniform(-180, 180), rng.uniform(0, 0.35), rng.uniform(-180, 180)),
                      p=sm.nominal_params(m=rng.uniform(70, 150), health=rng.uniform(0.6, 1.0, 4))))
mc = {c: dict(ok=[], t=[], fuel=[]) for c in CTRL}
for i, cs in enumerate(cases):
    for c, ctrl in CTRL.items():
        t, S, D, fuel, pulses = sm.simulate(ctrl, cs["s0"], p=cs["p"], T=T, noise=1.0, seed=i)
        m = sm.metrics(t, S, fuel, pulses)
        mc[c]["ok"].append(not np.isnan(m["t_all"])); mc[c]["t"].append(m["t_all"]); mc[c]["fuel"].append(m["fuel_g"])
out.append(f"\nMonte Carlo, {N} random separations (spin 5–45°/s, any attitude, drift ≤0.35 m/s, mass 70–150 kg, "
           f"each thruster 60–100 % healthy, sensor noise), 120 s limit:")
for c in CTRL:
    ok = np.array(mc[c]["ok"]); tt = np.array(mc[c]["t"]); fu = np.array(mc[c]["fuel"])
    out.append(f"  {c}: fully recovered {100*ok.mean():.0f}%  | median time {np.nanmedian(tt):.1f} s  | median fuel {np.median(fu):.0f} g")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for c in CTRL:
    tt = np.array(mc[c]["t"]); fu = np.array(mc[c]["fuel"]); ok = np.array(mc[c]["ok"])
    axs[0].scatter(np.where(ok, tt, T + 5), fu, s=12, color=COL[c], alpha=.6, label=f"{c} ({100*ok.mean():.0f}% recovered)")
axs[0].axvline(T, color="k", ls=":", lw=1); axs[0].text(T + 1, axs[0].get_ylim()[1] * 0.95, "failed", fontsize=7)
axs[0].set_xlabel("time to fully normal (s)"); axs[0].set_ylabel("fuel used (g)"); axs[0].legend(fontsize=7)
axs[0].set_title("Each dot = one random separation", loc="left", fontsize=9); axs[0].grid(alpha=.3)
names = list(CTRL)
axs[1].bar(names, [100 * np.mean(mc[c]["ok"]) for c in names], color=[COL[c] for c in names])
axs[1].set_ylabel("% fully recovered in 120 s"); axs[1].set_ylim(0, 105); axs[1].grid(alpha=.3, axis="y")
axs[1].set_title("Success rate", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig("montecarlo.png", dpi=120); plt.close(fig)

h = np.load("rl_policy.npz")["history"]
fig, a = plt.subplots(figsize=(6, 3))
a.plot(h, color=COL["RL"]); a.axvline(400, color="k", ls=":", lw=1); a.text(405, min(h) * 0.95, "refine stage", fontsize=7)
a.set_xlabel("training iteration"); a.set_ylabel("return"); a.set_title("RL learning curve (ARS)", loc="left"); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("learning_curve.png", dpi=120); plt.close(fig)

open("results.txt", "w").write("\n".join(out)); print("\n".join(out))

r = 1
fig, axs = plt.subplots(1, 3, figsize=(11, 4.2))
corners = np.array([[-sm.W / 2, -sm.H / 2], [sm.W / 2, -sm.H / 2], [sm.W / 2, sm.H / 2], [-sm.W / 2, sm.H / 2]])
art = {}
for a, c in zip(axs, CTRL):
    a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_facecolor("#0b1020"); a.set_title(c, color=COL[c], fontweight="bold")
    body = Polygon(corners, closed=True, fc="#c9ced6", ec="white", lw=1.5); a.add_patch(body)
    nose, = a.plot([], [], color="#ffd34d", lw=2)
    flames = [a.plot([], [], color="#ff7a1a", lw=4, solid_capstyle="round")[0] for _ in range(4)]
    trail, = a.plot([], [], color=COL[c], lw=1, alpha=.6)
    txt = a.text(-3.8, 3.8, "", color="white", fontsize=8, family="monospace", va="top")
    art[c] = (body, nose, flames, trail, txt)
fig.suptitle("Violent separation (-40°/s spin, 150° off, 0.3 m/s drift) → back to 0°/s, 0°  |  orange = thruster firing",
             fontsize=9)
step = 3
frames = range(0, int(60 / sm.CTRL_DT), step)


def draw(k):
    for c in CTRL:
        t, S, D = runs[(r, c)]
        x, y, vx, vy, th, om = S[k]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        body, nose, flames, trail, txt = art[c]
        body.set_xy(corners @ R.T)
        tip = R @ np.array([sm.W / 2 + 0.4, 0]); base = R @ np.array([sm.W / 2, 0])
        nose.set_data([base[0], tip[0]], [base[1], tip[1]])
        d = D[min(k, len(D) - 1)]
        for i, fl in enumerate(flames):
            if d[i] > 0.05:
                pos = R @ sm.THR_POS[i]; dirn = -(R @ sm.THR_DIR[i])
                end = pos + dirn * (0.3 + 0.9 * d[i])
                fl.set_data([pos[0], end[0]], [pos[1], end[1]])
            else:
                fl.set_data([], [])
        j = slice(max(0, k - 400), k + 1)
        trail.set_data(S[j, 0] - x, S[j, 1] - y)
        txt.set_text(f"t = {t[k]:5.1f} s\nspin {np.rad2deg(om):6.2f} °/s\natt  {np.rad2deg(sm.wrap(th)):6.1f} °\n"
                     f"drift {1000*np.hypot(vx, vy):5.0f} mm/s")
    return []


ani = animation.FuncAnimation(fig, draw, frames=frames)
ani.save("detumble.gif", writer=animation.PillowWriter(fps=25), dpi=80)
print("saved")
