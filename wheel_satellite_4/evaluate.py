import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon, Circle
import wheel_model as wm
import controllers_v2 as cv

deg = np.deg2rad
MK = {"PID": wm.TurnAndBurn, "LQR": wm.LQR, "RL v2": cv.RLv2, "RLv2→PID": cv.Hybrid, "PID+RLres": cv.ResidualRL}
COL = {"PID": "#1f77b4", "LQR": "#7f7f7f", "RL v2": "#d62728", "RLv2→PID": "#9467bd", "PID+RLres": "#2ca02c"}
T = 250.0
out = []


def s0(spin, att, v, vdir=30):
    a = deg(vdir); return [0, 0, v * np.cos(a), v * np.sin(a), deg(att), deg(spin)]


SCEN = {
    "1. Normal separation: 30°/s, 20° off, 0.18 m/s drift":           dict(s0=s0(30, 20, 0.18)),
    "2. Violent separation: -35°/s, 150° off, 0.3 m/s drift":         dict(s0=s0(-35, 150, 0.3, -120)),
    "3. One wheel dead + 30°/s spin":                                 dict(s0=s0(30, 20, 0.18), p=wm.params(wheels=(1, 0, 1, 1))),
    "4. 55°/s spin (was impossible with 3 wheels)":                  dict(s0=s0(55, 20, 0.18)),
    "4b. 70°/s spin: more than even 4 wheels can hold":              dict(s0=s0(70, 20, 0.18)),
    "5. Thruster misaligned 8 cm (big disturbance torque in burns)":  dict(s0=s0(30, 20, 0.18), p=wm.params(offset=0.08)),
    "6. Heavier than expected: 130 kg":                               dict(s0=s0(30, 20, 0.18), p=wm.params(m=130)),
    "7. Raw (unfiltered) sensor noise":                               dict(s0=s0(30, 20, 0.18), noise=1.0),
}
f_ = lambda v: "never" if np.isnan(v) else f"{v:.0f} s"
out.append(f"{'Scenario':64s} {'Ctrl':9s} {'Spin stopped':>12s} {'All normal':>10s} {'Fuel g':>6s} {'Peak rpm':>8s} {'Final °/s':>9s} {'Final °':>7s} {'Drift mm/s':>10s}")
fig, ax = plt.subplots(len(SCEN), 4, figsize=(17, 2.5 * len(SCEN)), sharex=True)
runs = {}
for r, (name, sc) in enumerate(SCEN.items()):
    for c, mk in MK.items():
        ctrl = mk()
        t, S, C, fuel, burns = wm.simulate(ctrl, sc["s0"], p=sc.get("p"), T=T, noise=sc.get("noise", 0.0))
        runs[(r, c)] = (t, S, C, list(ctrl.log))
        m = wm.metrics(t, S, fuel, burns)
        out.append(f"{name:64s} {c:9s} {f_(m['t_rate']):>12s} {f_(m['t_all']):>10s} {m['fuel_g']:6.0f} {m['peak_rpm']:8.0f} "
                   f"{m['final_rate']:9.3f} {m['final_att']:7.2f} {m['final_v_mm']:10.1f}")
        ax[r, 0].plot(t, np.rad2deg(S[:, 5]), color=COL[c], label=c)
        ax[r, 1].plot(t, np.rad2deg(wm.wrap(S[:, 4])), color=COL[c])
        ax[r, 2].plot(t, 1000 * np.hypot(S[:, 2], S[:, 3]), color=COL[c])
        ax[r, 3].plot(t, np.max(np.abs(S[:, 6:6 + wm.N_WHEELS]), 1) * 60 / (2 * np.pi), color=COL[c])
    ax[r, 3].axhline(6000, color="k", ls=":", lw=1)
    ax[r, 0].set_title(name, loc="left", fontsize=9)
    for a, lab in zip(ax[r], ["spin (°/s)", "attitude (°)", "drift (mm/s)", "max wheel rpm"]):
        a.set_ylabel(lab); a.grid(alpha=.3)
    ax[r, 0].legend(fontsize=7)
for a in ax[-1]: a.set_xlabel("time (s)")
fig.suptitle("4 reaction wheels + 1 thruster: detumble, turn-and-burn, return to 0°  (dotted = wheel speed limit)", fontweight="bold")
fig.tight_layout(); fig.savefig("scenarios.png", dpi=110); plt.close(fig)

rng = np.random.default_rng(21)
N = 100
cases = []
for i in range(N):
    wh = np.ones(wm.N_WHEELS)
    if rng.random() < 0.25: wh[rng.integers(wm.N_WHEELS)] = 0
    cases.append(dict(s0=s0(rng.uniform(5, 40) * rng.choice([-1, 1]), rng.uniform(-180, 180), rng.uniform(0, 0.3), rng.uniform(-180, 180)),
                      p=wm.params(m=rng.uniform(70, 140), wheels=wh, offset=rng.uniform(0, 0.08))))
mc = {c: dict(ok=[], t=[], fuel=[]) for c in MK}
for i, cs in enumerate(cases):
    for c, mk in MK.items():
        t, S, C, fuel, burns = wm.simulate(mk(), cs["s0"], p=cs["p"], T=T, noise=0.25, seed=i)
        m = wm.metrics(t, S, fuel, burns)
        mc[c]["ok"].append(not np.isnan(m["t_all"])); mc[c]["t"].append(m["t_all"]); mc[c]["fuel"].append(m["fuel_g"])
out.append(f"\nMonte Carlo, {N} random separations (spin 5–40°/s, any attitude, drift ≤0.3 m/s, mass 70–140 kg, "
           f"25 % with a dead wheel, misalignment 0–8 cm, filtered-navigation noise), {T:.0f} s limit:")
for c in MK:
    ok = np.array(mc[c]["ok"]); tt = np.array(mc[c]["t"]); fu = np.array(mc[c]["fuel"])
    out.append(f"  {c:9s}: fully recovered {100*ok.mean():3.0f}%  | median time (successes) {np.nanmedian(tt):5.1f} s  | median fuel {np.median(fu):3.0f} g")

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for c in MK:
    tt = np.array(mc[c]["t"]); fu = np.array(mc[c]["fuel"]); ok = np.array(mc[c]["ok"])
    axs[0].scatter(np.where(ok, tt, T + 8), fu, s=12, color=COL[c], alpha=.6, label=f"{c} ({100*ok.mean():.0f}%)")
axs[0].axvline(T, color="k", ls=":", lw=1)
axs[0].set_xlabel("time to fully normal (s)  — right of dotted line = failed"); axs[0].set_ylabel("fuel (g)")
axs[0].legend(fontsize=7); axs[0].grid(alpha=.3); axs[0].set_title("Each dot = one random separation", loc="left", fontsize=9)
axs[1].bar(list(MK), [100 * np.mean(mc[c]["ok"]) for c in MK], color=[COL[c] for c in MK])
axs[1].set_ylim(0, 105); axs[1].set_ylabel(f"% fully recovered in {T:.0f} s"); axs[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig("montecarlo.png", dpi=120); plt.close(fig)

open("results.txt", "w").write("\n".join(out)); print("\n".join(out))

r = 0
fig, axs = plt.subplots(1, 5, figsize=(16, 3.8))
corners = np.array([[-wm.W / 2, -wm.H / 2], [wm.W / 2, -wm.H / 2], [wm.W / 2, wm.H / 2], [-wm.W / 2, wm.H / 2]])
wpos = np.array([[-0.6, 0], [-0.2, 0], [0.2, 0], [0.6, 0]])
art = {}
for a, c in zip(axs, MK):
    a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_facecolor("#0b1020"); a.set_title(c, color=COL[c], fontweight="bold")
    body = Polygon(corners, closed=True, fc="#c9ced6", ec="white", lw=1.5); a.add_patch(body)
    wheels = [Circle((0, 0), 0.17, fc="#3a4150", ec="#ffd34d", lw=1) for _ in range(wm.N_WHEELS)]
    for wc in wheels: a.add_patch(wc)
    spokes = [a.plot([], [], color="#ffd34d", lw=1.5)[0] for _ in range(wm.N_WHEELS)]
    flame, = a.plot([], [], color="#ff7a1a", lw=5, solid_capstyle="round")
    trail, = a.plot([], [], color=COL[c], lw=1, alpha=.6)
    txt = a.text(-2.9, 2.9, "", color="white", fontsize=6.5, family="monospace", va="top")
    art[c] = (body, wheels, spokes, flame, trail, txt)
fig.suptitle("Normal separation (30°/s, 20° off, 0.18 m/s drift). Yellow = reaction wheels (spokes show wheel spin), "
             "orange = the single thruster firing", fontsize=8)
wang = {c: np.cumsum(runs[(r, c)][1][:, 6:6 + wm.N_WHEELS] * wm.CTRL_DT, axis=0) for c in MK}
frames = range(0, int(140 / wm.CTRL_DT), 5)


def draw(k):
    for c in MK:
        t, S, C, log = runs[(r, c)]
        x, y, vx, vy, th, om = S[k, :6]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        body, wheels, spokes, flame, trail, txt = art[c]
        body.set_xy(corners @ R.T)
        for i in range(wm.N_WHEELS):
            pc = R @ wpos[i]; wheels[i].center = tuple(pc)
            ang = th + wang[c][k, i] * 0.02
            spokes[i].set_data([pc[0] - 0.17 * np.cos(ang), pc[0] + 0.17 * np.cos(ang)],
                               [pc[1] - 0.17 * np.sin(ang), pc[1] + 0.17 * np.sin(ang)])
        d = C[min(k, len(C) - 1), 1]
        if d > 0.05:
            p0 = R @ wm.THR_POS; p1 = p0 - (R @ wm.THR_DIR) * (0.3 + 0.9 * d)
            flame.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        else:
            flame.set_data([], [])
        j = slice(max(0, k - 600), k + 1); trail.set_data(S[j, 0] - x, S[j, 1] - y)
        rpm = S[k, 6:6 + wm.N_WHEELS] * 60 / (2 * np.pi)
        mode = log[min(k, len(log) - 1)] if log else ""
        txt.set_text(f"t {t[k]:5.0f}s  {mode}\nspin {np.rad2deg(om):6.2f}°/s\natt  {np.rad2deg(wm.wrap(th)):6.1f}°\n"
                     f"drift {1000*np.hypot(vx, vy):4.0f} mm/s\nwheel {np.mean(np.abs(rpm)):4.0f} rpm")
    return []


animation.FuncAnimation(fig, draw, frames=frames).save("wheel4_sat.gif", writer=animation.PillowWriter(fps=20), dpi=75)
print("saved")
