import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import unicycle_model as um

CTRL = {"PID": um.PID(), "LQR": um.LQR(), "RL": um.RLPolicy()}
COL = {"PID": "#1f77b4", "LQR": "#7f7f7f", "RL": "#d62728"}
deg = np.deg2rad
out = []


def p_with(M):
    p = dict(um.P); p["M"] = M; return p


SCEN = {
    "1. Recover from 10° tilt (5 kg)":           dict(s0=[0, 0, deg(10), 0], p=p_with(5.0)),
    "2. 40 N shove at t=2 s (5 kg)":            dict(s0=[0, 0, 0, 0], p=p_with(5.0), push=(2.0, 40.0, 0.1)),
    "3. Heavier load 8 kg, 10° tilt (unknown to controller)": dict(s0=[0, 0, deg(10), 0], p=p_with(8.0)),
    "4. Lighter load 3 kg, 10° tilt":            dict(s0=[0, 0, deg(10), 0], p=p_with(3.0)),
    "5. 12 kg load (outside RL training range)": dict(s0=[0, 0, deg(10), 0], p=p_with(12.0)),
    "6. 10° tilt + noisy sensors":               dict(s0=[0, 0, deg(10), 0], p=p_with(5.0), noise=1.0),
}

fig, axes = plt.subplots(len(SCEN), 2, figsize=(11, 2.6 * len(SCEN)), sharex=True)
out.append(f"{'Scenario':58s} {'Ctrl':4s} {'Fell':5s} {'Settle(s)':>9s} {'MaxTilt°':>9s} {'Effort':>8s} {'Drift(m)':>8s}")
for row, (name, sc) in enumerate(SCEN.items()):
    for cname, c in CTRL.items():
        t, S, U, fell = um.simulate(c, sc["s0"], p=sc["p"], T=8, push=sc.get("push"),
                                    sensor_noise=sc.get("noise", 0.0))
        m = um.metrics(t, S, U, fell)
        out.append(f"{name:58s} {cname:4s} {str(m['fell']):5s} {m['settle_s']:9.2f} "
                   f"{m['max_tilt_deg']:9.1f} {m['effort']:8.1f} {m['final_x']:8.3f}")
        axes[row, 0].plot(t, np.rad2deg(S[:, 2]), color=COL[cname], label=cname + (" (FELL)" if fell else ""))
        axes[row, 1].plot(t[:-1], U, color=COL[cname], lw=0.8)
    axes[row, 0].set_title(name, fontsize=9, loc="left")
    axes[row, 0].set_ylabel("tilt (°)"); axes[row, 1].set_ylabel("torque (N·m)")
    axes[row, 0].legend(fontsize=7, loc="upper right"); axes[row, 0].grid(alpha=.3); axes[row, 1].grid(alpha=.3)
axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)")
fig.suptitle("Unicycle balancing a 5 kg mass — PID vs LQR vs RL", fontweight="bold")
fig.tight_layout(); fig.savefig("scenarios.png", dpi=130); plt.close(fig)

masses = np.arange(1.0, 20.5, 0.5)
fig, ax = plt.subplots(figsize=(7, 3.5))
out.append("\nMass sweep (10° start tilt): range of top mass each controller survives")
for cname, c in CTRL.items():
    st = []
    for M in masses:
        t, S, U, fell = um.simulate(c, [0, 0, deg(10), 0], p=p_with(M), T=8)
        m = um.metrics(t, S, U, fell)
        st.append(np.nan if fell else m["settle_s"])
    st = np.array(st); ok = masses[~np.isnan(st)]
    out.append(f"  {cname}: survives {ok.min():.1f}–{ok.max():.1f} kg" if len(ok) else f"  {cname}: none")
    ax.plot(masses, st, "o-", ms=3, color=COL[cname], label=cname)
ax.axvspan(3, 8, color="#d62728", alpha=.07, label="RL training range")
ax.axvline(5, color="k", ls=":", lw=1); ax.text(5.1, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 1, "design mass", fontsize=7)
ax.set_xlabel("top mass (kg)"); ax.set_ylabel("settling time (s)\n(gap = fell)")
ax.set_title("Robustness to an unknown load", loc="left"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("mass_sweep.png", dpi=130); plt.close(fig)

out.append("\nLargest start tilt recovered (5 kg):")
for cname, c in CTRL.items():
    best = 0
    for a in np.arange(1, 46, 1):
        _, S, U, fell = um.simulate(c, [0, 0, deg(a), 0], T=8)
        if fell: break
        best = a
    out.append(f"  {cname}: {best}°")

rng = np.random.default_rng(7)
N = 300
out.append(f"\nMonte Carlo, {N} random cases (mass 2–12 kg, tilt ±20°, shove ±50 N, noisy sensors): success rate")
cases = [(rng.uniform(2, 12), rng.uniform(-20, 20), rng.uniform(-50, 50), rng.uniform(1, 5)) for _ in range(N)]
mc = {}
for cname, c in CTRL.items():
    ok = 0
    for i, (M, a, F, tp) in enumerate(cases):
        _, S, U, fell = um.simulate(c, [0, 0, deg(a), 0], p=p_with(M), T=8, push=(tp, F, 0.1), sensor_noise=1.0, seed=i)
        ok += (not fell)
    mc[cname] = ok / N
    out.append(f"  {cname}: {100*ok/N:.1f}%")

h = np.load("rl_policy.npz")["history"]
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(h, color=COL["RL"]); ax.axhline(600, ls=":", color="k", lw=1)
ax.set_xlabel("training iteration"); ax.set_ylabel("episode return"); ax.set_title("RL learning curve (ARS)", loc="left")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("learning_curve.png", dpi=130); plt.close(fig)

open("results.txt", "w").write("\n".join(out))
print("\n".join(out))

p8 = p_with(8.0)
runs = {c: um.simulate(CTRL[c], [0, 0, deg(8), 0], p=p8, T=5, push=(2.0, 40.0, 0.1)) for c in CTRL}
fig, axs = plt.subplots(1, 3, figsize=(9, 3.4))
art = {}
for ax, c in zip(axs, CTRL):
    ax.set_xlim(-1, 1); ax.set_ylim(-0.05, 0.85); ax.set_aspect("equal"); ax.axhline(0, color="k", lw=1)
    ax.set_title(c, color=COL[c], fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    wheel = plt.Circle((0, um.P["r"]), um.P["r"], fc="none", ec="k", lw=2); ax.add_patch(wheel)
    pole, = ax.plot([], [], color=COL[c], lw=3)
    mass, = ax.plot([], [], "o", color=COL[c], ms=16)
    spoke, = ax.plot([], [], "k", lw=1)
    txt = ax.text(-0.95, 0.78, "", fontsize=8)
    art[c] = (wheel, pole, mass, spoke, txt)
fig.suptitle("8 kg load (controllers designed for 5 kg) + 40 N shove at t=2 s", fontsize=9)
frames = range(0, int(5 / um.DT), 3)


def draw(k):
    for c in CTRL:
        t, S, U, fell = runs[c]
        s = S[k] if not np.isnan(S[k, 0]) else S[~np.isnan(S[:, 0])][-1]
        x, _, th, _ = s
        x = np.clip(x, -0.8, 0.8)
        wheel, pole, mass, spoke, txt = art[c]
        wheel.center = (x, um.P["r"])
        tx, ty = x + um.P["L"] * np.sin(th), um.P["r"] + um.P["L"] * np.cos(th)
        pole.set_data([x, tx], [um.P["r"], ty]); mass.set_data([tx], [ty])
        phi = -s[0] / um.P["r"]
        spoke.set_data([x, x + um.P["r"] * np.sin(phi)], [um.P["r"], um.P["r"] + um.P["r"] * np.cos(phi)])
        fallen = np.isnan(S[k, 0])
        txt.set_text(f"t={k*um.DT:.2f}s" + ("  FELL" if fallen else "") + ("  SHOVE!" if 2.0 <= k*um.DT < 2.25 else ""))
    return []


ani = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
ani.save("unicycle_push.gif", writer=animation.PillowWriter(fps=33))
print("saved plots + gif")
