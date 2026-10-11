"""Maps of the Interstellar world: the Gargantua system and the tesseract."""
import json, numpy as np, mpmath as mp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from interstellar_physics import r_isco, dilation_circular, spin_needed
import tesseract_climax as TC

SPIN = 1 - spin_needed()
planets = {"Miller's planet": float(r_isco(SPIN)), "Mann's planet": 28.0, "Edmunds' planet": 160.0}
dil = {k: float(dilation_circular(r, SPIN)) for k, r in planets.items()}
json.dump({"spin": str(SPIN), "radii": planets, "dilation": dil}, open("results/gargantua_system.json", "w"), indent=1)
print(dil)

# ---------------- system map (radial distance shown on a log scale so all planets fit) ----------
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(11, 11)); ax.set_aspect("equal"); ax.axis("off")
def R(r): return np.log10(r) * 3.2 + 0.6        # display radius
th = np.linspace(0, 2 * np.pi, 400)
ax.add_patch(plt.Circle((0, 0), R(1.0), color="black", zorder=3))
for rr, a in [(1.0, 0.9), (1.5, 0.35)]:
    ax.plot(R(rr) * np.cos(th), R(rr) * np.sin(th), color="#f2b45c", lw=1.2, alpha=a)
for r_in in np.linspace(3, 11, 30):
    ax.plot(R(r_in) * np.cos(th), R(r_in) * np.sin(th) * 0.35, color="#e0a050", lw=0.6, alpha=0.25)
angles = {"Miller's planet": 2.3, "Mann's planet": -0.6, "Edmunds' planet": 0.9}
cols = {"Miller's planet": "#4fa3d9", "Mann's planet": "#d9d9d9", "Edmunds' planet": "#7cc47c"}
for name, r in planets.items():
    ax.plot(R(r) * np.cos(th), R(r) * np.sin(th), color=cols[name], lw=1, ls="--", alpha=0.6)
    x, y = R(r) * np.cos(angles[name]), R(r) * np.sin(angles[name])
    ax.scatter(x, y, s=180, color=cols[name], zorder=4, edgecolor="white")
    d = dil[name]
    txt = f"{name}\norbit r = {r:.6f} M\n1 hour here = {d * 1:,.0f} hours on Earth" if d > 100 else f"{name}\norbit r = {r:.0f} M\n1 hour here = {d:.2f} hours on Earth"
    off = {"Miller's planet": (0.5, 0.4), "Mann's planet": (0.6, -0.9), "Edmunds' planet": (0.5, 0.4)}[name]
    ax.text(x + off[0], y + off[1], txt, fontsize=11, color=cols[name])
wx, wy = R(900) * np.cos(-2.2), R(900) * np.sin(-2.2)
ax.scatter(wx, wy, s=420, facecolor="none", edgecolor="#b39ddb", lw=2.5); ax.text(wx - 2.4, wy - 1.0, "wormhole to Saturn\n(Endurance arrives here)", color="#b39ddb", fontsize=11)
route = [(wx, wy), (R(planets["Miller's planet"]) * np.cos(2.3) * 1.6, R(planets["Miller's planet"]) * np.sin(2.3) * 1.6),
         (R(28) * np.cos(-0.6), R(28) * np.sin(-0.6)), (R(2.2) * np.cos(-1.6), R(2.2) * np.sin(-1.6)), (R(160) * np.cos(0.9), R(160) * np.sin(0.9))]
rx, ry = zip(*route); ax.plot(rx, ry, color="#ffcc66", lw=1.8, alpha=0.8)
for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#ffcc66", lw=1.5))
ax.text(R(2.2) * np.cos(-1.6) - 5.2, R(2.2) * np.sin(-1.6) - 1.6, "slingshot past Gargantua:\nCooper falls in toward the tesseract", color="#ffcc66", fontsize=10.5)
ax.text(-1.9, -0.25, "Gargantua", color="#f2b45c", fontsize=12, weight="bold")
lim = R(1400); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
fig.savefig("../images/gargantua_system_map.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------------- the tesseract: Murph's bedroom across time ----------------
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(12, 9)); ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor("black")
years = np.arange(2019, 2068); books = np.arange(0, 24)
Bx, Yy = np.meshgrid(books, years)
ax.plot_wireframe(Bx, Yy, np.zeros_like(Bx), color="#8a6d3b", lw=0.3, alpha=0.5)
for yv in years[::4]: ax.plot(books, [yv] * len(books), [0] * len(books), color="#c8a46a", lw=0.6, alpha=0.6)
stay_y = np.full(4, 2019); stay_b = np.array([3, 7, 11, 15])
ax.scatter(stay_b, stay_y, [0.3] * 4, color="#ff6b6b", s=90, depthshade=False, label="'S-T-A-Y' pushed off the shelf (2019)")
rng = np.random.default_rng(2)
watch_y = rng.uniform(2046, 2052, 260); watch_b = np.full(260, 20) + rng.normal(0, 0.3, 260)
ax.scatter(watch_b, watch_y, rng.uniform(0.05, 1.2, 260), color="#7fd3ff", s=6, depthshade=False, label="watch second-hand twitches carrying the data (2040s)")
ax.scatter([5], [2019], [1.6], color="#ffd166", s=160, marker="*", depthshade=False, label="dust coordinates for NASA (2019)")
ax.set_xlabel("position along Murph's bookshelf"); ax.set_ylabel("year"); ax.set_zlabel("gravitational signal")
ax.view_init(elev=24, azim=-58); ax.legend(loc="upper left", fontsize=10)
for a in (ax.xaxis, ax.yaxis, ax.zaxis): a.pane.set_facecolor((0, 0, 0, 1))
fig.savefig("../images/tesseract_world.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------------- the watch channel: a real Morse signal and Murph's noisy reading -------------
plt.style.use("default")
MORSE = {"A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
         "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--.."}
msg = "QUANTUMDATA"
sig = []
for ch in msg:
    for sym in MORSE[ch]: sig += [1] * (1 if sym == "." else 3) + [0]
    sig += [0, 0]
sig = np.array(sig, float); t = np.arange(len(sig))
noisy = sig + rng.normal(0, 0.22, len(sig))
fig, ax = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
ax[0].step(t, sig, where="post", color="black"); ax[0].set_yticks([]); ax[0].set_title("What the tesseract sends: 'QUANTUMDATA' in Morse code on the watch's second hand", loc="left", fontsize=11)
ax[1].plot(t, noisy, color="tab:blue", lw=0.8); ax[1].set_yticks([]); ax[1].set_title("What Murph sees: the same twitches with timing jitter", loc="left", fontsize=11); ax[1].set_xlabel("time (ticks)")
fig.tight_layout(); fig.savefig("../images/watch_morse_channel.png", dpi=130); plt.close(fig)

# ---------------- plain chart: how many repetitions does Murph need? ------------------
rc = json.load(open("results/tesseract_climax.json"))["repetition_curve"]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot([r for r, _ in rc], [p for _, p in rc], "o-", color="tab:blue")
ax.set_xlabel("times each bit of the data is repeated"); ax.set_ylabel("P(Murph decodes all 240 bits)")
ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("../images/murph_repetitions.png", dpi=130); plt.close(fig)
print("visuals done")
