"""Ray-trace a non-spinning black hole with a thin accretion disc (units: Schwarzschild radius rs = 1)."""
import numpy as np
from PIL import Image

import sys
W, H = int(sys.argv[1]) if len(sys.argv) > 1 else 1280, int(sys.argv[2]) if len(sys.argv) > 2 else 720
FOV = 1.05                       # horizontal field of view (radians)
CAM_H = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
CAM = np.array([0.0, CAM_H, -np.sqrt(max(18.0**2 - CAM_H**2, 1))])  # camera on a sphere of radius 18
STEPS = 700
DISC_IN, DISC_OUT = 3.0, 11.0    # inner edge at the innermost stable orbit (3 rs)
rng = np.random.default_rng(4)

# camera basis looking at the hole
fwd = -CAM / np.linalg.norm(CAM); up0 = np.array([0, 1.0, 0])
right = np.cross(fwd, up0); right /= np.linalg.norm(right); up = np.cross(right, fwd)
xs = np.linspace(-1, 1, W) * np.tan(FOV / 2); ys = np.linspace(1, -1, H) * np.tan(FOV / 2) * H / W
X, Y = np.meshgrid(xs, ys)
d = fwd[None, None, :] + X[..., None] * right + Y[..., None] * up
d /= np.linalg.norm(d, axis=-1, keepdims=True)
pos = np.broadcast_to(CAM, d.shape).reshape(-1, 3).copy(); vel = d.reshape(-1, 3).copy()
h2 = np.sum(np.cross(pos, vel) ** 2, axis=1)          # conserved angular momentum squared

def accel(p):                                          # photon acceleration around a Schwarzschild hole
    r2 = np.sum(p * p, axis=1); return -1.5 * h2[:, None] * p / (r2 ** 2.5)[:, None]

n = len(pos); colour = np.zeros((n, 3)); alive = np.ones(n, bool); hit_disc = np.zeros(n, bool)
for _ in range(STEPS):
    idx = np.where(alive)[0]
    if len(idx) == 0: break
    p, v = pos[idx], vel[idx]; hh = h2[idx]
    rnow = np.linalg.norm(p, axis=1)
    DT = np.clip(0.035 * rnow, 0.02, 0.9)[:, None]       # adaptive step: fine near the hole, coarse far away
    def acc(q): r2 = np.sum(q * q, axis=1); return -1.5 * hh[:, None] * q / (r2 ** 2.5)[:, None]
    k1v = acc(p); k1p = v
    k2v = acc(p + 0.5 * DT * k1p); k2p = v + 0.5 * DT * k1v
    k3v = acc(p + 0.5 * DT * k2p); k3p = v + 0.5 * DT * k2v
    k4v = acc(p + DT * k3p); k4p = v + DT * k3v
    newp = p + DT / 6 * (k1p + 2 * k2p + 2 * k3p + k4p); newv = v + DT / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    # did the ray cross the disc plane (y = 0) this step?
    cross = (p[:, 1] * newp[:, 1] < 0)
    if cross.any():
        t = p[cross, 1] / (p[cross, 1] - newp[cross, 1]); hitp = p[cross] + (newp[cross] - p[cross]) * t[:, None]
        rr = np.hypot(hitp[:, 0], hitp[:, 2]); ok = (rr > DISC_IN) & (rr < DISC_OUT)
        ids = idx[cross][ok]; r_ok = rr[ok]
        temp = (r_ok / DISC_IN) ** -0.75                                  # thin-disc temperature profile
        # disc gas orbits counter-clockwise; light from gas moving toward the camera is boosted
        vorb = np.sqrt(0.5 / r_ok)                                       # orbital speed (units of c, M = 0.5)
        tang = np.stack([-hitp[ok, 2], np.zeros_like(r_ok), hitp[ok, 0]], 1) / r_ok[:, None]
        cosang = -np.sum(tang * newv[cross][ok] / np.linalg.norm(newv[cross][ok], axis=1)[:, None], axis=1)
        gamma = 1 / np.sqrt(1 - vorb ** 2); dop = 1 / (gamma * (1 - vorb * cosang))
        grav = np.sqrt(np.clip(1 - 1 / r_ok, 0, 1))
        ang = np.arctan2(hitp[ok, 2], hitp[ok, 0])
        texture = (0.86 + 0.06 * np.sin(r_ok * 7.3 + 1.7 * np.sin(ang * 2 + r_ok * 0.8))
                   + 0.05 * np.sin(r_ok * 17.9 + 3.1 * np.sin(ang * 4.0))
                   + 0.03 * np.sin(r_ok * 41.0 + ang * 7))                 # turbulent gas, irregular rings
        bright = np.clip((temp * dop ** 2 * grav) ** 1.1 * texture * 1.25, 0, 3)
        hot = np.clip((DISC_IN * 1.6 / r_ok), 0, 1)                         # inner disc is whiter and hotter
        c = np.stack([bright, bright * (0.55 + 0.4 * hot), bright * (0.22 + 0.55 * hot ** 2)], 1)
        colour[ids] = np.clip(c, 0, 1); alive[ids] = False; hit_disc[ids] = True
    pos[idx], vel[idx] = newp, newv
    r = np.linalg.norm(newp, axis=1)
    fell = r < 1.0; alive[idx[fell]] = False                            # crossed the event horizon: black
    escaped = r > 60; e_ids = idx[escaped]
    if len(e_ids):                                                     # map escaping direction to a star field
        dirn = newv[escaped] / np.linalg.norm(newv[escaped], axis=1)[:, None]
        theta = np.arccos(np.clip(dirn[:, 1], -1, 1)); phi = np.arctan2(dirn[:, 2], dirn[:, 0])
        q = np.floor(dirn * 380).astype(np.int64)                        # cube cells on the sky: no pole streaks
        hsh = ((q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)) % 1000003
        star = hsh < 2400
        mag = (hsh % 101) / 101.0
        glow = 0.004 + 0.018 * np.exp(-((theta - 1.25 - 0.15 * np.sin(phi * 2)) ** 2) / 0.02)
        val = np.where(star, 0.25 + 0.75 * mag ** 2, glow)
        colour[e_ids] = np.stack([val, val, val * 1.08], 1).clip(0, 1); alive[e_ids] = False

img = (colour.reshape(H, W, 3) ** (1 / 1.6) * 255).clip(0, 255).astype(np.uint8)
OUTFILE = sys.argv[4] if len(sys.argv) > 4 else "../images/gargantua_raytraced.png"
Image.fromarray(img).save(OUTFILE)
print("disc pixels", hit_disc.mean(), "black", (colour.sum(1) == 0).mean())
