"""How well can a hand alone calibrate the rig? Monte-Carlo ground truth for
local/handcal.py.

Six cameras on a ring look at a workspace; a hand (21 MediaPipe landmarks,
random finger curl) pauses at N random positions/orientations. Each
camera's detection gets pixel noise, extra error on landmarks hidden behind
the hand, occasional gross failures (fingers swapped, hand shifted),
misses when the palm is edge-on, and a little motion between the cameras'
capture times. MediaPipe's metric hand-size estimate is off by a random
~8%. Reports camera position/rotation error against the truth, separately
for shape (scale-free) and for the absolute scale.

    python test/hand_calib_sim.py --poses 6 10 15 25 40 --trials 8
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "local"))
import handcal  # noqa: E402

W, H = 1280, 720

# a right hand, mm, palm in the XY plane, fingers along +Y
BASE = {0: (0, 0), 1: (25, 25), 2: (40, 45), 3: (52, 62), 4: (60, 78),
        5: (22, 90), 6: (24, 130), 7: (25, 152), 8: (26, 170),
        9: (2, 92), 10: (2, 135), 11: (2, 160), 12: (2, 180),
        13: (-17, 86), 14: (-19, 124), 15: (-20, 147), 16: (-21, 165),
        17: (-33, 76), 18: (-38, 105), 19: (-40, 122), 20: (-42, 138)}
CHAINS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16),
          (17, 18, 19, 20)]


def hand_points(rng):
    P = np.array([[*BASE[k], 0.0] for k in range(21)], float)
    for chain in CHAINS:                       # curl each finger a bit
        curl = rng.uniform(0, 0.5)
        for j in range(1, len(chain)):
            pivot = P[chain[j - 1]].copy()
            a = curl
            Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                           [0, np.sin(a), np.cos(a)]])
            for k in chain[j:]:
                P[k] = pivot + Rx @ (P[k] - pivot)
    return P - P[[0, 5, 9, 13, 17]].mean(0)   # centred on the palm


def look_at(C, target):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.c_[x, y, z]
    return R_wc.T, -R_wc.T @ C                 # world -> camera


def make_rig(rng, radius=800.0):
    cams = {}
    for n, a in enumerate(np.linspace(0, 2 * np.pi, 7)[:6] + rng.uniform(-0.2, 0.2, 6)):
        C = np.array([radius * np.cos(a), radius * np.sin(a),
                      rng.uniform(200, 700)])
        R, t = look_at(C, np.array([0, 0, 0.0]) + rng.normal(0, 60, 3))
        f = 820.0 if n % 3 == 0 else 925.0
        K = np.array([[f, 0, W / 2 + rng.normal(0, 20)],
                      [0, f, H / 2 + rng.normal(0, 20)], [0, 0, 1]])
        cams[n] = (R, t, K)
    return cams


def rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def simulate(rng, cams, n_poses, px=2.5, hidden_px=6.0, gross=0.04,
             drift_mm_s=30.0, skew_s=0.04, scale_err=0.08):
    obs, truth_pts = [], []
    mp_scale = 1.0 + rng.normal(0, scale_err)    # MediaPipe size bias
    offsets = {n: rng.uniform(0, skew_s) for n in cams}
    while len(obs) < n_poses:
        hp = hand_points(rng)
        Rh = rand_rot(rng)
        c = rng.uniform([-250, -250, 0], [250, 250, 350])
        v = rng.normal(0, 1, 3)
        v *= drift_mm_s * rng.uniform(0, 1) / np.linalg.norm(v)
        world = (Rh @ hp.T).T + c
        normal = Rh @ np.array([0, 0, 1.0])
        pose = {}
        for n, (R, t, K) in cams.items():
            Xw = world + v * offsets[n]               # captured a bit later
            Xc = (R @ Xw.T).T + t
            if np.any(Xc[:, 2] < 100):
                continue
            uv = (K @ (Xc / Xc[:, 2:3]).T).T[:, :2]
            if not ((uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)).all():
                continue
            view = -(R.T @ t) - c
            edge = abs(normal @ view) / np.linalg.norm(view)
            if edge < 0.25 and rng.random() < 0.5:    # edge-on palm: often missed
                continue
            uv = uv + rng.normal(0, px, uv.shape)
            hidden = rng.random(21) < 0.3              # behind the hand
            uv[hidden] += rng.normal(0, hidden_px, (hidden.sum(), 2))
            if rng.random() < gross:                   # a bad detection
                if rng.random() < 0.5:
                    perm = list(range(21))
                    for a, b in ((5, 17), (6, 18), (7, 19), (8, 20), (9, 13),
                                 (10, 14), (11, 15), (12, 16)):
                        perm[a], perm[b] = perm[b], perm[a]
                    uv = uv[perm]
                else:
                    uv = uv + rng.normal(0, 40, 2)
            xy = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, None).reshape(-1, 2)
            wl = hp * mp_scale + rng.normal(0, 3, hp.shape)
            pose[n] = {"xy": xy, "f": float(K[0, 0]), "world": wl}
        if len(pose) >= 2:
            obs.append(pose)
            truth_pts.append(world)
    return obs


def evaluate(cams, res):
    T = res["T_world_cam"]
    nodes = sorted(T)
    C_est = np.array([T[n][:3, 3] for n in nodes])
    C_true = np.array([-cams[n][0].T @ cams[n][1] for n in nodes])
    # shape: best similarity fit (scale free); scale: ratio of that fit
    s, R, t = handcal.similarity_align(C_est, C_true)
    shape = np.linalg.norm((s * (R @ C_est.T)).T + t - C_true, axis=1)
    rot = []
    for n in nodes:
        R_est = R @ T[n][:3, :3]
        R_true = cams[n][0].T
        c = (np.trace(R_est.T @ R_true) - 1) / 2
        rot.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return shape, np.array(rot), 1.0 / s - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, nargs="+", default=[6, 10, 15, 25, 40])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--px", type=float, default=2.5)
    ap.add_argument("--gross", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    print(f"6 cameras, ring ~0.8 m; landmark noise {a.px} px (+6 px on ~30% "
          f"hidden), {a.gross:.0%} gross detections, 30 mm/s drift, 40 ms skew")
    for n in a.poses:
        pos, rot, scl, fails, tms = [], [], [], 0, []
        for tr in range(a.trials):
            rng = np.random.default_rng(a.seed * 1000 + tr)
            cams = make_rig(rng)
            obs = simulate(rng, cams, n, px=a.px, gross=a.gross)
            try:
                res = handcal.HandSolve(obs).solve()
            except (ValueError, np.linalg.LinAlgError, cv2.error) as e:
                fails += 1
                continue
            if res["unlinked"]:
                fails += 1
            sh, ro, sc = evaluate(cams, res)
            pos += list(sh)
            rot += list(ro)
            scl.append(abs(sc))
            tms.append(res["solve_s"])
        pos, rot = np.array(pos), np.array(rot)
        print(f"{n:3d} poses: camera position (shape) mean {pos.mean():6.1f} mm  "
              f"max {pos.max():6.1f} | rotation mean {rot.mean():.2f} deg max "
              f"{rot.max():.2f} | |scale err| mean {100 * np.mean(scl):.1f}% | "
              f"failures {fails}/{a.trials} | solve {np.mean(tms):.1f} s")


if __name__ == "__main__":
    main()
