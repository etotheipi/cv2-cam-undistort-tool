"""How accurately can a tag-adorned stylus be calibrated and tracked?
Monte-Carlo ground truth for local/stylus.py.

Six cameras (~0.8 m ring) with known poses. The stylus is a D12 carrying
20 mm tags on 11 faces (the 12th holds the shaft), stuck on imprecisely,
with the tip some distance beyond the shaft face. Calibration: the tip
rests in a divot and the stylus is turned through random orientations
(up to 60 deg from vertical, any spin); then tracking is tested on new,
freely placed poses, using the best 2-3 tags seen by 2+ cameras.

    python test/stylus_sim.py --snaps 8 15 25 40 --trials 6
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "local"))
import hand_calib_sim as hs   # noqa: E402  (rig, rotations)
import pose_solver_sim as ps  # noqa: E402  (D12 geometry)
import stylus                 # noqa: E402

TAG = 20.0
W, H = 1280, 720


def make_stylus(rng, shaft_mm):
    inr = ps.d12_inradius(TAG)
    d12 = ps.make_d12(rng, sticker_mm=1.0, tilt_deg=0.5)
    tags = ps.d12_body_tags(inr, d12)            # body = D12 centre
    shaft = min(tags, key=lambda f: tags[f][2, 3])   # lowest face holds the shaft
    del tags[shaft]
    n = ps.d12_face_frames(inr)[shaft][:3, 2]
    tip = n * (inr + shaft_mm)                   # tip beyond the shaft face
    return tags, tip


def rot_to(a, b):
    """Rotation taking unit vector a onto b."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def pose_with_axis(rng, tip_body, max_tilt_deg):
    """Body orientation whose tip->body axis is within max_tilt of +Z."""
    axis_b = -tip_body / np.linalg.norm(tip_body)     # tip -> centre, body frame
    th = np.radians(rng.uniform(0, max_tilt_deg))
    ph = rng.uniform(0, 2 * np.pi)
    up = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    R = rot_to(axis_b, up)
    spin = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(spin), np.sin(spin)
    K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    Rs = np.eye(3) + s * K + (1 - c) * K @ K
    return Rs @ R


def observe(rng, cams, T_wb, tags, px):
    """{node: {tid: pts}} for tags each camera can see and detect."""
    objp = stylus.tag_corners(TAG / 2)
    out = {}
    for n, (R, t, K) in cams.items():
        C = -R.T @ t
        for tid, Tb in tags.items():
            Tw = T_wb @ Tb
            ctr, nrm = Tw[:3, 3], Tw[:3, 2]
            view = C - ctr
            cosv = nrm @ view / np.linalg.norm(view)
            if cosv < 0.25:
                continue                               # edge-on or facing away
            Xc = (R @ ((Tw[:3, :3] @ objp.T).T + Tw[:3, 3]).T).T + t
            if np.any(Xc[:, 2] < 50):
                continue
            uv = (K @ (Xc / Xc[:, 2:3]).T).T[:, :2]
            if not ((uv > 2).all() and (uv[:, 0] < W - 2).all() and (uv[:, 1] < H - 2).all()):
                continue
            side = np.linalg.norm(uv[0] - uv[1])
            if side < 14 or (cosv < 0.45 and rng.random() < 0.4):
                continue                               # too small / marginal
            out.setdefault(n, {})[tid] = uv + rng.normal(0, px, uv.shape)
    return out


def triangulate_items(cams, obs):
    """What the live tracker produces: per tag seen by 2+ cameras, DLT
    triangulated corners."""
    by = {}
    for n, tags in obs.items():
        for tid, pts in tags.items():
            by.setdefault(tid, []).append((n, pts))
    items = []
    for tid, views in by.items():
        if len(views) < 2:
            continue
        pw = []
        for k in range(4):
            A = []
            for n, pts in views:
                R, t, K = cams[n]
                P = K @ np.c_[R, t]
                x, y = pts[k]
                A += [x * P[2] - P[0], y * P[2] - P[1]]
            X = np.linalg.svd(np.array(A))[2][-1]
            pw.append(X[:3] / X[3])
        items.append({"id": f"tag:{tid}", "n_views": len(views), "rms_px": 0.3,
                      "single_view": False, "points_world": [list(p) for p in pw]})
    return items


def run_trial(rng, n_snaps, px, shaft_mm=90.0, tests=60):
    cams = hs.make_rig(rng)
    cam_T = {}
    for n, (R, t, K) in cams.items():
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.T, -R.T @ t
        cam_T[n] = T
    tags, tip_b = make_stylus(rng, shaft_mm)
    pivot = np.array([0.0, 0.0, 0.0]) + rng.normal(0, 30, 3) * [1, 1, 0]
    snaps = []
    while len(snaps) < n_snaps:
        Rb = pose_with_axis(rng, tip_b, 60)
        T_wb = np.eye(4)
        T_wb[:3, :3] = Rb
        T_wb[:3, 3] = pivot + rng.normal(0, 0.05, 3) - Rb @ tip_b   # tip in divot
        obs = observe(rng, cams, T_wb, tags, px)
        if obs:
            snaps.append({"obs": {n: {t: {"pts": p} for t, p in tg.items()}
                                  for n, tg in obs.items()},
                          "k_of": {n: (cams[n][2], np.zeros(5)) for n in cams}})
    res = stylus.calibrate(snaps, cam_T, TAG / 2)
    model = stylus.StylusModel({"id": "s", "tip_diameter_mm": 2,
                                "calibration": res})
    errs, missing = [], 0
    for _ in range(tests):
        Rb = pose_with_axis(rng, tip_b, 80)
        T_wb = np.eye(4)
        T_wb[:3, :3] = Rb
        T_wb[:3, 3] = rng.uniform([-200, -200, 0], [200, 200, 250]) - Rb @ tip_b
        items = triangulate_items(cams, observe(rng, cams, T_wb, tags, px))
        loc = model.locate(items)
        if loc is None:
            missing += 1
            continue
        errs.append(np.linalg.norm(np.array(loc["tip"]) - (T_wb[:3, :3] @ tip_b + T_wb[:3, 3])))
    return res, np.array(errs), missing / tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", type=int, nargs="+", default=[8, 15, 25, 40])
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--px", type=float, default=0.3)
    ap.add_argument("--shaft", type=float, default=90.0)
    a = ap.parse_args()
    print(f"20 mm tags on a D12, tip {a.shaft:.0f} mm past the shaft face; corner "
          f"noise {a.px} px; 6 cameras ~0.8 m")
    for n in a.snaps:
        cal_rms, errs, miss, fails = [], [], [], 0
        for tr in range(a.trials):
            rng = np.random.default_rng(100 + tr)
            try:
                res, e, m = run_trial(rng, n, a.px, a.shaft)
            except Exception as ex:
                fails += 1
                continue
            cal_rms.append(res["stats"]["tip_rms_mm"])
            errs += list(e)
            miss.append(m)
        errs = np.array(errs)
        print(f"{n:3d} calibration snapshots: pivot residual {np.mean(cal_rms):.2f} mm | "
              f"tracked tip error mean {errs.mean():.2f} mm, 95% {np.percentile(errs, 95):.2f}, "
              f"max {errs.max():.2f} | no tip {100 * np.mean(miss):.0f}% of poses | "
              f"failures {fails}/{a.trials}")


if __name__ == "__main__":
    main()
