"""Monte-Carlo check of the camera pose solver (Tracker._solve_graph).

Builds a synthetic rig with a known ground truth — cameras around a room,
a cube with a tag on each face moved through several snapshots — projects
the tag corners, adds noise, and runs the real solver on it (including the
per-tag PnP that seeds it). Reports camera position / rotation errors
against the truth and, when the solver provides them, how well its own
uncertainty estimates match the actual errors.

    .venv/bin/python test/pose_solver_sim.py            # default scenario
    .venv/bin/python test/pose_solver_sim.py --trials 20 --outliers 0

Noise model knobs: --sigma (px, every camera), --bad-cam-sigma (px, one
camera, standing in for a poorly calibrated wide lens), --outliers (gross
corner errors per trial, 4-10 px).
"""
import argparse
import math
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "local"))
import tracker as tracker_mod  # noqa: E402

W, H = 1280, 720
MARKER = 40.0          # tag side, mm
CUBE = 60.0            # cube edge, mm (tag centred on each face)
PHI = (1 + 5 ** 0.5) / 2


def rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) * math.radians(deg))[0]


def look_at(C, target, up=(0, 0, 1)):
    """T_world<-cam for a camera at C looking at target (OpenCV axes)."""
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, (0, 1, 0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = C
    return T


def make_rig(rng, radius=1200.0):
    """6 cameras on a ring (default 1.2 m radius) at mixed heights, all
    aimed near the workspace centre. Four ~69 deg lenses, two ~95 deg wide
    ones with stronger distortion."""
    cams = {}
    k = radius / 1200.0
    for i in range(6):
        a = math.radians(60 * i + rng.uniform(-10, 10))
        C = np.array([radius * math.cos(a), radius * math.sin(a),
                      rng.uniform(400, 1400) * k])
        T_wc = look_at(C, rng.uniform(-100, 100, 3) * [k, k, 0])
        wide = i in (1, 4)
        f = 700.0 if wide else 925.0
        K = np.array([[f, 0, W / 2 + rng.uniform(-20, 20)],
                      [0, f, H / 2 + rng.uniform(-15, 15)], [0, 0, 1]])
        dist = (np.array([-0.21, 0.06, 0, 0, -0.01]) if wide
                else np.array([0.035, -0.05, 0, 0, 0.0]))
        cams[i] = {"T_wc": T_wc, "K": K, "dist": dist}
    return cams


def cube_tags(T_w_cube):
    """T_world<-tag for the 5 upward/side faces of a cube (ids 0-4)."""
    h = CUBE / 2
    faces = [((0, 0, 0), 0), ((0, 1, 0), 90), ((0, 1, 0), -90),
             ((1, 0, 0), 90), ((1, 0, 0), -90)]
    out = {}
    for tid, (axis, deg) in enumerate(faces):
        R = rot(axis, deg) if deg else np.eye(3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = R @ np.array([0, 0, h])     # face centre, normal = R z
        out[tid] = T_w_cube @ T
    return out


def d12_normals():
    """Outward face normals of a regular dodecahedron (= icosahedron
    vertex directions)."""
    v = []
    for a in (-1, 1):
        for b in (-1, 1):
            v += [(0, a, b * PHI), (a, b * PHI, 0), (a * PHI, 0, b)]
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def d12_inradius(marker):
    """Centre-to-face distance of the smallest regular dodecahedron whose
    pentagonal faces hold a centred square tag of this side (+5 % margin):
    face inradius >= the tag's half-diagonal."""
    face_inr = marker / 2 * 2 ** 0.5 * 1.05
    edge = face_inr / 0.688191              # pentagon inradius = 0.688 * edge
    return 1.113516 * edge                  # dodecahedron inradius = 1.1135 * edge


def d12_tags(T_w_body, inradius, spin):
    """T_world<-tag for all 12 faces (ids 0-11); tag z = outward normal,
    each tag spun in its face plane by a fixed per-face angle."""
    out = {}
    for tid, nrm in enumerate(d12_normals()):
        ref = np.array([0, 0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0, 0])
        x = np.cross(ref, nrm)
        x /= np.linalg.norm(x)
        y = np.cross(nrm, x)
        R = np.stack([x, y, nrm], axis=1) @ rot((0, 0, 1), spin[tid])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = nrm * inradius
        out[tid] = T_w_body @ T
    return out


def resting_d12_pose(rng, spread, inradius):
    """Body pose for a D12 resting on a random face at a random yaw."""
    down = d12_normals()[rng.integers(12)]
    # rotate `down` onto -z, then spin about z
    z = np.array([0, 0, -1.0])
    ax = np.cross(down, z)
    if np.linalg.norm(ax) < 1e-9:
        R = np.eye(3) if down @ z > 0 else rot((1, 0, 0), 180)
    else:
        R = cv2.Rodrigues(ax / np.linalg.norm(ax) * math.acos(np.clip(down @ z, -1, 1)))[0]
    R = rot((0, 0, 1), rng.uniform(0, 360)) @ R
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [*rng.uniform(-spread, spread, 2), inradius]
    return T


def simulate(cams, rng, n_snaps, sigma, bad_cam, bad_sigma, n_outliers, spread=250.0,
             obj="cube"):
    half = MARKER / 2
    objp = np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], np.float64)
    snaps, truth_tags = [], []
    outlier_slots = []
    inr = d12_inradius(MARKER)
    spin = rng.uniform(0, 360, 12)
    for si in range(n_snaps):
        if obj == "d12":
            tags = d12_tags(resting_d12_pose(rng, spread, inr), inr, spin)
        else:
            T_cube = np.eye(4)
            T_cube[:3, :3] = rot((0, 0, 1), rng.uniform(0, 360))
            T_cube[:3, 3] = [*rng.uniform(-spread, spread, 2), CUBE / 2]
            tags = cube_tags(T_cube)
        truth_tags.append(tags)
        obs, views, k_of = {}, {}, {}
        for n, c in cams.items():
            T_cw = np.linalg.inv(c["T_wc"])
            k_of[n] = (c["K"], c["dist"])
            views[n] = {}
            for tid, T_wt in tags.items():
                T_ct = T_cw @ T_wt
                normal = T_ct[:3, :3] @ [0, 0, 1]
                to_cam = -T_ct[:3, 3] / np.linalg.norm(T_ct[:3, 3])
                if normal @ to_cam < math.cos(math.radians(70)):
                    continue                    # face turned away / too oblique
                pts, _ = cv2.projectPoints(objp, cv2.Rodrigues(T_ct[:3, :3])[0],
                                           T_ct[:3, 3], c["K"], c["dist"])
                pts = pts.reshape(4, 2)
                if (pts < 5).any() or (pts[:, 0] > W - 5).any() or (pts[:, 1] > H - 5).any():
                    continue
                s = bad_sigma if n == bad_cam else sigma
                pts = pts + rng.normal(0, s, pts.shape)
                obs.setdefault(n, {})[tid] = pts.astype(np.float32)
                outlier_slots.append((si, n, tid))
        snaps.append({"obs": obs, "views": views, "k_of": k_of})
    # gross outliers: push one corner of a random observation 4-10 px away
    for idx in rng.choice(len(outlier_slots), size=min(n_outliers, len(outlier_slots)), replace=False):
        si, n, tid = outlier_slots[idx]
        k = rng.integers(4)
        ang = rng.uniform(0, 2 * math.pi)
        snaps[si]["obs"][n][tid][k] += rng.uniform(4, 10) * np.array([math.cos(ang), math.sin(ang)], np.float32)
    # per-tag PnP, exactly as the live capture does
    for s in snaps:
        for n, tags in list(s["obs"].items()):
            K, dist = s["k_of"][n]
            full = {}
            for tid, pts in tags.items():
                sol = tracker_mod.Tracker._pnp_square(objp.astype(np.float32), pts, K, dist)
                if sol is None:
                    continue
                rvec, tvec, _ = sol
                T = np.eye(4)
                T[:3, :3] = cv2.Rodrigues(rvec)[0]
                T[:3, 3] = tvec.ravel()
                x, y = pts[:, 0], pts[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                full[tid] = {"T": T, "pts": pts, "area": float(max(area, 1.0))}
            s["obs"][n] = full
    return snaps, truth_tags


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--snaps", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--bad-cam-sigma", type=float, default=1.2)
    ap.add_argument("--outliers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--radius", type=float, default=1200.0,
                    help="camera ring radius, mm (workspace scales with it)")
    ap.add_argument("--object", choices=["cube", "d12"], default="cube",
                    help="calibration object: 60 mm cube (5 tags) or D12 (12 tags)")
    ap.add_argument("--plain", action="store_true",
                    help="plain least squares (Tracker.ROBUST = False)")
    a = ap.parse_args()

    tr = tracker_mod.Tracker({})
    tr.ROBUST = not a.plain
    pos_err, ang_err, z2, times, rms, iters, layout = [], [], [], [], [], [], []
    flagged, flagged_bad, unflagged_bad, n_cams = 0, 0, 0, 0
    per_cam_pos = {}
    for t in range(a.trials):
        rng = np.random.default_rng(a.seed + t)
        cams = make_rig(rng, a.radius)
        snaps, truth = simulate(cams, rng, a.snaps, a.sigma, 1, a.bad_cam_sigma, a.outliers,
                                spread=250.0 * a.radius / 1200.0, obj=a.object)
        # world = the most-seen tag of snapshot 0 (the solver reports poses
        # in the root tag's frame at its earliest snapshot)
        seen0 = {}
        for tags in snaps[0]["obs"].values():
            for tid in tags:
                seen0[tid] = seen0.get(tid, 0) + 1
        root = max(seen0, key=lambda k: (seen0[k], -k))
        T_w0 = np.linalg.inv(truth[0][root])
        t0 = time.perf_counter()
        res = tr._solve_graph(snaps, root, MARKER)
        times.append(time.perf_counter() - t0)
        last_snaps = snaps
        rms.append(res["rms_px"])
        iters.append(res.get("iterations", 0))
        # frame-free accuracy: camera centres after the best rigid fit to
        # the truth (removes the reference tag's own orientation error)
        E = np.array([np.array(c["T"])[:3, 3] for c in res["cameras"]])
        Tt = np.array([(T_w0 @ cams[c["node"]]["T_wc"])[:3, 3] for c in res["cameras"]])
        U, _S, Vt = np.linalg.svd((E - E.mean(0)).T @ (Tt - Tt.mean(0)))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        Ra = Vt.T @ D @ U.T
        layout.extend(np.linalg.norm((E - E.mean(0)) @ Ra.T - (Tt - Tt.mean(0)), axis=1))
        for c in res["cameras"]:
            T_true = T_w0 @ cams[c["node"]]["T_wc"]
            T_est = np.array(c["T"])
            e = T_est[:3, 3] - T_true[:3, 3]
            pos_err.append(np.linalg.norm(e))
            n_cams += 1
            if c.get("ambiguous"):
                flagged += 1
                flagged_bad += np.linalg.norm(e) > 100
            else:
                unflagged_bad += np.linalg.norm(e) > 100
            per_cam_pos.setdefault(c["node"], []).append(np.linalg.norm(e))
            ang_err.append(rot_err_deg(T_est[:3, :3], T_true[:3, :3]))
            cov = c.get("pos_cov_mm2")
            if cov is not None:
                # the covariance is relative to the anchor tag: compare there
                Ta = next(np.array(tg["T"]) for tg in res["tags"]
                          if tg["id"] == res["anchor"] and tg["snap"] == res["anchor_snap"])
                Ta_true = truth[res["anchor_snap"]][res["anchor"]]
                e_rel = (np.linalg.inv(Ta) @ T_est)[:3, 3] - \
                        (np.linalg.inv(Ta_true) @ cams[c["node"]]["T_wc"])[:3, 3]
                C_rel = Ta[:3, :3].T @ np.array(cov) @ Ta[:3, :3]
                z2.append(float(e_rel @ np.linalg.solve(C_rel, e_rel)))
    pe, ae = np.array(pos_err), np.array(ang_err)
    n_seen = [len(t) for sn in last_snaps for t in sn["obs"].values()]
    print(f"{a.object}: {np.mean(n_seen):.1f} tags seen per camera per snapshot (last trial)")
    print(f"{a.trials} trials, {a.snaps} snapshots, ring radius {a.radius:.0f} mm, sigma {a.sigma} px, "
          f"camera 1 sigma {a.bad_cam_sigma} px, {a.outliers} outlier corners/trial")
    print(f"  camera position error  mean {pe.mean():6.2f} mm   median {np.median(pe):6.2f}   max {pe.max():6.2f}")
    print(f"  camera rotation error  mean {ae.mean():6.3f} deg  median {np.median(ae):6.3f}   max {ae.max():6.3f}")
    print("  per camera (mean mm):  " + "  ".join(
        f"c{n}:{np.mean(v):5.2f}" for n, v in sorted(per_cam_pos.items())))
    print(f"  camera layout error    mean {np.mean(layout):6.2f} mm   max {np.max(layout):6.2f}   (after best rigid fit: frame-free)")
    print(f"  solve time             mean {np.mean(times):5.2f} s    max {np.max(times):5.2f} s   "
          f"iterations mean {np.mean(iters):.0f} max {np.max(iters)}   rms {np.mean(rms):.3f} px")
    print(f"  gross failures (>100 mm)  {flagged_bad + unflagged_bad} of {n_cams} cameras: "
          f"{flagged_bad} flagged ambiguous, {unflagged_bad} NOT flagged   "
          f"({flagged} flagged in total)")
    if z2:
        z2 = np.array(z2)
        # e' Cov^-1 e ~ chi-square(3) if the reported covariance is honest
        print(f"  uncertainty check      mean e'C^-1e {z2.mean():5.2f} (ideal 3.0)   "
              f"within 95% ellipsoid {np.mean(z2 < 7.815) * 100:5.1f}% (ideal 95%)")


if __name__ == "__main__":
    main()
