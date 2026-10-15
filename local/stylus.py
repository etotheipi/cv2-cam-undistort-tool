"""Tag-adorned stylus: calibration (pivot) and live tip tracking.

Calibration runs after world calibration, so every camera's pose is known
and fixed. The tip sits in a divot while the stylus is turned through many
orientations; each snapshot sees some of its tags from some cameras.

  1. every (snapshot, tag) seen by any camera gets a world pose from ALL
     cameras that see it (the IPPE candidate that best fits every view,
     refined over all of them -- two views also settle a square's tilt);
  2. the tags' layout on the stylus is chained from tags seen together
     (robust mean of their relative poses), anchored on the most-seen tag;
  3. a rigid-body bundle adjustment refines the layout and every
     snapshot's stylus pose against all tag corners (Huber loss);
  4. pivot calibration: the tip is the one point of the stylus that stayed
     fixed in the world in every snapshot -- linear least squares for the
     tip (stylus frame) and the divot (world); then the BA is re-run with
     that constraint, a slipped tip in a snapshot being down-weighted;
  5. the stylus frame is re-centred on the tip, +Z pointing from the tip
     towards the tags.

The tip found is the centre of the ball; the contact point is one radius
further along -Z.

Live: the stylus pose comes from the best 2-3 of its tags that two or
more cameras see (triangulated corners, rigid fit), which in practice is
steadier than folding in every marginal detection.
test/stylus_sim.py checks the whole thing against a synthetic rig.
"""

import time

import cv2
import numpy as np


def _inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _T(r, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]
    T[:3, 3] = t
    return T


def _rt(T):
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def _rot_deg(A, B):
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def tag_corners(half):
    """Tag-frame corners in ArUco order (TL, TR, BR, BL; y up)."""
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], np.float64)


def robust_mean_pose(Ts):
    """Mean of the poses that agree with the most central one."""
    if len(Ts) > 2:
        med = min(Ts, key=lambda A: sum(_rot_deg(A, B) for B in Ts))
        Ts = [T for T in Ts if _rot_deg(T, med) < 10] or [med]
    U, _s, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    out = np.eye(4)
    out[:3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def kabsch(A, B):
    """T (4x4) with T @ A ~ B for matched 3-D points (rows)."""
    ma, mb = A.mean(0), B.mean(0)
    U, _s, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    T = np.eye(4)
    T[:3, :3] = U @ D @ Vt
    T[:3, 3] = mb - T[:3, :3] @ ma
    return T


def _reproj(T_w_obj, objp, views):
    """Stacked reprojection residuals (px) of object points in all views:
    views = [(T_cam<-world, K, dist, pts Nx2)]."""
    out = []
    for Tcw, K, d, pts in views:
        T = Tcw @ T_w_obj
        if T[2, 3] <= 0:
            out.append(np.full(2 * len(pts), 1e3))
            continue
        p, _ = cv2.projectPoints(objp, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, d)
        out.append((p.reshape(-1, 2) - pts).ravel())
    return np.concatenate(out)


def refine_pose(T, objp, views, iters=15):
    """Gauss-Newton on a 6-DOF pose against several fixed cameras."""
    x = _rt(T)
    for _ in range(iters):
        r = _reproj(_T(x[:3], x[3:]), objp, views)
        J = np.zeros((len(r), 6))
        for k in range(6):
            e = 1e-6 if k < 3 else 1e-4
            x2 = x.copy()
            x2[k] += e
            J[:, k] = (_reproj(_T(x2[:3], x2[3:]), objp, views) - r) / e
        try:
            dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x = x + dx
        if np.abs(dx).max() < 1e-7:
            break
    T = _T(x[:3], x[3:])
    r = _reproj(T, objp, views)
    return T, float(np.sqrt(np.mean(r ** 2)))


def tag_world_pose(views, half):
    """World pose of one tag seen by one or more fixed cameras.
    views = [(T_world<-cam, K, dist, pts 4x2)]. Returns (T, rms_px)."""
    objp = tag_corners(half)
    vv = [(_inv(Twc), K, d, np.asarray(p, np.float64)) for Twc, K, d, p in views]
    cands = []
    for Twc, K, d, p in views:
        try:
            n, rvecs, tvecs, _e = cv2.solvePnPGeneric(
                objp, np.asarray(p, np.float64), K, d,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            continue
        for i in range(n):
            cands.append(Twc @ _T(rvecs[i].ravel(), tvecs[i].ravel()))
    if not cands:
        return None, None

    def score(T):
        r = _reproj(T, objp, vv).reshape(-1, 2)
        return float(np.minimum(np.sum(r ** 2, axis=1), 400.0).sum())
    best = min(cands, key=score)
    return refine_pose(best, objp, vv)


# ------------------------------------------------------------ calibration
def calibrate(snapshots, cam_T, half, delta_px=2.0, tip_sigma_mm=0.3):
    """snapshots: [{"obs": {node: {tid: {"pts": 4x2}}}, "k_of": {node: (K, dist)}}]
    cam_T: {node: T_world<-cam}. Returns the stylus model and statistics."""
    t0 = time.perf_counter()
    objp = tag_corners(half)
    # 1. per (snapshot, tag) world poses
    poses = {}                       # (s, tid) -> (T, rms, n_views)
    for s, snap in enumerate(snapshots):
        by_tag = {}
        for n, tags in snap["obs"].items():
            if n not in cam_T or n not in snap["k_of"]:
                continue
            K, d = snap["k_of"][n]
            for tid, o in tags.items():
                by_tag.setdefault(int(tid), []).append(
                    (cam_T[n], np.asarray(K, float), np.asarray(d, float),
                     np.asarray(o["pts"], np.float64)))
        for tid, views in by_tag.items():
            T, rms = tag_world_pose(views, half)
            if T is not None:
                poses[(s, tid)] = (T, rms, len(views))
    good = {k: v for k, v in poses.items() if v[2] >= 2 and v[1] < 3.0}
    if not good:
        raise ValueError("no stylus tag was seen by two cameras at once")
    count = {}
    for (s, tid) in good:
        count[tid] = count.get(tid, 0) + 1
    ref = max(count, key=lambda t: (count[t], -t))
    # 2. layout: relative poses of tags seen together, chained from ref
    pair = {}
    snaps_of = {}
    for (s, tid), v in good.items():
        snaps_of.setdefault(s, {})[tid] = v[0]
    for s, tags in snaps_of.items():
        for a, Ta in tags.items():
            for b, Tb in tags.items():
                if a != b:
                    pair.setdefault((a, b), []).append(_inv(Ta) @ Tb)
    layout = {ref: np.eye(4)}
    frontier = [ref]
    while frontier:
        nxt = []
        for a in frontier:
            for (x, b), rels in sorted(pair.items(), key=lambda kv: -len(kv[1])):
                if x == a and b not in layout:
                    layout[b] = layout[a] @ robust_mean_pose(rels)
                    nxt.append(b)
        frontier = nxt
    # single-view-only tags, if consistent with a chained neighbour
    for (s, tid), v in poses.items():
        if tid in layout or v[1] > 1.5:
            continue
        others = [t for t in snaps_of.get(s, {}) if t in layout]
        if others:
            o = others[0]
            body = snaps_of[s][o] @ _inv(layout[o])
            layout[tid] = _inv(body) @ v[0]
    # body (stylus) pose per snapshot from its known tags
    bodies = {}
    for s in range(len(snapshots)):
        cand = [poses[(s, t)][0] @ _inv(layout[t]) for t in layout
                if (s, t) in poses and poses[(s, t)][1] < 3.0]
        if cand:
            bodies[s] = robust_mean_pose(cand)
    # 3-4. rigid-body BA, then pivot, then BA with the pivot constraint
    obs = []
    for s, snap in enumerate(snapshots):
        if s not in bodies:
            continue
        for n, tags in snap["obs"].items():
            if n not in cam_T or n not in snap["k_of"]:
                continue
            K, d = snap["k_of"][n]
            for tid, o in tags.items():
                if int(tid) in layout:
                    obs.append((s, int(tid), _inv(cam_T[n]), np.asarray(K, float),
                                np.asarray(d, float), np.asarray(o["pts"], np.float64)))
    bodies, layout, rms = _rigid_ba(obs, bodies, layout, ref, objp, delta_px)
    tip, pivot, tip_res = _pivot(bodies)
    bodies, layout, rms, tip, pivot = _rigid_ba(
        obs, bodies, layout, ref, objp, delta_px,
        tip=(tip, pivot, tip_sigma_mm))
    tip_res = np.array([np.linalg.norm(T[:3, :3] @ tip + T[:3, 3] - pivot)
                        for T in bodies.values()])
    # 5. stylus frame: origin at the tip, +Z towards the tags
    centres = np.array([T[:3, 3] for T in layout.values()])
    z = centres.mean(0) - tip
    z /= np.linalg.norm(z)
    xr = layout[ref][:3, 0] - (layout[ref][:3, 0] @ z) * z
    if np.linalg.norm(xr) < 1e-6:
        xr = np.cross(z, [1.0, 0, 0])
    x = xr / np.linalg.norm(xr)
    F = np.eye(4)
    F[:3, :3] = np.c_[x, np.cross(z, x), z]
    F[:3, 3] = tip
    model = {int(t): (_inv(F) @ T) for t, T in layout.items()}
    seen = {}
    for (s, tid, *_r) in obs:
        seen.setdefault(tid, set()).add(s)
    return {
        "tags": {str(t): np.round(T, 6).tolist() for t, T in model.items()},
        "tag_mm": 2 * half,
        "ref_tag": int(ref),
        "tip_to_tags_mm": round(float(np.linalg.norm(centres.mean(0) - tip)), 2),
        "stats": {
            "snapshots": len(snapshots), "snapshots_used": len(bodies),
            "tip_rms_mm": round(float(np.sqrt(np.mean(tip_res ** 2))), 3),
            "tip_max_mm": round(float(tip_res.max()), 3),
            "tip_residual_mm": {str(s): round(float(v), 3)
                                for s, v in zip(bodies, tip_res)},
            "reproj_rms_px": round(rms, 3),
            "tag_snapshots": {str(t): len(v) for t, v in sorted(seen.items())},
            "pivot_world": np.round(pivot, 2).tolist(),
            "solve_s": round(time.perf_counter() - t0, 2),
        },
    }


def _pivot(bodies):
    """Least squares for the tip t (body) and divot p (world):
    R_i t + t_i = p for every snapshot, re-solved without outliers."""
    Ts = list(bodies.values())
    keep = np.ones(len(Ts), bool)
    for _ in range(3):
        A = np.vstack([np.c_[T[:3, :3], -np.eye(3)] for T, k in zip(Ts, keep) if k])
        b = np.concatenate([-T[:3, 3] for T, k in zip(Ts, keep) if k])
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        t, p = sol[:3], sol[3:]
        res = np.array([np.linalg.norm(T[:3, :3] @ t + T[:3, 3] - p) for T in Ts])
        med = np.median(res)
        keep = res <= max(3 * med, 0.5)
    return t, p, res


def _rigid_ba(obs, bodies, layout, ref, objp, delta, tip=None, max_it=40):
    """Huber LM over snapshot stylus poses and tag placements (ref fixed);
    with tip=(t, p, sigma_mm) the pivot residual joins in (and t, p are
    solved too). Cameras stay fixed."""
    snaps, tags = sorted(bodies), sorted(t for t in layout if t != ref)
    blocks = [("b", s) for s in snaps] + [("t", t) for t in tags]
    if tip is not None:
        blocks.append(("tip", 0))
    off = {k: 6 * i for i, k in enumerate(blocks)}
    x = np.concatenate([_rt(bodies[s]) for s in snaps] +
                       [_rt(layout[t]) for t in tags] +
                       ([np.r_[tip[0], tip[1]]] if tip is not None else []))
    n_obs = len(obs)

    def body(xv, s):
        o = off[("b", s)]
        return _T(xv[o:o + 3], xv[o + 3:o + 6])

    def tagT(xv, t):
        if t == ref:
            return np.eye(4)
        o = off[("t", t)]
        return _T(xv[o:o + 3], xv[o + 3:o + 6])

    def res_obs(i, xv):
        s, t, Tcw, K, d, pts = obs[i]
        T = Tcw @ body(xv, s) @ tagT(xv, t)
        if T[2, 3] <= 0:
            return np.full(8, 1e3)
        p, _ = cv2.projectPoints(objp, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, d)
        return (p.reshape(-1, 2) - pts).ravel()

    def res_tip(j, xv):
        o = off[("tip", 0)]
        tt, pp = xv[o:o + 3], xv[o + 3:o + 6]
        B = body(xv, snaps[j])
        # in "pixel" units: tip_sigma corresponds to ~0.3 px of corner noise
        return (B[:3, :3] @ tt + B[:3, 3] - pp) * (0.3 / tip[2])

    def residuals(xv):
        r = [res_obs(i, xv) for i in range(n_obs)]
        if tip is not None:
            r += [res_tip(j, xv) for j in range(len(snaps))]
        return np.concatenate(r)

    # which residual rows each block touches
    rows = {k: [] for k in blocks}
    for i, (s, t, *_r) in enumerate(obs):
        rows[("b", s)].append(("o", i))
        if t != ref:
            rows[("t", t)].append(("o", i))
    if tip is not None:
        for j, s in enumerate(snaps):
            rows[("b", s)].append(("p", j))
            rows[("tip", 0)].append(("p", j))

    def row_slice(kind, i):
        return (slice(8 * i, 8 * i + 8) if kind == "o"
                else slice(8 * n_obs + 3 * i, 8 * n_obs + 3 * i + 3))

    def jac(xv, r):
        J = np.zeros((len(r), len(xv)))
        for k in blocks:
            o = off[k]
            for c in range(6):
                e = 1e-6 if (k[0] != "tip" and c < 3) else 1e-4
                x2 = xv.copy()
                x2[o + c] += e
                for kind, i in rows[k]:
                    sl = row_slice(kind, i)
                    new = res_obs(i, x2) if kind == "o" else res_tip(i, x2)
                    J[sl, o + c] = (new - r[sl]) / e
        return J

    def loss(r):
        pairs = r[:8 * n_obs].reshape(-1, 2)
        e = np.sqrt(np.sum(pairs ** 2, axis=1))
        big = e > delta
        w = np.where(big, delta / np.maximum(e, 1e-12), 1.0)
        rho = float(np.where(big, 2 * delta * e - delta ** 2, e ** 2).sum())
        sw = np.repeat(np.sqrt(w), 2)
        if tip is not None:
            tp = r[8 * n_obs:].reshape(-1, 3)
            et = np.linalg.norm(tp, axis=1)
            bt = et > delta
            wt = np.where(bt, delta / np.maximum(et, 1e-12), 1.0)
            rho += float(np.where(bt, 2 * delta * et - delta ** 2, et ** 2).sum())
            sw = np.r_[sw, np.repeat(np.sqrt(wt), 3)]
        return rho, sw

    r = residuals(x)
    cost, sw = loss(r)
    lam = 1e-3
    for _ in range(max_it):
        Jw = jac(x, r) * sw[:, None]
        A, g = Jw.T @ Jw, Jw.T @ (r * sw)
        D = np.diag(np.diag(A) + 1e-9)
        step = None
        for _try in range(8):
            try:
                dx = np.linalg.solve(A + lam * D, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r2 = residuals(x + dx)
            c2, sw2 = loss(r2)
            if c2 < cost:
                step = (dx, r2, c2, sw2)
                lam = max(lam / 3, 1e-9)
                break
            lam *= 5
        if step is None:
            break
        dx, r, c2, sw = step
        x = x + dx
        done = cost - c2 <= 1e-7 * cost
        cost = c2
        if done:
            break
    bodies = {s: body(x, s) for s in snaps}
    layout = {t: tagT(x, t) for t in [ref] + tags}
    rms = float(np.sqrt(np.mean(r[:8 * n_obs] ** 2)))
    if tip is None:
        return bodies, layout, rms
    o = off[("tip", 0)]
    return bodies, layout, rms, x[o:o + 3].copy(), x[o + 3:o + 6].copy()


# --------------------------------------------------------------- tracking
class StylusModel:
    """A calibrated stylus ready for tracking."""

    def __init__(self, cfg):
        cal = cfg.get("calibration") or {}
        self.id = cfg.get("id")
        self.name = cfg.get("name") or self.id
        self.radius = float(cfg.get("tip_diameter_mm") or 0) / 2
        # the divot the stylus was calibrated in is the origin of its
        # coordinates: easy to measure other points against. (World frame
        # at calibration time -- recalibrate after a new World Calibration.)
        pw = (cal.get("stats") or {}).get("pivot_world")
        self.origin = np.asarray(pw, float) if pw else None
        half = float(cal.get("tag_mm") or cfg.get("tag_mm") or 20) / 2
        objp = tag_corners(half)
        self.corners = {}
        for t, T in (cal.get("tags") or {}).items():
            T = np.asarray(T, float)
            self.corners[f"tag:{int(t)}"] = (T[:3, :3] @ objp.T).T + T[:3, 3]

    def _out(self, T, extra):
        tip = T[:3, 3]
        axis = T[:3, 2]                            # tip -> tags
        rel = tip - self.origin if self.origin is not None else None
        out = {"kind": "stylus", "id": f"stylus:{self.id}", "label": self.name,
               "tip": [round(float(v), 2) for v in tip],
               "origin": (self.origin.round(2).tolist()
                          if self.origin is not None else None),
               "tip_rel": [round(float(v), 2) for v in rel] if rel is not None else None,
               "dist_mm": round(float(np.linalg.norm(rel)), 2) if rel is not None else None,
               "contact": [round(float(v), 2) for v in tip - self.radius * axis],
               "axis": [round(float(v), 4) for v in axis],
               "T": np.round(T, 5).tolist(),
               "radius_mm": self.radius, "localized": True,
               "center": [round(float(v), 2) for v in tip]}
        out.update(extra)
        return out

    def locate_ml(self, items, obs_by_cam, geoms, prev_T=None, delta=2.0):
        """Maximum-likelihood stylus pose: the one 6-DOF pose that best
        reprojects EVERY stylus tag corner in EVERY camera that sees it --
        single-camera tags included (a corner seen once still pins a ray;
        the rigid model does the rest, so no camera has to judge depth).
        Huber loss; a tag sighting that still disagrees badly (a misread)
        is dropped and the pose refitted."""
        views = []
        for n, obs in obs_by_cam.items():
            g = geoms.get(n)
            if g is None:
                continue
            Tcw = np.eye(4)
            Tcw[:3, :3], Tcw[:3, 3] = g["R_cw"], g["t_cw"]
            for o in obs:
                if o.get("kind") == "aruco" and o.get("id") in self.corners:
                    views.append((n, o["id"], Tcw, g["K"], g["dist"],
                                  self.corners[o["id"]],
                                  np.asarray(o["points"], np.float64)))
        if not views:
            return None

        def resid(T, vv):
            out = []
            for _n, _t, Tcw, K, d, M, pts in vv:
                C = Tcw @ T
                Xc = (C[:3, :3] @ M.T).T + C[:3, 3]
                if (Xc[:, 2] <= 1).any():
                    out.append(np.full(2 * len(M), 1e3))
                    continue
                p, _ = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), K, d)
                out.append((p.reshape(-1, 2) - pts).ravel())
            return np.concatenate(out)

        def cost(T, vv):
            e = np.linalg.norm(resid(T, vv).reshape(-1, 2), axis=1)
            return float(np.minimum(e, 30.0).sum())

        # starting points: the triangulated fit, the last frame, and any
        # camera that sees 2+ stylus tags (non-planar: no flip ambiguity)
        starts = []
        k = self.locate(items, max_tags=99)
        if k is not None:
            starts.append(np.asarray(k["T"], float))
        if prev_T is not None:
            starts.append(prev_T)
        by_cam = {}
        for v in views:
            by_cam.setdefault(v[0], []).append(v)
        for n, vv in by_cam.items():
            if len(vv) < 2:
                continue
            M = np.vstack([v[5] for v in vv]).astype(np.float64)
            P = np.vstack([v[6] for v in vv]).astype(np.float64)
            ok, rv, tv = cv2.solvePnP(M, P, vv[0][3], vv[0][4], flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                starts.append(_inv(vv[0][2]) @ _T(rv.ravel(), tv.ravel()))
        if not starts:
            return None
        T = min(starts, key=lambda S: cost(S, views))

        def refine(T, vv):
            x = _rt(T)
            for _ in range(12):
                Tx = _T(x[:3], x[3:])
                r = resid(Tx, vv)
                e = np.repeat(np.linalg.norm(r.reshape(-1, 2), axis=1), 2)
                w = np.sqrt(np.where(e > delta, delta / np.maximum(e, 1e-9), 1.0))
                J = np.zeros((len(r), 6))
                for c in range(6):
                    h = 1e-6 if c < 3 else 1e-4
                    x2 = x.copy()
                    x2[c] += h
                    J[:, c] = (resid(_T(x2[:3], x2[3:]), vv) - r) / h
                dx = np.linalg.lstsq(J * w[:, None], -r * w, rcond=None)[0]
                x = x + dx
                if np.abs(dx).max() < 1e-7:
                    break
            return _T(x[:3], x[3:])

        T = refine(T, views)
        per = [float(np.sqrt(np.mean(resid(T, [v]) ** 2))) for v in views]
        med = float(np.median(per))
        keep = [v for v, e in zip(views, per) if e <= max(4.0, 4 * med)]
        if len(keep) < len(views) and keep:
            T = refine(T, keep)
            views = keep
        r = resid(T, views)
        return self._out(T, {
            "mode": "ml",
            "fit_px": round(float(np.sqrt(np.mean(r ** 2))), 3),
            "fit_mm": None,
            "tags_used": sorted({v[1] for v in views}),
            "corners_used": 4 * len(views),
            "cams_used": len({v[0] for v in views}),
            "n_views": len({v[0] for v in views})})

    def locate(self, items, max_tags=3):
        """Tip pose from the best (most views, lowest error) 2-3 stylus tags
        that at least two cameras see; None if there are none."""
        cands = [it for it in items
                 if it.get("id") in self.corners and (it.get("n_views") or 0) >= 2
                 and not it.get("single_view")
                 and it.get("points_world") and all(p is not None for p in it["points_world"])]
        if not cands:
            return None
        cands.sort(key=lambda it: (-it["n_views"], it.get("rms_px") or 99))
        use = cands[:max_tags]
        A = np.vstack([self.corners[it["id"]] for it in use])
        B = np.vstack([np.asarray(it["points_world"], float) for it in use])
        T = kabsch(A, B)
        fit = np.linalg.norm((T[:3, :3] @ A.T).T + T[:3, 3] - B, axis=1)
        return self._out(T, {
            "mode": "kabsch",
            "tags_used": [it["id"] for it in use],
            "fit_mm": round(float(np.sqrt(np.mean(fit ** 2))), 3),
            "n_views": min(it["n_views"] for it in use)})


# ---------------------------------------------------------------- capture
class StylusCalSession:
    """Snapshots for stylus calibration, taken through the tracker (same
    streams and intrinsics as the pose tab)."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.snaps = []

    def start(self):
        self.snaps = []

    def snap(self, tag_mm):
        obs, _views, k_of = self.tracker._capture_observations(float(tag_mm))
        self.snaps.append({"obs": {n: {t: {"pts": o["pts"]} for t, o in tags.items()}
                                   for n, tags in obs.items()},
                           "k_of": k_of})
        per_tag = {}
        for n, tags in obs.items():
            for t in tags:
                per_tag[t] = per_tag.get(t, 0) + 1
        return {"ok": True, "index": len(self.snaps),
                "tags": {str(t): c for t, c in sorted(per_tag.items())}}

    def solve(self, cam_T, tag_mm):
        if len(self.snaps) < 6:
            return {"ok": False, "error": f"only {len(self.snaps)} snapshots — "
                    "take at least 6, ideally 20+, turning the stylus through "
                    "every direction"}
        try:
            res = calibrate(self.snaps, cam_T, float(tag_mm) / 2)
        except (ValueError, np.linalg.LinAlgError, cv2.error) as e:
            return {"ok": False, "error": str(e)}
        res["ok"] = True
        return res
