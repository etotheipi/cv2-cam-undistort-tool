"""World calibration from a moving hand -- no calibration object needed.

The hand is treated like a motion-capture wand with 21 markers whose
3-D layout is unknown and changes from pose to pose: every (pose,
landmark) is its own free 3-D point, and only the cameras are shared.
That is classic structure from motion with known intrinsics:

  1. relative pose of the best-connected camera pair from the essential
     matrix (RANSAC over every landmark of every shared pose);
  2. cameras added one at a time by PnP against the points triangulated
     so far (RANSAC again);
  3. a bundle adjustment over all cameras and points, solved with the
     Schur complement (cameras are few, points are many) under a Huber
     loss, so a bad detection is down-weighted instead of dragging every
     camera with it; gross outliers are then dropped and it is re-run.

A hand carries no absolute scale. It comes from MediaPipe's metric
"world landmarks" (a model estimate of the hand in metres, typically
within ~10-15% for an adult hand): the rigid palm segments triangulated
here are matched to the same segments there. The direction of up and
the origin come from a reference camera, as for the tag solve.

Observations are taken in undistorted normalized coordinates; residuals
are scaled by each camera's focal length so they read in pixels.
test/hand_calib_sim.py checks all of this against a synthetic rig.
"""

import time

import cv2
import numpy as np

# wrist + finger bases: the part of the hand that does not flex
PALM_SEGMENTS = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13),
                 (13, 17), (5, 17)]
HUBER_K = 2.5            # Huber threshold, in units of the noise sigma
OUTLIER_Z = 4.0          # observations beyond this many sigma are dropped
INIT_THRESH_PX = 6.0     # RANSAC threshold for the initial geometry


def _skew(v):
    """Batch of 3-vectors -> batch of cross-product matrices."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1),
                     np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], 1)


def _triangulate(Ps, xys):
    """Linear DLT from normalized views. Ps: [3x4 world->cam], xys: [(x, y)]."""
    A = []
    for P, (x, y) in zip(Ps, xys):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _u, _s, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


class HandSolve:
    """obs: list of poses; each pose is {node: {"xy": (21, 2) normalized
    undistorted coords, "f": focal length in px, "world": (21, 3) mm or
    None}}. Solves T_world_cam for every camera it can link."""

    def __init__(self, obs):
        self.obs = obs
        self.nodes = sorted({n for pose in obs for n in pose})
        self.log = []

    # ------------------------------------------------------------ helpers
    def _pairs(self, a, b):
        pa, pb, keys = [], [], []
        for i, pose in enumerate(self.obs):
            if a in pose and b in pose:
                pa.append(pose[a]["xy"])
                pb.append(pose[b]["xy"])
                keys += [(i, k) for k in range(len(pose[a]["xy"]))]
        if not pa:
            return None, None, []
        return np.vstack(pa), np.vstack(pb), keys

    def _f(self, n):
        return float(np.median([p[n]["f"] for p in self.obs if n in p]))

    def _P(self, n):
        R, t = self.cams[n]
        return np.hstack([R, t.reshape(3, 1)])

    def _triangulate_all(self, min_views=2, max_err_px=12.0):
        """(Re)triangulate every (pose, landmark) seen by >= min_views
        registered cameras; keeps those in front of every camera with a
        sane reprojection."""
        pts = {}
        for i, pose in enumerate(self.obs):
            views = [n for n in pose if n in self.cams]
            if len(views) < min_views:
                continue
            for k in range(len(pose[views[0]]["xy"])):
                Ps = [self._P(n) for n in views]
                xys = [pose[n]["xy"][k] for n in views]
                X = _triangulate(Ps, xys)
                if X is None:
                    continue
                ok = True
                for n, P, xy in zip(views, Ps, xys):
                    c = P[:, :3] @ X + P[:, 3]
                    if c[2] <= 1e-6 or (np.linalg.norm(c[:2] / c[2] - xy)
                                        * self._f(n) > max_err_px):
                        ok = False
                        break
                if ok:
                    pts[(i, k)] = X
        return pts

    # ------------------------------------------------------------- init
    def _init(self):
        best = None
        for ia, a in enumerate(self.nodes):
            for b in self.nodes[ia + 1:]:
                pa, pb, keys = self._pairs(a, b)
                if pa is None or len({k[0] for k in keys}) < 3 or len(pa) < 30:
                    continue
                thr = INIT_THRESH_PX / max(self._f(a), self._f(b))
                E, mask = cv2.findEssentialMat(
                    pa, pb, np.eye(3), method=cv2.RANSAC, prob=0.999,
                    threshold=thr)
                if E is None or E.shape != (3, 3):
                    continue
                n_in, R, t, mask2 = cv2.recoverPose(E, pa, pb, np.eye(3),
                                                    mask=mask.copy())
                if best is None or n_in > best[0]:
                    best = (n_in, a, b, R, t.ravel())
        if best is None:
            raise ValueError("no two cameras saw the hand together in at "
                             "least 3 poses")
        n_in, a, b, R, t = best
        self.log.append(f"seed pair video{a}+video{b}: {n_in} inliers")
        self.cams = {a: (np.eye(3), np.zeros(3)), b: (R, t)}
        self.root = a
        self.points = self._triangulate_all()
        self._ba(iters=15)
        while True:
            pending = [n for n in self.nodes if n not in self.cams]
            if not pending:
                break
            cand = []
            for n in pending:
                X, xy = [], []
                for (i, k), P in self.points.items():
                    if n in self.obs[i]:
                        X.append(P)
                        xy.append(self.obs[i][n]["xy"][k])
                cand.append((len(X), n, X, xy))
            cand.sort(key=lambda c: -c[0])
            added = False
            for cnt, n, X, xy in cand:
                if cnt < 12:
                    continue
                ok, rv, tv, inl = cv2.solvePnPRansac(
                    np.asarray(X, np.float64), np.asarray(xy, np.float64),
                    np.eye(3), None, reprojectionError=INIT_THRESH_PX / self._f(n),
                    iterationsCount=500, flags=cv2.SOLVEPNP_SQPNP)
                if not ok or inl is None or len(inl) < 10:
                    continue
                self.cams[n] = (cv2.Rodrigues(rv)[0], tv.ravel())
                self.log.append(f"added video{n}: {len(inl)}/{cnt} inliers")
                self.points = self._triangulate_all()
                self._ba(iters=15)
                added = True
                break
            if not added:
                break

    # ------------------------------------------------ bundle adjustment
    def _arrays(self):
        cams = sorted(self.cams)
        ci_of = {n: j for j, n in enumerate(cams)}
        keys = sorted(self.points)
        pi_of = {k: j for j, k in enumerate(keys)}
        ci, pi, xy, f = [], [], [], []
        for (i, k) in keys:
            for n, o in self.obs[i].items():
                if n in ci_of and (i, k) not in self.dropped.get(n, ()):
                    ci.append(ci_of[n])
                    pi.append(pi_of[(i, k)])
                    xy.append(o["xy"][k])
                    f.append(o["f"])
        return (cams, keys, np.array(ci), np.array(pi), np.array(xy),
                np.array(f, float))

    def _residuals(self, R, t, X, ci, pi, xy, f):
        RX = np.einsum("nij,nj->ni", R[ci], X[pi])
        Xc = RX + t[ci]
        uv = Xc[:, :2] / Xc[:, 2:3]
        return RX, Xc, (uv - xy) * f[:, None]

    def _ba(self, iters=40, sigma=None):
        """Huber-weighted LM over all cameras and points (Schur complement
        on the cameras); the root camera is held fixed, and the overall
        scale -- which hand data cannot fix -- is re-normalised each step."""
        if not hasattr(self, "dropped"):
            self.dropped = {}
        cams, keys, ci, pi, xy, f = self._arrays()
        if len(keys) == 0:
            return None
        nc, npnt = len(cams), len(keys)
        R = np.array([self.cams[n][0] for n in cams])
        t = np.array([self.cams[n][1] for n in cams])
        X = np.array([self.points[k] for k in keys])
        root = cams.index(self.root)
        # observation pairs sharing a point, for the reduced camera system
        by_pt = [[] for _ in range(npnt)]
        for o, p in enumerate(pi):
            by_pt[p].append(o)
        o1 = np.array([a for lst in by_pt for a in lst for _b in lst], int)
        o2 = np.array([b for lst in by_pt for _a in lst for b in lst], int)

        def cost_w(r, sig):
            s = np.linalg.norm(r, axis=1)
            k = HUBER_K * sig
            w = np.where(s <= k, 1.0, k / np.maximum(s, 1e-12))
            rho = np.where(s <= k, 0.5 * s * s, k * (s - 0.5 * k))
            return float(rho.sum()), w

        _RX, _Xc, r = self._residuals(R, t, X, ci, pi, xy, f)
        if sigma is None:
            sigma = max(0.3, float(np.median(np.linalg.norm(r, axis=1)))
                        / np.sqrt(2 * np.log(2)))
        cost, w = cost_w(r, sigma)
        lam = 1e-3
        for _it in range(iters):
            RX, Xc, r = self._residuals(R, t, X, ci, pi, xy, f)
            cost, w = cost_w(r, sigma)
            z = Xc[:, 2]
            Juv = np.zeros((len(ci), 2, 3))
            Juv[:, 0, 0] = 1 / z
            Juv[:, 0, 2] = -Xc[:, 0] / z ** 2
            Juv[:, 1, 1] = 1 / z
            Juv[:, 1, 2] = -Xc[:, 1] / z ** 2
            Juv *= f[:, None, None]
            Jc = np.concatenate([Juv @ -_skew(RX), Juv], axis=2)   # (no,2,6)
            Jp = Juv @ R[ci]                                       # (no,2,3)
            wJc, wJp = Jc * w[:, None, None], Jp * w[:, None, None]
            Hcc = np.zeros((nc, 6, 6))
            np.add.at(Hcc, ci, np.einsum("oai,oaj->oij", wJc, Jc))
            Hpp = np.zeros((npnt, 3, 3))
            np.add.at(Hpp, pi, np.einsum("oai,oaj->oij", wJp, Jp))
            Hcp = np.einsum("oai,oaj->oij", wJc, Jp)               # (no,6,3)
            gc = np.zeros((nc, 6))
            np.add.at(gc, ci, np.einsum("oai,oa->oi", wJc, r))
            gp = np.zeros((npnt, 3))
            np.add.at(gp, pi, np.einsum("oai,oa->oi", wJp, r))
            improved, rel = False, 0.0
            for _try in range(8):
                Hcc_d = Hcc + lam * np.einsum("nii->ni", Hcc)[:, :, None] * np.eye(6)
                Hcc_d += 1e-9 * np.eye(6)
                Hpp_d = Hpp + lam * np.einsum("nii->ni", Hpp)[:, :, None] * np.eye(3)
                Hpp_d += 1e-9 * np.eye(3)
                Hpp_inv = np.linalg.inv(Hpp_d)
                Y = np.einsum("oij,ojk->oik", Hcp, Hpp_inv[pi])    # (no,6,3)
                S = np.zeros((nc, nc, 6, 6))
                S[np.arange(nc), np.arange(nc)] = Hcc_d
                np.add.at(S, (ci[o1], ci[o2]),
                          -np.einsum("oij,okj->oik", Y[o1], Hcp[o2]))
                bc = gc.copy()
                np.add.at(bc, ci, -np.einsum("oij,oj->oi", Y, gp[pi]))
                keep = [j for j in range(nc) if j != root]
                idx = np.concatenate([np.arange(6 * j, 6 * j + 6) for j in keep])
                Sf = S.transpose(0, 2, 1, 3).reshape(6 * nc, 6 * nc)
                try:
                    dck = np.linalg.solve(Sf[np.ix_(idx, idx)], -bc.reshape(-1)[idx])
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                dc = np.zeros(6 * nc)
                dc[idx] = dck
                dc = dc.reshape(nc, 6)
                hpc = np.zeros((npnt, 3))
                np.add.at(hpc, pi, np.einsum("oij,oi->oj", Hcp, dc[ci]))
                dp = np.einsum("pij,pj->pi", Hpp_inv, -gp - hpc)
                R2 = np.array([cv2.Rodrigues(dc[j, :3])[0] @ R[j] for j in range(nc)])
                t2 = t + dc[:, 3:]
                X2 = X + dp
                # hold the scale still: mean camera distance from the root
                C2 = np.einsum("nji,nj->ni", R2, -t2)
                s = np.mean(np.linalg.norm(C2 - C2[root], axis=1)[[j for j in keep]])
                C1 = np.einsum("nji,nj->ni", R, -t)
                s0 = np.mean(np.linalg.norm(C1 - C1[root], axis=1)[[j for j in keep]])
                if s > 1e-9 and s0 > 1e-9:
                    k = s0 / s
                    t2, X2 = t2 * k, X2 * k
                _a, Xc2, r2 = self._residuals(R2, t2, X2, ci, pi, xy, f)
                if np.all(Xc2[:, 2] > 0):
                    c2, _w = cost_w(r2, sigma)
                    if c2 < cost:
                        R, t, X = R2, t2, X2
                        lam = max(lam / 3, 1e-7)
                        improved = True
                        rel = (cost - c2) / max(cost, 1e-12)
                        cost = c2
                        break
                lam *= 10
            if not improved or rel < 1e-7:
                break
        for j, n in enumerate(cams):
            self.cams[n] = (R[j], t[j])
        for j, k in enumerate(keys):
            self.points[k] = X[j]
        return sigma

    # ----------------------------------------------------------- solve
    def solve(self):
        t0 = time.perf_counter()
        self.dropped = {}
        self._init()
        sigma = self._ba(iters=60)
        # drop gross outliers (bad detections), then refine again
        for _round in range(2):
            cams, keys, ci, pi, xy, f = self._arrays()
            R = np.array([self.cams[n][0] for n in cams])
            t = np.array([self.cams[n][1] for n in cams])
            X = np.array([self.points[k] for k in keys])
            _a, _b, r = self._residuals(R, t, X, ci, pi, xy, f)
            s = np.linalg.norm(r, axis=1)
            sig = max(0.3, float(np.median(s)) / np.sqrt(2 * np.log(2)))
            bad = np.nonzero(s > OUTLIER_Z * sig)[0]
            for o in bad:
                self.dropped.setdefault(cams[ci[o]], set()).add(keys[pi[o]])
            # points left with < 2 views go
            self.points = {k: v for k, v in self.points.items()
                           if sum(1 for n in self.obs[k[0]]
                                  if n in self.cams and k not in self.dropped.get(n, ()))
                           >= 2}
            sigma = self._ba(iters=60)
        self.solve_s = time.perf_counter() - t0
        return self._report()

    def _scale_mm(self):
        """mm per solver unit, from MediaPipe's metric palm segments."""
        ratios = []
        for (i, pose) in enumerate(self.obs):
            for a, b in PALM_SEGMENTS:
                if (i, a) not in self.points or (i, b) not in self.points:
                    continue
                tri = np.linalg.norm(self.points[(i, a)] - self.points[(i, b)])
                mm = [np.linalg.norm(o["world"][a] - o["world"][b])
                      for o in pose.values() if o.get("world") is not None]
                if tri > 1e-9 and mm:
                    ratios.append(float(np.median(mm)) / tri)
        return float(np.median(ratios)) if ratios else None

    def _report(self):
        cams, keys, ci, pi, xy, f = self._arrays()
        R = np.array([self.cams[n][0] for n in cams])
        t = np.array([self.cams[n][1] for n in cams])
        X = np.array([self.points[k] for k in keys])
        _a, _b, r = self._residuals(R, t, X, ci, pi, xy, f)
        s = np.linalg.norm(r, axis=1)
        scale = self._scale_mm() or 1.0
        T = {}
        for n in cams:
            Rn, tn = self.cams[n]
            M = np.eye(4)
            M[:3, :3] = Rn.T
            M[:3, 3] = -Rn.T @ (tn * scale)
            T[n] = M
        per = {}
        for j, n in enumerate(cams):
            sel = ci == j
            per[n] = {"noise_px": float(np.median(s[sel]) / np.sqrt(2 * np.log(2)))
                      if sel.any() else None,
                      "points": int(sel.sum()),
                      "dropped": len(self.dropped.get(n, ())),
                      "poses": sum(1 for p in self.obs if n in p)}
        return {"T_world_cam": T, "scale_mm": scale,
                "scale_from": "mediapipe" if self._scale_mm() else None,
                "rms_px": float(np.sqrt(np.mean(s ** 2))) if len(s) else None,
                "noise_px": float(np.median(s) / np.sqrt(2 * np.log(2)))
                if len(s) else None,
                "per_cam": per, "points": len(keys),
                "unlinked": [n for n in self.nodes if n not in self.cams],
                "hand_points": {k: v * scale for k, v in self.points.items()},
                "log": self.log, "solve_s": getattr(self, "solve_s", None)}


def similarity_align(A, B):
    """Umeyama: s, R, t minimising |s R A + t - B| (rows are points)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (A0 ** 2).sum())
    return s, R, mb - s * R @ ma


# ---------------------------------------------------------------- capture
class HandCalSession:
    """Collects hand observations from the pose tab's camera streams.

    A sample is kept when one hand (and only one -- a second would be
    ambiguous with no extrinsics to tell them apart) is seen by at least two
    cameras, is moving slowly (free-running cameras capture tens of ms
    apart, so a fast hand is in a different place for each), and has moved
    to a new spot since the last sample (repeats add no geometry)."""

    STILL_PX = 12.0       # palm motion per tick, px, to count as slow
    NEW_PX = 45.0         # palm must move this far (some camera) for a new sample
    TICK_S = 0.1

    def __init__(self, streams, k_for, detector_factory):
        self.streams = streams
        self.k_for = k_for                  # (node, w, h) -> (K, dist, approx)
        self.detector_factory = detector_factory
        self.samples, self.running, self.note = [], False, ""
        self._thread = None
        self._lock = None

    def start(self, nodes_rot):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self.stop()
        self.nodes_rot = {int(n): int(r or 0) % 360 for n, r in nodes_rot.items()}
        self.det = self.detector_factory()
        self.samples, self.note = [], ""
        self.seen = {n: 0 for n in self.nodes_rot}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.nodes_rot)),
                                        thread_name_prefix="handcal")
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if getattr(self, "det", None) is not None:
            try:
                self.det.close()
            except Exception:
                pass
            self.det = None

    def _one(self, node):
        st = self.streams.get(node)
        if st is None or not st.started:
            return node, None
        frame, _seq, ts = st.get_frame(0, timeout=0.2)
        if frame is None:
            return node, None
        h, w = frame.shape[:2]
        K, dist, approx = self.k_for(node, w, h)
        if approx:
            return node, None                  # uncalibrated: unusable
        hands = self.det.detect(frame, {"node": node, "ts": ts,
                                        "rot": self.nodes_rot.get(node, 0)})
        if len(hands) != 1:
            return node, ("many" if len(hands) > 1 else None)
        o = hands[0]
        px = np.asarray(o["points"], np.float64)
        xy = cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        wp = o.get("world_points")
        return node, {"xy": xy, "f": float((K[0, 0] + K[1, 1]) / 2),
                      "world": np.asarray(wp, np.float64) if wp else None,
                      "palm": px[[0, 5, 9, 13, 17]].mean(0)}

    def _loop(self):
        prev, last = {}, None
        while self.running:
            t0 = time.time()
            got = dict(self._pool.map(self._one, list(self.nodes_rot)))
            views = {n: o for n, o in got.items() if isinstance(o, dict)}
            many = [n for n, o in got.items() if o == "many"]
            slow = all(n in prev and np.linalg.norm(o["palm"] - prev[n]) < self.STILL_PX
                       for n, o in views.items())
            new = last is None or any(
                n in last and np.linalg.norm(o["palm"] - last[n]) > self.NEW_PX
                for n, o in views.items()) or not (set(views) & set(last or {}))
            if many:
                self.note = ("more than one hand in view of " +
                             ", ".join(f"video{n}" for n in many))
            elif len(views) < 2:
                self.note = "show one hand to at least two cameras"
            elif not slow:
                self.note = "moving fast — slow down"
            elif not new:
                self.note = "move to a new spot"
            else:
                with self._lock:
                    self.samples.append({n: {k: v for k, v in o.items() if k != "palm"}
                                         for n, o in views.items()})
                    for n in views:
                        self.seen[n] += 1
                last = {n: o["palm"] for n, o in views.items()}
                self.note = "sample kept"
            prev = {n: o["palm"] for n, o in views.items()}
            time.sleep(max(0.0, self.TICK_S - (time.time() - t0)))

    def status(self):
        pairs = {}
        with self._lock if self._lock else _Null():
            for s in self.samples:
                ns = sorted(s)
                for i, a in enumerate(ns):
                    for b in ns[i + 1:]:
                        pairs[f"{a}-{b}"] = pairs.get(f"{a}-{b}", 0) + 1
            return {"running": self.running, "samples": len(self.samples),
                    "per_cam": {str(n): c for n, c in getattr(self, "seen", {}).items()},
                    "pairs": pairs, "note": self.note}

    def solve(self, world_ref=None, compare=None, keep_scale=False,
              frame_from_camera=None, tag_mm=None):
        self.stop()
        with self._lock if self._lock else _Null():
            obs = list(self.samples)
        if len(obs) < 5:
            return {"ok": False, "error": f"only {len(obs)} samples — need "
                    "at least 5, ideally 20-30, spread through every "
                    "camera's view"}
        try:
            res = HandSolve(obs).solve()
        except (ValueError, np.linalg.LinAlgError, cv2.error) as e:
            return {"ok": False, "error": f"hand solve failed: {e}"}
        T = dict(res["T_world_cam"])
        scale_note = {"source": "MediaPipe hand-size estimate (±10%)"
                      if res["scale_from"] else "none (arbitrary units)"}
        cmp = None
        cmp_nodes = sorted(n for n in T if compare and str(n) in compare)
        if len(cmp_nodes) >= 3:
            A = np.array([T[n][:3, 3] for n in cmp_nodes])
            B = np.array([np.asarray(compare[str(n)], float)[:3, 3] for n in cmp_nodes])
            s, R, t = similarity_align(A, B)
            d = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
            cmp = {"cameras": len(cmp_nodes),
                   "position_rms_mm": round(float(np.sqrt(np.mean(d ** 2))), 1),
                   "position_max_mm": round(float(d.max()), 1),
                   "per_cam_mm": {str(n): round(float(v), 1) for n, v in zip(cmp_nodes, d)},
                   "scale_ratio": round(float(s), 4)}
            if keep_scale:
                for n in T:
                    T[n] = T[n].copy()
                    T[n][:3, 3] *= s
                res["hand_points"] = {k: v * s for k, v in res["hand_points"].items()}
                scale_note = {"source": "current world calibration",
                              "mediapipe_vs_current": round(1 / s, 4)}
        ref_info = None
        if world_ref and world_ref.get("node") is not None and frame_from_camera:
            X, info = frame_from_camera(T, int(world_ref["node"]),
                                        world_ref.get("mode") or "topdown",
                                        int(world_ref.get("yaw_quadrant") or 0))
            if X is None:
                ref_info = {"error": info}
            else:
                T = {n: X @ M for n, M in T.items()}
                res["hand_points"] = {k: (X[:3, :3] @ v + X[:3, 3])
                                      for k, v in res["hand_points"].items()}
                ref_info = dict(info, node=int(world_ref["node"]))
        palms = {}
        for (i, k), v in res["hand_points"].items():
            if k in (0, 5, 9, 13, 17):
                palms.setdefault(i, []).append(v)
        cams_out = []
        for n, M in sorted(T.items()):
            pc = res["per_cam"][n]
            cams_out.append({"node": n, "T": np.round(M, 5).tolist(),
                             "pos": np.round(M[:3, 3], 1).tolist(), "seen": [],
                             "pos_sigma_mm": None, "pos_cov_mm2": None,
                             "rot_sigma_deg": None,
                             "noise_px": round(pc["noise_px"], 3)
                             if pc["noise_px"] is not None else None,
                             "corners": pc["points"],
                             "outlier_corners": pc["dropped"],
                             "ambiguous": False})
        return {"ok": True, "method": "hands", "root": None,
                "marker_mm": tag_mm, "world_ref": ref_info,
                "anchor": None, "anchor_snap": None,
                "rms_px": round(res["rms_px"], 3) if res["rms_px"] else None,
                "noise_px": round(res["noise_px"], 3) if res["noise_px"] else None,
                "snap_count": len(obs),
                "outlier_corners": sum(p["dropped"] for p in res["per_cam"].values()),
                "solve_s": round(res["solve_s"], 2), "iterations": None,
                "cameras": cams_out, "tags": [],
                "hands": [np.round(np.mean(v, axis=0), 1).tolist()
                          for v in palms.values()],
                "views_by_snap": [], "unlinked_cameras": res["unlinked"],
                "unlinked_tags": [], "scale": scale_note, "compare": cmp,
                "log": res["log"]}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
