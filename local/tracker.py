"""Server-side ArUco (AprilTag 36h11) tag tracker for the Tracking tab.

Detection runs natively here rather than in the browser's Pyodide: it is
multi-core, ~5x faster, and reflects what an eventual deployment (e.g. a
Raspberry Pi tracker) actually runs — so the load metrics this module
reports are meaningful for capacity planning.

One tracker thread sweeps every tracked camera at the configured rate,
detecting on each camera's newest frame (frames between ticks are simply
skipped). Per-tag distance comes from solvePnP(IPPE_SQUARE) on the four
corners with the camera's calibration; uncalibrated cameras fall back to
a rough focal guess (f = 0.8*width, ~64 deg HFOV) and are flagged approx.
"""

import base64
import itertools
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


class Metrics:
    """Process/system usage from /proc, sampled between calls."""

    def __init__(self):
        self.ncpu = os.cpu_count() or 1
        self.clk = os.sysconf("SC_CLK_TCK")
        self._last = None
        self._cpu = {}          # last computed CPU figures (1s-averaged)

    def _read(self):
        with open("/proc/self/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        proc = (int(parts[11]) + int(parts[12])) / self.clk   # utime+stime
        with open("/proc/stat") as f:
            cpu = f.readline().split()[1:]
        total = sum(int(x) for x in cpu)
        idle = int(cpu[3]) + int(cpu[4])
        rss = 0
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
        return {"t": time.time(), "proc": proc, "total": total,
                "idle": idle, "rss": rss}

    def sample(self):
        cur = self._read()
        out = {"rss_mb": round(cur["rss"] / 1048576, 1),
               "ncpu": self.ncpu,
               "loadavg": [round(v, 2) for v in os.getloadavg()]}
        try:
            with open("/proc/meminfo") as f:
                memtotal = int(f.readline().split()[1]) * 1024
            out["rss_pct"] = round(100.0 * cur["rss"] / memtotal, 1)
        except (OSError, ValueError):
            pass
        # CPU figures averaged over >=1s windows so frequent polls don't
        # produce jittery (or missing) numbers
        if self._last is None:
            self._last = cur
        elif cur["t"] - self._last["t"] >= 1.0:
            last = self._last
            dt = cur["t"] - last["t"]
            self._cpu = {"proc_cpu_pct_one_core": round(
                100.0 * (cur["proc"] - last["proc"]) / dt, 1)}
            self._cpu["proc_cpu_pct_machine"] = round(
                self._cpu["proc_cpu_pct_one_core"] / self.ncpu, 1)
            dtotal = cur["total"] - last["total"]
            didle = cur["idle"] - last["idle"]
            if dtotal > 0:
                self._cpu["system_cpu_pct"] = round(
                    100.0 * (1 - didle / dtotal), 1)
            self._last = cur
        out.update(self._cpu)
        return out


class Tracker:
    def __init__(self, streams):
        self.streams = streams          # server's node -> CameraStream dict
        self.detector = cv2.aruco.ArucoDetector(DICTIONARY)
        self.metrics = Metrics()
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self.track_fps = 5.0
        self.marker_mm = 40.0
        self.warmup_s = 3.0
        self.cams = {}                  # node -> {"K","dist","cal_size"}
        self.results = {}               # node -> latest detection payload
        self.stats = {"achieved_fps": 0.0, "duty_pct": 0.0}

    def configure(self, track_fps=None, marker_mm=None):
        with self._lock:
            if track_fps:
                self.track_fps = float(track_fps)
            if marker_mm:
                self.marker_mm = float(marker_mm)

    def start(self, cams, track_fps, marker_mm, warmup_s=3.0):
        self.stop()
        self.cams = cams
        self.track_fps = float(track_fps)
        self.marker_mm = float(marker_mm)
        self.warmup_s = float(warmup_s)
        self.results = {}
        self.stats = {"achieved_fps": 0.0, "duty_pct": 0.0}
        if not cams:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self):
        return self._running

    def _K_for(self, node, w, h):
        c = self.cams.get(node) or {}
        if c.get("K"):
            cw, ch = c.get("cal_size") or (w, h)
            K = np.array(c["K"], np.float64)
            if (cw, ch) != (w, h):
                K = np.diag([w / cw, h / ch, 1.0]) @ K
            return K, np.array(c.get("dist") or [0] * 5, np.float64), False
        f = 0.8 * w                    # rough default for uncalibrated cams
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        return K, np.zeros(5), True

    @staticmethod
    def _pnp_square(objp, pts, K, dist):
        """Robust single-tag pose. cv2 5.0's IPPE_SQUARE can return badly
        wrong poses on near-degenerate (small/axis-aligned) quads, so every
        candidate is verified by reprojection, SQPNP is the fallback, and
        the winner gets an LM polish. Returns (rvec, tvec, rms) or None."""
        best = None
        for flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_SQPNP):
            try:
                ok, rvec, tvec = cv2.solvePnP(objp, pts, K, dist, flags=flag)
            except cv2.error:
                continue
            if not ok:
                continue
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            err = float(np.sqrt(np.mean(
                np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
            if best is None or err < best[0]:
                best = (err, rvec, tvec)
            if err < 1.5:
                break
        if best is None:
            return None
        err, rvec, tvec = best
        try:
            rvec, tvec = cv2.solvePnPRefineLM(objp, pts, K, dist, rvec, tvec)
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            err = float(np.sqrt(np.mean(
                np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
        except cv2.error:
            pass
        return rvec, tvec, err

    def _detect(self, node, frame, marker_mm):
        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rej = self.detector.detectMarkers(gray)
        h, w = gray.shape
        tags = []
        if ids is not None and len(ids):
            K, dist, approx = self._K_for(node, w, h)
            half = marker_mm / 2.0
            # order matches aruco corners: TL, TR, BR, BL (y up on the tag)
            objp = np.array([[-half, half, 0], [half, half, 0],
                             [half, -half, 0], [-half, -half, 0]], np.float32)
            for c4, tid in zip(corners, ids.ravel()):
                pts = c4.reshape(4, 2)
                entry = {"id": int(tid),
                         "corners": np.round(pts, 1).tolist()}
                sol = self._pnp_square(objp, pts.astype(np.float32), K, dist)
                if sol is not None:
                    entry["distance_mm"] = round(
                        float(np.linalg.norm(sol[1])), 1)
                    entry["approx"] = bool(approx)
                tags.append(entry)
        return {"ts": round(time.time(), 3), "n": len(tags), "tags": tags,
                "size": [w, h],
                "detect_ms": round((time.time() - t0) * 1000, 2)}

    def _loop(self):
        time.sleep(self.warmup_s)      # camera startup + auto-exposure
        seqs = {}
        ticks = deque(maxlen=20)
        busys = deque(maxlen=20)
        while self._running:
            t0 = time.time()
            with self._lock:
                fps = self.track_fps
                marker = self.marker_mm
            for node in list(self.cams.keys()):
                if not self._running:
                    return
                st = self.streams.get(node)
                if st is None or not st.started:
                    continue
                frame, seq, _ts = st.get_frame(seqs.get(node, 0), timeout=0.005)
                if frame is None or seq == seqs.get(node):
                    continue
                seqs[node] = seq
                self.results[node] = self._detect(node, frame, marker)
            busy = time.time() - t0
            ticks.append(t0)
            busys.append(busy)
            if len(ticks) >= 2:
                span = ticks[-1] - ticks[0] + busy
                if span > 0:
                    self.stats["achieved_fps"] = round(
                        (len(ticks) - 1) / (ticks[-1] - ticks[0]), 2) \
                        if ticks[-1] > ticks[0] else 0.0
                    self.stats["duty_pct"] = round(
                        100.0 * sum(busys) / span, 1)
            time.sleep(max(0.0, 1.0 / max(fps, 0.1) - busy))

    def snapshot(self):
        return {"running": self._running,
                "stats": dict(self.stats),
                "config": {"track_fps": self.track_fps,
                           "marker_mm": self.marker_mm},
                "results": dict(self.results)}

    # ------------------------------------------------- world pose graph
    @staticmethod
    def _inv(T):
        R, t = T[:3, :3], T[:3, 3]
        out = np.eye(4)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    def _capture_observations(self, marker):
        """Grab the newest frame from every tracked camera and detect tags
        with full poses. Returns (obs, views, k_of):
        obs[node][tid] = {"T": T_cam<-tag, "pts": 4x2, "area": px^2}."""
        half = marker / 2.0
        objp = np.array([[-half, half, 0], [half, half, 0],
                         [half, -half, 0], [-half, -half, 0]], np.float32)
        obs, views, k_of = {}, {}, {}
        for node in list(self.cams.keys()):
            st = self.streams.get(node)
            if st is None or not st.started:
                continue
            frame, _seq, _ts = st.get_frame(0, timeout=0.5)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _rej = self.detector.detectMarkers(gray)
            h, w = gray.shape
            K, dist, approx = self._K_for(node, w, h)
            k_of[node] = (K, dist)
            tag2d, t_obs = [], {}
            if ids is not None:
                for c4, tid in zip(corners, ids.ravel()):
                    pts = c4.reshape(4, 2).astype(np.float32)
                    entry = {"id": int(tid),
                             "corners": np.round(pts, 1).tolist()}
                    sol = self._pnp_square(objp, pts, K, dist)
                    if sol is not None:
                        rvec, tvec, _err = sol
                        R, _ = cv2.Rodrigues(rvec)
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = tvec.ravel()
                        x, y = pts[:, 0], pts[:, 1]
                        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) -
                                         np.dot(y, np.roll(x, -1)))
                        t_obs[int(tid)] = {"T": T, "pts": pts,
                                           "area": float(max(area, 1.0))}
                        entry["distance_mm"] = round(
                            float(np.linalg.norm(tvec)), 1)
                        entry["approx"] = bool(approx)
                    tag2d.append(entry)
            ok_enc, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])
            views[node] = {
                "jpg_b64": base64.b64encode(jpg.tobytes()).decode()
                           if ok_enc else None,
                "size": [w, h], "tags": tag2d, "approx": bool(approx)}
            if t_obs:
                obs[node] = t_obs
        return obs, views, k_of

    # ------------------------------------------------ pose-graph helpers
    HUBER_K = 2.5       # Huber threshold, in units of the corner noise sigma
    OUTLIER_Z = 3.5     # report corners beyond this many sigma as outliers
    ROBUST = True       # stage 2 (per-camera noise + Huber); False = plain LS

    @staticmethod
    def _Tmat(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.ravel(t)
        return T

    @staticmethod
    def _tag_objp(half):
        return np.array([[-half, half, 0], [half, half, 0],
                         [half, -half, 0], [-half, -half, 0]], np.float64)

    @staticmethod
    def _ippe_candidates(objp, pts, K, dist):
        """Both poses a small planar square is ambiguous between (IPPE's
        two solutions), as T_cam<-tag, best-fitting first."""
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                objp, pts.astype(np.float64), K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return []
        errs = np.ravel(errs) if errs is not None else np.zeros(n)
        out = [(Tracker._Tmat(cv2.Rodrigues(rvecs[i])[0], tvecs[i]), errs[i])
               for i in range(n) if float(np.ravel(tvecs[i])[2]) > 0]
        return [T for T, _e in sorted(out, key=lambda c: c[1])]

    @staticmethod
    def _reproj_sq(T_ct, objp, pts, K, dist):
        """Summed squared corner error of a tag at T_cam<-tag."""
        if T_ct[2, 3] <= 0:
            return np.inf                       # behind the camera
        rv, _ = cv2.Rodrigues(T_ct[:3, :3])
        p, _ = cv2.projectPoints(objp, rv, T_ct[:3, 3], K, dist)
        return float(np.sum((p.reshape(-1, 2) - pts) ** 2))

    @staticmethod
    def _robust_kabsch(P, Q, min_spread):
        """T_a<-b from matched 3-D points (P in frame a, Q in frame b,
        P ~ R Q + t) by SVD (Kabsch), with a pass of outlier rejection.
        (None, 0) if fewer than 3 points survive or they are near-collinear
        (rotation about their line would be unconstrained)."""
        keep = np.ones(len(P), bool)
        R = t = None
        for _ in range(3):
            n = int(keep.sum())
            if n < 3:
                return None, 0
            Pk, Qk = P[keep], Q[keep]
            mp, mq = Pk.mean(0), Qk.mean(0)
            if np.linalg.svd(Qk - mq, compute_uv=False)[1] / np.sqrt(n) < min_spread / 2:
                return None, 0
            U, _S, Vt = np.linalg.svd((Qk - mq).T @ (Pk - mp))
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            t = mp - R @ mq
            d = np.linalg.norm(P - (Q @ R.T + t), axis=1)
            new = d < max(3 * np.median(d[keep]), min_spread)
            if (new == keep).all():
                break
            keep = new
        return Tracker._Tmat(R, t), int(keep.sum())

    @staticmethod
    def _reproj_tr(T_ct, objp, pts, K, dist, tau):
        """Reprojection cost with each corner's squared error capped at
        tau^2, so one wild corner cannot dominate a comparison."""
        if T_ct[2, 3] <= 0:
            return 4.0 * tau * tau                  # behind the camera
        rv, _ = cv2.Rodrigues(T_ct[:3, :3])
        p, _ = cv2.projectPoints(objp, rv, T_ct[:3, 3], K, dist)
        e2 = np.sum((p.reshape(-1, 2) - pts) ** 2, axis=1)
        return float(np.minimum(e2, tau * tau).sum())

    # Pose-space agreement scale for the initialisation's discrete choices:
    # two views of one tag "agree" when its implied positions differ by
    # well under INIT_POS_MM and orientations by well under INIT_ANG_DEG.
    # Unrefined hypotheses are good to a few degrees; an IPPE flip is off
    # by tens of degrees.
    INIT_POS_MM = 40.0
    INIT_ANG_DEG = 8.0

    @classmethod
    def _pose_disagree(cls, A, B):
        """Normalised disagreement of two poses of the same tag, capped at 1
        (one fully contradicting tag costs 1, however wrong it is)."""
        dp = np.linalg.norm(A[:3, 3] - B[:3, 3]) / cls.INIT_POS_MM
        c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1.0) / 2.0
        da = np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) / cls.INIT_ANG_DEG
        return min(1.0, dp * dp + da * da)

    def _init_graph(self, cam_ids, tag_keys, anchor, obs_list, half,
                    mode="hypotheses", top_k=2):
        """Candidate starting points for the bundle adjustment: a list of
        (T_world<-cam, T_world<-tag) in the anchor-tag gauge, most
        self-consistent first. The caller solves from each and
        keeps the lowest-cost result, so these heuristics only have to put
        the right answer among a few candidates, not pick it outright.

        Single-tag PnP is reliable in POSITION but not in TILT: a small
        square looks almost the same tilted either way (IPPE's two-fold
        ambiguity). Chaining raw single-tag poses flipped ~1 in 4 tags in
        simulation, and a flip puts a camera metres off — too far for LM to
        recover. So the discrete choices are hypothesis tests judged in
        POSE space (do two views agree on a tag's position/orientation?);
        pixel error can't judge them, since an unrefined but correct
        hypothesis is already tens of pixels off at a distance.

          * camera pairs: shared tags propose relative poses (both IPPE
            solutions from each side), plus a rigid alignment of the shared
            tag CENTRES (Kabsch; flip-free) when there are >= 3. mode
            "positions" takes Kabsch whenever it exists (strongest with
            several snapshots); mode "hypotheses" scores every proposal on
            every shared tag (copes with fewer shared tags). Cameras are
            linked strongest pair first (a maximum spanning tree).
          * links left ambiguous (typically one shared tag) are resolved
            jointly: a wrong flip breaks every loop through it, so flip
            combinations are ranked by global tag agreement; the top_k
            become starting points, plus — for each link that no loop can
            decide — the start with that link switched, so the solve can
            report whether the data tells the two apart.
          * tags take the orientation their cameras agree on; cameras are
            re-fit (PnP) to all their tags' corners; repeat."""
        objp = self._tag_objp(half)
        by_cam = {n: {} for n in cam_ids}
        by_tag = {tk: {} for tk in tag_keys}
        cands = {}
        for node, tk, o, K, dist in obs_list:
            by_cam[node][tk] = (o, K, dist)
            by_tag[tk][node] = (o, K, dist)
            cs = []
            for T in [o["T"]] + self._ippe_candidates(objp, o["pts"], K, dist):
                # drop near-duplicates (the stored pose is usually one of
                # IPPE's two solutions)
                if all(np.linalg.norm(cv2.Rodrigues(T[:3, :3] @ U[:3, :3].T)[0]) > 0.02
                       for U in cs):
                    cs.append(T)
            cands[(node, tk)] = cs
        dis = self._pose_disagree

        def rot_deg(A, B):
            return np.degrees(np.linalg.norm(cv2.Rodrigues(A[:3, :3] @ B[:3, :3].T)[0]))

        def pair_edge(a, b, shared):
            order = sorted(shared, key=lambda k: -min(by_cam[a][k][0]["area"],
                                                      by_cam[b][k][0]["area"]))
            hyps = []
            if len(shared) >= 3:
                P = np.array([cands[(a, tk)][0][:3, 3] for tk in shared])
                Q = np.array([cands[(b, tk)][0][:3, 3] for tk in shared])
                T_ab, _n = self._robust_kabsch(P, Q, 2 * half)
                if T_ab is not None:
                    if mode == "positions":
                        return (len(shared) + 1000, 0.0), [T_ab]
                    hyps.append(T_ab)
            for tk in order[:6 if mode == "hypotheses" else 1]:
                for Ca in cands[(a, tk)]:
                    for Cb in cands[(b, tk)]:
                        hyps.append(Ca @ self._inv(Cb))     # T_a<-b
            scored = []
            for h in hyps:
                cost, agree = 0.0, 0
                for tk in order[:20]:
                    c = min(dis(Ca, h @ Cb) for Ca in cands[(a, tk)]
                            for Cb in cands[(b, tk)])
                    cost += c
                    agree += c < 0.5
                scored.append(((agree, -cost), h))
            scored.sort(key=lambda e: e[0], reverse=True)
            # every hypothesis that makes as many tags agree as the best is a
            # live alternative; distinct only if flip-sized apart IN
            # ROTATION (correct hypotheses differ by a few degrees, which a
            # 1-2 m lever arm turns into a large offset)
            top = scored[0][0][0]
            alts = []
            for (agree, _negc), h in scored:
                if agree == top and all(rot_deg(h, g) > 12.0 for g in alts):
                    alts.append(h)
            return scored[0][0], alts

        edges = []
        for i, a in enumerate(cam_ids):
            for b in cam_ids[i + 1:]:
                shared = [tk for tk in by_cam[a] if tk in by_cam[b]]
                if shared:
                    wgt, alts = pair_edge(a, b, shared)
                    edges.append((wgt, a, b, alts))
        # maximum spanning tree (Prim): strongest links first
        root = max(cam_ids, key=lambda n: len(by_cam[n]))
        placed, tree = {root}, []           # (parent, child, [T_parent<-child])
        while len(placed) < len(cam_ids):
            best = None
            for e in edges:
                if (e[1] in placed) != (e[2] in placed) and (best is None or e[0] > best[0]):
                    best = e
            if best is None:
                break
            _w, a, b, alts = best
            if a in placed:
                tree.append((a, b, alts))
                placed.add(b)
            else:
                tree.append((b, a, [self._inv(h) for h in alts]))
                placed.add(a)

        def build(choice):
            cams = {root: np.eye(4)}         # world = root camera for now
            for (par, ch, alts), k in zip(tree, choice):
                cams[ch] = cams[par] @ alts[k]
            return cams

        def place_tags(cams):
            """Each tag's world pose = the candidate (any camera, either
            IPPE solution) its cameras agree on best. Returns the poses and
            the total disagreement (0 = every multi-camera tag consistent)."""
            out, total = {}, 0.0
            for tk, seen in by_tag.items():
                views = [[cams[m] @ C for C in cands[(m, tk)]]
                         for m in seen if m in cams]
                if not views:
                    continue
                best = None
                for Ws in views:
                    for P in Ws:
                        c = sum(min(dis(P, W) for W in Ws2) for Ws2 in views)
                        if best is None or c < best[0]:
                            best = (c, P)
                out[tk] = best[1]
                total += best[0]
            return out, total

        def cam_err(n, T_cw, tags, tau=20.0):
            return sum(self._reproj_tr(T_cw @ tags[tk], objp, o["pts"], K, dist, tau)
                       for tk, (o, K, dist) in by_cam[n].items() if tk in tags)

        def refine(choice):
            cams_T = build(choice)
            tags_T = place_tags(cams_T)[0]
            for _round in range(4):
                changed = False
                for n in cam_ids:
                    if n not in cams_T:
                        continue
                    own = [tk for tk in by_cam[n] if tk in tags_T]
                    if len(own) < 2:
                        continue
                    P3 = np.concatenate([(tags_T[tk][:3, :3] @ objp.T).T + tags_T[tk][:3, 3]
                                         for tk in own])
                    P2 = np.concatenate([by_cam[n][tk][0]["pts"]
                                         for tk in own]).astype(np.float64)
                    _o, K, dist = by_cam[n][own[0]]
                    T_cw = self._inv(cams_T[n])
                    best_T, best_e = T_cw, cam_err(n, T_cw, tags_T)
                    rv, _ = cv2.Rodrigues(T_cw[:3, :3])
                    for kw in (dict(rvec=rv, tvec=T_cw[:3, 3].reshape(3, 1).copy(),
                                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE),
                               dict(flags=cv2.SOLVEPNP_SQPNP)):
                        try:
                            ok, rv2, tv2 = cv2.solvePnP(P3, P2, K, dist, **kw)
                        except cv2.error:
                            continue
                        if not ok:
                            continue
                        T2 = self._Tmat(cv2.Rodrigues(rv2)[0], tv2)
                        e2 = cam_err(n, T2, tags_T)
                        if e2 < best_e * 0.999:
                            best_T, best_e = T2, e2
                    if best_T is not T_cw:
                        cams_T[n] = self._inv(best_T)
                        changed = True
                tags_T = place_tags(cams_T)[0]
                if not changed:
                    break
            M = self._inv(tags_T[anchor]) if anchor in tags_T else np.eye(4)
            return ({n: M @ T for n, T in cams_T.items()},
                    {tk: M @ T for tk, T in tags_T.items()})

        # flip combinations over the tree's ambiguous links: all of them
        # when few, else a random sample plus the all-best one
        sizes = [len(alts) for _p, _c, alts in tree]
        combos = [tuple(0 for _ in sizes)]
        n_all = int(np.prod(sizes)) if sizes else 1
        if n_all > 1:
            if n_all <= 256:
                combos = list(itertools.product(*[range(k) for k in sizes]))
            else:
                rng = np.random.default_rng(0)
                combos += [tuple(int(rng.integers(k)) for k in sizes) for _ in range(255)]
        ranked = sorted(((place_tags(build(c))[1], i, c) for i, c in enumerate(combos)),
                        key=lambda e: (e[0], e[1]))     # all-best (index 0) wins ties
        chosen = [c for _t, _i, c in ranked[:top_k]]
        best_total, _i, best_c = ranked[0]
        for ei, (_par, _ch, alts) in enumerate(tree):
            for k in range(len(alts)):
                if k == best_c[ei]:
                    continue
                c = list(best_c)
                c[ei] = k
                c = tuple(c)
                # a link no loop can decide: switching it agrees as well
                if c not in chosen and place_tags(build(c))[1] < best_total + 0.5:
                    chosen.append(c)
        return [refine(c) for c in chosen[:8]]

    SELECT_PX = 1.0     # fixed noise scale for comparing starting points

    def _bundle_adjust(self, cams_T, tags_T, anchor, obs_list, half, quick=False):
        """Joint refinement of every camera pose and every non-anchor tag
        pose, minimising corner reprojection error.

        Stage 1 is plain least squares — the maximum-likelihood estimate
        if every corner had the same Gaussian noise. Stage 2 drops both of
        those assumptions:
          * per-camera noise: sigma_c is estimated from each camera's
            residuals (degrees-of-freedom corrected, shrunk toward the
            global value when a camera has few corners) and every residual
            is whitened by it — a poorly calibrated camera counts less;
          * robustness: a Huber loss (quadratic up to HUBER_K sigma, then
            linear) so a misdetected corner can't drag the solution;
        re-estimating sigma_c and re-solving until it settles.

        Tags seen by a single camera are left out of the joint solve: their
        own six parameters can reproduce their corners for ANY camera pose,
        so they carry no information about the cameras — only flat,
        weakly-determined directions that make LM crawl. They are re-placed
        from their camera's refined pose afterwards.

        quick=True runs stage 1 only and returns the poses plus
        `select_cost`: the Huber cost at a FIXED noise scale (SELECT_PX), so
        results from different starting points compare on one objective.

        Otherwise returns the refined poses (anchor gauge), the plain
        residuals, and per-camera noise / outlier / covariance figures.
        Covariance is the Gauss-Newton approximation inv(J^T W J) at the
        optimum, i.e. uncertainty RELATIVE TO THE ANCHOR TAG; directions
        the data cannot determine (null space) get infinite variance."""
        corners = self._tag_objp(half)
        cam_ids = sorted(n for n in cams_T)
        n_seen = {}
        for node, tk, *_rest in obs_list:
            if node in cams_T and tk in tags_T:
                n_seen[tk] = n_seen.get(tk, 0) + 1
        lone = {tk for tk, k in n_seen.items() if k == 1 and tk != anchor}
        lone_rel = {}                    # tag relative to the camera that saw it
        for node, tk, *_rest in obs_list:
            if tk in lone and node in cams_T:
                lone_rel[tk] = (node, self._inv(cams_T[node]) @ tags_T[tk])
        tag_ids = [k for k in sorted(tags_T) if k != anchor and k not in lone]
        ci = {n: i for i, n in enumerate(cam_ids)}
        ti = {k: len(cam_ids) + i for i, k in enumerate(tag_ids)}
        nb = len(cam_ids) + len(tag_ids)
        x = np.zeros(6 * nb)
        for n in cam_ids:
            T = self._inv(cams_T[n])                       # T_cam<-world
            x[6 * ci[n]:6 * ci[n] + 3] = cv2.Rodrigues(T[:3, :3])[0].ravel()
            x[6 * ci[n] + 3:6 * ci[n] + 6] = T[:3, 3]
        for k in tag_ids:
            T = tags_T[k]                                  # T_world<-tag
            x[6 * ti[k]:6 * ti[k] + 3] = cv2.Rodrigues(T[:3, :3])[0].ravel()
            x[6 * ti[k] + 3:6 * ti[k] + 6] = T[:3, 3]
        obs = [(ci[node], ti.get(tk), o["pts"].astype(np.float64), K, dist, node, tk)
               for node, tk, o, K, dist in obs_list
               if node in ci and tk in tags_T and tk not in lone]
        touching = [[] for _ in range(nb)]
        for i, ob in enumerate(obs):
            touching[ob[0]].append(i)
            if ob[1] is not None:
                touching[ob[1]].append(i)
        corner_cam = np.repeat([ob[5] for ob in obs], 4)
        # how many cameras saw each tag node: a node seen by k cameras
        # spends 6 of its 8k residual dof on its own pose
        k_of_tag = {}
        for ob in obs:
            k_of_tag[ob[6]] = k_of_tag.get(ob[6], 0) + 1
        corner_k = np.repeat([k_of_tag[ob[6]] for ob in obs], 4)

        def res_obs(i, xv):
            bc, bt, pts, K, dist, _n, _tk = obs[i]
            if bt is None:
                wc = corners
            else:
                Rt, _ = cv2.Rodrigues(xv[6 * bt:6 * bt + 3])
                wc = corners @ Rt.T + xv[6 * bt + 3:6 * bt + 6]
            p, _ = cv2.projectPoints(wc, xv[6 * bc:6 * bc + 3],
                                     xv[6 * bc + 3:6 * bc + 6], K, dist)
            return (p.reshape(-1, 2) - pts).ravel()

        def residuals(xv):
            return (np.concatenate([res_obs(i, xv) for i in range(len(obs))])
                    if obs else np.zeros(0))

        def jacobian(xv, r):
            # each 6-param block moves only the observations it touches:
            # finite differences per block, not over the whole problem
            J = np.zeros((len(r), len(xv)))
            for b in range(nb):
                for j in range(6 * b, 6 * b + 6):
                    eps = 1e-6 if j % 6 < 3 else 1e-4
                    x2 = xv.copy()
                    x2[j] += eps
                    for i in touching[b]:
                        J[8 * i:8 * i + 8, j] = (res_obs(i, x2) - r[8 * i:8 * i + 8]) / eps
            return J

        def loss(r, scale, huber):
            s = np.sum((r.reshape(-1, 2) / scale[:, None]) ** 2, axis=1)
            if not huber:
                return float(s.sum()), np.ones_like(s)
            k2 = self.HUBER_K ** 2
            big = s > k2
            rt = np.sqrt(np.maximum(s, 1e-300))
            rho = np.where(big, 2 * self.HUBER_K * rt - k2, s)
            return float(rho.sum()), np.where(big, self.HUBER_K / rt, 1.0)

        def lm(xv, scale, huber, max_it=60):
            r = residuals(xv)
            cost, w = loss(r, scale, huber)
            lam, its = 1e-3, 0
            for its in range(1, max_it + 1):
                J = jacobian(xv, r)
                rs = np.repeat(np.sqrt(w) / scale, 2)
                Jw, rw = J * rs[:, None], r * rs
                A, g = Jw.T @ Jw, Jw.T @ rw
                Dg = np.diag(np.diag(A) + 1e-9)
                step = None
                for _try in range(8):
                    try:
                        dx = np.linalg.solve(A + lam * Dg, -g)
                    except np.linalg.LinAlgError:
                        lam *= 10
                        continue
                    r2 = residuals(xv + dx)
                    c2, w2 = loss(r2, scale, huber)
                    if c2 < cost:
                        step = (dx, r2, c2, w2)
                        lam = max(lam / 3, 1e-9)
                        break
                    lam *= 5
                if step is None:
                    break
                dx, r, c2, w = step
                xv = xv + dx
                # stop once a step buys < 1e-6 of the cost: far below what
                # corner noise can resolve, and Huber's IRLS tail otherwise
                # crawls on for hundreds of iterations
                done = (cost - c2) <= 1e-6 * max(cost, 1e-12) or np.linalg.norm(dx) < 1e-6
                cost = c2
                if done:
                    break
            return xv, r, w, its

        n_corners = 4 * len(obs)
        ones = np.ones(n_corners)
        x, r, _w, it1 = lm(x, ones, False)

        def unpack_poses(xv):
            cams = {}
            for n in cam_ids:
                b = ci[n]
                R, _ = cv2.Rodrigues(xv[6 * b:6 * b + 3])
                cams[n] = self._inv(self._Tmat(R, xv[6 * b + 3:6 * b + 6]))
            tags = {anchor: np.eye(4)}
            for k in tag_ids:
                b = ti[k]
                tags[k] = self._Tmat(cv2.Rodrigues(xv[6 * b:6 * b + 3])[0],
                                     xv[6 * b + 3:6 * b + 6])
            for k, (node, T_rel) in lone_rel.items():
                tags[k] = cams[node] @ T_rel
            return cams, tags

        if quick:
            cams_q, tags_q = unpack_poses(x)
            sel, _w = loss(r, np.full(n_corners, self.SELECT_PX), True)
            return {"cams_T": cams_q, "tags_T": tags_q, "select_cost": sel,
                    "iterations": it1}

        def noise_model(r):
            """Robust per-camera corner sigma. For 2-D Gaussian noise the
            median of |e|^2 is 2 ln2 sigma^2; dividing by the node's
            residual-dof fraction (1 - 6/(8k)) undoes the part a tag's own
            pose absorbs. Nodes seen by one camera carry almost no noise
            information and are left out unless nothing else exists."""
            e2 = np.sum(r.reshape(-1, 2) ** 2, axis=1)
            f = 1.0 - 6.0 / (8.0 * corner_k)
            usable = corner_k >= 2 if (corner_k >= 2).any() else np.ones_like(corner_k, bool)
            s2 = e2 / f
            g2 = max(np.median(s2[usable]) / (2 * np.log(2)), 0.05 ** 2)
            sig = {}
            for n in cam_ids:
                m = usable & (corner_cam == n)
                nc = int(m.sum())
                c2 = np.median(s2[m]) / (2 * np.log(2)) if nc else g2
                sig[n] = float(np.sqrt(max((nc * c2 + 16 * g2) / (nc + 16), 0.05 ** 2)))
            return sig, float(np.sqrt(g2))

        its = it1
        sig, g = noise_model(r)
        if not self.ROBUST:                      # plain least squares only
            sig = {n: g for n in cam_ids}
        for _outer in range(4 if self.ROBUST else 0):
            scale = np.array([sig[n] for n in corner_cam])
            x, r, w, it = lm(x, scale, True)
            its += it
            sig2, g = noise_model(r)
            settled = all(abs(sig2[n] - sig[n]) < 0.02 * sig[n] for n in cam_ids)
            sig = sig2
            if settled:
                break
        scale = np.array([sig[n] for n in corner_cam])
        _c, w = loss(r, scale, self.ROBUST)
        z = np.sqrt(np.sum((r.reshape(-1, 2) / scale[:, None]) ** 2, axis=1))

        # ---- covariance at the optimum: inv(J^T W J), whitened units ----
        J = jacobian(x, r)
        rs = np.repeat(np.sqrt(w) / scale, 2)
        Jw = J * rs[:, None]
        Hm = Jw.T @ Jw
        lam_h, V = np.linalg.eigh(Hm)
        null = lam_h <= max(lam_h.max(), 1e-300) * 1e-10
        cov = (V[:, ~null] / lam_h[~null]) @ V[:, ~null].T
        # parameters with any weight in the null space are undetermined
        undetermined = np.abs(V[:, null]).max(axis=1) > 1e-6 if null.any() \
            else np.zeros(len(x), bool)

        cams_out, tags_out, stats = {}, {anchor: np.eye(4)}, {}
        for n in cam_ids:
            b = ci[n]
            rv, tv = x[6 * b:6 * b + 3], x[6 * b + 3:6 * b + 6]
            R, _ = cv2.Rodrigues(rv)
            cams_out[n] = self._inv(self._Tmat(R, tv))     # T_world<-cam
            # camera centre C = -R^T t and its covariance via a numeric
            # Jacobian; rotation uncertainty as a small-angle vector
            G = np.zeros((3, 6))
            Mr = np.zeros((3, 3))
            C0 = -R.T @ tv
            for j in range(6):
                d = np.zeros(6)
                d[j] = 1e-6
                R2, _ = cv2.Rodrigues(rv + d[:3])
                G[:, j] = ((-R2.T @ (tv + d[3:])) - C0) / 1e-6
                if j < 3:
                    # small-angle log map: the antisymmetric part of the
                    # rotation delta (cv2.Rodrigues rounds deltas this tiny
                    # to exactly zero)
                    Dr = R2 @ R.T
                    Mr[:, j] = np.array([Dr[2, 1] - Dr[1, 2], Dr[0, 2] - Dr[2, 0],
                                         Dr[1, 0] - Dr[0, 1]]) / 2.0 / 1e-6
            Cb = cov[6 * b:6 * b + 6, 6 * b:6 * b + 6]
            m = corner_cam == n
            known = not undetermined[6 * b:6 * b + 6].any()
            stats[n] = {
                "noise_px": sig[n],
                "corners": int(m.sum()),
                "outlier_corners": int(np.sum(z[m] > self.OUTLIER_Z)),
                "pos_cov": G @ Cb @ G.T if known else None,
                "rot_sigma_deg": float(np.degrees(np.sqrt(max(
                    np.linalg.eigvalsh(Mr @ Cb[:3, :3] @ Mr.T).max(), 0.0))))
                    if known else None,
            }
        for k in tag_ids:
            b = ti[k]
            tags_out[k] = self._Tmat(cv2.Rodrigues(x[6 * b:6 * b + 3])[0],
                                     x[6 * b + 3:6 * b + 6])
        for k, (node, T_rel) in lone_rel.items():
            tags_out[k] = cams_out[node] @ T_rel
        return {"cams_T": cams_out, "tags_T": tags_out, "r": r,
                "stats": stats, "noise_px": g, "iterations": its,
                "outlier_corners": int(np.sum(z > self.OUTLIER_Z))}

    def _solve_graph(self, snapshots, root_id, marker, world_ref=None):
        """Joint pose-graph solve over N snapshots. Cameras have ONE pose
        shared by every snapshot; each (tag, snapshot) is its own free
        node (the reference block moves between snapshots). Camera nodes
        carry constraints across snapshots, so a camera that never sees
        the root still links in through any snapshot's shared tags.

        Pipeline: _init_graph proposes starting points that survive the
        single-tag tilt ambiguity; each is solved (plain least squares) and
        the lowest robust cost wins; _bundle_adjust then refines it with
        per-camera noise weights and a Huber loss and reports per-camera
        noise, outliers and covariance. Residuals live in corner-pixel
        space, so bearing information is tight and single-tag tilt is
        loose without any hand-tuned weighting. test/pose_solver_sim.py
        checks all of it against a synthetic ground truth.

        One snapshot of a small block is inherently fragile (cameras often
        share only one ambiguous tag); three or more are reliable.

        Gauge: anchored on the most-observed (tag, snapshot) node in the
        root's component, then re-expressed in the frame of the root tag
        at its EARLIEST visible snapshot."""
        half = marker / 2.0
        # bipartite connectivity: cameras <-> (tag, snap)
        if root_id is None:
            # No designated tag: seed from the most-observed (tag, snapshot)
            # anywhere. The seed only picks which connected component gets
            # solved and provides a temporary gauge — the final frame comes
            # from the reference camera, so no tag has special status.
            best, root_key = -1, None
            for si, s in enumerate(snapshots):
                seen = {}
                for tags in s["obs"].values():
                    for tid in tags:
                        seen[tid] = seen.get(tid, 0) + 1
                for tid, c in seen.items():
                    if c > best:
                        best, root_key = c, (int(tid), si)
            if root_key is None:
                return None
            root_id, root_si = root_key
        else:
            root_id = int(root_id)
            root_si = next((si for si, s in enumerate(snapshots)
                            if any(root_id in t for t in s["obs"].values())),
                           None)
            if root_si is None:
                return None
            root_key = (root_id, root_si)
        comp_tags, comp_cams = {root_key}, set()
        changed = True
        while changed:
            changed = False
            for si, s in enumerate(snapshots):
                for node, tags in s["obs"].items():
                    linked = any((tid, si) in comp_tags for tid in tags)
                    if node in comp_cams and not linked:
                        for tid in tags:
                            if (tid, si) not in comp_tags:
                                comp_tags.add((tid, si))
                                changed = True
                    elif linked:
                        if node not in comp_cams:
                            comp_cams.add(node)
                            changed = True
                        for tid in tags:
                            if (tid, si) not in comp_tags:
                                comp_tags.add((tid, si))
                                changed = True
        counts = {tk: sum(1 for n in comp_cams
                          if tk[0] in snapshots[tk[1]]["obs"].get(n, {}))
                  for tk in comp_tags}
        anchor = max(comp_tags,
                     key=lambda tk: (counts[tk], tk == root_key,
                                     -tk[0], -tk[1]))
        obs_list = []                   # (node, tag key, obs, K, dist)
        for si, s in enumerate(snapshots):
            for node, tags in s["obs"].items():
                if node not in comp_cams or node not in s["k_of"]:
                    continue
                K, dist = s["k_of"][node]
                for tid, o in tags.items():
                    if (tid, si) in comp_tags:
                        obs_list.append((node, (tid, si), o, K, dist))
        corners_h = np.array([[-half, half, 0, 1], [half, half, 0, 1],
                              [half, -half, 0, 1], [-half, -half, 0, 1]],
                             np.float64).T
        t_solve = time.perf_counter()
        # Multi-start: every initialisation heuristic proposes starting
        # points, each is solved (plain stage), and the objective itself —
        # robust reprojection cost at one fixed noise scale — picks the
        # winner. A camera is AMBIGUOUS if another start converged to a
        # flip-sized different pose for it at essentially the same cost:
        # the data fits both, and no solver can tell them apart.
        starts = []
        for mode in ("positions", "hypotheses"):
            try:
                starts += self._init_graph(sorted(comp_cams), sorted(comp_tags),
                                           anchor, obs_list, half, mode=mode)
            except (np.linalg.LinAlgError, cv2.error, ValueError, KeyError):
                pass                    # a failed heuristic just adds no start
        results = sorted((self._bundle_adjust(c0, t0, anchor, obs_list, half, quick=True)
                          for c0, t0 in starts), key=lambda q: q["select_cost"])
        best_q = results[0]
        ambiguous = set()
        tol = max(4.0, 0.02 * best_q["select_cost"])
        for q in results[1:]:
            if q["select_cost"] > best_q["select_cost"] + tol:
                break
            for n, T in q["cams_T"].items():
                Tb = best_q["cams_T"][n]
                c = (np.trace(T[:3, :3].T @ Tb[:3, :3]) - 1.0) / 2.0
                if (np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) > 12.0
                        or np.linalg.norm(T[:3, 3] - Tb[:3, 3]) > 150.0):
                    ambiguous.add(n)
        ba = self._bundle_adjust(best_q["cams_T"], best_q["tags_T"], anchor, obs_list, half)
        cams_T, tags_T, r = ba["cams_T"], ba["tags_T"], ba["r"]
        solve_s = time.perf_counter() - t_solve
        rms = (round(float(np.sqrt(np.mean(r ** 2))), 3)
               if len(r) else None)
        shift = self._inv(tags_T[root_key])
        tags_T = {tk: shift @ T for tk, T in tags_T.items()}
        cams_T = {n: shift @ T for n, T in cams_T.items()}
        R_frame = shift[:3, :3]          # anchor gauge -> reported frame
        # Re-anchor on a camera if one was designated. The tag gauge above
        # is only a temporary handle; the frame the user actually gets is
        # a property of the rig, so it survives the markers being moved.
        ref_info = None
        if world_ref and world_ref.get("node") is not None:
            X, info = self._frame_from_camera(
                cams_T, int(world_ref["node"]),
                world_ref.get("mode") or "topdown",
                int(world_ref.get("yaw_quadrant") or 0),
                level_tags=list(tags_T.values()), tag_mm=marker)
            if X is None:
                ref_info = {"error": info}
            else:
                tags_T = {tk: X @ T for tk, T in tags_T.items()}
                cams_T = {n: X @ T for n, T in cams_T.items()}
                R_frame = X[:3, :3] @ R_frame
                ref_info = dict(info, node=int(world_ref["node"]))
        tags_out = [{"id": tk[0], "snap": tk[1],
                     "corners_world": np.round((T @ corners_h).T[:, :3],
                                               1).tolist(),
                     "T": np.round(T, 5).tolist()}
                    for tk, T in tags_T.items()]
        cams_out = []
        for node, T in cams_T.items():
            seen = sorted({tid for s in snapshots
                           for tid in s["obs"].get(node, {})})
            st = ba["stats"][node]
            # 1-sigma, relative to the anchor tag, in the reported world
            # axes; None when the data leaves this camera undetermined
            pcov = (R_frame @ st["pos_cov"] @ R_frame.T
                    if st["pos_cov"] is not None else None)
            cams_out.append({"node": node, "T": np.round(T, 5).tolist(),
                             "pos": np.round(T[:3, 3], 1).tolist(),
                             "seen": seen,
                             "pos_sigma_mm": (np.round(np.sqrt(np.maximum(
                                 np.diag(pcov), 0)), 2).tolist()
                                 if pcov is not None else None),
                             "pos_cov_mm2": (np.round(pcov, 4).tolist()
                                             if pcov is not None else None),
                             "rot_sigma_deg": (round(st["rot_sigma_deg"], 3)
                                               if st["rot_sigma_deg"] is not None else None),
                             "noise_px": round(st["noise_px"], 3),
                             "corners": st["corners"],
                             "outlier_corners": st["outlier_corners"],
                             # linked only through a tag whose flip no
                             # other observation can decide
                             "ambiguous": node in ambiguous})
        all_cams, all_ids = set(), set()
        for s in snapshots:
            all_cams.update(s["views"].keys())
            for t in s["obs"].values():
                all_ids.update(t.keys())
        return {"ok": True, "root": root_id, "marker_mm": marker,
                "world_ref": ref_info,
                "anchor": int(anchor[0]), "anchor_snap": int(anchor[1]),
                "rms_px": rms, "snap_count": len(snapshots),
                "noise_px": round(ba["noise_px"], 3),
                "outlier_corners": ba["outlier_corners"],
                "solve_s": round(solve_s, 2),
                "iterations": ba["iterations"],
                "cameras": cams_out, "tags": tags_out,
                "views_by_snap": [s["views"] for s in snapshots],
                "unlinked_cameras": sorted(all_cams - set(cams_T)),
                "unlinked_tags": sorted(
                    all_ids - {tk[0] for tk in tags_T})}

    def world_solve(self, root_id, marker_mm=None, world_ref=None):
        """Single-snapshot world solve (the 'Show World Coordinate
        System' button)."""
        marker = float(marker_mm or self.marker_mm)
        obs, views, k_of = self._capture_observations(marker)
        snap = {"obs": obs, "views": views, "k_of": k_of}
        res = self._solve_graph([snap], root_id, marker, world_ref)
        if res is None:
            return {"ok": False, "views_by_snap": [views],
                    "error": (f"Tag {int(root_id)} is not visible to any camera"
                              if root_id is not None else
                              "No tags visible to any camera — put the "
                              "reference block where the cameras can see it")}
        return res

    # ------------------------------------------------ pose verification
    @staticmethod
    def _triangulate(rays, min_parallax_deg=0.75):
        """Linear DLT over N views. rays: [(P 3x4, xn, yn)] with points in
        normalized camera coords. Returns the world point or None.

        Rejects the two ways a solve can be nonsense rather than merely
        imprecise: a point that lands behind a camera that supposedly saw
        it, and rays so nearly parallel that depth is unconstrained (the
        homogeneous solution heads for infinity and X[:3]/X[3] blows up).
        Neither is a divide-by-zero in practice -- X[3] stays comfortably
        non-zero -- which is why an epsilon check alone never caught them.
        """
        if len(rays) < 2:
            return None
        A = []
        for P, x, y in rays:
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        A = np.array(A, np.float64)
        # svd raises on non-finite input, and this runs on the tracker
        # thread where an exception would kill the loop outright
        if not np.all(np.isfinite(A)):
            return None
        try:
            _u, _s, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return None
        X = Vt[-1]
        if abs(X[3]) < 1e-12:
            return None
        X = X[:3] / X[3]
        dirs = []
        for P, _x, _y in rays:
            if float((P @ np.append(X, 1.0))[2]) <= 0:
                return None                      # behind this camera
            C = -P[:, :3].T @ P[:, 3]
            v = X - C
            n = np.linalg.norm(v)
            if n < 1e-9:
                return None
            dirs.append(v / n)
        best = 0.0
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                best = max(best, float(np.degrees(np.arccos(
                    np.clip(float(dirs[i] @ dirs[j]), -1.0, 1.0)))))
        if best < min_parallax_deg:
            return None                          # depth is unconstrained
        return X

    @staticmethod
    def _ray_gap(Ca, ra, Cb, rb):
        """Closest approach between two world-space rays, in mm."""
        w0 = Ca - Cb
        aa, bb, cc = ra @ ra, ra @ rb, rb @ rb
        dd, ee = ra @ w0, rb @ w0
        den = aa * cc - bb * bb
        if abs(den) < 1e-9:                 # parallel: no useful constraint
            return None
        s = (bb * ee - cc * dd) / den
        t = (aa * ee - bb * dd) / den
        return float(np.linalg.norm(w0 + s * ra - t * rb))

    def verify_poses(self, poses, marker_mm=None, tol_px=3.0, tol_mm=8.0):
        """Check saved camera extrinsics against a live view of the test
        block.

        Judged on *pairwise* geometric agreement, not on per-camera tag
        poses. A single tag's PnP pose carries the planar two-fold
        ambiguity, so comparing tag poses flags huge disagreements even
        when nothing has moved; the closest approach between two cameras'
        back-projected corner rays has no such ambiguity and is zero when
        both poses are right.

        Two cameras agree if their rays to the same corner meet within
        tolerance. The largest group of mutually-agreeing cameras (a
        connected component of the agreement graph) becomes the reference,
        and anything outside it moved. That keeps one bumped camera from
        contaminating the verdict on the others, which a plain
        triangulate-from-everyone check cannot do.

        Two limits worth knowing: with only two posed cameras in view a
        disagreement cannot be attributed to either one, and a rigid motion
        of the whole rig is invisible to a check that compares cameras only
        against each other.

        poses: {node: 4x4 T_world<-cam}. Cameras with no pose, or with no
        tags in view, are reported rather than silently dropped.
        """
        marker = float(marker_mm or self.marker_mm)
        tol_px, tol_mm = float(tol_px), float(tol_mm)
        obs, views, k_of = self._capture_observations(marker)

        cams = {}
        for n, T in (poses or {}).items():
            n = int(n)
            if T is None or n not in k_of:
                continue
            try:
                Twc = np.array(T, np.float64).reshape(4, 4)
            except (ValueError, TypeError):
                continue
            Tcw = self._inv(Twc)
            K, dist = k_of[n]
            cams[n] = {"K": K, "dist": dist,
                       "R_cw": Tcw[:3, :3], "t_cw": Tcw[:3, 3],
                       "P": np.hstack([Tcw[:3, :3], Tcw[:3, 3:4]]),
                       "C": Twc[:3, 3], "R_wc": Twc[:3, :3]}

        # (tag id, corner index) -> {node: (observed px, normalized xy, ray)}
        pts = {}
        for node, tags in obs.items():
            c = cams.get(node)
            if c is None:
                continue
            for tid, o in tags.items():
                raw = np.asarray(o["pts"], np.float64).reshape(-1, 1, 2)
                und = cv2.undistortPoints(raw, c["K"], c["dist"]).reshape(-1, 2)
                for ci in range(min(4, len(und))):
                    r = c["R_wc"] @ np.array([und[ci][0], und[ci][1], 1.0])
                    pts.setdefault((int(tid), ci), {})[node] = (
                        np.asarray(o["pts"][ci], np.float64), und[ci],
                        r / np.linalg.norm(r))

        # ---- pairwise agreement ----
        gaps, tags_of = {}, {n: set() for n in cams}
        shared_tags = set()
        for (tid, _ci), d in pts.items():
            nodes = sorted(d)
            if len(nodes) < 2:
                continue
            shared_tags.add(tid)
            for i, a in enumerate(nodes):
                for b in nodes[i + 1:]:
                    g = self._ray_gap(cams[a]["C"], d[a][2],
                                      cams[b]["C"], d[b][2])
                    if g is None:
                        continue
                    gaps.setdefault(frozenset((a, b)), []).append(g)
                    tags_of[a].add(tid)
                    tags_of[b].add(tid)
        score = {k: float(np.sqrt(np.mean(np.square(v))))
                 for k, v in gaps.items()}

        # ---- largest mutually-agreeing group (union-find over good edges) --
        parent = {n: n for n in cams}

        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for key, s in score.items():
            if s <= tol_mm:
                a, b = sorted(key)
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        groups = {}
        for n in cams:
            if tags_of[n]:
                groups.setdefault(find(n), []).append(n)
        ref = max(groups.values(), key=len) if groups else []
        ref_set = set(ref) if len(ref) >= 2 else set()

        # ---- reprojection residual against the reference group ----
        reproj = {n: [] for n in cams}
        for (tid, _ci), d in pts.items():
            for n in d:
                others = [m for m in d if m in ref_set and m != n]
                if len(others) < 2:
                    continue
                X = self._triangulate(
                    [(cams[m]["P"], d[m][1][0], d[m][1][1]) for m in others])
                if X is None:
                    continue
                c = cams[n]
                rv, _ = cv2.Rodrigues(c["R_cw"])
                proj, _ = cv2.projectPoints(
                    X.reshape(1, 3), rv, c["t_cw"], c["K"], c["dist"])
                reproj[n].append(
                    float(np.linalg.norm(proj.reshape(2) - d[n][0])))

        def worst_gap(n, against):
            vals = [s for k, s in score.items()
                    if n in k and (k - {n}) & against]
            return max(vals) if vals else None

        cameras, verified, moved = [], [], []
        for node in sorted(set(views) | set(cams)):
            seen = sorted(int(t) for t in obs.get(node, {}))
            entry = {"node": node, "tags_seen": seen}
            if node not in cams:
                entry["status"] = "no_pose"
                entry["detail"] = ("no saved extrinsic — run pose estimation "
                                   "for this camera")
                cameras.append(entry)
                continue
            if not seen:
                entry["status"] = "not_seen"
                entry["detail"] = ("no tags in view — not covered by this "
                                   "check")
                cameras.append(entry)
                continue
            if not tags_of[node]:
                entry["status"] = "unverifiable"
                entry["detail"] = ("saw only tags no other posed camera saw — "
                                   "move the block into a shared view")
                cameras.append(entry)
                continue

            entry["shared_tags"] = sorted(tags_of[node])
            if reproj[node]:
                entry["reproj_rms_px"] = round(float(np.sqrt(
                    np.mean(np.square(reproj[node])))), 2)
                entry["corners"] = len(reproj[node])
            gap_ref = worst_gap(node, ref_set - {node})
            gap_any = worst_gap(node, set(cams) - {node})
            gap = gap_ref if gap_ref is not None else gap_any
            if gap is not None:
                entry["max_offset_mm"] = round(gap, 2)

            if not ref_set:
                # nothing formed a mutually-agreeing pair: everyone who was
                # compared disagreed, and with this little overlap the blame
                # cannot be assigned
                partners = sorted({m for k in score if node in k
                                   for m in (k - {node})})
                entry["status"] = "disagree"
                entry["detail"] = (
                    f"disagrees with {', '.join('video%d' % m for m in partners)}"
                    f" by up to {gap:.1f} mm — one of them moved, but with no "
                    "agreeing group there is nothing to judge against; add a "
                    "third view of the block")
                moved.append(node)
            elif node in ref_set:
                px = (f"{entry['reproj_rms_px']:.2f} px / "
                      if "reproj_rms_px" in entry else "")
                entry["status"] = "ok"
                entry["detail"] = (
                    f"agrees with {len(ref_set) - 1} other camera(s) to "
                    f"{px}{gap:.1f} mm")
                verified.append(node)
            else:
                entry["status"] = "moved"
                entry["detail"] = (
                    f"disagrees with the {len(ref_set)} agreeing cameras by "
                    f"up to {gap:.1f} mm (tolerance {tol_mm:g} mm) — this "
                    "camera appears to have been bumped; re-run pose "
                    "estimation")
                moved.append(node)
            cameras.append(entry)

        # where the block actually is, from the agreeing cameras — lets the
        # 3D view show what was just checked instead of a stale solve
        tag_pts = {}
        for (tid, ci), d in pts.items():
            use = [n for n in d if n in ref_set] or list(d)
            if len(use) >= 2:
                X = self._triangulate(
                    [(cams[m]["P"], d[m][1][0], d[m][1][1]) for m in use])
                if X is not None:
                    tag_pts.setdefault(int(tid), {})[int(ci)] = \
                        [round(float(v), 1) for v in X]
        tags_out = [{"id": t, "corners_world": [c[i] for i in sorted(c)]}
                    for t, c in sorted(tag_pts.items()) if len(c) == 4]
        return {"ok": True, "marker_mm": marker,
                "tol_px": tol_px, "tol_mm": tol_mm,
                "cameras": cameras, "views": views,
                "verified": verified, "moved": moved,
                "reference_group": sorted(ref_set),
                "tags": tags_out,
                "camera_poses": {str(n): np.round(cams[n]["C"], 1).tolist()
                                 for n in cams},
                "shared_tags": sorted(shared_tags)}
    # ------------------------------------------------ world re-orientation
    @staticmethod
    def _frame_from_camera(P, ref, mode="topdown", yaw_quadrant=0,
                           level_tags=None, tag_mm=40.0):
        """4x4 taking the current world frame to one defined by camera `ref`.

        Anchoring on a camera rather than a tag means the world frame is a
        property of the rig, not of where some marker happened to be lying.
        The rig doesn't move; the marker does.

          topdown  the camera looks at the floor, so its view direction is
                   the floor normal: new +Z is straight up, +Y is the
                   camera's image-up.
          forward  the camera looks across the workspace: its image-up is
                   world up (+Z), and its view direction is +X (forward).

        A side-facing camera is just a forward-facing one turned 90
        degrees, which is what yaw_quadrant is for. Z = 0 sits at the
        lowest camera either way, so heights read positive.

        Returns (X, info) or (None, error string).
        """
        if ref not in P:
            return None, f"reference camera video{ref} has no pose"
        R_wc = P[ref][:3, :3]
        view = R_wc @ np.array([0.0, 0.0, 1.0])       # camera looks along +Z
        up = -(R_wc @ np.array([0.0, 1.0, 0.0]))      # image-up is -Y
        if np.linalg.norm(view) < 1e-9:
            return None, "reference camera pose is degenerate"
        view = view / np.linalg.norm(view)

        # A camera mounted "top-down" is only ever approximately vertical,
        # so taking its optical axis as the floor normal bakes that error
        # into every height. The calibration object always presents one
        # exactly horizontal face, and its topmost tag IS that face — so
        # when tag poses are available, level on the tag instead and keep
        # the camera only for deciding which way is up and where yaw sits.
        leveled = None
        if mode == "topdown" and level_tags:
            prov = -view                      # provisional up, from the camera
            best_h, best_n = None, None
            for T in level_tags:
                T = np.asarray(T, np.float64)
                n = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
                if float(n @ prov) < 0:
                    n = -n                    # orient out of the block, upward
                half = float(tag_mm) / 2.0
                for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
                    c = T[:3, :3] @ np.array([sx * half, sy * half, 0.0]) + T[:3, 3]
                    h = float(c @ prov)
                    if best_h is None or h > best_h:
                        best_h, best_n = h, n / np.linalg.norm(n)
            if best_n is not None:
                tilt = float(np.degrees(np.arccos(
                    np.clip(float(best_n @ prov), -1.0, 1.0))))
                # a horizontal face should be within a few degrees of the
                # camera axis; much more than that and we picked up a side
                # face, so keep the camera's own axis rather than tilt the
                # whole world onto a bad guess
                if tilt <= 25.0:
                    view = -best_n
                    leveled = round(tilt, 2)

        if mode == "forward":
            e3 = up / np.linalg.norm(up)              # world up = image up
            e1 = view - float(view @ e3) * e3         # forward, levelled
            if np.linalg.norm(e1) < 1e-6:
                return None, ("that camera points along its own up axis — "
                              "it is not usable as a forward reference")
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(e3, e1)                     # right-handed
            R = np.array([e1, e2, e3])
        else:                                          # topdown
            e3 = -view                                 # world up
            e2 = up - float(up @ e3) * e3
            if np.linalg.norm(e2) < 1e-6:
                alt = R_wc @ np.array([1.0, 0.0, 0.0])
                e2 = alt - float(alt @ e3) * e3
            e2 /= np.linalg.norm(e2)
            e1 = np.cross(e2, e3)
            R = np.array([e1, e2, e3])
        for _ in range(int(yaw_quadrant) % 4):
            R = np.array([[0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]]) @ R

        heights = {n: float((R @ T[:3, 3])[2]) for n, T in P.items()}
        low = min(heights, key=heights.get)
        X = np.eye(4)
        X[:3, :3] = R
        X[:3, 3] = [0.0, 0.0, -heights[low]]
        # how far off the assumed orientation the camera actually is: for
        # topdown, angle from vertical; for forward, how much it is tilted
        # out of level. Says whether the assumption was fair.
        off = (np.degrees(np.arccos(np.clip(float(view @ np.array([0, 0, -1.0])),
                                            -1.0, 1.0)))
               if mode == "topdown" else
               np.degrees(np.arcsin(np.clip(abs(float(view @ e3)), 0.0, 1.0))))
        return X, {"floor_node": low, "off_axis_deg": round(float(off), 2),
                   "mode": mode, "yaw_quadrant": int(yaw_quadrant) % 4,
                   "leveled_on_tag_deg": leveled}

    @staticmethod
    def repose_world(poses, reference_node, yaw_quadrant=0, mode="topdown"):
        """Re-express saved camera poses in a frame defined by one camera.

        See _frame_from_camera for what the modes mean. Applying the result
        to the stored extrinsics moves everything downstream with it.
        """
        ref = int(reference_node)
        P = {}
        for n, T in (poses or {}).items():
            if T is None:
                continue
            try:
                P[int(n)] = np.array(T, np.float64).reshape(4, 4)
            except (ValueError, TypeError):
                continue
        X, info = Tracker._frame_from_camera(P, ref, mode, yaw_quadrant)
        if X is None:
            return {"ok": False, "error": info}
        out, new_h = {}, {}
        for n, T in P.items():
            Tn = X @ T
            out[str(n)] = np.round(Tn, 6).tolist()
            new_h[n] = round(float(Tn[2, 3]), 1)
        return {"ok": True, "reference_node": ref,
                "transform": np.round(X, 6).tolist(),
                "poses": out, "heights_mm": new_h,
                "floor_node": info["floor_node"],
                "mode": info["mode"],
                "yaw_quadrant": info["yaw_quadrant"],
                "off_axis_deg": info["off_axis_deg"],
                "note": ("world Z is up, Z=0 at the lowest camera; "
                         + ("+Y is the reference camera's image-up"
                            if mode == "topdown"
                            else "+X is the reference camera's view direction")
                         + ", turned by yaw_quadrant x 90 degrees")}

    # ---------------------------------------- world calibration session
    def wcal_start(self):
        self.wcal = []

    def wcal_snap(self, marker_mm=None):
        marker = float(marker_mm or self.marker_mm)
        obs, views, k_of = self._capture_observations(marker)
        if not hasattr(self, "wcal"):
            self.wcal = []
        self.wcal.append({"obs": obs, "views": views, "k_of": k_of})
        return {"ok": True, "index": len(self.wcal),
                "summary": {str(n): len(t) for n, t in obs.items()}}

    def wcal_solve(self, root_id, marker_mm=None, world_ref=None):
        snaps = getattr(self, "wcal", [])
        if not snaps:
            return {"ok": False, "error": "No snapshots captured"}
        marker = float(marker_mm or self.marker_mm)
        res = self._solve_graph(snaps, root_id, marker, world_ref)
        if res is None:
            return {"ok": False,
                    "views_by_snap": [s["views"] for s in snaps],
                    "error": (f"Tag {int(root_id)} is not visible in any snapshot"
                              if root_id is not None else
                              "No tags in any snapshot — the reference block "
                              "was not visible to any camera")}
        return res
