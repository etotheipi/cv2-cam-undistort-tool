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

    def _solve_graph(self, snapshots, root_id, marker, world_ref=None):
        """Joint pose-graph solve over N snapshots. Cameras have ONE pose
        shared by every snapshot; each (tag, snapshot) is its own free
        node (the reference block moves between snapshots). Camera nodes
        carry constraints across snapshots, so a camera that never sees
        the root still links in through any snapshot's shared tags.

        Residuals live in corner-pixel space: the camera update is one
        solvePnP over ALL corners it observed across ALL snapshots, which
        is exactly maximum likelihood under isotropic Gaussian corner
        noise — bearing information is tight, single-tag tilt is loose,
        with no hand-tuned per-pose weighting needed. Tag updates average
        across cameras weighted by observed pixel area (~cos(tilt)/d^2).

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
        tags_T = {anchor: np.eye(4)}
        cams_T = {}
        changed = True
        while changed:
            changed = False
            for si, s in enumerate(snapshots):
                for node in comp_cams:
                    tags = s["obs"].get(node, {})
                    if node not in cams_T:
                        for tid, o in tags.items():
                            if (tid, si) in tags_T:
                                cams_T[node] = (tags_T[(tid, si)] @
                                                self._inv(o["T"]))
                                changed = True
                                break
                    Tc = cams_T.get(node)
                    if Tc is None:
                        continue
                    for tid, o in tags.items():
                        if (tid, si) in comp_tags and (tid, si) not in tags_T:
                            tags_T[(tid, si)] = Tc @ o["T"]
                            changed = True
        corners_h = np.array([[-half, half, 0, 1], [half, half, 0, 1],
                              [half, -half, 0, 1], [-half, -half, 0, 1]],
                             np.float64).T
        corners_tag = corners_h.T[:, :3]
        # ---- joint bundle adjustment (damped LM over all poses) ----
        # Block-alternation slides along the weakly-constrained tilt
        # directions of single-tag views; joint damped least squares over
        # (all camera poses + all non-anchor tag poses) with corner-pixel
        # residuals does not. Small problem => dense numeric Jacobian.
        cam_ids = sorted(n for n in comp_cams if n in cams_T)
        tag_ids = [tk for tk in sorted(tags_T) if tk != anchor]
        obs_list = []
        for si, s in enumerate(snapshots):
            for node, tags in s["obs"].items():
                if node not in cams_T or node not in s["k_of"]:
                    continue
                K, dist = s["k_of"][node]
                for tid, o in tags.items():
                    if (tid, si) in tags_T:
                        obs_list.append((node, (tid, si), o["pts"], K, dist))

        def pack():
            xs = []
            for n in cam_ids:
                Tcw = self._inv(cams_T[n])
                rv, _ = cv2.Rodrigues(Tcw[:3, :3])
                xs += list(rv.ravel()) + list(Tcw[:3, 3])
            for tkk in tag_ids:
                T = tags_T[tkk]
                rv, _ = cv2.Rodrigues(T[:3, :3])
                xs += list(rv.ravel()) + list(T[:3, 3])
            return np.array(xs, np.float64)

        def unpack(x):
            cams_cw, tags_w = {}, {anchor: np.eye(4)}
            i = 0
            for n in cam_ids:
                R, _ = cv2.Rodrigues(x[i:i + 3])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = x[i + 3:i + 6]
                cams_cw[n] = T                 # T_cam<-world
                i += 6
            for tkk in tag_ids:
                R, _ = cv2.Rodrigues(x[i:i + 3])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = x[i + 3:i + 6]
                tags_w[tkk] = T                # T_world<-tag
                i += 6
            return cams_cw, tags_w

        def residuals(x):
            cams_cw, tags_w = unpack(x)
            out = []
            for node, tkk, pts, K, dist in obs_list:
                Tw = tags_w[tkk]
                wc = (Tw[:3, :3] @ corners_tag.T).T + Tw[:3, 3]
                Tcw = cams_cw[node]
                rv, _ = cv2.Rodrigues(Tcw[:3, :3])
                proj, _ = cv2.projectPoints(wc, rv, Tcw[:3, 3], K, dist)
                out.append((proj.reshape(-1, 2) - pts).ravel())
            return np.concatenate(out) if out else np.zeros(0)

        x = pack()
        r = residuals(x)
        if len(r) and len(x):
            cost = float(r @ r)
            lam = 1e-3
            for _it in range(15):
                J = np.empty((len(r), len(x)))
                for j in range(len(x)):
                    eps = 1e-5 if (j % 6) < 3 else 1e-3
                    x2 = x.copy()
                    x2[j] += eps
                    J[:, j] = (residuals(x2) - r) / eps
                g = J.T @ r
                A = J.T @ J
                D = np.diag(np.diag(A) + 1e-9)
                improved = False
                dx = None
                for _try in range(6):
                    try:
                        dx = np.linalg.solve(A + lam * D, -g)
                    except np.linalg.LinAlgError:
                        lam *= 10
                        continue
                    r2 = residuals(x + dx)
                    c2 = float(r2 @ r2)
                    if c2 < cost:
                        x = x + dx
                        r, cost = r2, c2
                        lam = max(lam / 3, 1e-7)
                        improved = True
                        break
                    lam *= 5
                if not improved:
                    break
                if dx is not None and float(np.linalg.norm(dx)) < 1e-5:
                    break
            cams_cw, tags_w = unpack(x)
            cams_T = {n: self._inv(T) for n, T in cams_cw.items()}
            tags_T = tags_w
        rms = (round(float(np.sqrt(np.mean(r ** 2))), 3)
               if len(r) else None)
        shift = self._inv(tags_T[root_key])
        tags_T = {tk: shift @ T for tk, T in tags_T.items()}
        cams_T = {n: shift @ T for n, T in cams_T.items()}
        # Re-anchor on a camera if one was designated. The tag gauge above
        # is only a temporary handle; the frame the user actually gets is
        # a property of the rig, so it survives the markers being moved.
        ref_info = None
        if world_ref and world_ref.get("node") is not None:
            X, info = self._frame_from_camera(
                cams_T, int(world_ref["node"]),
                world_ref.get("mode") or "topdown",
                int(world_ref.get("yaw_quadrant") or 0))
            if X is None:
                ref_info = {"error": info}
            else:
                tags_T = {tk: X @ T for tk, T in tags_T.items()}
                cams_T = {n: X @ T for n, T in cams_T.items()}
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
            cams_out.append({"node": node, "T": np.round(T, 5).tolist(),
                             "pos": np.round(T[:3, 3], 1).tolist(),
                             "seen": seen})
        all_cams, all_ids = set(), set()
        for s in snapshots:
            all_cams.update(s["views"].keys())
            for t in s["obs"].values():
                all_ids.update(t.keys())
        return {"ok": True, "root": root_id, "marker_mm": marker,
                "world_ref": ref_info,
                "anchor": int(anchor[0]), "anchor_snap": int(anchor[1]),
                "rms_px": rms, "snap_count": len(snapshots),
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
    def _frame_from_camera(P, ref, mode="topdown", yaw_quadrant=0):
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
                   "mode": mode, "yaw_quadrant": int(yaw_quadrant) % 4}

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
